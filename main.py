import numpy as np
import os
from typing import List, Dict, Any, Optional
from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType
from Bio import SeqIO
from io import StringIO
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProteinRAGService:
    """Core Protein RAG service using Milvus Lite"""

    def __init__(self,
                 db_path: str = "./milvus_lite.db",
                 collection_name: str = "protein_collection"):
        """
        Initialize service - Milvus Lite

        Args:
            db_path: Milvus Lite database file path
            collection_name: Collection name
        """
        # Ensure database file path correctness
        if not db_path.endswith('.db'):
            db_path = os.path.join(db_path, 'milvus_lite.db')

        self.db_path = os.path.abspath(db_path)
        self.collection_name = collection_name
        self.collection = None
        self.embedding_model = None
        self.tokenizer = None
        self._db_connected = False
        self._model_loaded = False

        # Ensure directory exists
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        logger.info(f"Database file path: {self.db_path}")

    def connect_database(self) -> bool:
        """Connect to Milvus Lite database"""
        if self._db_connected:
            return True

        try:
            # Disconnect existing connection if any
            try:
                connections.disconnect("default")
                logger.info("Previous connection disconnected")
            except:
                pass  # Ignore if not connected

            # Ensure directory exists
            db_dir = os.path.dirname(self.db_path)
            os.makedirs(db_dir, exist_ok=True)

            # Connect to Milvus Lite
            uri_path = self.db_path
            connections.connect(
                "default",
                uri=uri_path
            )
            logger.info(f"Successfully connected to Milvus Lite: {self.db_path}")
            self._db_connected = True
            return True
        except Exception as e:
            logger.error(f"Failed to connect database: {str(e)}")
            # Try alternative connection style
            try:
                connections.connect(
                    "default",
                    uri=f"file:{self.db_path}"
                )
                logger.info(f"Connected using file: prefix: {self.db_path}")
                self._db_connected = True
                return True
            except Exception as e2:
                logger.error(f"Second connection attempt failed: {str(e2)}")
                try:
                    # Third attempt: standard server mode
                    connections.connect(
                        "default",
                        host="localhost",
                        port="19530"
                    )
                    logger.info("Connected using standard server mode")
                    self._db_connected = True
                    return True
                except Exception as e3:
                    logger.error(f"All connection attempts failed: {str(e3)}")
                    return False

    def create_collection_if_not_exists(self) -> bool:
        """Create collection if it does not exist"""
        if not self._db_connected:
            if not self.connect_database():
                return False

        try:
            if utility.has_collection(self.collection_name):
                logger.info(f"Collection {self.collection_name} already exists, using it")
                self.collection = Collection(self.collection_name)
                self.collection.load()
                return True

            fields = [
                FieldSchema(
                    name="id",
                    dtype=DataType.INT64,
                    is_primary=True,
                    auto_id=True,
                    description="Primary Key"
                ),
                FieldSchema(
                    name="protein_id",
                    dtype=DataType.VARCHAR,
                    max_length=100,
                    description="Protein Identifier"
                ),
                FieldSchema(
                    name="sequence",
                    dtype=DataType.VARCHAR,
                    max_length=10000,
                    description="Protein Sequence"
                ),
                FieldSchema(
                    name="description",
                    dtype=DataType.VARCHAR,
                    max_length=1000,
                    description="Protein Description"
                ),
                FieldSchema(
                    name="length",
                    dtype=DataType.INT64,
                    description="Sequence Length"
                ),
                FieldSchema(
                    name="embedding",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=320,  # ESM2_t6_8M embedding dimension
                    description="ESM2 Protein Embedding Vector"
                )
            ]

            schema = CollectionSchema(
                fields=fields,
                description="Protein sequences with ESM2 embeddings"
            )

            self.collection = Collection(
                name=self.collection_name,
                schema=schema
            )
            logger.info(f"Created new collection: {self.collection_name}")

            self._create_indexes()
            self.collection.load()
            logger.info(f"Collection {self.collection_name} loaded into memory")
            return True
        except Exception as e:
            logger.error(f"Failed to create collection: {str(e)}")
            return False

    def _create_indexes(self) -> bool:
        """Create indexes"""
        try:
            index_params = {
                "metric_type": "L2",
                "index_type": "AUTOINDEX",  # Milvus Lite supported
                "params": {}
            }

            logger.info("Creating index for embedding field...")
            self.collection.create_index(
                field_name="embedding",
                index_params=index_params
            )

            logger.info("Creating index for protein_id field...")
            try:
                self.collection.create_index(
                    field_name="protein_id",
                    index_params={"index_type": "INVERTED"}
                )
            except Exception as e:
                logger.warning(f"protein_id index creation failed (ignored): {str(e)}")

            logger.info("Indexes created")
            return True
        except Exception as e:
            logger.error(f"Failed to create index: {str(e)}")
            logger.warning("Index creation failed but continuing with core functionality...")
            return True

    def initialize_database(self) -> bool:
        """Initialize database (connect and create collection if needed)"""
        if not self.connect_database():
            return False
        if not self.create_collection_if_not_exists():
            return False
        return True

    def load_esm2_model(self):
        """Load ESM2 model"""
        if self._model_loaded:
            return True
        try:
            import torch  # noqa: F401
            from transformers import EsmModel, EsmTokenizer
            model_name = "facebook/esm2_t6_8M_UR50D"
            self.tokenizer = EsmTokenizer.from_pretrained(model_name)
            self.embedding_model = EsmModel.from_pretrained(model_name)
            self.embedding_model.eval()
            logger.info("ESM2 model loaded successfully")
            self._model_loaded = True
            return True
        except Exception as e:
            logger.error(f"Failed to load ESM2 model: {str(e)}")
            return False

    def get_protein_embedding(self, sequence: str) -> Optional[np.ndarray]:
        """
        Get embedding for a protein sequence

        Args:
            sequence: Protein sequence

        Returns:
            Embedding vector or None
        """
        try:
            import torch  # noqa: F401
            max_length = 1000
            if len(sequence) > max_length:
                sequence = sequence[:max_length]
            inputs = self.tokenizer(sequence, return_tensors="pt", truncation=True, max_length=1024)
            with torch.no_grad():
                outputs = self.embedding_model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :].numpy()
            return embedding.flatten()
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            return None

    def process_fasta_file(self, file_content: str) -> List[Dict[str, Any]]:
        """
        Process FASTA file content

        Args:
            file_content: FASTA file content

        Returns:
            List of processed protein data
        """
        if not self._model_loaded:
            if not self.load_esm2_model():
                logger.error("Model load failed, cannot process file")
                return []

        protein_data = []
        try:
            for record in SeqIO.parse(StringIO(file_content), "fasta"):
                protein_id = record.id
                sequence = str(record.seq)
                description = record.description
                embedding = self.get_protein_embedding(sequence)
                if embedding is None:
                    logger.warning(f"Skipping protein {protein_id}, embedding generation failed")
                    continue
                protein_data.append({
                    "protein_id": protein_id,
                    "sequence": sequence,
                    "description": description,
                    "length": len(sequence),
                    "embedding": embedding.tolist()
                })
            logger.info(f"Successfully processed {len(protein_data)} protein sequences")
            return protein_data
        except Exception as e:
            logger.error(f"Failed to process FASTA file: {str(e)}")
            return []

    def insert_proteins(self, protein_data: List[Dict[str, Any]]) -> int:
        """
        Insert protein data into database

        Args:
            protein_data: Protein data list

        Returns:
            Number of records inserted
        """
        if not protein_data:
            return 0
        if not self._db_connected:
            if not self.connect_database():
                logger.error("Database connection failed, cannot insert")
                return 0
        if not self.collection:
            if not self.create_collection_if_not_exists():
                logger.error("Collection creation failed, cannot insert")
                return 0
        try:
            logger.info(f"Preparing to insert {len(protein_data)} records")
            if protein_data:
                logger.debug(f"Sample record: {protein_data[0]})")
            entities = []
            for p in protein_data:
                entity = {
                    "protein_id": str(p["protein_id"]),
                    "sequence": str(p["sequence"]),
                    "description": str(p["description"]),
                    "length": int(p["length"]),
                    "embedding": p["embedding"]
                }
                entities.append(entity)
            result = self.collection.insert(entities)
            self.collection.flush()
            logger.info(f"Inserted {len(protein_data)} records into database")
            logger.debug(f"Insert result: {result}")
            return len(protein_data)
        except Exception as e:
            logger.error(f"Failed to insert data: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            if protein_data:
                logger.debug(f"Original sample: {protein_data[0]}")
                for key, value in protein_data[0].items():
                    logger.debug(f"  {key}: {type(value)} = {str(value)[:100]}...")
            return 0

    def search_similar_proteins(self,
                              query_sequence: str,
                              top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search similar protein sequences

        Args:
            query_sequence: Query sequence
            top_k: Number of results

        Returns:
            List of similar proteins
        """
        if not self._model_loaded:
            if not self.load_esm2_model():
                logger.error("Model load failed, cannot search")
                return []
        if not self._db_connected:
            if not self.connect_database():
                logger.error("Database connection failed, cannot search")
                return []
        if not self.collection:
            if not self.create_collection_if_not_exists():
                logger.error("Collection does not exist, cannot search")
                return []
        try:
            query_embedding = self.get_protein_embedding(query_sequence)
            if query_embedding is None:
                logger.error("Query embedding generation failed")
                return []
            search_params = {
                "metric_type": "L2",
                "params": {"ef": 128}
            }
            results = self.collection.search(
                data=[query_embedding.tolist()],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["protein_id", "sequence", "description", "length"]
            )
            similar_proteins = []
            for hits in results:
                for hit in hits:
                    similar_proteins.append({
                        "protein_id": hit.entity.get("protein_id"),
                        "sequence": hit.entity.get("sequence"),
                        "description": hit.entity.get("description"),
                        "length": hit.entity.get("length"),
                        "distance": hit.distance,
                        "similarity_score": 1 / (1 + hit.distance)
                    })
            logger.info(f"Search complete, found {len(similar_proteins)} similar proteins")
            return similar_proteins
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        if not self._db_connected:
            return {"total_proteins": 0, "collection_name": self.collection_name, "is_loaded": False}
        try:
            if not self.collection:
                if not self.create_collection_if_not_exists():
                    return {"total_proteins": 0, "collection_name": self.collection_name, "is_loaded": False}
            stats = {
                "total_proteins": self.collection.num_entities,
                "collection_name": self.collection_name,
                "is_loaded": True
            }
            return stats
        except Exception as e:
            logger.error(f"Failed to get stats: {str(e)}")
            return {"total_proteins": 0, "collection_name": self.collection_name, "is_loaded": False}

    def check_database_connection(self) -> Dict[str, Any]:
        """Check database connection state (non-invasive)"""
        return {
            "db_connected": self._db_connected,
            "model_loaded": self._model_loaded,
            "collection_name": self.collection_name
        }

    def clear_database(self) -> bool:
        """Clear all data from database"""
        if not self._db_connected:
            if not self.connect_database():
                logger.error("Database connection failed, cannot clear database")
                return False
        try:
            if not utility.has_collection(self.collection_name):
                logger.info("Collection does not exist, nothing to clear")
                return True
            utility.drop_collection(self.collection_name)
            logger.info(f"Dropped collection: {self.collection_name}")
            self.collection = None
            if self.create_collection_if_not_exists():
                logger.info("Re-created empty collection successfully")
                return True
            else:
                logger.error("Failed to recreate collection")
                return False
        except Exception as e:
            logger.error(f"Failed to clear database: {str(e)}")
            return False

# Global service instance
protein_service = None

def get_protein_service() -> ProteinRAGService:
    """Get singleton protein service instance"""
    global protein_service
    if protein_service is None:
        protein_service = ProteinRAGService()
    return protein_service

def initialize_service() -> bool:
    """Initialize service (connect to database at startup)"""
    service = get_protein_service()
    logger.info("Initializing Protein RAG service...")
    if not service.initialize_database():
        logger.error("Database initialization failed")
        return False
    logger.info("Protein RAG service initialization complete")
    return True
