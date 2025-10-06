#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Protein RAG system database creation workflow
Initialize Milvus database, create collection, build indexes
"""

import sys
import logging
import os
from typing import Dict, Any
from pymilvus import (
    connections, Collection, FieldSchema, CollectionSchema,
    DataType, utility
)
from langchain.embeddings import HuggingFaceEmbeddings
import argparse
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('db_creation.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Read embedding model path/name from environment variable with fallback.
# Can be either a local directory path or a HuggingFace model ID.
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "facebook/esm2_t6_8M_UR50D")
if not EMBEDDING_MODEL_PATH:
    logger.warning("EMBEDDING_MODEL_PATH not set; using default facebook/esm2_t6_8M_UR50D")
    EMBEDDING_MODEL_PATH = "facebook/esm2_t6_8M_UR50D"

class ProteinDatabaseCreator:
    """Protein database creator"""

    def __init__(self,
                 host: str = "localhost",
                 port: str = "19530",
                 collection_name: str = "protein_collection",
                 embedding_model: str = EMBEDDING_MODEL_PATH):
        """
        Initialize database creator

        Args:
            host: Milvus server host
            port: Milvus server port
            collection_name: Collection name
            embedding_model: Embedding model path/name (default ESM2)
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self.collection = None
        self.embedding_model = None

    def connect_to_milvus(self) -> bool:
        """Connect to Milvus"""
        try:
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port
            )
            logger.info(f"Connected to Milvus {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect Milvus: {str(e)}")
            return False

    def load_embedding_model(self) -> bool:
        """Load embedding model"""
        try:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name
            )
            logger.info("Embedding model loaded")
            return True
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            return False

    def create_collection_schema(self) -> CollectionSchema:
        """Create collection schema"""
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
        return schema

    def create_collection(self, drop_if_exists: bool = False) -> bool:
        """Create collection"""
        try:
            if utility.has_collection(self.collection_name):
                if drop_if_exists:
                    logger.info(f"Dropping existing collection: {self.collection_name}")
                    utility.drop_collection(self.collection_name)
                else:
                    logger.info(f"Collection {self.collection_name} exists, using it")
                    self.collection = Collection(self.collection_name)
                    return True
            schema = self.create_collection_schema()
            self.collection = Collection(
                name=self.collection_name,
                schema=schema
            )
            logger.info(f"Created collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to create collection: {str(e)}")
            return False

    def create_index(self) -> bool:
        """Create index for vector field"""
        try:
            index_params = {
                "metric_type": "L2",
                "index_type": "HNSW",
                "params": {
                    "M": 16,
                    "efConstruction": 200
                }
            }
            logger.info("Creating index for embedding field...")
            self.collection.create_index(
                field_name="embedding",
                index_params=index_params
            )
            logger.info("Creating index for protein_id field...")
            self.collection.create_index(
                field_name="protein_id",
                index_params={"index_type": "INVERTED"}
            )
            logger.info("Indexes created")
            return True
        except Exception as e:
            logger.error(f"Failed to create index: {str(e)}")
            return False

    def load_collection(self) -> bool:
        """Load collection into memory"""
        try:
            self.collection.load()
            logger.info(f"Collection {self.collection_name} loaded")
            return True
        except Exception as e:
            logger.error(f"Failed to load collection: {str(e)}")
            return False

    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection info"""
        if not self.collection:
            return {}
        try:
            info = {
                "name": self.collection.name,
                "description": self.collection.description,
                "num_entities": self.collection.num_entities,
                "schema": {
                    field.name: {
                        "type": field.dtype.name,
                        "description": field.description
                    } for field in self.collection.schema.fields
                },
                "indexes": []
            }
            for field in self.collection.schema.fields:
                try:
                    index_info = self.collection.index(field.name)
                    if index_info:
                        info["indexes"].append({
                            "field": field.name,
                            "index": index_info.params
                        })
                except:
                    pass
            return info
        except Exception as e:
            logger.error(f"Failed to get collection info: {str(e)}")
            return {}

    def validate_database(self) -> bool:
        """Validate database settings"""
        try:
            if not self.connect_to_milvus():
                return False
            if not utility.has_collection(self.collection_name):
                logger.error(f"Collection {self.collection_name} does not exist")
                return False
            collection = Collection(self.collection_name)
            try:
                collection.load()
                logger.info("Collection loaded successfully")
            except Exception as e:
                logger.warning(f"Issue loading collection: {str(e)}")
            logger.info("Database validation passed")
            return True
        except Exception as e:
            logger.error(f"Database validation failed: {str(e)}")
            return False

    def run_workflow(self, drop_if_exists: bool = False) -> bool:
        """Run full database creation workflow"""
        logger.info("Starting database creation workflow...")
        if not self.connect_to_milvus():
            return False
        if not self.load_embedding_model():
            return False
        if not self.create_collection(drop_if_exists):
            return False
        if not self.create_index():
            return False
        if not self.load_collection():
            return False
        info = self.get_collection_info()
        if info:
            logger.info(f"Collection info: {info}")
        logger.info("Database creation workflow completed")
        return True


def main():
    """Main entry"""
    parser = argparse.ArgumentParser(description="Protein RAG DB creation tool")
    parser.add_argument("--host", default="localhost", help="Milvus server host")
    parser.add_argument("--port", default="19530", help="Milvus server port")
    parser.add_argument("--collection", default="protein_collection", help="Collection name")
    parser.add_argument("--model", default="facebook/esm2_t6_8M_UR50D", help="Embedding model name")
    parser.add_argument("--drop", action="store_true", help="Drop existing collection if present")
    parser.add_argument("--validate", action="store_true", help="Only validate database setup")

    args = parser.parse_args()

    creator = ProteinDatabaseCreator(
        host=args.host,
        port=args.port,
        collection_name=args.collection,
        embedding_model=args.model
    )

    if args.validate:
        success = creator.validate_database()
        if success:
            print("✅ Database validation succeeded")
        else:
            print("❌ Database validation failed")
        sys.exit(0 if success else 1)

    success = creator.run_workflow(drop_if_exists=args.drop)
    if success:
        print("✅ Database created successfully!")
        print(f"📊 Collection: {args.collection}")
        print(f"🔗 Endpoint: {args.host}:{args.port}")
    else:
        print("❌ Database creation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
