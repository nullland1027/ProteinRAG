# ProteinRAG

Blazing-fast protein sequence similarity search using learned protein language model embeddings + Milvus Lite vector database. Move beyond traditional sequence alignment to embedding‑based semantic retrieval.

## ✨ Key Features
- **ESM2 embeddings** (default: `facebook/esm2_t6_8M_UR50D`) for compact sequence representations (320-dim)
- **Milvus Lite** local file mode (no external server required)
- **Automatic collection & index creation** on first use
- **Streamlit web UI** for uploading FASTA files and running similarity search
- **Top‑K nearest neighbor retrieval** with configurable K (1–20)
- **Secure destructive operation** (database clear requires 6‑digit verification)
- **On-demand model loading** (lazy until first embedding generation) + option to preload
- **Programmatic Python API** (reusable service class `ProteinRAGService`)

## 🧱 Architecture Overview
```
app.py (Streamlit UI)
 ├─ uses main.get_protein_service() -> singleton ProteinRAGService
main.py
 ├─ ProteinRAGService
 │   ├─ connect_database() -> Milvus Lite (file-backed)
 │   ├─ create_collection_if_not_exists()
 │   ├─ load_esm2_model() (ESM2 via HuggingFace transformers)
 │   ├─ process_fasta_file() -> parse + embed
 │   ├─ insert_proteins() -> vector + scalar fields
 │   ├─ search_similar_proteins() -> ANN search (L2)
 │   ├─ clear_database() -> drop & recreate collection
 │   └─ get_collection_stats()
create_db.py (optional standalone workflow / validation)
```

## 📂 Project Structure
```
ProteinRAG/
├─ app.py                # Streamlit application
├─ main.py               # Core service (Milvus + embeddings)
├─ create_db.py          # Optional DB init/validation workflow
├─ test_milvus_lite.py   # Basic connectivity & collection test
├─ requirements.txt      # Python dependencies
├─ milvus_lite.db        # Milvus Lite local storage (created after first run)
├─ README.md             # Documentation
└─ .env (optional)       # Environment variables (not tracked)
```

## 🛠 Requirements
- Python 3.10+ (tested with 3.12)
- macOS / Linux (Windows should work but not yet validated)
- Sufficient RAM (embeddings are small; model ~15–150MB depending on variant)
- (Optional) GPU w/ CUDA for larger ESM models (current default is small CPU-friendly)

## 📦 Installation
```bash
# 1. Clone repository
git clone <your-fork-or-origin-url> ProteinRAG
cd ProteinRAG

# 2. Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Upgrade pip & install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## 🔐 Environment Variables (.env)
Create a `.env` file in the project root if you need to override defaults.

| Variable | Description | Default |
|----------|-------------|---------|
| `EMBEDDING_MODEL_PATH` | HuggingFace model ID or local path to ESM2 checkpoint | `facebook/esm2_t6_8M_UR50D` |

Example `.env`:
```
EMBEDDING_MODEL_PATH=facebook/esm2_t6_8M_UR50D
# or local path, e.g.
# EMBEDDING_MODEL_PATH=/data/models/esm2_t30_150M_UR50D
```

> If unset, the service falls back to the small ESM2-t6 8M model (fast & low memory). You may swap to larger variants for improved biological sensitivity.

## ▶️ Running the Web App
```bash
streamlit run app.py
```
Open the URL shown (typically http://localhost:8501).

### UI Workflow
1. Upload a FASTA (`.fasta`, `.fa`, `.txt`) file.
2. System parses sequences → generates ESM2 embeddings → inserts into Milvus Lite.
3. Switch to “Protein Sequence Search” tab and paste a query sequence.
4. Choose K (1–20) and run similarity search.
5. Review ranked hits with similarity scores (1 / (1 + L2 distance)).

## 🧪 Basic Test (Milvus Lite Sanity)
```bash
python test_milvus_lite.py
```
Expected: All steps succeed & stats printed.

## 🧬 Programmatic Usage
```python
from main import get_protein_service

service = get_protein_service()
service.initialize_database()          # Ensure DB + collection

# Insert from FASTA string
fasta_content = ">seq1\nMKT...\n>seq2\nGAVL..."
records = service.process_fasta_file(fasta_content)
service.insert_proteins(records)

# Search
results = service.search_similar_proteins("MKTVRQE...", top_k=5)
for r in results:
    print(r['protein_id'], r['similarity_score'])
```

## 📊 Data Model (Milvus Collection Fields)
| Field | Type | Notes |
|-------|------|-------|
| id | INT64 (auto) | Primary key |
| protein_id | VARCHAR(100) | FASTA record ID |
| sequence | VARCHAR(10000) | Raw amino acid sequence (truncated if pre-processing added) |
| description | VARCHAR(1000) | FASTA header description |
| length | INT64 | Sequence length used for display/validation |
| embedding | FLOAT_VECTOR(320) | ESM2 embedding (CLS token) |

## 🧮 Similarity Metric
- Embeddings compared with **L2 distance**
- UI displays a transformed similarity score: `similarity = 1 / (1 + distance)` (bounded (0,1])

## 🗃 Milvus Lite Notes
- Local file created automatically (default: `milvus_lite.db` in project root)
- No separate Milvus server process needed
- Index type: `AUTOINDEX` for compatibility (Lite mode)
- For server Milvus deployment you could switch to IVF/HNSW, but Lite restricts options

## 🧹 Clearing the Database
- Trigger via UI “Clear Database” button
- Requires 6-digit verification code (random each time)
- Drops & recreates collection (schema preserved)

## 🚀 Performance Tips
| Strategy | Benefit |
|----------|---------|
| Batch larger FASTA uploads | Fewer model forward passes overhead |
| Preload model (call `load_esm2_model`) | Avoid first-request latency |
| Use larger ESM2 variant | Better semantic sensitivity (slower) |
| Pin Python + deps | Reproducibility |

## 🧷 Edge Cases & Handling
| Case | Behavior |
|------|----------|
| Empty FASTA | Inserts nothing; warning in logs |
| Invalid characters | Currently accepted by tokenizer (may produce lower quality embeddings); consider validation if needed |
| Duplicate protein_id | Each treated as separate row; add uniqueness logic if desired |
| Very long sequence (>1000 aa) | Truncated to 1000 aa before embedding (adjust in code) |
| DB file deleted during run | Reconnect + recreation will occur on insert/search attempt |

## 🛠 Development Workflow
```bash
# Run app
streamlit run app.py

# Run tests / sanity
python test_milvus_lite.py

# Lint (optional if you add tools like ruff/flake8)
ruff .   # example
```

Recommended branching model:
- `main`: stable
- feature branches: `feat/<name>`
- bugfix branches: `fix/<issue>`

## 🐞 Troubleshooting
| Symptom | Possible Cause | Fix |
|---------|----------------|-----|
| "Failed to connect database" | File permission / path invalid | Ensure write permission / delete stale lock file |
| Model download slow | Network / HuggingFace rate limit | Pre-download model & set `EMBEDDING_MODEL_PATH` to local path |
| High latency first request | Lazy model load | Trigger `service.load_esm2_model()` at startup |
| Search returns empty | Empty collection | Upload FASTA first |
| Clear DB not working | Wrong verification code | Re-enter 6-digit code |

## 🔄 Future Enhancements (Ideas)
- Add sequence validity filter (A,C,D,... only)
- Support larger ESM2 variants with optional GPU
- Add cosine similarity option
- Export results as CSV/JSON
- REST API wrapper (FastAPI) for non-UI integration
- Duplicate detection / merging

## 🔒 Security Considerations
- No authentication layer (local dev usage assumed)
- Add user/session auth before deploying remotely
- Sensitive model paths not committed (use `.env`)

## 📄 License
See [LICENSE](./LICENSE).

## 🙌 Contributions
Issues & PRs welcome: fixes, performance, validation, features.

---
**If you use this project in research or production, consider stating the original ESM & Milvus projects in attribution.**
