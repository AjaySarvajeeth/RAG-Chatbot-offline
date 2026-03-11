# retriever.py (Updated and Optimized)

import os
import json
from pathlib import Path
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# ---------------- CONFIG ---------------- #
INDEX_DIR = Path(os.getenv("INDEX_DIR", "./index")).expanduser().resolve()
EMBEDDING_MODEL = Path(os.getenv("EMBEDDING_MODEL", "./models/all-MiniLM-L6-v2")).expanduser().resolve()
TOP_K = int(os.getenv("TOP_K", 6))

# Offline mode
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

# Device
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# ---------------- EMBEDDER ---------------- #
if not EMBEDDING_MODEL.exists():
    raise FileNotFoundError(f"Embedding model not found: {EMBEDDING_MODEL}")

embedder = SentenceTransformer(str(EMBEDDING_MODEL), device=DEVICE, trust_remote_code=False)

# ---------------- INDEX + META ---------------- #
_index_path = INDEX_DIR / "faiss.index"
_meta_path = INDEX_DIR / "meta.jsonl"

if not _index_path.exists():
    raise FileNotFoundError(f"Missing FAISS index: {_index_path}")
if not _meta_path.exists():
    raise FileNotFoundError(f"Missing metadata: {_meta_path}")

_index = faiss.read_index(str(_index_path))

# --- MEMORY OPTIMIZATION ---
# Instead of loading all JSON, just read the lines into memory.
# This is significantly more RAM-efficient.
with open(_meta_path, "r", encoding="utf-8") as f:
    _meta_lines = f.readlines()
# --- END OPTIMIZATION ---

# ---------------- RETRIEVAL ---------------- #
def retrieve(query: str, k: int | None = None, normalize_embeddings: bool = False):
    """Retrieve top-k results for query."""
    total = _index.ntotal
    if total <= 0:
        return []
    k = min(int(k or TOP_K), total)

    q_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=normalize_embeddings)
    D, I = _index.search(np.asarray(q_emb, dtype="float32"), k)

    results = []
    for pos, idx in enumerate(I[0]):
        # --- MEMORY OPTIMIZATION ---
        # Parse the JSON only for the lines we actually need.
        if 0 <= idx < len(_meta_lines):
            m = json.loads(_meta_lines[idx])
            # --- END OPTIMIZATION ---
            dist = float(D[0][pos])
            score = 1.0 / (1.0 + dist)
            m["_score"] = score
            results.append(m)
    return results

# ---------------- RETRIEVER FACTORY ---------------- #
def get_retriever():
    """Returns a function that takes a query and returns plain text chunks."""
    def _retriever(query: str, top_k: int = TOP_K):
        results = retrieve(query, k=top_k)
        texts = [r.get("text", "") for r in results if "text" in r]
        return texts
    return _retriever