"""
Phase 2 - Step 1: Embed chunks and build a FAISS vector index.

Run this from the project root:
    python src/03_build_index.py
"""

import json
import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = "data/chunks_metadata_aware.jsonl"  # richest metadata, use this for retrieval
INDEX_PATH = "data/faiss_index.bin"
METADATA_PATH = "data/chunk_metadata.pkl"

# Load chunks we built in Phase 1
print(f"Loading chunks from {CHUNKS_FILE}...")
chunks = []
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        chunks.append(json.loads(line))
print(f"Loaded {len(chunks)} chunks")

# Same embedding model we used for semantic chunking - keeps everything consistent
print("Loading embedding model...")
embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Embed all chunk texts in one batch (much faster than one at a time)
texts = [c["text"] for c in chunks]
print(f"Embedding {len(texts)} chunks...")
embeddings = embed_model.encode(
    texts,
    normalize_embeddings=True,   # so we can use cosine similarity via dot product
    show_progress_bar=True,
    batch_size=64,
)
embeddings = np.array(embeddings).astype("float32")

# Build FAISS index - IndexFlatIP does exact search using inner product
# (= cosine similarity since our vectors are normalized). Exact search is
# fine at this scale (a few thousand chunks) and keeps retrieval simple.
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)
print(f"Built FAISS index with {index.ntotal} vectors, dimension {dimension}")

# Save the index itself
faiss.write_index(index, INDEX_PATH)
print(f"Saved index -> {INDEX_PATH}")

# Save the chunk metadata separately, in the SAME ORDER as the index,
# so index position i always maps to chunks[i]
with open(METADATA_PATH, "wb") as f:
    pickle.dump(chunks, f)
print(f"Saved chunk metadata -> {METADATA_PATH}")

print("\nDone. Index is ready for retrieval testing.")