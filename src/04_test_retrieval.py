"""
Phase 2 - Step 2: Test retrieval - type a query, see what comes back.
"""

import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

INDEX_PATH = "data/faiss_index.bin"
METADATA_PATH = "data/chunk_metadata.pkl"
TOP_K = 3

print("Loading index and metadata...")
index = faiss.read_index(INDEX_PATH)
with open(METADATA_PATH, "rb") as f:
    chunks = pickle.load(f)

print("Loading embedding model...")
embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def search(query: str, k: int = TOP_K):
    query_vec = embed_model.encode([query], normalize_embeddings=True).astype("float32")
    scores, indices = index.search(query_vec, k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        chunk = chunks[idx]
        results.append({"score": float(score), "text": chunk["text"], "query_id": chunk["query_id"]})
    return results


dataset = load_dataset("ai4bharat/IndicMSMARCO", "hi", split="train")

print("\n=== Testing retrieval on 5 sample queries ===\n")
hits = 0
for i in range(5):
    query = dataset[i]["query"]
    correct_query_id = dataset[i]["query_id"]

    print(f"Query: {query}")

    results = search(query)
    top_result_correct = results[0]["query_id"] == correct_query_id
    hits += int(top_result_correct)

    for rank, r in enumerate(results, start=1):
        match = " <-- CORRECT PASSAGE" if r["query_id"] == correct_query_id else ""
        print(f"  #{rank} [score={r['score']:.3f}]{match}")
        print(f"       {r['text'][:100]}...")
    print()

print(f"Top-1 accuracy on this sample: {hits}/5")

print("=== Try your own query (type 'exit' to quit) ===")
while True:
    query = input("\nYour query (Hindi): ").strip()
    if query.lower() == "exit":
        break
    results = search(query)
    for rank, r in enumerate(results, start=1):
        print(f"  #{rank} [score={r['score']:.3f}] {r['text'][:100]}...")