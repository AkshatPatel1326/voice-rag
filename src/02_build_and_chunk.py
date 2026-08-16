"""
Phase 1 - Step 3: Build the corpus and run all 3 chunking strategies.

Run this from the project root:
    python src/02_build_and_chunk.py
"""

import json
import os
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from chunking import fixed_size_chunk, semantic_chunk, metadata_aware_chunk

LANGUAGE = "hi"
OUTPUT_DIR = "data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Loading IndicMSMARCO ({LANGUAGE})...")
dataset = load_dataset("ai4bharat/IndicMSMARCO", LANGUAGE, split="train")

print("Loading embedding model for semantic chunking (small model, downloads once)...")
embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

fixed_chunks = []
semantic_chunks = []
metadata_chunks = []

print(f"Chunking {len(dataset)} passages with all 3 strategies...")
for row in dataset:
    passage_text = row["passage"]
    base_meta = {
        "passage_id": row["passage_id"],
        "query_id": row["query_id"],
        "title": row["title"],
        "source": row["source"],
    }

    for i, chunk in enumerate(fixed_size_chunk(passage_text)):
        fixed_chunks.append({"text": chunk, "chunk_index": i, **base_meta})

    for i, chunk in enumerate(semantic_chunk(passage_text, embed_model)):
        semantic_chunks.append({"text": chunk, "chunk_index": i, **base_meta})

    metadata_chunks.extend(metadata_aware_chunk(passage_text, base_meta))

for name, chunks in [
    ("fixed", fixed_chunks),
    ("semantic", semantic_chunks),
    ("metadata_aware", metadata_chunks),
]:
    path = os.path.join(OUTPUT_DIR, f"chunks_{name}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"Saved {len(chunks)} chunks -> {path}")

print("\n--- Example: same passage, 3 strategies ---")
sample_passage = dataset[0]["passage"]
print(f"\nOriginal passage ({len(sample_passage)} chars):\n{sample_passage}\n")

print(f"Fixed-size chunks ({len(fixed_size_chunk(sample_passage))}):")
for c in fixed_size_chunk(sample_passage):
    print(f"  - [{len(c)} chars] {c[:80]}...")

print(f"\nSemantic chunks ({len(semantic_chunk(sample_passage, embed_model))}):")
for c in semantic_chunk(sample_passage, embed_model):
    print(f"  - [{len(c)} chars] {c[:80]}...")