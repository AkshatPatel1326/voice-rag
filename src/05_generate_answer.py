"""
Phase 3: Retrieve chunks + generate a grounded answer using Groq.

Run this from the project root:
    python src/05_generate_answer.py
"""

import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()  # reads your .env file

INDEX_PATH = "data/faiss_index.bin"
METADATA_PATH = "data/chunk_metadata.pkl"
TOP_K = 3
GROQ_MODEL = "llama-3.1-8b-instant"  # fast + free tier, good for our latency target

print("Loading index and metadata...")
index = faiss.read_index(INDEX_PATH)
with open(METADATA_PATH, "rb") as f:
    chunks = pickle.load(f)

print("Loading embedding model...")
embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def retrieve(query: str, k: int = TOP_K):
    query_vec = embed_model.encode([query], normalize_embeddings=True).astype("float32")
    scores, indices = index.search(query_vec, k)
    return [
        {"text": chunks[idx]["text"], "score": float(score)}
        for score, idx in zip(scores[0], indices[0])
    ]


def generate_answer(query: str, retrieved_chunks: list[dict]) -> str:
    context = "\n\n".join(f"[{i+1}] {c['text']}" for i, c in enumerate(retrieved_chunks))

    system_prompt = (
        "You are a helpful assistant answering questions in Hindi. "
        "Answer ONLY using the information in the provided context below. "
        "If the answer is not contained in the context, say clearly in Hindi "
        "that you don't have enough information to answer — do not guess or "
        "make anything up. Keep answers concise, 1-3 sentences."
    )

    user_prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,  # low temperature = more grounded, less creative drift
        max_tokens=300,
    )
    return response.choices[0].message.content


def ask(query: str):
    print(f"\nQuery: {query}")
    retrieved = retrieve(query)
    print(f"Retrieved {len(retrieved)} chunks (top score: {retrieved[0]['score']:.3f})")

    answer = generate_answer(query, retrieved)
    print(f"\nAnswer: {answer}")
    return answer


if __name__ == "__main__":
    # Test with a couple of real queries first
    from datasets import load_dataset

    dataset = load_dataset("ai4bharat/IndicMSMARCO", "hi", split="train")

    print("=== Testing generation on 2 sample queries ===")
    for i in [0, 1]:
        ask(dataset[i]["query"])
        print("-" * 60)

    # Interactive mode
    print("\n=== Try your own question (type 'exit' to quit) ===")
    while True:
        q = input("\nYour question (Hindi): ").strip()
        if q.lower() == "exit":
            break
        ask(q)