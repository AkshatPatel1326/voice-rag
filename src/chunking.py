"""
Phase 1 - Step 2: Three chunking strategies.
"""

import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np


# ---------- Strategy 1: Fixed-size with overlap ----------

def fixed_size_chunk(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """Splits text into overlapping fixed-size chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", "।", ". ", " ", ""],  # "।" = Hindi sentence terminator (danda)
    )
    return splitter.split_text(text)


# ---------- Strategy 2: Semantic chunking ----------

def _split_into_sentences(text: str) -> list[str]:
    """Splits Hindi/English mixed text into sentences using Devanagari
    and Latin sentence terminators (।, ., !, ?)."""
    sentences = re.split(r"(?<=[।.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def semantic_chunk(text: str, embed_model: SentenceTransformer, similarity_threshold: float = 0.6) -> list[str]:
    """
    Splits text into sentences, embeds each, and breaks into a new
    chunk wherever similarity to the previous sentence drops below
    the threshold (i.e. the topic shifts).
    """
    sentences = _split_into_sentences(text)
    if len(sentences) <= 1:
        return sentences

    embeddings = embed_model.encode(sentences, normalize_embeddings=True)

    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        similarity = float(np.dot(embeddings[i], embeddings[i - 1]))
        if similarity < similarity_threshold:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# ---------- Strategy 3: Metadata-aware chunking ----------

def metadata_aware_chunk(text: str, metadata: dict, chunk_size: int = 300, overlap: int = 50) -> list[dict]:
    """
    Same splitting as Strategy 1, but returns dicts with metadata
    attached to every chunk.
    """
    raw_chunks = fixed_size_chunk(text, chunk_size=chunk_size, overlap=overlap)
    return [
        {
            "text": chunk,
            "chunk_index": i,
            "total_chunks": len(raw_chunks),
            **metadata,
        }
        for i, chunk in enumerate(raw_chunks)
    ]