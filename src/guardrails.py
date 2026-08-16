"""
Phase 4: Guardrails for the RAG pipeline.
"""

import re

# --- 1. Unsafe input filter ---
# Covers BOTH English and Hindi since our queries are Hindi.
UNSAFE_PATTERNS = [
    # English
    r"\bbomb\b", r"\bweapon\b", r"\bkill\b", r"\bsuicide\b",
    r"\bhack\b.*\bpassword\b", r"\bmake.*\bdrug\b",
    # Hindi (Devanagari)
    r"बम", r"हथियार", r"हत्या", r"आत्महत्या", r"मारने", r"मारना",
    r"नशीली दवा", r"बनाना.*सिखाओ",  # "teach me how to make ___" pattern
]

def is_unsafe_input(query: str) -> bool:
    query_lower = query.lower()
    return any(re.search(pattern, query_lower) for pattern in UNSAFE_PATTERNS)


# --- 2. Low-confidence / off-topic check ---
def is_low_confidence(retrieved_chunks: list[dict], threshold: float = 0.45) -> bool:
    if not retrieved_chunks:
        return True
    return retrieved_chunks[0]["score"] < threshold


# --- 3. Grounding check ---
# Two-tier: first check if the model already declined on its own (it was
# told to do this in the prompt) - if so, that's a valid, intentional
# refusal, not a hallucination. Only run the word-overlap check otherwise,
# and use a lenient threshold since Hindi word forms vary a lot.
DECLINE_PHRASES = [
    "पर्याप्त जानकारी नहीं", "जानकारी नहीं है", "नहीं पता", "मालूम नहीं",
    "उत्तर नहीं दे सकता", "जवाब नहीं दे सकता",
]

def model_declined(answer: str) -> bool:
    return any(phrase in answer for phrase in DECLINE_PHRASES)

def is_grounded(answer: str, retrieved_chunks: list[dict], min_overlap: float = 0.08) -> bool:
    if model_declined(answer):
        return True  # intentional decline, not a hallucination - counts as fine

    context_text = " ".join(c["text"] for c in retrieved_chunks)
    context_words = set(re.findall(r"\w+", context_text.lower()))
    answer_words = set(re.findall(r"\w+", answer.lower()))

    if not answer_words:
        return False

    overlap = len(answer_words & context_words) / len(answer_words)
    return overlap >= min_overlap


# --- Combined checks used by the pipeline ---
def check_input(query: str) -> dict:
    if is_unsafe_input(query):
        return {"ok": False, "reason": "unsafe_input"}
    return {"ok": True, "reason": None}


def check_retrieval(retrieved_chunks: list[dict]) -> dict:
    if is_low_confidence(retrieved_chunks):
        return {"ok": False, "reason": "low_confidence"}
    return {"ok": True, "reason": None}


def check_output(answer: str, retrieved_chunks: list[dict]) -> dict:
    if model_declined(answer):
        return {"ok": True, "reason": None, "self_declined": True}
    if not is_grounded(answer, retrieved_chunks):
        return {"ok": False, "reason": "not_grounded"}
    return {"ok": True, "reason": None}