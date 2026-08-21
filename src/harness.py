"""
Phase 5: Harness - structured I/O, retries, and error handling
around the guarded RAG pipeline.
"""

import os
import time
import pickle
import logging
from typing import Optional

import faiss
from groq import Groq
from dotenv import load_dotenv
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from guardrails import check_input, check_retrieval, check_output

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rag_harness")

INDEX_PATH = "data/faiss_index.bin"
METADATA_PATH = "data/chunk_metadata.pkl"
TOP_K = 3
GROQ_MODEL = "openai/gpt-oss-20b"

# "torch" (default): full sentence-transformers + PyTorch, used for local dev/
#   testing/benchmarking scripts where RAM isn't a constraint.
# "onnx": lightweight onnxruntime-based embedder, used on Render (set via the
#   EMBED_BACKEND environment variable) where the free tier's 512MB RAM can't
#   fit full PyTorch + the float32 model.
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "torch")

REFUSAL_MESSAGES = {
    "unsafe_input": "मुझे खेद है, मैं इस प्रकार के प्रश्न का उत्तर नहीं दे सकता।",
    "low_confidence": "मेरे पास इस प्रश्न का उत्तर देने के लिए पर्याप्त जानकारी नहीं है।",
    "not_grounded": "मुझे खेद है, मैं इस प्रश्न का विश्वसनीय उत्तर नहीं दे पा रहा।",
    "pipeline_error": "क्षमा करें, कुछ तकनीकी समस्या हुई। कृपया दोबारा प्रयास करें।",
}


# ---------- Structured I/O schemas ----------

class QueryRequest(BaseModel):
    query: str


class PipelineResponse(BaseModel):
    query: str
    answer: str
    status: str                    # "answered" | "refused" | "error"
    reason: Optional[str] = None
    chunks_used: int = 0
    top_score: Optional[float] = None
    latency_ms: Optional[float] = None
    retries_used: int = 0


# ---------- The harness itself ----------

class RAGHarness:
    def __init__(self):
        logger.info("Loading index, metadata, embedding model...")
        self.index = faiss.read_index(INDEX_PATH)
        with open(METADATA_PATH, "rb") as f:
            self.chunks = pickle.load(f)

        if EMBED_BACKEND == "onnx":
            logger.info("Using lightweight ONNX embedding backend (deployment mode).")
            from onnx_embedder import OnnxQueryEmbedder
            self.embed_model = OnnxQueryEmbedder()
        else:
            logger.info("Using full sentence-transformers/PyTorch embedding backend (local mode).")
            from sentence_transformers import SentenceTransformer
            self.embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        logger.info("Harness ready.")

    def _retrieve(self, query: str, k: int = TOP_K):
        query_vec = self.embed_model.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = self.index.search(query_vec, k)
        return [
            {"text": self.chunks[idx]["text"], "score": float(score)}
            for score, idx in zip(scores[0], indices[0])
        ]

    # Retries on network/API failures specifically - not on bad input,
    # since retrying a malformed request won't help.
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _generate_answer(self, query: str, retrieved_chunks: list[dict]) -> str:
        context = "\n\n".join(f"[{i+1}] {c['text']}" for i, c in enumerate(retrieved_chunks))
        system_prompt = (
            "You are a helpful assistant answering questions in Hindi. "
            "Use the provided context to answer the question. If the context "
            "contains information related to the question, use it to give the "
            "best possible answer, even if it's not a perfect or complete match. "
            "Only say in Hindi that you don't have enough information if the "
            "context is genuinely unrelated to the question. Keep answers concise."
        )
        user_prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

        response = self.groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=300,
        )
        return response.choices[0].message.content

    def run(self, request: QueryRequest) -> PipelineResponse:
        """Structured-input, structured-output entry point with full
        error recovery - never raises, always returns a valid response."""
        start = time.perf_counter()
        query = request.query

        try:
            # Guardrail 1: unsafe input
            input_check = check_input(query)
            if not input_check["ok"]:
                logger.info(f"Blocked (unsafe_input): {query}")
                return self._refused(query, input_check["reason"], start)

            # Retrieve
            retrieved = self._retrieve(query)

            # Guardrail 2: low confidence
            retrieval_check = check_retrieval(retrieved)
            if not retrieval_check["ok"]:
                logger.info(f"Blocked (low_confidence): {query}")
                return self._refused(query, retrieval_check["reason"], start,
                                       chunks_used=0, top_score=retrieved[0]["score"] if retrieved else None)

            # Generate (with automatic retry on failure, built into _generate_answer)
            answer = self._generate_answer(query, retrieved)

            # Guardrail 3: grounding
            output_check = check_output(answer, retrieved)
            if not output_check["ok"]:
                logger.info(f"Blocked (not_grounded): {query}")
                return self._refused(query, output_check["reason"], start,
                                       chunks_used=len(retrieved), top_score=retrieved[0]["score"])

            latency = (time.perf_counter() - start) * 1000
            logger.info(f"Answered in {latency:.1f}ms: {query}")
            return PipelineResponse(
                query=query, answer=answer, status="answered",
                chunks_used=len(retrieved), top_score=retrieved[0]["score"],
                latency_ms=latency,
            )

        except Exception as e:
            # Catches: Groq API down, network failure, embedding failure,
            # or anything unexpected - pipeline degrades gracefully
            # instead of crashing the whole app.
            logger.error(f"Pipeline error for query '{query}': {e}")
            latency = (time.perf_counter() - start) * 1000
            return PipelineResponse(
                query=query, answer=REFUSAL_MESSAGES["pipeline_error"],
                status="error", reason=str(e), latency_ms=latency,
            )

    def _refused(self, query, reason, start, chunks_used=0, top_score=None) -> PipelineResponse:
        latency = (time.perf_counter() - start) * 1000
        return PipelineResponse(
            query=query, answer=REFUSAL_MESSAGES[reason], status="refused",
            reason=reason, chunks_used=chunks_used, top_score=top_score, latency_ms=latency,
        )