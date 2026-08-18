# ---- Cloud Run deployment for voice-rag backend ----
FROM python:3.11-slim

WORKDIR /app

# Build deps needed by faiss / torch native wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch explicitly (the default PyPI wheel pulls CUDA
# libs you don't need and bloats the image by ~2GB+)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ ./src/

# Pre-download the embedding model INTO the image at build time.
# This avoids a Hugging Face download on every cold start (slow + a
# potential point of failure) and means first-request latency is just
# model load-from-disk, not load-from-network.
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

# Fix for the import-path issue you hit on Render:
# api.py does 'from harness import ...' and harness.py does
# 'from guardrails import ...' — these are siblings inside src/, so we
# need src/ on the Python path AND run uvicorn from inside src/
# rather than as 'src.api:app' from the repo root.
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Cloud Run injects PORT at runtime (defaults to 8080) — the app MUST
# bind to this, not a hardcoded port
ENV PORT=8080
EXPOSE 8080

WORKDIR /app/src
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT}"]