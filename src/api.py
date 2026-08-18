"""
Phase 8: FastAPI backend for the voice-RAG pipeline.

Exposes two endpoints for the frontend (built separately with v0/Lovable, deployed
on Vercel) to call:
  POST /ask-voice   multipart/form-data, field "audio" -> transcribes + runs pipeline
  POST /ask-text     JSON {"query": "..."}             -> runs pipeline directly

Run locally from the project root:
    uvicorn src.api:app --reload --port 8000
(or, from inside src/:  uvicorn api:app --reload --port 8000)

Deploy on Google Cloud Run:
  - Handled entirely by the Dockerfile at the repo root (see Dockerfile).
  - It runs `uvicorn api:app --host 0.0.0.0 --port $PORT` from inside src/,
    with src/ added to PYTHONPATH, so the sys.path.insert below is a
    belt-and-suspenders fallback, not strictly required by that path.
  - Deploy with:
      gcloud run deploy ragtag-goa --source . --region asia-south1 \
        --memory 2Gi --cpu 1 --allow-unauthenticated \
        --set-env-vars GROQ_API_KEY=...,ELEVENLABS_API_KEY=...
  - GROQ_API_KEY and ELEVENLABS_API_KEY are passed via --set-env-vars above
    (Cloud Run's equivalent of Render's Environment Variables panel).
"""

import os
import sys
import tempfile

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Make sibling imports (harness, stt, and harness's own "from guardrails import ...")
# work regardless of whether this is run as "python src/api.py", as the package
# "src.api:app", or as "api:app" from inside src/ (how the Cloud Run Dockerfile
# launches it).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harness import RAGHarness, QueryRequest
from stt import transcribe_audio

app = FastAPI(title="Voice RAG API")

# Allow the Vercel frontend to call this API from the browser.
# Once you have your real Vercel domain, replace "*" with it for tighter security,
# e.g. ["https://your-app.vercel.app"].
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading RAG harness (index, embedding model, Groq client)...")
harness = RAGHarness()
print("Ready.")


class TextQuery(BaseModel):
    query: str


class PipelineOutput(BaseModel):
    transcript: str
    answer: str
    status: str
    reason: str | None = None
    chunks_used: int = 0
    top_score: float | None = None
    latency_ms: float | None = None
    retries_used: int = 0


def _run(query_text: str) -> PipelineOutput:
    response = harness.run(QueryRequest(query=query_text))
    return PipelineOutput(
        transcript=query_text,
        answer=response.answer,
        status=response.status,
        reason=response.reason,
        chunks_used=response.chunks_used,
        top_score=response.top_score,
        latency_ms=response.latency_ms,
        retries_used=response.retries_used,
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask-text", response_model=PipelineOutput)
def ask_text(body: TextQuery):
    return _run(body.query)


@app.post("/ask-voice", response_model=PipelineOutput)
async def ask_voice(audio: UploadFile = File(...)):
    # Save the uploaded audio to a temp file since transcribe_audio() expects a path.
    suffix = os.path.splitext(audio.filename or "")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        transcript = transcribe_audio(tmp_path)
    finally:
        os.remove(tmp_path)

    return _run(transcript)