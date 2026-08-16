# Voice-Enabled RAG Model — HH Goa 2026 Task #2

Speak a question → transcribed → retrieved from IndicMSMARCO (Hindi) → grounded answer.

Deadline: Aug 22, 2026, 11:59 PM IST. `#RAGInGoa` required on all promo posts.

## Stack (free-first)
- **STT**: ElevenLabs (Scribe v1) — free tier
  (originally planned Sarvam AI, switched due to 0 free credits on Sarvam)
- **Embeddings**: sentence-transformers, `paraphrase-multilingual-MiniLM-L12-v2` (local, free)
- **Vector DB**: FAISS, `IndexFlatIP` (local, free)
- **LLM**: Groq, `llama-3.1-8b-instant` (free tier)
- **Harness**: pydantic + tenacity retries + logging

## Dataset
`ai4bharat/IndicMSMARCO`, `hi` config, 1000 rows.
(Originally planned `ai4bharat/MSMARCO-XI` — swapped due to size, 3.79GB was too
large for available disk space.)

## Project structure
```
voice-rag/
├── data/           # dataset + saved vector index + benchmark results
├── src/            # pipeline code
├── notebooks/      # exploration / testing
├── tests/          # latency + guardrail tests
├── static/         # simple web UI (mic button)
├── requirements.txt
└── .env.example
```

## Setup
1. `python -m venv venv && source venv/bin/activate` (Windows: `venv\Scripts\activate`)
2. `pip install -r requirements.txt`
3. `cp .env.example .env` and fill in your API keys (`GROQ_API_KEY`, `ELEVENLABS_API_KEY`)
4. Run phase scripts in order from `src/` (see phase docs / progress notes)

## Phases
- [x] Phase 0 — Setup (venv, requirements, .env)
- [x] Phase 1 — Data + Chunking (fixed-size, semantic, metadata-aware — 1000 passages → 1887 chunks)
- [x] Phase 2 — Embedding + Retrieval (FAISS index over metadata-aware chunks)
- [x] Phase 3 — Answer Generation (retrieval + Groq generation)
- [x] Phase 4 — Guardrails (unsafe input, low-confidence, grounding checks)
- [x] Phase 5 — Harness (`RAGHarness`, pydantic I/O, tenacity retries, logging)
- [x] Phase 6 — Voice Input (ElevenLabs STT → full pipeline, confirmed end-to-end, ~941ms single-call latency)
- [x] Phase 7 — Latency Benchmarking (see results below)
- [ ] Phase 8 — UI + Deployment
- [ ] Phase 9 — Videos + Submission

## Phase 7 — Latency Benchmarking

Measured P50/P70/P100 latency across 25 randomly sampled queries, separating
**retrieval-only** latency (local FAISS + local sentence-transformer embeddings)
from **full-pipeline** latency (retrieval + guardrails + a live Groq LLM call).

| Metric | Retrieval-only | Full pipeline (retrieval + guardrails + Groq LLM) |
|---|---|---|
| P50 | 73.0 ms | 372.9 ms |
| P70 | 107.5 ms | 565.0 ms |
| P100 (max) | 149.5 ms | 1312.9 ms |
| mean | 74.8 ms | 458.4 ms |

Retrieval-only comfortably meets the 200ms target — it's entirely local computation
with no network dependency.

Full-pipeline latency is higher because it includes a live network round-trip to the
Groq API. We consider the 200ms target unrealistic once a live LLM call is in the
loop, and report this honestly rather than hiding it or only measuring the local
components.

We also confirmed (via Groq's dashboard, cross-referenced against our own results)
that Groq's free-tier rate limit (30 requests/minute) meaningfully affects latency
under burst load: back-to-back testing without pacing triggered `429` responses and
automatic retries, inflating full-pipeline P50 to over 4 seconds. Pacing requests to
stay under the rate limit (~15 RPM) brought latency back down to the numbers above,
which we consider representative of realistic (non-burst) usage such as a live demo.

Benchmark script: `src/10_benchmark_latency.py`. Raw per-query results:
`data/benchmark_results_clean.csv`.

## Known quirks
- Dataset fields: `query_id`, `query`, `passage`, `passage_id` (empty), `language`,
  `answer`, `title` (empty), `url`, `query_type`, `relevance_score`, `is_selected`
  (all True), `text`, `meta`, `dataset`, `source`. 1 passage per query, no candidates.
  Note: `chunks_metadata_aware.jsonl` keeps `query_id` per chunk but not the query
  text itself — join back to the original dataset to recover query text.
- Groq package needed `--upgrade` to fix an `httpx` "proxies" TypeError.
- Windows needed the LongPathsEnabled registry fix for pip installs.