"""
Phase 7 — Latency Benchmarking
================================
Measures retrieval-only latency AND full-pipeline (retrieval + guardrails + Groq LLM)
latency on the SAME set of ~25 randomly sampled queries, and reports P50 / P70 / P100
(=max) for each, plus a CSV of every individual run.

WHY TWO NUMBERS: the 200ms target is realistic for retrieval alone (local FAISS +
local embedding model), but not once you add a live Groq API call over the network.
This script measures both honestly so the submission can show the breakdown instead
of hiding it.

--------------------------------------------------------------------------------
ADAPT-THIS markers below point to the ~2 spots most likely to need a tweak if your
actual function names in src/harness.py or src/05_generate_answer.py differ slightly
from what's assumed here (based on PROGRESS.md). Everything else should run as-is.
--------------------------------------------------------------------------------

Usage:
    python src/10_benchmark_latency.py

Outputs:
    data/benchmark_results.csv   (one row per query: retrieval_ms, full_pipeline_ms)
    Printed summary with P50 / P70 / P100 for both.
"""

import json
import random
import statistics
import time
import csv
from pathlib import Path

# --- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_FILE = PROJECT_ROOT / "data" / "chunks_metadata_aware.jsonl"
OUTPUT_CSV = PROJECT_ROOT / "data" / "benchmark_results_clean.csv"

N_QUERIES = 25
RANDOM_SEED = 42  # reproducible sample
DELAY_BETWEEN_QUERIES_SEC = 4.0  # ~15 RPM, safely under Groq free-tier 30 RPM cap


def load_sample_queries(n=N_QUERIES):
    """
    chunks_metadata_aware.jsonl keeps query_id per chunk but NOT the query text
    (dropped to avoid repeating it across every chunk of the same passage).
    So: 1) collect the set of query_ids that actually have chunks, then
        2) load the original dataset to map query_id -> query text, then
        3) sample n of the ones that have chunks.
    """
    chunk_query_ids = set()
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            qid = row.get("query_id")
            if qid is not None:
                chunk_query_ids.add(qid)

    from datasets import load_dataset  # same source used in 01_explore_data.py
    ds = load_dataset("ai4bharat/IndicMSMARCO", "hi")
    split = ds["train"] if "train" in ds else list(ds.values())[0]

    qid_to_query = {}
    for row in split:
        qid = row.get("query_id")
        q = row.get("query")
        if qid in chunk_query_ids and qid not in qid_to_query and q:
            qid_to_query[qid] = q

    unique_queries = list(qid_to_query.items())  # [(query_id, query), ...]
    random.seed(RANDOM_SEED)
    sample_size = min(n, len(unique_queries))
    if sample_size == 0:
        raise RuntimeError(
            "No query text could be matched to chunk query_ids. Check that "
            "ai4bharat/IndicMSMARCO 'hi' config has a 'query_id' field matching "
            "the one stored in chunks_metadata_aware.jsonl."
        )
    return random.sample(unique_queries, sample_size)


def time_call(fn, *args, **kwargs):
    """Run fn once, return (result, elapsed_ms)."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return result, elapsed_ms


def percentile(data, pct):
    """pct in [0, 100]. P100 == max."""
    if not data:
        return float("nan")
    data_sorted = sorted(data)
    if pct >= 100:
        return data_sorted[-1]
    idx = int(round((pct / 100) * (len(data_sorted) - 1)))
    return data_sorted[idx]


def main():
    print(f"Loading {N_QUERIES} random sample queries from {CHUNKS_FILE.name} ...")
    samples = load_sample_queries(N_QUERIES)
    print(f"Sampled {len(samples)} unique queries (seed={RANDOM_SEED}).\n")

    # Retrieval-only and full-pipeline calls, using your actual harness.py API:
    # - harness._retrieve(query)              -> retrieval only (no LLM call)
    # - harness.run(QueryRequest(query=query)) -> full guarded pipeline (LLM included)
    try:
        from harness import RAGHarness, QueryRequest  # sibling import (script run from inside src/)
    except ModuleNotFoundError:
        from src.harness import RAGHarness, QueryRequest  # fallback if run as a package from project root

    harness = RAGHarness()

    def retrieval_only(query: str):
        # harness._retrieve() is the private retrieval step used inside run() —
        # calling it directly gives us retrieval latency without the Groq call.
        return harness._retrieve(query)

    def full_pipeline(query: str):
        # run() takes a QueryRequest pydantic object, not a raw string.
        return harness.run(QueryRequest(query=query))

    # --- Run benchmark -------------------------------------------------------
    rows = []
    retrieval_times = []
    full_times = []
    status_counts = {}

    for i, (qid, query) in enumerate(samples, 1):
        print(f"[{i}/{len(samples)}] {query[:60]}...")

        try:
            _, retrieval_ms = time_call(retrieval_only, query)
        except Exception as e:
            print(f"  ! retrieval failed: {e}")
            retrieval_ms = None

        response = None
        try:
            response, full_ms = time_call(full_pipeline, query)
        except Exception as e:
            print(f"  ! full pipeline failed: {e}")
            full_ms = None

        if retrieval_ms is not None:
            retrieval_times.append(retrieval_ms)
        if full_ms is not None:
            full_times.append(full_ms)

        status = response.status if response is not None else "hard_error"
        reason = response.reason if response is not None else "exception raised in benchmark script"
        retries_used = response.retries_used if response is not None else None
        status_counts[status] = status_counts.get(status, 0) + 1

        if status != "answered":
            print(f"  ! status={status} reason={reason}")

        rows.append({
            "query_id": qid,
            "query": query,
            "retrieval_ms": round(retrieval_ms, 1) if retrieval_ms is not None else "ERROR",
            "full_pipeline_ms": round(full_ms, 1) if full_ms is not None else "ERROR",
            "status": status,
            "reason": reason or "",
            "retries_used": retries_used if retries_used is not None else "",
        })

        if i < len(samples):
            time.sleep(DELAY_BETWEEN_QUERIES_SEC)

    # --- Write CSV -------------------------------------------------------------
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["query_id", "query", "retrieval_ms", "full_pipeline_ms", "status", "reason", "retries_used"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote per-query results to {OUTPUT_CSV}")
    print(f"Status breakdown across {len(rows)} queries: {status_counts}")

    # --- Summary -----------------------------------------------------------
    def summarize(name, times):
        if not times:
            print(f"{name}: no successful runs")
            return
        print(f"\n{name} (n={len(times)}):")
        print(f"  P50  : {percentile(times, 50):.1f} ms")
        print(f"  P70  : {percentile(times, 70):.1f} ms")
        print(f"  P100 : {percentile(times, 100):.1f} ms  (max)")
        print(f"  mean : {statistics.mean(times):.1f} ms")

    print("\n" + "=" * 60)
    print("LATENCY BENCHMARK SUMMARY")
    print("=" * 60)
    summarize("Retrieval-only", retrieval_times)
    summarize("Full pipeline (retrieval + guardrails + Groq LLM)", full_times)
    print("\nNote: the 200ms target is realistic for retrieval-only (local FAISS +")
    print("local embedding model). Full-pipeline latency is higher because it includes")
    print("a live network call to the Groq API — this is expected and will be explained")
    print("in the submission rather than hidden.")
    print(f"\nRuns were paced {DELAY_BETWEEN_QUERIES_SEC}s apart to stay under Groq's free-tier")
    print("30 requests/minute limit, avoiding retry/backoff noise inflating the numbers.")


if __name__ == "__main__":
    main()