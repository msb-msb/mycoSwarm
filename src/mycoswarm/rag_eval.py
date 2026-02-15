"""RAG evaluation framework for mycoSwarm.

Measures retrieval quality using a curated eval set of query/expected pairs.
Computes hit@1, hit@5, MRR, NDCG@5, and keyword overlap metrics.
Results are saved with timestamps for delta comparison across runs.

Usage:
    python tests/rag_eval.py                  Run evaluation (standalone)
    python tests/rag_eval.py --json           Output results as JSON
    mycoswarm library eval                    Run via CLI
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

# Default eval set: tests/rag_eval_set.json relative to project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_SET_PATH = _PROJECT_ROOT / "tests" / "rag_eval_set.json"
RESULTS_PATH = Path("~/.config/mycoswarm/eval_results.json").expanduser()


def load_eval_set(path: Path | None = None) -> list[dict]:
    """Load evaluation pairs from JSON file."""
    p = path or EVAL_SET_PATH
    data = json.loads(p.read_text())
    return data["eval_pairs"]


def load_previous_results() -> dict | None:
    """Load the most recent evaluation results, or None."""
    if not RESULTS_PATH.exists():
        return None
    try:
        data = json.loads(RESULTS_PATH.read_text())
        runs = data.get("runs", [])
        return runs[-1] if runs else None
    except (json.JSONDecodeError, KeyError, IndexError):
        return None


def save_results(results: dict) -> None:
    """Append results to the eval results file."""
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    if RESULTS_PATH.exists():
        try:
            data = json.loads(RESULTS_PATH.read_text())
        except (json.JSONDecodeError, KeyError):
            data = {"runs": []}
    else:
        data = {"runs": []}

    data["runs"].append(results)

    # Keep last 50 runs
    if len(data["runs"]) > 50:
        data["runs"] = data["runs"][-50:]

    RESULTS_PATH.write_text(json.dumps(data, indent=2))


def _hit_at_k(expected_sources: list[str], retrieved_sources: list[str], k: int) -> float:
    """1.0 if any expected source appears in top-k retrieved, else 0.0."""
    top_k = retrieved_sources[:k]
    for src in expected_sources:
        if src in top_k:
            return 1.0
    return 0.0


def _reciprocal_rank(expected_sources: list[str], retrieved_sources: list[str]) -> float:
    """1/rank of first expected source found, or 0.0 if not found."""
    for i, src in enumerate(retrieved_sources):
        if src in expected_sources:
            return 1.0 / (i + 1)
    return 0.0


def _dcg_at_k(relevance: list[float], k: int) -> float:
    """Discounted Cumulative Gain at k."""
    dcg = 0.0
    for i in range(min(k, len(relevance))):
        dcg += relevance[i] / math.log2(i + 2)  # i+2 because log2(1)=0
    return dcg


def _ndcg_at_k(expected_sources: list[str], retrieved_sources: list[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at k."""
    relevance = [
        1.0 if src in expected_sources else 0.0
        for src in retrieved_sources[:k]
    ]

    dcg = _dcg_at_k(relevance, k)

    # Ideal: all relevant docs at top
    ideal = sorted(relevance, reverse=True)
    idcg = _dcg_at_k(ideal, k)

    if idcg == 0:
        return 0.0
    return dcg / idcg


def _keyword_overlap(expected_keywords: list[str], retrieved_texts: list[str]) -> float:
    """Fraction of expected keywords found in any retrieved text."""
    if not expected_keywords:
        return 1.0

    combined = " ".join(retrieved_texts).lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in combined)
    return found / len(expected_keywords)


def run_eval(
    eval_pairs: list[dict] | None = None,
    n_results: int = 5,
    model: str | None = None,
    verbose: bool = False,
) -> dict:
    """Run the full evaluation and return metrics.

    Returns {
        "timestamp": ISO string,
        "n_queries": int,
        "n_results": int,
        "metrics": {"hit_at_1": float, "hit_at_5": float, "mrr": float,
                     "ndcg_at_5": float, "keyword_overlap": float},
        "per_query": [{"id": str, "hit_at_1": float, ...}, ...],
        "duration_seconds": float,
    }
    """
    from mycoswarm.library import search_all

    if eval_pairs is None:
        eval_pairs = load_eval_set()

    start = time.time()
    per_query: list[dict] = []

    total_hit1 = 0.0
    total_hit5 = 0.0
    total_mrr = 0.0
    total_ndcg5 = 0.0
    total_kw_overlap = 0.0

    for pair in eval_pairs:
        query = pair["query"]
        expected_sources = pair["expected_sources"]
        expected_keywords = pair.get("expected_keywords", [])

        # Run retrieval (no re-ranking — measure raw retrieval quality)
        doc_hits, session_hits, _ = search_all(
            query, n_results=n_results, model=model, do_rerank=False,
        )

        # Extract sources and texts from results
        retrieved_sources = [h.get("source", "") for h in doc_hits]
        retrieved_texts = [h.get("text", "") for h in doc_hits]
        # Include session summaries in keyword overlap
        retrieved_texts += [h.get("summary", "") for h in session_hits]

        # Compute per-query metrics
        h1 = _hit_at_k(expected_sources, retrieved_sources, 1)
        h5 = _hit_at_k(expected_sources, retrieved_sources, 5)
        rr = _reciprocal_rank(expected_sources, retrieved_sources)
        ndcg = _ndcg_at_k(expected_sources, retrieved_sources, 5)
        kw = _keyword_overlap(expected_keywords, retrieved_texts)

        total_hit1 += h1
        total_hit5 += h5
        total_mrr += rr
        total_ndcg5 += ndcg
        total_kw_overlap += kw

        entry = {
            "id": pair["id"],
            "query": query,
            "hit_at_1": h1,
            "hit_at_5": h5,
            "mrr": round(rr, 4),
            "ndcg_at_5": round(ndcg, 4),
            "keyword_overlap": round(kw, 4),
            "retrieved_sources": retrieved_sources[:5],
        }
        per_query.append(entry)

        if verbose:
            status = "HIT" if h5 > 0 else "MISS"
            print(f"  [{status}] {pair['id']}: h@1={h1:.0f} h@5={h5:.0f} mrr={rr:.2f} kw={kw:.2f}")

    n = len(eval_pairs)
    duration = round(time.time() - start, 2)

    metrics = {
        "hit_at_1": round(total_hit1 / n, 4) if n else 0.0,
        "hit_at_5": round(total_hit5 / n, 4) if n else 0.0,
        "mrr": round(total_mrr / n, 4) if n else 0.0,
        "ndcg_at_5": round(total_ndcg5 / n, 4) if n else 0.0,
        "keyword_overlap": round(total_kw_overlap / n, 4) if n else 0.0,
    }

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_queries": n,
        "n_results": n_results,
        "metrics": metrics,
        "per_query": per_query,
        "duration_seconds": duration,
    }


def format_delta(current: float, previous: float) -> str:
    """Format a metric delta as a string like '+0.05' or '-0.10'."""
    delta = current - previous
    if abs(delta) < 0.0001:
        return "  —"
    sign = "+" if delta > 0 else ""
    return f"{sign}{delta:.4f}"


def print_results(results: dict, previous: dict | None = None) -> None:
    """Print evaluation results as a formatted table."""
    metrics = results["metrics"]

    print("\n📊 RAG Evaluation Results")
    print("=" * 55)
    print(f"  Queries:  {results['n_queries']}")
    print(f"  Top-K:    {results['n_results']}")
    print(f"  Duration: {results['duration_seconds']}s")
    print(f"  Time:     {results['timestamp'][:19]}")
    print()

    # Metrics table
    prev_metrics = previous["metrics"] if previous else None

    header = f"  {'Metric':<20} {'Score':>8}"
    if prev_metrics:
        header += f"  {'Delta':>8}  {'Previous':>8}"
    print(header)
    print(f"  {'─' * 20} {'─' * 8}", end="")
    if prev_metrics:
        print(f"  {'─' * 8}  {'─' * 8}", end="")
    print()

    for key, label in [
        ("hit_at_1", "Hit@1"),
        ("hit_at_5", "Hit@5"),
        ("mrr", "MRR"),
        ("ndcg_at_5", "NDCG@5"),
        ("keyword_overlap", "Keyword Overlap"),
    ]:
        val = metrics[key]
        line = f"  {label:<20} {val:>8.4f}"
        if prev_metrics and key in prev_metrics:
            prev = prev_metrics[key]
            delta = format_delta(val, prev)
            line += f"  {delta:>8}  {prev:>8.4f}"
        print(line)

    print()

    # Per-query breakdown — show misses
    misses = [q for q in results["per_query"] if q["hit_at_5"] == 0]
    if misses:
        print(f"  Misses ({len(misses)}):")
        for m in misses:
            print(f"    {m['id']}: {m['query'][:60]}")
            print(f"      Retrieved: {m['retrieved_sources'][:3]}")
        print()

    print(f"  Results saved to: {RESULTS_PATH}")
    print("=" * 55)
