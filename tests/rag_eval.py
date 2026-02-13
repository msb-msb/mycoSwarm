#!/usr/bin/env python3
"""Standalone RAG evaluation runner.

Usage:
    python tests/rag_eval.py                  Run evaluation
    python tests/rag_eval.py --json           Output results as JSON
    python tests/rag_eval.py -v               Verbose per-query output
    mycoswarm library eval                    Run via CLI
"""

import json
import sys

from mycoswarm.rag_eval import (  # noqa: F401 — re-exported for tests
    run_eval,
    save_results,
    load_previous_results,
    load_eval_set,
    print_results,
    format_delta,
    _hit_at_k,
    _reciprocal_rank,
    _ndcg_at_k,
    _keyword_overlap,
    EVAL_SET_PATH,
    RESULTS_PATH,
)


def main():
    """CLI entry point."""
    as_json = "--json" in sys.argv
    verbose = "-v" in sys.argv or "--verbose" in sys.argv

    previous = load_previous_results()

    if not as_json:
        print("🔬 Running RAG evaluation...")

    results = run_eval(verbose=verbose)
    save_results(results)

    if as_json:
        print(json.dumps(results, indent=2))
    else:
        print_results(results, previous)


if __name__ == "__main__":
    main()
