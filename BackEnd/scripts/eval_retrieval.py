"""Run retrieval evaluation. This script does not score gold labels.

Example:
    python -m scripts.eval_retrieval --queries queries.jsonl --mode hybrid --output hits.jsonl

The query file contributes only query_id, query, and optional doc_ids.
"""

from __future__ import annotations

import argparse

from knowledge.graph_rag.eval_retrieval import MODES, run_query_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Retrieval-only evaluation. No gold metrics are computed.")
    parser.add_argument("--queries", required=True, help="JSONL of {query_id, query, doc_ids?}")
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--output", required=True, help="JSONL of candidates, spans, trace, and timings")
    parser.add_argument("--candidate-limit", type=int, default=8)
    parser.add_argument("--context-chars", type=int, default=5600)
    parser.add_argument("--passage-chars", type=int, default=900)
    parser.add_argument("--rerank-pool", type=int, default=16)
    parser.add_argument("--tool-calls", type=int, default=0, help="Override the tool cap. 0 keeps the facet rule.")
    args = parser.parse_args(argv)
    rows = run_query_file(
        args.queries,
        args.mode,
        args.output,
        candidates=args.candidate_limit,
        context_chars=args.context_chars,
        passage_chars=args.passage_chars,
        rerank_pool=args.rerank_pool,
        tool_calls=args.tool_calls or None,
    )
    print(f"wrote {len(rows)} {args.mode} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
