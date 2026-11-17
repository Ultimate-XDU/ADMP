"""Adapt graph answers to the existing /api/qa payload.

multi and agent both run the LangGraph agent. graph runs the same controller
with the observation policy. Complex auto uses that agent only when the graph
already has nodes. Otherwise it stays a single hybrid search and says why.
Neither path calls the old two-hop coordinator.
"""

from __future__ import annotations

from typing import Optional

from knowledge.graph_rag.service import get_service


def graph_ready() -> bool:
    """True when extracted nodes exist. Failures stay on the single-search path."""
    try:
        service = get_service()
        with service.catalog.lock:
            nodes = service.projection.get("nodes") or []
        return len(nodes) > 0
    except Exception:
        return False


def prepare_graph_dispatch(query: str, history, retrieval_mode: str) -> tuple[str, Optional[str], str]:
    """Choose the graph controller after the same resolver the single path uses.

    Returns the contextual query, the graph mode (``graph`` or ``agent``), and
    ``graph_not_built`` when a complex auto question has no graph to search.
    """
    from qa.prompt import _build_contextual_search_query
    from search.retrieval_coordinator import resolve_retrieval_mode

    search_query = _build_contextual_search_query(query, history)
    mode = str(retrieval_mode or "auto")
    if mode == "graph":
        return search_query, "graph", ""
    if mode in {"multi", "agent"}:
        return search_query, "agent", ""
    if mode == "auto" and resolve_retrieval_mode(search_query, "auto") == "multi":
        if graph_ready():
            return search_query, "agent", ""
        return search_query, None, "graph_not_built"
    return search_query, None, ""


def answer_with_graph(question: str, mode: str = "agent") -> dict:
    result = get_service().ask(question, "graph" if mode == "graph" else "agent")
    stop = str(result.get("stopReason") or "")
    sources = list(result.get("sources") or [])
    unavailable = stop == "graph_unavailable"
    abstain = ""
    if unavailable:
        abstain = "graph_unavailable"
    elif stop == "partial_evidence":
        abstain = "partial_evidence"
    elif stop in {"insufficient_evidence", "uncited_answer"}:
        abstain = "no_evidence"
    elif stop == "error":
        abstain = "llm_unavailable"
    scores = [float(row.get("score") or 0.0) for row in sources]
    confidence = max(scores) if scores and not abstain else 0.0
    trace_steps = []
    for step in result.get("trace") or []:
        trace_steps.append(
            {
                "hop": int(step.get("step") or 0),
                "query": str(step.get("query") or ""),
                "resultCount": len(step.get("sourceIds") or []),
                "basisChunkIds": list(step.get("sourceIds") or []),
            }
        )
    return {
        "answer": str(result.get("answer") or ""),
        "sources": sources,
        "confidence": confidence,
        "abstainReason": abstain,
        "evidenceIds": [str(row.get("chunk_id") or "") for row in sources if row.get("chunk_id")],
        "queryType": "graph",
        "retrievalTrace": {
            "steps": trace_steps,
            "mode": "graph_unavailable" if unavailable else mode,
            "planner": "langgraph",
            "reason": stop,
        },
        "visualContext": {"used": False, "pages": [], "reason": ""},
        "stopReason": stop,
        "graph": result.get("graph") or {},
        "graphTrace": list(result.get("trace") or []),
    }
