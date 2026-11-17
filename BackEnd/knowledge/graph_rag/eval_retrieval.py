"""Retrieval-only evaluation. This module does not score gold labels.

Modes
-----
baseline
    Frozen package at ``work/eval20/baseline-code``. A fresh interpreter
    imports that tree and runs its agent. The frozen files are not edited.
hybrid
    Windowed BM25 + dense ranks fused by reciprocal rank, then one rerank batch.
hybrid_graph
    The same hybrid seeds, plus graph bridges that cover a facet the seeds miss.
agent
    End-to-end planner and answer generation. This mode does call the answer model.

Query files may only contribute ``query_id``, ``query``, and optional ``doc_ids``.
Any gold or evidence keys are ignored and are never sent to a model.
Returned scores are the implementation's own ranks. Recall, precision, MRR, and
nDCG are not computed here.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from knowledge.graph_rag.agent import run_agent
from knowledge.graph_rag.config import MAX_TOOL_CALLS, REPO_ROOT
from knowledge.graph_rag.hybrid_index import (
    budget_for,
    budget_note,
    covers,
    pack_context,
    question_facets,
    retrieve_spans,
)
from knowledge.graph_rag.service import (
    _answer_from_evidence,
    _planner_model,
    _select_answer_sources,
    _tools,
)

BASELINE_BACKEND = REPO_ROOT / "work" / "eval20" / "baseline-code" / "BackEnd"
MODES = ("baseline", "hybrid", "hybrid_graph", "agent")
_METRIC_KEYS = {"recall", "precision", "mrr", "ndcg", "f1", "gold", "label", "labels"}


def load_queries(path: str | Path) -> list[dict]:
    """Read a JSONL file of ``{query_id, query, doc_ids?}``. Gold keys are dropped."""
    rows = []
    text = Path(path).read_text(encoding="utf-8")
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError(f"line {line_no} is not an object")
        query = str(item.get("query") or "").strip()
        if not query:
            raise ValueError(f"line {line_no} is missing query")
        row = {"query_id": str(item.get("query_id") or f"q{line_no}"), "query": query}
        if "doc_ids" in item and item.get("doc_ids") is not None:
            doc_ids = item.get("doc_ids")
            if not isinstance(doc_ids, list):
                raise ValueError(f"line {line_no} doc_ids must be a list")
            row["doc_ids"] = [str(doc_id) for doc_id in doc_ids]
        rows.append(row)
    return rows


def retrieval_budget(
    query: str,
    *,
    candidates: int = 8,
    context_chars: int = 5600,
    passage_chars: int = 900,
    rerank_pool: int = 16,
    tool_calls: Optional[int] = None,
) -> dict:
    """Shared per-query limits. ``tool_calls`` is the cap, not a quality score."""
    state = {"question": query, "retrieval_budget": {}}
    if tool_calls:
        state["retrieval_budget"]["tool_calls"] = int(tool_calls)
    return {
        "candidates": max(1, int(candidates)),
        "context_chars": max(200, int(context_chars)),
        "passage_chars": max(80, int(passage_chars)),
        "rerank_pool": max(1, int(rerank_pool)),
        "tool_calls": budget_for(state),
        "tool_call_base": MAX_TOOL_CALLS,
        "note": budget_note(state),
        "tool_calls_override": bool(tool_calls),
    }


def run_query(
    query: dict,
    mode: str,
    snapshot: Optional[dict] = None,
    *,
    candidates: int = 8,
    context_chars: int = 5600,
    passage_chars: int = 900,
    rerank_pool: int = 16,
    tool_calls: Optional[int] = None,
) -> dict:
    """Run one query. Retrieval modes set ``answer`` to null and do not generate."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode}")
    query_id = str(query.get("query_id") or "")
    text = str(query.get("query") or "").strip()
    if not text:
        raise ValueError("query is empty")
    doc_ids = query.get("doc_ids")
    if doc_ids is not None and not isinstance(doc_ids, list):
        raise ValueError("doc_ids must be a list")
    budget = retrieval_budget(
        text,
        candidates=candidates,
        context_chars=context_chars,
        passage_chars=passage_chars,
        rerank_pool=rerank_pool,
        tool_calls=tool_calls,
    )
    started = time.perf_counter()
    if mode == "baseline":
        payload = _run_baseline(text, doc_ids, snapshot or {}, budget)
    elif mode == "agent":
        payload = _run_agent_mode(text, doc_ids, snapshot, budget)
    else:
        payload = _run_improved(text, doc_ids, snapshot or {}, budget, mode)
    payload["timings"] = {"seconds": time.perf_counter() - started}
    payload["query_id"] = query_id
    payload["mode"] = mode
    payload["budgets"] = _public_budget(budget, mode)
    for key in list(payload):
        if key.lower() in _METRIC_KEYS:
            del payload[key]
    return payload


def run_query_file(
    path: str | Path,
    mode: str,
    output: str | Path,
    snapshot: Optional[dict] = None,
    **limits,
) -> list[dict]:
    """Evaluate every query row. The output JSONL is spans and timings, not metrics."""
    if snapshot is None and mode != "agent":
        from knowledge.graph_rag.service import get_service

        snapshot = get_service()._snapshot()
    rows = []
    for query in load_queries(path):
        try:
            rows.append(run_query(query, mode, snapshot, **limits))
        except Exception as exc:
            budget = retrieval_budget(str(query.get("query") or ""), **_limit_kwargs(limits))
            rows.append(
                {
                    "query_id": query.get("query_id") or "",
                    "mode": mode,
                    "generated": False,
                    "answer": None,
                    "candidates": [],
                    "packed_spans": [],
                    "trace": [],
                    "timings": {"seconds": 0.0},
                    "budgets": _public_budget(budget, mode),
                    "error": str(exc),
                }
            )
    handle = Path(output)
    handle.parent.mkdir(parents=True, exist_ok=True)
    handle.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return rows


def _limit_kwargs(limits: dict) -> dict:
    allowed = {"candidates", "context_chars", "passage_chars", "rerank_pool", "tool_calls"}
    return {key: value for key, value in limits.items() if key in allowed}


def _public_budget(budget: dict, mode: str) -> dict:
    note = budget["note"]
    if mode == "baseline":
        note = (
            note
            + " Baseline harness truncates frozen output to these candidate and context budgets."
            + " Frozen ranking still embeds the first 2000 characters and keeps eight hits before that truncation."
        )
    return {
        "candidates": budget["candidates"],
        "context_chars": budget["context_chars"],
        "passage_chars": budget["passage_chars"],
        "rerank_pool": budget["rerank_pool"],
        "tool_calls": budget["tool_calls"],
        "tool_call_base": budget["tool_call_base"],
        "note": note,
    }


def _working_snapshot(snapshot: dict, budget: dict) -> dict:
    working = dict(snapshot or {})
    working.pop("_span_cache", None)
    stored = {
        "candidates": budget["candidates"],
        "context_chars": budget["context_chars"],
        "passage_chars": budget["passage_chars"],
        "rerank_pool": budget["rerank_pool"],
    }
    if budget.get("tool_calls_override"):
        stored["tool_calls"] = budget["tool_calls"]
    working["retrieval_budget"] = stored
    return working


def _run_improved(query: str, doc_ids, snapshot: dict, budget: dict, mode: str) -> dict:
    working = _working_snapshot(snapshot, budget)
    seeds = retrieve_spans(
        working,
        query,
        doc_ids,
        limit=budget["candidates"],
        rerank_pool=budget["rerank_pool"],
    )
    trace = []
    tool_calls_made = 0
    if mode == "hybrid":
        chosen = list(seeds)
        trace.append(
            {
                "step": 1,
                "action": "hybrid_retrieve",
                "query": query,
                "summary": f"窗口混合检索返回 {len(chosen)} 条。",
                "nodeIds": [],
                "sourceIds": [str(span.get("unit_id") or "") for span in chosen],
            }
        )
    else:
        def answer_fn(state: dict) -> dict:
            return {
                "answer": "",
                "stop_reason": "retrieval_only",
                "sources": _select_answer_sources(state),
            }

        result = run_agent(
            query,
            doc_ids,
            "graph",
            _tools(working),
            answer_fn,
            None,
            _agent_budget(budget),
        )
        extras = _bridge_spans(working, query, seeds, result)
        chosen = _merge_graph_candidates(seeds, extras, query, budget["candidates"])
        trace = _safe_trace(result.get("trace") or [])
        tool_calls_made = sum(1 for step in trace if step.get("action") != "finish")
    packed = _pack_units(query, chosen, working, budget)
    return {
        "generated": False,
        "answer": None,
        "candidates": [_candidate(span) for span in chosen],
        "packed_spans": packed,
        "trace": trace,
        "tool_calls_made": tool_calls_made,
    }


def _agent_budget(budget: dict) -> dict:
    stored = {
        "candidates": budget["candidates"],
        "context_chars": budget["context_chars"],
        "passage_chars": budget["passage_chars"],
        "rerank_pool": budget["rerank_pool"],
    }
    if budget.get("tool_calls_override"):
        stored["tool_calls"] = budget["tool_calls"]
    return stored


def _run_agent_mode(query: str, doc_ids, snapshot: Optional[dict], budget: dict) -> dict:
    """End-to-end path. This is the only mode that generates an answer."""
    if snapshot is None:
        from knowledge.graph_rag.service import get_service

        payload = get_service().ask(query, "agent", doc_ids)
        return {
            "generated": True,
            "answer": payload.get("answer"),
            "stop_reason": payload.get("stopReason"),
            "candidates": [],
            "packed_spans": [],
            "trace": _safe_trace(payload.get("trace") or []),
            "tool_calls_made": sum(1 for step in payload.get("trace") or [] if step.get("action") != "finish"),
        }
    working = _working_snapshot(snapshot, budget)

    def answer_fn(state: dict) -> dict:
        return _answer_from_evidence(state)

    result = run_agent(
        query,
        doc_ids,
        "agent",
        _tools(working),
        answer_fn,
        _planner_model,
        _agent_budget(budget),
    )
    sources = list(result.get("evidence") or [])
    packed = pack_context(
        query,
        sources,
        context_chars=budget["context_chars"],
        passage_chars=budget["passage_chars"],
        facets=question_facets(query),
    )
    return {
        "generated": True,
        "answer": result.get("answer"),
        "stop_reason": result.get("stop_reason"),
        "candidates": [_candidate_from_source(row, working) for row in sources][: budget["candidates"]],
        "packed_spans": packed,
        "trace": _safe_trace(result.get("trace") or []),
        "tool_calls_made": int(result.get("tool_calls") or 0),
    }


def _bridge_spans(snapshot: dict, query: str, seeds: list[dict], result: dict) -> list[dict]:
    seed_units = {str(span.get("unit_id") or "") for span in seeds}
    found = []
    seen = set()
    chunks = snapshot.get("chunks") or {}
    for observation in result.get("observations") or []:
        for span in observation.get("spans") or []:
            unit_id = str(span.get("unit_id") or "")
            if not unit_id or unit_id in seed_units:
                continue
            key = (unit_id, int(span.get("start") or 0), int(span.get("end") or 0))
            if key in seen:
                continue
            seen.add(key)
            found.append(dict(span))
        if observation.get("action") != "graph_expand":
            continue
        for unit_id in observation.get("sourceIds") or []:
            unit_id = str(unit_id)
            if not unit_id or unit_id in seed_units or unit_id not in chunks:
                continue
            packed = pack_context(
                query,
                [_source_view(chunks[unit_id])],
                context_chars=10**9,
                passage_chars=int((snapshot.get("retrieval_budget") or {}).get("passage_chars") or 900),
                facets=question_facets(query),
            )
            for span in packed:
                key = (span["unit_id"], int(span["start"]), int(span["end"]))
                if key in seen:
                    continue
                seen.add(key)
                found.append(span)
    return found


def _merge_graph_candidates(seeds: list[dict], extras: list[dict], query: str, limit: int) -> list[dict]:
    """Keep hybrid seeds. A bridge may fill a free slot or replace a redundant seed."""
    facets = question_facets(query)
    chosen: list[dict] = []
    seen = set()

    def identity(span: dict) -> tuple:
        return (str(span.get("unit_id") or ""), int(span.get("start") or 0), int(span.get("end") or 0))

    for span in seeds:
        if len(chosen) >= limit:
            break
        key = identity(span)
        if key in seen:
            continue
        seen.add(key)
        chosen.append(span)
    for extra in extras:
        key = identity(extra)
        if key in seen:
            continue
        new_facets = [
            facet
            for facet in facets
            if covers(facet, str(extra.get("text") or "")) and not _facet_covered(facet, chosen)
        ]
        if not new_facets:
            continue
        if len(chosen) < limit:
            chosen.append(extra)
            seen.add(key)
            continue
        for index in range(len(chosen) - 1, 0, -1):
            row = chosen[index]
            row_facets = [facet for facet in facets if covers(facet, str(row.get("text") or ""))]
            if not row_facets:
                continue
            if all(_facet_covered(facet, chosen[:index]) for facet in row_facets):
                chosen[index] = extra
                seen.add(key)
                break
    return chosen[:limit]


def _facet_covered(facet: str, spans: list[dict]) -> bool:
    return any(covers(facet, str(span.get("text") or "")) for span in spans)


def _pack_units(query: str, spans: list[dict], snapshot: dict, budget: dict) -> list[dict]:
    chunks = snapshot.get("chunks") or {}
    sources = []
    seen = set()
    for span in spans:
        unit_id = str(span.get("unit_id") or "")
        if not unit_id or unit_id in seen:
            continue
        seen.add(unit_id)
        chunk = chunks.get(unit_id) or {}
        sources.append(_source_view(chunk, unit_id, span))
    packed = pack_context(
        query,
        sources,
        context_chars=budget["context_chars"],
        passage_chars=budget["passage_chars"],
        facets=question_facets(query),
    )
    used = 0
    kept = []
    for span in packed:
        text = str(span.get("text") or "")
        if kept and used + len(text) > budget["context_chars"]:
            continue
        content = str((chunks.get(span.get("unit_id")) or {}).get("content") or "")
        start = int(span.get("start") or 0)
        end = int(span.get("end") or 0)
        if content and content[start:end] != text:
            continue
        kept.append(span)
        used += len(text)
    return kept


def _source_view(chunk: dict, unit_id: str = "", span: Optional[dict] = None) -> dict:
    span = span or {}
    return {
        "chunk_id": unit_id or str(chunk.get("chunk_id") or span.get("unit_id") or ""),
        "content": str(chunk.get("content") or ""),
        "doc_id": str(span.get("doc_id") or chunk.get("doc_id") or ""),
        "doc_name": str(span.get("doc_name") or chunk.get("doc_name") or ""),
        "page_number": int(span["page"]) if span.get("page") is not None else int(chunk["page_number"]) if chunk.get("page_number") is not None else 1,
        "section_title": str(span.get("section") or chunk.get("section_title") or ""),
        "heading": str(chunk.get("heading") or ""),
    }


def _candidate(span: dict) -> dict:
    return {
        "unit_id": str(span.get("unit_id") or ""),
        "doc_id": str(span.get("doc_id") or ""),
        "doc_name": str(span.get("doc_name") or ""),
        "section": str(span.get("section") or ""),
        "page": int(span["page"]) if span.get("page") is not None else 1,
        "start": int(span.get("start") or 0),
        "end": int(span.get("end") or 0),
        "text": str(span.get("text") or ""),
        "score": float(span.get("score") or 0.0),
        "ranks": {str(key): int(value) for key, value in dict(span.get("ranks") or {}).items()},
    }


def _candidate_from_source(row: dict, snapshot: dict) -> dict:
    unit_id = str(row.get("chunk_id") or "")
    chunk = (snapshot.get("chunks") or {}).get(unit_id) or {}
    content = str(chunk.get("content") or row.get("content") or "")
    text = str(row.get("content") or "")
    start = content.find(text) if text else 0
    if start < 0:
        start = 0
        text = content[:900]
    return _candidate(
        {
            "unit_id": unit_id,
            "doc_id": str(chunk.get("doc_id") or ""),
            "doc_name": str(row.get("doc_name") or chunk.get("doc_name") or ""),
            "section": str(row.get("section_title") or chunk.get("section_title") or ""),
            "page": int(row["page_number"]) if row.get("page_number") is not None else int(chunk["page_number"]) if chunk.get("page_number") is not None else 1,
            "start": start,
            "end": start + len(text),
            "text": content[start : start + len(text)] if content else text,
            "score": float(row.get("score") or 0.0),
            "ranks": {},
        }
    )


def _safe_trace(steps: list) -> list[dict]:
    safe = []
    for index, step in enumerate(steps or [], start=1):
        if not isinstance(step, dict):
            continue
        safe.append(
            {
                "step": int(step.get("step") or index),
                "action": str(step.get("action") or ""),
                "query": str(step.get("query") or ""),
                "summary": str(step.get("summary") or ""),
                "nodeIds": [str(item) for item in step.get("nodeIds") or []],
                "sourceIds": [str(item) for item in step.get("sourceIds") or []],
            }
        )
    return safe


def _run_baseline(query: str, doc_ids, snapshot: dict, budget: dict) -> dict:
    if not (BASELINE_BACKEND / "knowledge" / "graph_rag" / "service.py").is_file():
        return _empty_retrieval(f"frozen baseline is missing at {BASELINE_BACKEND}")
    payload = {
        "backend": str(BASELINE_BACKEND),
        "query": query,
        "doc_ids": doc_ids,
        "snapshot": _jsonable_snapshot(snapshot),
        "limits": {
            "candidates": budget["candidates"],
            "context_chars": budget["context_chars"],
            "passage_chars": budget["passage_chars"],
        },
    }
    env = os.environ.copy()
    # The frozen tree must be imported before the live backend on PYTHONPATH.
    env["PYTHONPATH"] = str(BASELINE_BACKEND)
    env.pop("PYTHONSAFEPATH", None)
    try:
        completed = subprocess.run(
            [sys.executable, "-c", _BASELINE_DRIVER],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            timeout=120,
            env=env,
            cwd=str(BASELINE_BACKEND),
        )
    except subprocess.TimeoutExpired as exc:
        return _empty_retrieval(f"baseline timed out: {exc}")
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        return _empty_retrieval(detail[-2000:] or f"baseline exited {completed.returncode}")
    try:
        body = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return _empty_retrieval((completed.stderr or completed.stdout or "baseline returned no JSON")[-2000:])
    if body.get("error"):
        return _empty_retrieval(str(body.get("error")))
    return {
        "generated": False,
        "answer": None,
        "candidates": list(body.get("candidates") or []),
        "packed_spans": list(body.get("packed_spans") or []),
        "trace": _safe_trace(body.get("trace") or []),
        "tool_calls_made": int(body.get("tool_calls_made") or 0),
    }


def _empty_retrieval(error: str) -> dict:
    return {
        "generated": False,
        "answer": None,
        "candidates": [],
        "packed_spans": [],
        "trace": [],
        "tool_calls_made": 0,
        "error": error,
    }


def _jsonable_snapshot(snapshot: dict) -> dict:
    return {
        "chunks": snapshot.get("chunks") or {},
        "graph": snapshot.get("graph") or {"nodes": [], "edges": []},
        "passage_ids": [str(item) for item in snapshot.get("passage_ids") or []],
        "entity_ids": [str(item) for item in snapshot.get("entity_ids") or []],
        "passage_matrix": _matrix_to_lists(snapshot.get("passage_matrix")),
        "entity_matrix": _matrix_to_lists(snapshot.get("entity_matrix")),
    }


def _matrix_to_lists(matrix):
    if matrix is None:
        return None
    if hasattr(matrix, "tolist"):
        return matrix.tolist()
    if isinstance(matrix, list):
        return matrix
    return None


_BASELINE_DRIVER = r"""
import json, sys, time
payload = json.load(sys.stdin)
sys.path.insert(0, payload["backend"])
try:
    from knowledge.graph_rag.agent import run_agent
    from knowledge.graph_rag.service import _relevant_window, _select_answer_sources, _tools
except Exception as exc:
    json.dump({"error": "baseline import failed: " + str(exc)}, sys.stdout)
    raise SystemExit(0)

snapshot = payload["snapshot"]
try:
    import numpy as np
except Exception:
    np = None
for key in ("passage_matrix", "entity_matrix"):
    value = snapshot.get(key)
    if isinstance(value, list) and np is not None:
        snapshot[key] = np.asarray(value, dtype="float32")
    else:
        snapshot[key] = None

query = payload["query"]
doc_ids = payload.get("doc_ids")
limits = payload["limits"]

def answer_fn(state):
    return {"answer": "", "stop_reason": "retrieval_only", "sources": _select_answer_sources(state)}

started = time.perf_counter()
result = run_agent(query, doc_ids, "graph", _tools(snapshot), answer_fn, None)
chunks = snapshot.get("chunks") or {}
candidates = []
packed = []
used = 0
for source in list(result.get("evidence") or []):
    if len(candidates) >= int(limits["candidates"]):
        break
    unit_id = str(source.get("chunk_id") or "")
    content = str((chunks.get(unit_id) or {}).get("content") or source.get("content") or "")
    window = _relevant_window(content, query, int(limits["passage_chars"]))
    start = content.find(window) if window else -1
    if start < 0 or content[start:start + len(window)] != window:
        continue
    end = start + len(window)
    text = content[start:end]
    if candidates and used + len(text) > int(limits["context_chars"]):
        continue
    row = {
        "unit_id": unit_id,
        "doc_id": str((chunks.get(unit_id) or {}).get("doc_id") or ""),
        "doc_name": str(source.get("doc_name") or ""),
        "section": str(source.get("section_title") or ""),
        "page": int(source.get("page_number") or 1),
        "start": start,
        "end": end,
        "text": text,
        "score": float(source.get("score") or 0.0),
        "ranks": {"frozen": len(candidates) + 1},
    }
    candidates.append(row)
    packed.append(dict(row))
    used += len(text)
trace = []
for step in result.get("trace") or []:
    trace.append({
        "step": step.get("step"),
        "action": step.get("action"),
        "query": step.get("query"),
        "summary": step.get("summary"),
        "nodeIds": list(step.get("nodeIds") or []),
        "sourceIds": list(step.get("sourceIds") or []),
    })
json.dump({
    "candidates": candidates,
    "packed_spans": packed,
    "trace": trace,
    "tool_calls_made": int(result.get("tool_calls") or 0),
    "seconds": time.perf_counter() - started,
}, sys.stdout)
"""
