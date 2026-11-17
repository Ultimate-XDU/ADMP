"""LangGraph controller for graph-backed multi-hop retrieval.

The planner sees tool observations and chooses the next action. Tool nodes
write the trace. Hidden model reasoning is never copied into that trace.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from knowledge.graph_rag.config import ABSTAIN_TEXT, MAX_TOOL_CALLS
from knowledge.graph_rag.hybrid_index import budget_for, missing_aspects

ActionFn = Callable[[dict], str]
AnswerFn = Callable[[dict], dict]
ToolFn = Callable[[dict], dict]

_ACTIONS = {"graph_search", "graph_expand", "document_retrieve", "inspect_evidence", "finish"}
_JSON_OBJECT = re.compile(r"\{.*\}", re.S)


class AgentState(TypedDict, total=False):
    """Separate channels so a node can return only the keys it changes.

    LangGraph treats a bare ``dict`` schema as one root value. A node that
    returns ``{"pending": ...}`` then deletes the question, tool count, and
    evidence. Each field below is its own channel and survives a partial return.
    """

    question: str
    doc_ids: Optional[list[str]]
    mode: str
    tool_calls: int
    observations: list
    pending: dict
    trace: list
    graph_parts: list
    evidence: list
    answer: str
    stop_reason: str
    retrieval_budget: dict
    context_spans: list
    context_limits: dict


def finish_action(query: str = "") -> dict:
    return {"action": "finish", "query": query, "nodeIds": [], "chunkIds": [], "hops": 1}


def normalize_query(query: str) -> str:
    """Compare retrieval queries after spacing and case, without changing their meaning."""
    return re.sub(r"\s+", " ", str(query or "").strip()).casefold()


def parse_model_action(text: str) -> Optional[dict]:
    """Read one JSON action. Reasoning text around it is ignored."""
    raw = str(text or "").strip()
    if not raw:
        return None
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        match = _JSON_OBJECT.search(raw)
        if not match:
            return None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(payload, dict):
        return None
    action = str(payload.get("action") or "").strip()
    if action not in _ACTIONS:
        return None
    node_ids = payload.get("nodeIds") or payload.get("node_ids") or []
    chunk_ids = payload.get("chunkIds") or payload.get("chunk_ids") or []
    try:
        hops = int(payload.get("hops") or 1)
    except (TypeError, ValueError):
        hops = 1
    return {
        "action": action,
        "query": str(payload.get("query") or ""),
        "nodeIds": [str(item) for item in node_ids if str(item).strip()],
        "chunkIds": [str(item) for item in chunk_ids if str(item).strip()],
        "hops": hops,
    }


def bridge_labels(observations: list[dict]) -> list[str]:
    """Labels that an expansion introduced and that earlier steps had not seen."""
    seen: set[str] = set()
    bridges: list[str] = []
    for observation in observations:
        labels = observation.get("labels") or {}
        current = [str(item) for item in observation.get("nodeIds") or []]
        if observation.get("action") == "graph_expand":
            for node_id in current:
                if node_id in seen:
                    continue
                label = str(labels.get(node_id) or "").strip()
                if label and label not in bridges:
                    bridges.append(label)
        seen.update(current)
    return bridges


def _collected_sources(observations: list[dict]) -> list[str]:
    sources: list[str] = []
    for item in observations:
        for source_id in item.get("sourceIds") or []:
            if source_id not in sources:
                sources.append(str(source_id))
    return sources


def _issued_retrieve_queries(observations: list[dict]) -> set[str]:
    return {
        normalize_query(item.get("query") or "")
        for item in observations
        if item.get("action") == "document_retrieve"
    }


def _retrieve_missing(state: dict, observations: list[dict], missing: list[str]) -> Optional[dict]:
    issued = _issued_retrieve_queries(observations)
    for facet in missing:
        if normalize_query(facet) not in issued:
            return {"action": "document_retrieve", "query": facet, "nodeIds": [], "chunkIds": [], "hops": 1}
    return None


def policy_action(state: dict) -> dict:
    """Choose the next tool from observations. Domain facts are not hardcoded."""
    if state.get("mode") == "single":
        observations = list(state.get("observations") or [])
        retrieved = any(item.get("action") == "document_retrieve" for item in observations)
        if retrieved or int(state.get("tool_calls") or 0) >= 1:
            return finish_action()
        return {
            "action": "document_retrieve",
            "query": str(state.get("question") or ""),
            "nodeIds": [],
            "chunkIds": [],
            "hops": 1,
        }
    tool_calls = int(state.get("tool_calls") or 0)
    limit = budget_for(state)
    if tool_calls >= limit:
        return finish_action()
    observations = list(state.get("observations") or [])
    searched_labels: set[str] = set()
    for item in observations:
        if item.get("action") == "graph_search":
            searched_labels.update(part for part in str(item.get("query") or "").split() if part)
    inspected = any(item.get("action") == "inspect_evidence" for item in observations)
    retrieved = any(item.get("action") == "document_retrieve" for item in observations)
    question = str(state.get("question") or "")
    sources = _collected_sources(observations)
    missing = missing_aspects(state)
    # Keep the last allowed call for reading passages already found.
    if sources and not inspected and tool_calls >= limit - 1:
        return {
            "action": "inspect_evidence",
            "query": question,
            "nodeIds": [],
            "chunkIds": sources[:8],
            "hops": 1,
        }
    if not observations:
        return {"action": "graph_search", "query": question, "nodeIds": [], "chunkIds": [], "hops": 1}
    last = observations[-1]
    action = last.get("action")
    if action == "graph_search":
        if last.get("nodeIds"):
            return {
                "action": "graph_expand",
                "query": str(last.get("query") or ""),
                "nodeIds": list(last.get("nodeIds") or [])[:8],
                "chunkIds": [],
                "hops": 2,
            }
        pending_facet = _retrieve_missing(state, observations, missing)
        if pending_facet:
            return pending_facet
        if not retrieved:
            return {"action": "document_retrieve", "query": question, "nodeIds": [], "chunkIds": [], "hops": 1}
        return finish_action()
    if action == "graph_expand":
        # A missing aspect is queried before an unrelated bridge label.
        pending_facet = _retrieve_missing(state, observations, missing)
        if pending_facet:
            return pending_facet
        fresh = [label for label in bridge_labels(observations) if label and label not in searched_labels]
        if fresh:
            return {
                "action": "graph_search",
                "query": " ".join(fresh[:4]),
                "nodeIds": [],
                "chunkIds": [],
                "hops": 1,
            }
        if last.get("sourceIds") and not inspected:
            return {
                "action": "inspect_evidence",
                "query": "",
                "nodeIds": [],
                "chunkIds": list(last.get("sourceIds") or [])[:8],
                "hops": 1,
            }
        if not retrieved:
            return {"action": "document_retrieve", "query": question, "nodeIds": [], "chunkIds": [], "hops": 1}
        return finish_action()
    if action == "document_retrieve":
        if tool_calls < limit - 1:
            pending_facet = _retrieve_missing(state, observations, missing)
            if pending_facet:
                return pending_facet
        if last.get("sourceIds") and not inspected:
            return {
                "action": "inspect_evidence",
                "query": question,
                "nodeIds": [],
                "chunkIds": list(last.get("sourceIds") or [])[:8],
                "hops": 1,
            }
        return finish_action()
    fresh = [label for label in bridge_labels(observations) if label and label not in searched_labels]
    if fresh and tool_calls < limit and not missing:
        return {"action": "graph_search", "query": " ".join(fresh[:4]), "nodeIds": [], "chunkIds": [], "hops": 1}
    if tool_calls < limit:
        pending_facet = _retrieve_missing(state, observations, missing)
        if pending_facet:
            return pending_facet
    return finish_action()


def validate_action(candidate: Optional[dict], state: dict) -> Optional[dict]:
    if not candidate:
        return None
    if int(state.get("tool_calls") or 0) >= budget_for(state):
        return finish_action()
    action = candidate.get("action")
    if action not in _ACTIONS:
        return None
    if action == "finish":
        return finish_action(str(candidate.get("query") or ""))
    observations = list(state.get("observations") or [])
    known_nodes = {node_id for item in observations for node_id in item.get("nodeIds") or []}
    known_chunks = {chunk_id for item in observations for chunk_id in item.get("sourceIds") or []}
    if action == "graph_expand":
        node_ids = [node_id for node_id in candidate.get("nodeIds") or [] if node_id in known_nodes]
        if not node_ids:
            return None
        candidate = dict(candidate)
        candidate["nodeIds"] = node_ids[:8]
        candidate["hops"] = max(1, min(int(candidate.get("hops") or 1), 3))
        return candidate
    if action == "inspect_evidence":
        chunk_ids = [chunk_id for chunk_id in candidate.get("chunkIds") or [] if chunk_id in known_chunks]
        if not chunk_ids:
            return None
        candidate = dict(candidate)
        candidate["chunkIds"] = chunk_ids[:8]
        return candidate
    if action == "graph_search":
        query = str(candidate.get("query") or "").strip()
        if not query:
            return None
        searched = {str(item.get("query") or "") for item in observations if item.get("action") == "graph_search"}
        if query in searched:
            return None
        candidate = dict(candidate)
        candidate["query"] = query
        return candidate
    if action == "document_retrieve":
        query = re.sub(r"\s+", " ", str(candidate.get("query") or state.get("question") or "").strip())
        normalized = normalize_query(query)
        if not normalized:
            return None
        issued = {
            normalize_query(item.get("query") or "")
            for item in observations
            if item.get("action") == "document_retrieve"
        }
        if normalized in issued:
            return None
        candidate = dict(candidate)
        candidate["query"] = query
        return candidate
    return None


def initial_state(question: str, doc_ids: Optional[list[str]], mode: str, retrieval_budget: Optional[dict] = None) -> dict:
    return {
        "question": question,
        "doc_ids": doc_ids,
        "mode": mode,
        "tool_calls": 0,
        "observations": [],
        "pending": {},
        "trace": [],
        "graph_parts": [],
        "evidence": [],
        "answer": "",
        "stop_reason": "",
        "retrieval_budget": dict(retrieval_budget or {}),
        "context_spans": [],
        "context_limits": {},
    }


def _compile(tools: dict[str, ToolFn], answer_fn: AnswerFn, action_fn: Optional[ActionFn]):
    def planner(state: dict) -> dict:
        suggestion = None
        limit = budget_for(state)
        if state.get("mode") == "agent" and action_fn is not None and int(state.get("tool_calls") or 0) < limit:
            try:
                suggestion = parse_model_action(action_fn(state))
            except Exception:
                suggestion = None
        chosen = validate_action(suggestion, state) or policy_action(state)
        tool_calls = int(state.get("tool_calls") or 0)
        if tool_calls >= limit:
            chosen = finish_action()
        else:
            reserved = policy_action(state)
            # The last allowed call reads passages already found. A model that
            # asks for another hop must not consume that call.
            if (
                tool_calls >= limit - 1
                and reserved.get("action") == "inspect_evidence"
                and chosen.get("action") not in {"finish", "inspect_evidence"}
            ):
                chosen = reserved
        return {"pending": chosen}

    def route(state: dict) -> str:
        action = str((state.get("pending") or {}).get("action") or "finish")
        if action not in _ACTIONS:
            return "finish"
        if action != "finish" and int(state.get("tool_calls") or 0) >= budget_for(state):
            return "finish"
        return action

    def finish(state: dict) -> dict:
        evidence = list(state.get("evidence") or [])
        trace = list(state.get("trace") or [])
        if not evidence:
            result = {"answer": ABSTAIN_TEXT, "stop_reason": "insufficient_evidence"}
        else:
            try:
                result = answer_fn(state) or {}
            except Exception:
                result = {"answer": "", "stop_reason": "error"}
        answer = str(result.get("answer") or "")
        stop_reason = str(result.get("stop_reason") or ("answered" if answer else "insufficient_evidence"))
        if stop_reason == "answered":
            summary = "根据已核对的原文生成回答。"
        elif stop_reason == "partial_evidence":
            summary = "原文只覆盖了问题的一部分，未支持的部分没有写成结论。"
        elif stop_reason == "retrieval_only":
            summary = "只返回检索片段，不生成回答。"
        else:
            summary = "缺少可引用的原文，停止作答。"
        summary = f"{summary} {budget_for(state)} 次工具上限，基础 {MAX_TOOL_CALLS}。"
        returned_sources = result.get("sources")
        if isinstance(returned_sources, list):
            evidence = returned_sources
        trace.append(
            {
                "step": len(trace) + 1,
                "action": "finish",
                "query": str(state.get("question") or ""),
                "summary": summary,
                "nodeIds": [],
                "sourceIds": [str(item.get("chunk_id") or "") for item in evidence if item.get("chunk_id")],
            }
        )
        return {
            "answer": answer,
            "stop_reason": stop_reason,
            "trace": trace,
            "evidence": evidence,
            "context_spans": list(result.get("context_spans") or []),
            "context_limits": dict(result.get("context_limits") or {}),
        }

    builder = StateGraph(AgentState)
    builder.add_node("planner", planner)
    for name, tool in tools.items():
        builder.add_node(name, tool)
        builder.add_edge(name, "planner")
    builder.add_node("finish", finish)
    builder.add_edge(START, "planner")
    builder.add_conditional_edges(
        "planner",
        route,
        {
            "graph_search": "graph_search",
            "graph_expand": "graph_expand",
            "document_retrieve": "document_retrieve",
            "inspect_evidence": "inspect_evidence",
            "finish": "finish",
        },
    )
    builder.add_edge("finish", END)
    return builder.compile()


def run_agent(
    question: str,
    doc_ids: Optional[list[str]],
    mode: str,
    tools: dict[str, ToolFn],
    answer_fn: AnswerFn,
    action_fn: Optional[ActionFn] = None,
    retrieval_budget: Optional[dict] = None,
) -> dict[str, Any]:
    graph = _compile(tools, answer_fn, action_fn if mode == "agent" else None)
    state = initial_state(question, doc_ids, mode, retrieval_budget)
    return graph.invoke(state, config={"recursion_limit": 30})
