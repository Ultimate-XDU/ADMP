
from typing import Any, Dict, List, Optional

from search.common import logger
from search.hybrid_search import HybridSearch
from search.retrieval_coordinator import retrieve

from .prompt import (
    BASE_SYSTEM_PROMPT,
    build_context,
    build_user_prompt,
    call_qwen,
)
from .response import (
    _abstain_message,
    _estimate_confidence,
    _llm_unavailable_message,
    _normalize_response_source,
    _search_error_message,
    attach_retrieval_fields,
)
from .vision import visual_context_for_sources

def ask(query: str, engine: HybridSearch = None) -> str:
    payload = ask_structured(query, engine=engine)
    answer = payload.get("answer", "")
    sources = payload.get("sources", [])
    lines = ["=" * 60, f"问题：{query}", "=" * 60, "", answer, "", "-" * 60, "参考来源："]
    for index, row in enumerate(sources, start=1):
        section = row.get("section_title") or row.get("heading") or row.get("parent_heading") or "未标注章节"
        lines.append(f"[{index}] {row.get('doc_name', '未知文档')} | 第 {row.get('page_number', 1)} 页 | {section}")
    lines.append("=" * 60)
    return "\n".join(lines)


def _apply_graph_fallback(payload: Dict, reason: str) -> Dict:
    if reason != "graph_not_built":
        return payload
    trace = payload.get("retrievalTrace")
    if not isinstance(trace, dict):
        trace = {"steps": [], "mode": "single", "planner": "none", "reason": reason}
    else:
        trace = dict(trace)
        trace["mode"] = "single"
        trace["planner"] = "none"
        trace["reason"] = reason
    updated = dict(payload)
    updated["retrievalTrace"] = trace
    return updated


def ask_structured(
    query: str,
    engine: HybridSearch = None,
    history: Optional[List[Dict[str, Any]]] = None,
    retrieval_mode: str = "auto",
    include_visual: bool = False,
) -> Dict:
    payload = _compose_structured_answer(
        query,
        engine=engine,
        history=history,
        retrieval_mode=retrieval_mode,
        include_visual=include_visual,
    )
    return attach_retrieval_fields(payload)


def _compose_structured_answer(
    query: str,
    engine: HybridSearch = None,
    history: Optional[List[Dict[str, Any]]] = None,
    retrieval_mode: str = "auto",
    include_visual: bool = False,
) -> Dict:
    query_type = "general"
    from knowledge.graph_rag.qa_bridge import answer_with_graph, prepare_graph_dispatch

    search_query, graph_mode, fallback_reason = prepare_graph_dispatch(query, history, retrieval_mode)
    if graph_mode:
        # Explicit graph/agent/multi, and complex auto when a graph exists,
        # stay on the LangGraph controller. A missing graph is reported
        # directly and does not fall through to the old hop planner.
        try:
            return answer_with_graph(search_query, mode=graph_mode)
        except Exception:
            logger.exception("图谱问答失败 query=%s", query)
            return {
                "answer": "知识图谱当前不可用，无法执行图谱检索。",
                "sources": [],
                "confidence": 0.0,
                "abstainReason": "graph_unavailable",
                "evidenceIds": [],
                "queryType": "graph",
                "retrievalTrace": {
                    "steps": [],
                    "mode": "graph_unavailable",
                    "planner": "langgraph",
                    "reason": "graph_unavailable",
                },
                "visualContext": {"used": False, "pages": [], "reason": ""},
                "stopReason": "graph_unavailable",
                "graph": {},
                "graphTrace": [],
            }
    # Complex auto with no graph is one hybrid search, not the old second hop.
    effective_mode = "single" if fallback_reason == "graph_not_built" else retrieval_mode
    if engine is None:
        with HybridSearch(hyde_llm_call=call_qwen) as local_engine:
            payload = _compose_structured_answer(
                query,
                engine=local_engine,
                history=history,
                retrieval_mode=effective_mode,
                include_visual=include_visual,
            )
        return _apply_graph_fallback(payload, fallback_reason)
    try:
        search_results, trace = retrieve(engine, search_query, mode=effective_mode)
    except Exception:
        logger.exception("问答检索失败 query=%s", query)
        return _apply_graph_fallback(
            {
                "answer": _search_error_message(),
                "sources": [],
                "confidence": 0.0,
                "abstainReason": "search_error",
                "evidenceIds": [],
                "queryType": query_type,
                "visualContext": {"used": False, "pages": [], "reason": ""},
            },
            fallback_reason,
        )
    selected_sources = [_normalize_response_source(source) for source in search_results]
    evidence_ids = [row.get("chunk_id", "") for row in selected_sources if row.get("chunk_id")]
    confidence = _estimate_confidence(selected_sources)
    visual = {"used": False, "pages": [], "reason": ""}
    if not search_results:
        return _apply_graph_fallback(
            {
                "answer": _abstain_message(query_type),
                "sources": [],
                "confidence": 0.0,
                "abstainReason": "no_evidence",
                "evidenceIds": [],
                "queryType": query_type,
                "retrievalTrace": trace,
                "visualContext": visual,
            },
            fallback_reason,
        )

    context = build_context(selected_sources)
    user_prompt = build_user_prompt(query, context, query_type, history=history)
    # One completion when page images exist: the text evidence prompt plus those images.
    # Text generation runs once only when no usable page is available, or the vision call fails.
    answer = ""
    if include_visual:
        seen = visual_context_for_sources(query, selected_sources, user_prompt=user_prompt)
        visual = {
            "used": bool(seen.get("used")),
            "pages": seen.get("pages") or [],
            "reason": str(seen.get("reason") or ""),
        }
        if seen.get("used") and seen.get("answer"):
            answer = str(seen["answer"]).strip()
    if not answer:
        try:
            answer = call_qwen(BASE_SYSTEM_PROMPT, user_prompt).strip()
        except Exception as exc:
            logger.warning("调用大模型失败 query=%s: %s", query, exc)
            return _apply_graph_fallback(
                {
                    "answer": _llm_unavailable_message(selected_sources, query_type),
                    "sources": selected_sources,
                    "confidence": confidence,
                    "abstainReason": "llm_unavailable",
                    "evidenceIds": evidence_ids,
                    "queryType": query_type,
                    "retrievalTrace": trace,
                    "visualContext": visual,
                },
                fallback_reason,
            )
        if not answer:
            answer = _abstain_message(query_type)
    return _apply_graph_fallback(
        {
            "answer": answer,
            "sources": selected_sources,
            "confidence": confidence,
            "abstainReason": "",
            "evidenceIds": evidence_ids,
            "queryType": query_type,
            "retrievalTrace": trace,
            "visualContext": visual,
        },
        fallback_reason,
    )


def main():
    while True:
        query = input("\n请输入问题（直接回车退出）: ").strip()
        if not query:
            break
        print(ask(query))

