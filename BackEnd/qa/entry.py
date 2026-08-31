
from typing import Any, Dict, List, Optional

from search.common import logger
from search.hybrid_search import HybridSearch
from search.query_analysis import analyze_query_intent
from utils.retrieval_contract import apply_contract, build_retrieval_contract
from utils.structure import (
    answer_structural_classification,
    answer_structural_count,
    answer_structural_definition,
    load_all_structural_records,
)

from .evidence import _repair_response_sources, _select_prompt_evidence
from .prompt import (
    BASE_SYSTEM_PROMPT,
    _build_contextual_search_query,
    build_context,
    build_user_prompt,
    call_qwen,
    prepare_prompt_evidence,
)
from .response import (
    _abstain_message,
    _abstain_payload,
    _estimate_confidence,
    _finalized_contract_answer_or_none,
    _llm_unavailable_message,
    _normalize_response_source,
    _search_error_message,
)
from .structured import (
    _definition_query_term,
    _is_valid_query,
    _try_answer_article_count,
    _try_answer_document_applicability,
)

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


def ask_structured(query: str, engine: HybridSearch = None, history: Optional[List[Dict[str, Any]]] = None) -> Dict:

    if not _is_valid_query(query):
        query_type = "general"
        return {
            "answer": "请输入有效的技术问题。问题需包含至少两个有意义的字符。",
            "sources": [],
            "confidence": 0.0,
            "abstainReason": "invalid_query",
            "evidenceIds": [],
            "queryType": query_type,
        }

    contract = build_retrieval_contract(query)
    query_profile = analyze_query_intent(query)
    query_type = query_profile["query_type"]
    if contract.block_reason:
        return _abstain_payload(query_type, contract.block_reason, contract=contract)
    if engine is None:
        with HybridSearch(hyde_llm_call=call_qwen) as local_engine:
            return ask_structured(query, engine=local_engine, history=history)
    structural_records = load_all_structural_records()
    structural_count_answer = answer_structural_count(query, structural_records)
    finalized_structural = _finalized_contract_answer_or_none(structural_count_answer, contract, query_type, engine)
    if finalized_structural:
        return finalized_structural

    structural_classification_answer = answer_structural_classification(query, structural_records)
    finalized_structural = _finalized_contract_answer_or_none(structural_classification_answer, contract, query_type, engine)
    if finalized_structural:
        return finalized_structural

    structural_definition_answer = answer_structural_definition(query, structural_records)
    finalized_structural = _finalized_contract_answer_or_none(structural_definition_answer, contract, query_type, engine)
    if finalized_structural:
        return finalized_structural
    definition_term = _definition_query_term(query)
    if definition_term and not contract.article_variants and not (contract.required_terms or contract.focus_terms or contract.targets):
        return {
            "answer": f"当前证据不足，无法确认“{definition_term}”在已索引文档中有明确的术语定义。建议补充所属文件、章节或具体业务场景后再查询。",
            "sources": [],
            "confidence": 0.0,
            "abstainReason": "no_definition_evidence",
            "evidenceIds": [],
            "queryType": query_type,
        }

    applicability_answer = _try_answer_document_applicability(query, query_type)
    finalized_direct = _finalized_contract_answer_or_none(applicability_answer, contract, query_type, engine)
    if finalized_direct:
        return finalized_direct

    direct_answer = _try_answer_article_count(query, query_type)
    finalized_direct = _finalized_contract_answer_or_none(direct_answer, contract, query_type, engine)
    if finalized_direct:
        return finalized_direct

    try:
        search_query = _build_contextual_search_query(query, history)
        search_results = engine.search_for_qa(search_query)
    except Exception as exc:
        logger.exception("问答检索失败 query=%s", query)
        return {
            "answer": _search_error_message(),
            "sources": [],
            "confidence": 0.0,
            "abstainReason": "search_error",
            "evidenceIds": [],
            "queryType": query_type,
        }
    search_results = apply_contract(contract, search_results, keep_failed_when_no_hard_constraints=True)
    selected_sources = _select_prompt_evidence(search_query, search_results, query_type, contract=contract)
    selected_sources = apply_contract(
        contract,
        selected_sources,
        keep_failed_when_no_hard_constraints=True,
        preserve_order=True,
    )
    selected_sources = _repair_response_sources(engine, selected_sources)
    selected_sources = [_normalize_response_source(source) for source in selected_sources]
    evidence_ids = [row.get("chunk_id", "") for row in selected_sources if row.get("chunk_id")]
    confidence = _estimate_confidence(selected_sources)

    if not search_results:
        return {
            "answer": _abstain_message(query_type),
            "sources": [],
            "confidence": 0.0,
            "abstainReason": "no_evidence",
            "evidenceIds": [],
            "queryType": query_type,
            "retrievalContract": contract.to_dict(),
        }

    if confidence < 0.46:
        return {
            "answer": _abstain_message(query_type),
            "sources": selected_sources,
            "confidence": confidence,
            "abstainReason": "low_confidence",
            "evidenceIds": evidence_ids,
            "queryType": query_type,
            "retrievalContract": contract.to_dict(),
        }

    prompt_results = prepare_prompt_evidence(selected_sources, query_type)
    context = build_context(prompt_results)
    user_prompt = build_user_prompt(query, context, query_type, history=history)
    try:
        answer = call_qwen(BASE_SYSTEM_PROMPT, user_prompt).strip()
    except Exception as exc:
        logger.warning("调用大模型失败 query=%s: %s", query, exc)
        return {
            "answer": _llm_unavailable_message(prompt_results or search_results, query_type),
            "sources": selected_sources,
            "confidence": confidence,
            "abstainReason": "llm_unavailable",
            "evidenceIds": evidence_ids,
            "queryType": query_type,
        }
    if not answer:
        answer = _abstain_message(query_type)

    return {
        "answer": answer,
        "sources": selected_sources,
        "confidence": confidence,
        "abstainReason": "",
        "evidenceIds": evidence_ids,
        "queryType": query_type,
    }


def main():
    while True:
        query = input("\n请输入问题（直接回车退出）: ").strip()
        if not query:
            break
        print(ask(query))

