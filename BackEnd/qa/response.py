import logging
import math
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz
from openai import OpenAI

from utils.evidence_scores import ensure_evidence_score_fields, evidence_display_score, evidence_ranking_score
from utils.structure import answer_structural_classification, answer_structural_count, answer_structural_definition, load_all_structural_records
from aerospace_search_main import HybridSearch, analyze_query_intent, infer_query_precision_hints
from utils.retrieval_contract import RetrievalContract, apply_contract, build_retrieval_contract

from search.hybrid_search import HybridSearch

def _is_noisy_highlight_line(value: str) -> bool:
    line = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(line) < 4:
        return True
    if re.fullmatch(r"[A-Za-z]{4,14}", line) and line.lower() in {
        "standard",
        "aircraft",
        "appendix",
        "section",
        "chapter",
        "general",
        "landing",
        "manual",
        "program",
        "maintenance",
    }:
        return True
    if re.fullmatch(r"\d+[/.-]\d+[/.-]\d+", line):
        return True
    if re.fullmatch(r"(?:AC|FAA|CCAR)\s*[-\w. ]{1,12}", line, flags=re.I):
        return True
    if re.fullmatch(r"(?:Page|Par)\s+[\w() -]+", line, flags=re.I):
        return True
    return False


def _meaningful_highlight_lines(value: str, max_lines: int = 12) -> List[str]:
    lines: List[str] = []
    for raw_line in str(value or "").splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line or line == "[SECTION]" or _is_noisy_highlight_line(line):
            continue
        lines.append(line)
        if len(lines) >= max_lines:
            break
    return lines


def _select_source_highlight(source: Dict) -> str:
    content = str(source.get("content") or "")
    section = str(source.get("section_title") or source.get("heading") or "")
    raw_highlight = str(source.get("highlight_text") or "")
    haystack = _compact_evidence_text(f"{content}\n{section}")

    raw_compact = _compact_evidence_text(raw_highlight)
    if raw_highlight and len(raw_highlight) <= 220 and raw_compact and raw_compact in haystack:
        return raw_highlight

    for line in _meaningful_highlight_lines(raw_highlight):
        compact_line = _compact_evidence_text(line)
        if compact_line and compact_line in haystack:
            return line[:220]

    for line in _meaningful_highlight_lines(content):
        if len(line) >= 6:
            return line[:220]

    for line in _meaningful_highlight_lines(section):
        return line[:220]

    return ""


def _normalize_response_source(source: Dict) -> Dict:
    normalized = ensure_evidence_score_fields(source)
    normalized["highlight_text"] = _select_source_highlight(normalized)
    return normalized


def _finalize_source_payload(payload: Optional[Dict], engine: Optional[HybridSearch]) -> Optional[Dict]:
    if not payload:
        return payload
    sources = payload.get("sources")
    if isinstance(sources, list):
        finalized = dict(payload)
        finalized["sources"] = [
            _normalize_response_source(source)
            for source in _repair_response_sources(engine, sources)
        ]
        finalized["evidenceIds"] = [row.get("chunk_id", "") for row in finalized["sources"] if row.get("chunk_id")]
        return finalized
    return payload


def _abstain_payload(query_type: str, reason: str, *, contract: Optional[RetrievalContract] = None, evidence_ids: Optional[List[str]] = None) -> Dict:
    payload = {
        "answer": _abstain_message(query_type),
        "sources": [],
        "confidence": 0.0,
        "abstainReason": reason,
        "evidenceIds": evidence_ids or [],
        "queryType": query_type,
    }
    if contract is not None:
        payload["retrievalContract"] = contract.to_dict()
    return payload


def _apply_contract_to_answer(payload: Optional[Dict], contract: RetrievalContract, query_type: str) -> Optional[Dict]:
    if not payload:
        return None
    contracted_sources = apply_contract(
        contract,
        payload.get("sources", []),
        keep_failed_when_no_hard_constraints=True,
    )
    if not contracted_sources:
        return _abstain_payload(query_type, "contract_no_supporting_evidence", contract=contract)
    enriched = dict(payload)
    enriched["sources"] = contracted_sources
    enriched["queryType"] = query_type
    enriched["retrievalContract"] = contract.to_dict()
    enriched["evidenceIds"] = [row.get("chunk_id", "") for row in contracted_sources if row.get("chunk_id")]
    return enriched


def _finalized_contract_answer_or_none(payload: Optional[Dict], contract: RetrievalContract, query_type: str, engine: Optional[HybridSearch]) -> Optional[Dict]:
    contracted = _apply_contract_to_answer(payload, contract, query_type) if payload else None
    if not contracted or contracted.get("abstainReason"):
        return None
    return _finalize_source_payload(contracted, engine)


def _estimate_confidence(search_results: List[Dict]) -> float:
    if not search_results:
        return 0.0
    score_component = evidence_display_score(search_results[0])
    parse_component = sum(float(row.get("parse_confidence", 0.0)) for row in search_results[:3]) / max(min(len(search_results), 3), 1)
    diversity_component = min(len(search_results) / 3.0, 1.0)
    confidence = 0.55 * score_component + 0.25 * parse_component + 0.20 * diversity_component
    contract_scores = [
        float(row.get("contract_score", 0.0) or 0.0)
        for row in search_results[:3]
        if row.get("contract_passed") is not False
    ]
    if contract_scores:
        contract_component = min(max(sum(contract_scores) / (3.0 * len(contract_scores)), 0.0), 1.0)
        confidence = 0.72 * confidence + 0.28 * contract_component
    elif any(row.get("contract_passed") is False for row in search_results[:3]):
        confidence *= 0.55
    return round(confidence, 4)


def _abstain_message(query_type: str) -> str:
    if query_type == "fault_diagnosis":
        return "当前证据不足，无法给出可靠的故障排查结论。建议补充故障现象、相关章节或对应机型手册后再查询。"
    if query_type == "article_lookup":
        return "当前证据不足，无法确认对应条款或适用条件。建议补充更明确的条款编号、章节号或文档名称。"
    if query_type == "procedure":
        return "当前证据不足，无法给出可靠的操作步骤。建议补充具体部件、章节或维护任务编号。"
    return "当前证据不足，无法给出可靠结论。建议补充更具体的问题描述或相关章节信息。"


def _llm_unavailable_message(search_results: List[Dict], query_type: str) -> str:
    if not search_results:
        return _abstain_message(query_type)

    top = search_results[0]
    doc_name = top.get("doc_name", "检索文档")
    page_number = top.get("page_number", 1)
    section = top.get("section_title") or top.get("heading") or top.get("parent_heading") or "相关章节"
    return f"已检索到相关证据，但当前生成服务不可用。可先查看《{doc_name}》第 {page_number} 页的“{section}”原文。"


def _search_error_message() -> str:
    return "检索服务暂时不可用，当前无法返回可靠答案。请稍后重试并检查 Elasticsearch、Milvus 与模型服务状态。"

