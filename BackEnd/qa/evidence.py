
from search.common import logger
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

from search.evidence_scoring import _has_unmodified_subject, logger
from search.hybrid_search import HybridSearch, logger
from search.query_analysis import analyze_query_intent, infer_query_precision_hints, logger

def _compact_evidence_text(value: str) -> str:
    return re.sub(r"[^\u4e00-\u9fffA-Za-z0-9.]+", "", str(value or "").lower())


def _query_terms(query: str) -> List[str]:
    terms = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9.]{2,}", str(query or ""))
    stop_terms = {"什么", "多少", "共有", "几个", "几条", "内容", "具体", "规定", "要求", "标准"}
    deduped: List[str] = []
    for term in terms:
        if term in stop_terms or term in deduped:
            continue
        deduped.append(term)
    return deduped


def _evidence_haystack(row: Dict) -> str:
    return "\n".join(
        str(row.get(key) or "")
        for key in (
            "doc_name",
            "section_title",
            "heading",
            "parent_heading",
            "highlight_text",
            "clause_path",
            "content",
        )
    )


def _is_structural_evidence(row: Dict) -> bool:
    source_type = str(row.get("source_type") or "")
    layout_type = str(row.get("layout_type") or "")
    return source_type.startswith("structural") or layout_type == "structural" or bool(row.get("structural_node_id"))


def _has_unmodified_subject(text: str, subject: str, modifiers: tuple[str, ...] = ("出口", "外国", "特殊", "标准", "临时", "限制", "生产", "型号")) -> bool:
    compact_text = _compact_evidence_text(text)
    compact_subject = _compact_evidence_text(subject)
    if not compact_text or not compact_subject:
        return False
    start = 0
    while True:
        index = compact_text.find(compact_subject, start)
        if index < 0:
            return False
        prefix = compact_text[max(0, index - 4) : index]
        if not any(prefix.endswith(_compact_evidence_text(modifier)) for modifier in modifiers):
            return True
        start = index + len(compact_subject)


def _direct_title_text(row: Dict) -> str:
    title = str(row.get("section_title") or row.get("heading") or row.get("highlight_text") or "")
    if ">" in title:
        title = title.split(">")[-1]
    return title


def _qa_title_alignment_score(row: Dict, precision_hints: Dict) -> float:
    route = str(precision_hints.get("route") or "")
    if route not in {"classification", "issuance_requirement", "application_requirement", "procedure", "method", "definition", "scope", "document_scope", "time_limit", "numeric_threshold"}:
        return 0.0
    subject_terms = [term for term in precision_hints.get("subject_terms", []) if term]
    intent_terms = [_compact_evidence_text(term) for term in precision_hints.get("intent_terms", []) if term]
    if not subject_terms:
        return 0.0
    title = _direct_title_text(row)
    compact_title = _compact_evidence_text(title)
    has_subject = any(_has_unmodified_subject(compact_title, term) for term in subject_terms)
    has_modified_subject = any(_compact_evidence_text(term) in compact_title for term in subject_terms) and not has_subject
    has_intent = any(term and term in compact_title for term in intent_terms)
    score = 0.0
    if has_subject:
        score += 1.0
    if has_subject and has_intent:
        score += 1.35
    elif has_intent and not has_subject:
        score -= 0.95
    if has_modified_subject:
        score -= 1.15
    return score


def _evidence_quality_score(query: str, row: Dict, profile: Dict) -> float:
    raw_score = evidence_ranking_score(row)
    parse_confidence = max(min(float(row.get("parse_confidence", 0.0) or 0.0), 1.0), 0.0)
    haystack = _compact_evidence_text(_evidence_haystack(row))
    terms = _query_terms(query)
    matched_terms = sum(1 for term in terms if _compact_evidence_text(term) in haystack)
    term_score = matched_terms / max(len(terms), 1)
    # Normalize raw_score with sigmoid to compress into [0, 1] range
    normalized_score = 1.0 / (1.0 + math.exp(-max(min(raw_score, 8.0), -8.0)))
    score = normalized_score + 0.35 * parse_confidence + 0.65 * term_score

    if _is_structural_evidence(row):
        score += 0.35
    if row.get("bbox"):
        score += 0.12
    if str(row.get("chunk_role") or "") in set(profile.get("target_roles", [])):
        score += 0.18

    target_doc_name = str(profile.get("target_doc_name") or "")
    if target_doc_name:
        score += 0.30 if str(row.get("doc_name") or "") == target_doc_name else -0.35

    query_signature = profile.get("query_signature", {})
    article_variants = query_signature.get("article_label_variants", []) or query_signature.get("article_id_variants", [])
    if article_variants and any(_compact_evidence_text(variant) in haystack for variant in article_variants):
        score += 0.50
    elif article_variants:
        score -= 0.30

    if str(row.get("source_type") or "").endswith("toc"):
        score -= 0.50

    precision_hints = profile.get("precision_hints") or infer_query_precision_hints(query)
    haystack_compact = _compact_evidence_text(_evidence_haystack(row))
    article_number = str(row.get("article_id") or row.get("structural_number") or "").replace("第", "").replace("条", "")
    if precision_hints.get("target_doc_name"):
        score += 0.45 if str(row.get("doc_name") or "") == precision_hints["target_doc_name"] else -0.65
    score += _qa_title_alignment_score(row, precision_hints)
    preferred_articles = {str(value) for value in precision_hints.get("preferred_articles", []) if value}
    if preferred_articles:
        score += 1.1 if article_number in preferred_articles else -0.25
    if article_number in {str(value) for value in precision_hints.get("related_articles", []) if value}:
        score += 0.25
    if article_number in {str(value) for value in precision_hints.get("negative_articles", []) if value}:
        score -= 1.3
    required_terms = [_compact_evidence_text(term) for term in precision_hints.get("required_terms", []) if term]
    if required_terms:
        matched = sum(1 for term in required_terms if term in haystack_compact)
        if precision_hints.get("strict_term_match") and matched == 0:
            score -= 1.6
        else:
            score += 0.18 * matched
    forbidden_terms = [_compact_evidence_text(term) for term in precision_hints.get("forbidden_terms", []) if term]
    if forbidden_terms and any(term in haystack_compact for term in forbidden_terms):
        score -= 2.5
    return round(score, 6)


def _multi_target_doc_names(contract: Optional[RetrievalContract]) -> List[str]:
    if contract is None or not contract.allow_multi_doc:
        return []
    return [str(name or "") for name in contract.synced_target_doc_names if name]


def _source_identity_key(row: Dict) -> tuple:
    return (
        str(row.get("doc_name") or ""),
        int(row.get("page_number") or 1),
        str(row.get("chunk_id") or row.get("structural_node_id") or row.get("article_id") or row.get("section_title") or row.get("heading") or ""),
    )


def _ensure_multi_target_prompt_coverage(selected: List[Dict], candidates: List[Dict], contract: Optional[RetrievalContract], max_items: int) -> List[Dict]:
    target_doc_names = _multi_target_doc_names(contract)
    if len(target_doc_names) < 2:
        return selected

    output = [dict(row) for row in selected]
    seen_docs = {str(row.get("doc_name") or "") for row in output}
    seen_keys = {_source_identity_key(row) for row in output}
    additions: List[Dict] = []

    for target_doc_name in target_doc_names:
        if target_doc_name in seen_docs:
            continue
        target_row = next((row for row in candidates if row.get("doc_name") == target_doc_name), None)
        if target_row is None:
            continue
        key = _source_identity_key(target_row)
        if key in seen_keys:
            continue
        additions.append(dict(target_row))
        seen_docs.add(target_doc_name)
        seen_keys.add(key)

    if not additions:
        return output[:max_items]

    merged: List[Dict] = []
    seen_keys.clear()

    def append(row: Dict) -> None:
        key = _source_identity_key(row)
        if key in seen_keys:
            return
        seen_keys.add(key)
        merged.append(row)

    if output:
        append(output[0])
    for row in additions:
        append(row)
    for row in output[1:]:
        append(row)
        if len(merged) >= max_items:
            break
    return merged[:max_items]


def _select_prompt_evidence(query: str, search_results: List[Dict], query_type: str, contract: Optional[RetrievalContract] = None) -> List[Dict]:
    if not search_results:
        return []

    profile = analyze_query_intent(query)
    target_doc_name = _resolve_doc_name_from_text(query)
    if target_doc_name:
        profile["target_doc_name"] = target_doc_name
    max_items = 4 if query_type in {"fault_diagnosis", "procedure"} or len(_multi_target_doc_names(contract)) >= 2 else 3
    scored = []
    for rank, row in enumerate(search_results):
        rank_bonus = max(0.0, 0.32 - 0.05 * rank)
        if rank == 0 and _is_structural_evidence(row):
            rank_bonus += 0.18
        scored.append((_evidence_quality_score(query, row, profile) + rank_bonus, row))
    scored.sort(key=lambda item: item[0], reverse=True)
    if not scored:
        return []

    top_score, top_row = scored[0]
    top_doc = str(top_row.get("doc_name") or "")
    structural_top = _is_structural_evidence(top_row)
    min_keep_score = max(0.58, top_score * (0.52 if not structural_top else 0.74))
    selected: List[Dict] = []
    seen_keys = set()
    target_roles = {role for role in profile.get("target_roles", []) if role != "text"}

    for quality, row in scored:
        if selected and quality < min_keep_score:
            continue
        if structural_top and selected and top_doc and row.get("doc_name") != top_doc and quality < top_score * 0.85:
            continue
        if structural_top and selected and not _is_structural_evidence(row) and top_doc and row.get("doc_name") != top_doc and quality < top_score * 0.60:
            continue
        if selected and target_roles and str(row.get("chunk_role") or "") not in target_roles and quality < top_score * 0.95:
            continue
        key = (
            row.get("doc_name", ""),
            int(row.get("page_number") or 1),
            row.get("article_id") or row.get("structural_node_id") or row.get("section_title") or row.get("heading") or "",
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        enriched = dict(row)
        enriched["evidence_quality"] = quality
        selected.append(enriched)
        if len(selected) >= max_items:
            break

    selected = selected or [dict(top_row)]
    return _ensure_multi_target_prompt_coverage(selected, [row for _, row in scored], contract, max_items)


def _repair_response_sources(engine: Optional[HybridSearch], sources: List[Dict]) -> List[Dict]:
    if not sources or engine is None or not hasattr(engine, "repair_evidence_location"):
        return sources
    repaired_sources: List[Dict] = []
    for source in sources:
        try:
            repaired_sources.append(engine.repair_evidence_location(source))
        except Exception as exc:
            logger.debug("证据定位修复失败 chunk_id=%s: %s", source.get("chunk_id"), exc)
            repaired_sources.append(source)
    return repaired_sources

