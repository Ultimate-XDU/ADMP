import concurrent.futures
import hashlib
import math
import os
import re
import threading
import time
import warnings
import json
import logging
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from search.document_pipeline import (
    _derive_section_title,
    _extract_highlight_text,
    _restore_article_reference,
    _split_text_candidates,
    logger,
)
from search.query_analysis import _contract_focus_terms, logger
from search.common import (
    ACTION_VARIANT_MODIFIERS,
    ROUTE_TITLE_ACTION_TERMS,
    SPECIFICITY_MODIFIERS,
    _article_reference_matches,
    _compact_match_text,
    _extract_highlight_text,
    _looks_like_toc_or_index_listing,
    _ngram_similarity,
    _token_overlap_score,
    clean_block_text,
    logger,
)

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ["HF_ENABLE_PARALLEL_LOADING"] = os.environ.get("AEROSPACE_HF_ENABLE_PARALLEL_LOADING", "false")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import fitz
import numpy as np
import torch
from docx import Document
from elasticsearch import Elasticsearch, helpers
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility
from sentence_transformers import CrossEncoder, SentenceTransformer

from utils.evidence_scores import calibrate_ranking_score, normalize_structural_score
from utils.es_query_builder import (
    AliasClauseProfile,
    ArticleReferenceProfile,
    CodeVariantProfile,
    EsQueryPlan,
    PrecisionHintProfile,
    WeightedField,
    build_alias_clauses,
    build_article_reference_clauses,
    build_code_variant_clauses,
    build_precision_hint_clauses,
    es_hit_to_channel_result,
    match_clause,
    match_phrase_clause,
    terms_filter,
)
from utils.structure import (
    answer_structural_classification,
    answer_structural_count,
    answer_structural_definition,
    attach_structure_to_blocks,
    build_structural_records,
    clear_structural_records,
    load_all_structural_records,
    remove_structural_records,
    save_structural_records,
)
from utils.retrieval_contract import (
    RetrievalContract,
    apply_contract,
    build_retrieval_contract,
    code_query_variants,
    contract_needs_multi_target_recall,
)
from utils.query_utils import compact_query_text, cross_lingual_query_terms, classify_query_intent

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
def _query_alignment_score(query: str, evidence: dict, profile: dict) -> float:
    highlight = clean_block_text(evidence.get("highlight_text", ""))
    section = clean_block_text(evidence.get("section_title", ""))
    content = clean_block_text(evidence.get("content", ""))[:420]
    combined = "\n".join(part for part in [section, highlight, content] if part)
    query_signature = profile.get("query_signature", {})

    score = 0.85 * _token_overlap_score(query, highlight) + 0.6 * _token_overlap_score(query, section) + 0.2 * _ngram_similarity(query, combined)
    if _article_reference_matches(evidence.get("article_id", ""), evidence.get("article_label", ""), query_signature):
        score += 0.9
    if query_signature.get("normalized_heading") and evidence.get("normalized_heading") == query_signature.get("normalized_heading"):
        score += 0.45
    if evidence.get("chunk_role") in profile.get("target_roles", []):
        score += 0.18
    if evidence.get("chunk_role") == "effective_date" and "effective_date" in profile.get("target_roles", []):
        score += 0.65
    focus_terms = _contract_focus_terms(profile)
    if focus_terms:
        focus_haystack = _compact_match_text(combined)
        focus_scores = [
            max(_token_overlap_score(term, focus_haystack), _ngram_similarity(term, focus_haystack))
            for term in focus_terms
            if term
        ]
        if focus_scores:
            score += 0.55 * (sum(focus_scores) / len(focus_scores))
    target_doc_name = profile.get("target_doc_name", "")
    if target_doc_name:
        if evidence.get("doc_name") == target_doc_name:
            score += 0.65
        else:
            score -= 0.7

    return round(score, 6)


def _front_matter_penalty(metadata: dict, content: str, profile: dict) -> float:
    text = clean_block_text(
        "\n".join(
            part
            for part in [
                metadata.get("section_title", ""),
                metadata.get("heading_text", ""),
                metadata.get("title", ""),
                metadata.get("label", ""),
                metadata.get("structural_path", ""),
                metadata.get("scope_anchor", ""),
                content[:500],
            ]
            if part
        )
    )
    compact = _compact_match_text(text)
    if not compact:
        return 0.0

    penalty = 0.0
    if _looks_like_toc_or_index_listing(text):
        penalty += 0.72
    if any(token in compact for token in ("preface", "contents", "acknowledgment", "figurepage", "tableofcontents", "目录", "前言", "封面")):
        penalty += 0.28
    if any(
        token in compact
        for token in (
            "advisorycircularfeedbackform",
            "feedbackform",
            "relatedmaterial",
            "bibliography",
            "appendixa",
            "appendixb",
            "contactus",
            "recommendparagraph",
            "submittedby",
            "参考文献",
            "参考资料",
            "附录",
            "反馈表",
            "联系方式",
        )
    ):
        penalty += 0.42
    elif "references" in compact and len(re.findall(r"\b(?:available at|retrieved from|https?://|www\.|\(\d{4}\)|\d{4}\.)", text, flags=re.I)) >= 3:
        penalty += 0.42

    try:
        page_number = int(metadata.get("page_number") or 1)
    except Exception:
        page_number = 1
    if page_number <= 5 and profile.get("query_type") in {"procedure", "fault_diagnosis", "general"}:
        if any(token in compact for token in ("contents", "figurepage", "section1section2", "preface", "acknowledgment")):
            penalty += 0.25
    try:
        page_count = int(metadata.get("page_count") or 0)
    except Exception:
        page_count = 0
    if page_count and page_number >= max(page_count - 1, 1):
        if any(token in compact for token in ("feedbackform", "submittedby", "recommendparagraph", "contact", "references", "参考文献", "反馈表")):
            penalty += 0.28

    if len(compact) < 24:
        penalty += 0.12

    source_type = str(metadata.get("source_type") or "").lower()
    if source_type.startswith("structural"):
        return min(1.35, penalty)
    return min(1.15, penalty)


def _looks_like_reference_material(metadata: dict, content: str) -> bool:
    text = clean_block_text(
        "\n".join(
            str(part or "")
            for part in (
                metadata.get("section_title", ""),
                metadata.get("highlight_text", ""),
                metadata.get("heading", ""),
                metadata.get("heading_text", ""),
                metadata.get("structural_path", ""),
                content[:900],
            )
            if part
        )
    )
    compact = _compact_match_text(text)
    if not compact:
        return False
    if any(token in compact for token in ("advisorycircularfeedbackform", "feedbackform", "submittedby", "recommendparagraph", "参考文献", "反馈表")):
        return True
    if any(token in compact for token in ("appendixa", "relatedmaterial", "bibliography", "参考资料", "附录")):
        return True
    citation_hits = len(re.findall(r"\b(?:available at|retrieved from|https?://|www\.|doi:|\(\d{4}\)|\d{4}\.)", text, flags=re.I))
    short_numbered_refs = len(re.findall(r"(?:^|\s)\d{1,2}\.\s+[A-Z][A-Za-z .,'’&()-]{12,}\.\s*(?:\(\d{4}\)|\d{4}\.)", text))
    return citation_hits >= 3 or short_numbered_refs >= 3 or ("references" in compact and citation_hits >= 2)


def _row_article_number(row: dict) -> str:
    return clean_block_text(
        str(
            row.get("article_id")
            or row.get("structural_number")
            or row.get("number")
            or row.get("article_label")
            or ""
        )
    ).replace("第", "").replace("条", "")


def _row_section_number(row: dict) -> str:
    return clean_block_text(str(row.get("section_id") or row.get("structural_number") or row.get("number") or ""))


def _row_precision_haystack(row: dict, fallback: str = "") -> str:
    return _compact_match_text(
        "\n".join(
            str(row.get(key) or "")
            for key in (
                "doc_name",
                "section_title",
                "heading",
                "heading_text",
                "parent_heading",
                "highlight_text",
                "clause_path",
                "structural_path",
                "scope_anchor",
                "heading_path_text",
                "article_label",
                "title",
                "label",
                "content",
                "retrieval_text",
                "body_text",
            )
        )
        + "\n"
        + str(fallback or "")
    )


def _has_unmodified_subject(text: str, subject: str, modifiers: Tuple[str, ...] = SPECIFICITY_MODIFIERS) -> bool:
    compact_text = _compact_match_text(text)
    compact_subject = _compact_match_text(subject)
    if not compact_text or not compact_subject:
        return False
    start = 0
    while True:
        index = compact_text.find(compact_subject, start)
        if index < 0:
            return False
        prefix = compact_text[max(0, index - 4) : index]
        if not any(prefix.endswith(_compact_match_text(modifier)) for modifier in modifiers):
            return True
        start = index + len(compact_subject)


def _definition_anchor_matches(row: dict, term: str, fallback: str = "") -> bool:
    compact_term = _compact_match_text(term)
    if not compact_term:
        return False
    raw = clean_block_text(
        "\n".join(
            str(row.get(key) or "")
            for key in ("title", "label", "heading", "section_title", "highlight_text", "body_text", "content")
        )
        + "\n"
        + str(fallback or "")
    )
    compact_head = _compact_match_text(raw[:180])
    pos = compact_head.find(compact_term)
    if pos < 0 or pos > 18:
        return False
    after = compact_head[pos + len(compact_term) : pos + len(compact_term) + 6]
    return after.startswith(("指", "是指", "为", "定义为")) or "指" in after[:3]


def _node_title_text(row: dict) -> str:
    return _compact_match_text(
        " ".join(
            str(row.get(key) or "")
            for key in ("label", "title", "article_label", "heading_text")
        )
    )


def _title_alignment_score(row: dict, hints: dict, structural: bool = False) -> float:
    route = str(hints.get("route") or "")
    if route not in {
        "classification",
        "issuance_requirement",
        "application_requirement",
        "procedure",
        "definition",
        "scope",
        "document_scope",
        "method",
        "time_limit",
        "numeric_threshold",
    }:
        return 0.0

    title_text = _node_title_text(row)
    if not title_text:
        return 0.0

    subject_terms = [term for term in hints.get("subject_terms", []) if term]
    intent_terms = [_compact_match_text(term) for term in hints.get("intent_terms", []) if term]
    title_keyphrases = [_compact_match_text(term) for term in hints.get("title_keyphrases", []) if term]
    if not subject_terms and not title_keyphrases:
        return 0.0

    has_subject = any(_has_unmodified_subject(title_text, term) for term in subject_terms)
    has_modified_subject = any(_compact_match_text(term) in title_text for term in subject_terms) and not has_subject
    has_intent = any(term and term in title_text for term in intent_terms)
    keyphrase_hits = sum(1 for term in title_keyphrases if term and term in title_text)

    weight = 1.0 if structural else 0.45
    score = 0.0
    if keyphrase_hits:
        score += min(5.4, 2.4 * keyphrase_hits) * weight
    if has_subject:
        score += 2.6 * weight
    if has_subject and has_intent:
        score += 3.4 * weight
    elif has_intent and not has_subject:
        score -= 2.4 * weight
    if has_modified_subject:
        score -= 2.8 * weight

    return round(score, 6)


def _route_title_action_score(row: dict, hints: dict, structural: bool = False) -> float:
    route = str(hints.get("route") or "")
    action_terms = ROUTE_TITLE_ACTION_TERMS.get(route)
    if not action_terms:
        return 0.0

    title_text = _node_title_text(row)
    if not title_text:
        return 0.0

    subject_terms = [_compact_match_text(term) for term in hints.get("subject_terms", []) if term]
    if subject_terms and not any(_has_unmodified_subject(title_text, subject) for subject in subject_terms):
        return 0.0

    scale = 1.0 if structural else 0.34
    matched_actions = [term for term in action_terms if _compact_match_text(term) in title_text]
    if matched_actions:
        score = 2.4 * scale
    elif route in {"issuance_requirement", "application_requirement"}:
        score = -3.0 * scale
    else:
        return 0.0

    compact_query = str(hints.get("compact_query") or "")
    for modifier in ACTION_VARIANT_MODIFIERS:
        modifier_key = _compact_match_text(modifier)
        if modifier_key and modifier_key in title_text and modifier_key not in compact_query:
            score -= 3.4 * scale
    return round(score, 6)


def _precision_hint_score(row: dict, profile: dict, fallback: str = "", structural: bool = False) -> float:
    hints = profile.get("precision_hints") or {}
    if not hints:
        return 0.0
    score = 0.0
    haystack = _row_precision_haystack(row, fallback)
    article_number = _row_article_number(row)
    section_number = _row_section_number(row)

    target_doc_name = str(hints.get("target_doc_name") or "")
    if target_doc_name:
        score += 0.95 if str(row.get("doc_name") or "") == target_doc_name else -1.0

    preferred_articles = {str(value) for value in hints.get("preferred_articles", []) if value}
    related_articles = {str(value) for value in hints.get("related_articles", []) if value}
    negative_articles = {str(value) for value in hints.get("negative_articles", []) if value}
    if preferred_articles:
        if article_number in preferred_articles:
            score += 4.4 if structural else 1.45
        else:
            score -= 0.55 if structural else 0.25
    if related_articles and article_number in related_articles:
        score += 0.65
    if negative_articles and article_number in negative_articles:
        score -= 4.0 if structural else 1.8

    preferred_sections = {str(value) for value in hints.get("preferred_sections", []) if value}
    if preferred_sections:
        if section_number in preferred_sections or any(value in haystack for value in preferred_sections):
            score += 3.2 if structural else 1.1
        else:
            score -= 0.25

    required_terms = [_compact_match_text(term) for term in hints.get("required_terms", []) if term]
    if required_terms:
        matched = sum(1 for term in required_terms if term and term in haystack)
        score += 0.42 * matched
        if hints.get("strict_term_match") and matched == 0:
            score -= 2.2
        elif matched == 0:
            score -= 0.75
        if hints.get("route") == "definition":
            for raw_term in hints.get("required_terms", []):
                if _definition_anchor_matches(row, raw_term, fallback):
                    score += 2.4 if structural else 0.9
                    break

    intent_terms = [_compact_match_text(term) for term in hints.get("intent_terms", []) if term]
    if intent_terms:
        matched_intents = sum(1 for term in intent_terms if term and term in haystack)
        score += min(1.4 if structural else 0.55, 0.28 * matched_intents)
        if matched_intents == 0 and hints.get("route") not in {"numeric_threshold", "time_limit"}:
            score -= 0.35
    title_text = _node_title_text(row)
    title_keyphrases = [_compact_match_text(term) for term in hints.get("title_keyphrases", []) if term]
    if title_keyphrases:
        title_hits = sum(1 for term in title_keyphrases if term and term in title_text)
        body_hits = sum(1 for term in title_keyphrases if term and term in haystack)
        if title_hits:
            score += min(6.0 if structural else 1.8, (2.2 if structural else 0.65) * title_hits)
        elif body_hits:
            score += min(1.2 if structural else 0.4, 0.25 * body_hits)
        elif str(hints.get("route") or "") in {"method", "time_limit", "numeric_threshold", "application_requirement"}:
            score -= 0.65 if structural else 0.2
    score += _title_alignment_score(row, hints, structural=structural)

    route = str(hints.get("route") or "")
    record_type = str(row.get("record_type") or row.get("structural_record_type") or "")
    semantic_role = str(row.get("semantic_role") or row.get("chunk_role") or "")
    if route == "document_scope":
        if semantic_role == "scope":
            score += 1.8 if structural else 0.7
        if record_type in {"article", "section"}:
            score += 0.35
        try:
            page_number = max(int(row.get("page_start") or row.get("page_number") or 1), 1)
            score += max(0.0, 2.4 - 0.12 * (page_number - 1))
            if page_number > 12:
                score -= min(2.5, 0.05 * (page_number - 12))
        except Exception:
            pass
    elif route == "classification" and semantic_role == "classification":
        score += 1.2 if structural else 0.45
    elif route in {"issuance_requirement", "application_requirement"} and semantic_role == "requirement":
        score += 0.8 if structural else 0.35
    elif route == "procedure" and semantic_role == "procedure":
        score += 0.8 if structural else 0.35
    elif route == "method" and semantic_role in {"requirement", "classification", "scope", "procedure"}:
        score += 0.65 if structural else 0.28
    elif route == "time_limit":
        time_haystack = fallback or _row_precision_haystack(row)
        if any(token in time_haystack for token in ("小时", "分钟", "日内", "工作日", "个工作日", "时限")):
            score += 1.5 if structural else 0.55
        if any(token in time_haystack for token in ("法律责任", "罚款", "责令限期改正", "逾期不改正")):
            score -= 1.8 if structural else 0.7

    score += _route_title_action_score(row, hints, structural=structural)

    compact_query_subjects = [_compact_match_text(term) for term in hints.get("subject_terms", []) if term]
    specificity_modifiers = SPECIFICITY_MODIFIERS
    for subject in compact_query_subjects:
        if not subject:
            continue
        has_unmodified_subject = _has_unmodified_subject(haystack, subject, specificity_modifiers)
        for modifier in specificity_modifiers:
            candidate_phrase = f"{modifier}{subject}"
            if (
                candidate_phrase in haystack
                and candidate_phrase not in _compact_match_text("".join(hints.get("subject_terms", [])))
                and not has_unmodified_subject
            ):
                score -= 2.8 if structural else 0.9

    forbidden_terms = [_compact_match_text(term) for term in hints.get("forbidden_terms", []) if term]
    if forbidden_terms and any(term and term in haystack for term in forbidden_terms):
        score -= 4.0

    return round(score, 6)


def _first_meaningful_anchor(text: str, max_chars: int = 120) -> str:
    cleaned = clean_block_text(text)
    if not cleaned:
        return ""
    for line in cleaned.splitlines():
        line = clean_block_text(line)
        if 6 <= len(line) <= max_chars:
            return line
        if len(line) > max_chars:
            return line[:max_chars].rstrip()
    return cleaned[:max_chars].rstrip()


def _structural_highlight_text(record: dict, heading: str, content: str) -> str:
    record_type = str(record.get("record_type") or "")
    semantic_role = str(record.get("semantic_role") or "")
    body_anchor = _first_meaningful_anchor(content, 140)
    if record_type in {"clause", "item"} and body_anchor:
        return body_anchor
    if semantic_role in {"definition", "scope", "effective_date", "requirement", "classification"} and body_anchor:
        return body_anchor
    return heading or body_anchor or clean_block_text(record.get("article_label", ""))


def _select_query_aligned_highlight(query: str, metadata: dict, content: str, fallback: str = "") -> str:
    scored: List[Tuple[float, str]] = []
    query_has_measurement = any(token in query for token in ("厘米", "毫米", "米", "千克", "公斤", "英尺"))
    classification_query = any(token in query for token in ("类别", "分类", "分为", "分成", "包括"))

    seed_candidates = [
        metadata.get("highlight_text", ""),
        _derive_section_title(metadata, content),
        metadata.get("heading_text", ""),
        metadata.get("section_title", ""),
        metadata.get("article_label", ""),
        metadata.get("clause_path", ""),
    ]
    seed_candidates.extend(_split_text_candidates(content))

    seen = set()
    for candidate in seed_candidates:
        restored_candidates = [clean_block_text(candidate), _restore_article_reference(candidate, metadata)]
        for restored in restored_candidates:
            restored = clean_block_text(restored)
            if not restored or restored in seen:
                continue
            seen.add(restored)

            score = _token_overlap_score(query, restored) + 0.9 * _ngram_similarity(query, restored)
            if classification_query and any(token in restored for token in ("类别", "分类", "分为", "分成")):
                score += 0.85
            if restored.startswith("第") and "条" in restored:
                score += 0.18
            if not query_has_measurement and any(token in restored for token in ("厘米", "毫米", "米", "英尺", "公斤", "千克")):
                score -= 0.55
            if 12 <= len(restored) <= 110:
                score += 0.12
            elif len(restored) > 150:
                score -= 0.18

            scored.append((score, restored[:180]))

    if scored:
        scored.sort(key=lambda item: item[0], reverse=True)
        if scored[0][0] > 0.12:
            return scored[0][1]

    return clean_block_text(fallback or _extract_highlight_text(content))[:180]

