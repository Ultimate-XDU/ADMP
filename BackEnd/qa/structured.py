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

from search.common import DOC_DIR, logger

def _normalize_lookup_text(value: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", str(value or "").lower())


def _chinese_article_number_to_int(value: str) -> Optional[int]:
    token = re.sub(r"\s+", "", str(value or ""))
    if not token:
        return None
    if token.isdigit():
        return int(token)
    if all(ch in _CN_NUMERAL_VALUES for ch in token):
        total = 0
        for ch in token:
            total = total * 10 + _CN_NUMERAL_VALUES[ch]
        return total
    if "\u5341" in token:
        left, _, right = token.partition("\u5341")
        tens = _CN_NUMERAL_VALUES.get(left, 1) if left else 1
        ones = _CN_NUMERAL_VALUES.get(right, 0) if right else 0
        return tens * 10 + ones
    return None


def _find_pdf_named_in_query(query: str) -> Optional[Path]:
    query_key = _normalize_lookup_text(query)
    candidates = []
    for file_path in DOC_DIR.glob("*.pdf"):
        stem_key = _normalize_lookup_text(file_path.stem)
        if stem_key and stem_key in query_key:
            candidates.append((len(stem_key), file_path))
    if not candidates:
        return None
    candidates.sort(reverse=True, key=lambda item: item[0])
    return candidates[0][1]


def _find_pdf_by_query_reference(query: str) -> Optional[Path]:
    named = _find_pdf_named_in_query(query)
    if named:
        return named

    date_match = re.search(r"(\d{4})\s*\u5e74\s*(\d{1,2})\s*\u6708\s*(\d{1,2})\s*\u65e5", query)
    if not date_match:
        return None
    normalized_date = f"{int(date_match.group(1))}\u5e74{int(date_match.group(2))}\u6708{int(date_match.group(3))}\u65e5"

    for file_path in DOC_DIR.glob("*.pdf"):
        try:
            document = fitz.open(file_path)
            try:
                front_text = "\n".join(
                    document.load_page(index).get_text("text")
                    for index in range(min(document.page_count, 2))
                )
            finally:
                document.close()
        except Exception:
            continue
        if normalized_date in re.sub(r"\s+", "", front_text):
            return file_path
    return None


@lru_cache(maxsize=128)
def _count_pdf_articles(file_name: str) -> Optional[Dict]:
    file_path = DOC_DIR / Path(file_name).name
    if not file_path.exists() or file_path.suffix.lower() != ".pdf":
        return None

    articles = []
    try:
        document = fitz.open(file_path)
        try:
            for page_index in range(document.page_count):
                page_text = document.load_page(page_index).get_text("text")
                for match in re.finditer(r"\u7b2c\s*([0-9\u96f6\u3007\u4e00\u4e8c\u4e24\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u767e\u5343]+)\s*\u6761", page_text):
                    number = _chinese_article_number_to_int(match.group(1))
                    if number is None or number <= 0 or number > 300:
                        continue
                    start = max(0, match.start() - 80)
                    end = min(len(page_text), match.start() + 360)
                    articles.append(
                        {
                            "number": number,
                            "label": match.group(0),
                            "page_number": page_index + 1,
                            "excerpt": re.sub(r"\s+", " ", page_text[start:end]).strip(),
                        }
                    )
        finally:
            page_count = document.page_count
            document.close()
    except Exception as exc:
        logger.debug("Failed to count PDF articles for %s: %s", file_name, exc)
        return None

    unique = {}
    for item in articles:
        unique.setdefault(item["number"], item)
    if not unique:
        return None

    numbers = sorted(unique)
    max_number = numbers[-1]
    is_consecutive = numbers == list(range(1, max_number + 1))
    count = max_number if is_consecutive else len(numbers)
    last = unique[max_number]
    labels = "\u3001".join(unique[number]["label"] for number in numbers)
    content = (
        f"\u4ece PDF \u539f\u6587\u8bc6\u522b\u5230\u7684\u6761\u6b3e\u7f16\u53f7\uff1a{labels}\u3002\n"
        f"\u6700\u540e\u4e00\u6761\u4e3a {last['label']}\uff1a{last['excerpt']}"
    )
    return {
        "count": count,
        "last": last,
        "content": content,
        "page_count": page_count,
        "numbers": numbers,
    }


def _try_answer_article_count(query: str, query_type: str) -> Optional[Dict]:
    if not any(marker in query for marker in _ARTICLE_COUNT_MARKERS):
        return None
    file_path = _find_pdf_by_query_reference(query)
    if not file_path:
        return None
    counted = _count_pdf_articles(file_path.name)
    if not counted:
        return None

    stem = file_path.stem
    count = int(counted["count"])
    last = counted["last"]
    source = {
        "content": counted["content"],
        "doc_name": file_path.name,
        "heading": stem,
        "parent_heading": "",
        "score": 1.0,
        "chunk_id": f"pdf-article-count:{file_path.name}",
        "page_number": int(last.get("page_number", 1)),
        "page_count": int(counted.get("page_count", 1)),
        "section_title": stem,
        "highlight_text": str(last.get("label", "")),
        "bbox": None,
        "source_type": "pdf_text",
        "layout_type": "single_column",
        "parse_confidence": 1.0,
        "document_type": "REG",
        "is_ocr_fallback": False,
    }
    answer = (
        f"\u300a{stem}\u300b\u5171\u6709 {count} \u6761\u3002\n\n"
        f"\u4f9d\u636e PDF \u539f\u6587\u7684\u6761\u6b3e\u7f16\u53f7\u7edf\u8ba1\uff0c"
        f"\u8be5\u89c4\u5b9a\u4ece\u7b2c\u4e00\u6761\u7f16\u81f3{last.get('label', '')}\u3002"
    )
    return {
        "answer": answer,
        "sources": [source],
        "confidence": 1.0,
        "abstainReason": "",
        "evidenceIds": [source["chunk_id"]],
        "queryType": query_type,
    }


@lru_cache(maxsize=128)
def _extract_pdf_applicability_section(file_name: str) -> Optional[Dict]:
    file_path = DOC_DIR / Path(file_name).name
    if not file_path.exists() or file_path.suffix.lower() != ".pdf":
        return None

    try:
        document = fitz.open(file_path)
        try:
            page_count = document.page_count
            pages = []
            for page_index in range(min(page_count, 4)):
                pages.append((page_index + 1, document.load_page(page_index).get_text("text")))
        finally:
            document.close()
    except Exception as exc:
        logger.debug("Failed to extract applicability section for %s: %s", file_name, exc)
        return None

    combined = "\n".join(text for _, text in pages)
    pattern = re.compile(
        r"(?P<label>(?:\d+|[一二三四五六七八九十]+)\s*[．\.\u3001]\s*\u9002\u7528\u8303\u56f4)\s*"
        r"(?P<body>.*?)(?=\n\s*(?:\d+|[一二三四五六七八九十]+)\s*[．\.\u3001]\s*[\u4e00-\u9fff]{2,20}|$)",
        re.S,
    )
    match = pattern.search(combined)
    if not match:
        # Fallback: match 适用范围 with colon or space separator, no number prefix required
        pattern2 = re.compile(
            r"(?P<label>适用范围)\s*[:：\s]+"
            r"(?P<body>.+?)(?=\n\s*(?:[一-鿿]{2,8}[:：]|$))",
            re.S,
        )
        match = pattern2.search(combined)
    if not match:
        return None

    label = re.sub(r"\s+", "", match.group("label"))
    raw_section = f"{match.group('label')}\n{match.group('body')}"
    section = re.sub(r"\n\s*\u2014\s*\d+\s*\u2014\s*", "\n", raw_section)
    section = re.sub(r"[ \t]+", " ", section)
    section = re.sub(r"\n{3,}", "\n\n", section).strip()
    # Clean up TOC noise (dot leaders, page numbers)
    section = re.sub(r"[.\u00b7\u3002\uff0e]{3,}\s*-?\s*\d+\s*-?\s*", "", section)
    section = re.sub(r"^\s*\d+\s*$", "", section, flags=re.M)
    section = re.sub(r"\n{3,}", "\n\n", section).strip()
    compact_section = re.sub(r"\s+", "", section)

    page_number = 1
    for current_page, text in pages:
        if "\u9002\u7528\u8303\u56f4" in text:
            page_number = current_page
            break

    return {
        "label": label,
        "section": section,
        "compact_section": compact_section,
        "page_number": page_number,
        "page_count": page_count,
    }


def _try_answer_document_applicability(query: str, query_type: str) -> Optional[Dict]:
    if "\u9002\u7528\u8303\u56f4" not in query:
        return None
    # Accept queries that mention a document name OR contain scope-related markers
    has_doc_reference = _find_pdf_by_query_reference(query) is not None
    has_scope_marker = any(marker in query for marker in ("\u4e0b\u53d1", "\u6587\u4ef6", "\u54a8\u8be2\u901a\u544a", "\u672c\u6587\u4ef6", "\u300a", "\u300b"))
    if not has_doc_reference and not has_scope_marker:
        return None

    file_path = _find_pdf_by_query_reference(query)
    if not file_path:
        return None
    extracted = _extract_pdf_applicability_section(file_path.name)
    if not extracted:
        return None

    stem = file_path.stem
    section = extracted["section"]
    answer_body = re.sub(r"^\s*(?:\d+|[一二三四五六七八九十]+)\s*[．\.\u3001]\s*\u9002\u7528\u8303\u56f4\s*", "", section, flags=re.S).strip()
    source = {
        "content": section,
        "doc_name": file_path.name,
        "heading": extracted["label"],
        "parent_heading": stem,
        "score": 1.0,
        "chunk_id": f"pdf-applicability:{file_path.name}",
        "page_number": int(extracted["page_number"]),
        "page_count": int(extracted["page_count"]),
        "section_title": extracted["label"],
        "highlight_text": extracted["label"],
        "bbox": None,
        "source_type": "pdf_text",
        "layout_type": "single_column",
        "parse_confidence": 1.0,
        "document_type": "AC",
        "is_ocr_fallback": False,
    }
    return {
        "answer": f"\u300a{stem}\u300b\u7684\u9002\u7528\u8303\u56f4\u4e3a\uff1a\n\n{answer_body}",
        "sources": [source],
        "confidence": 1.0,
        "abstainReason": "",
        "evidenceIds": [source["chunk_id"]],
        "queryType": query_type,
    }


def _is_valid_query(query: str) -> bool:
    """Reject empty, whitespace-only, or nonsensical queries."""
    compact = re.sub(r"\s+", "", str(query or ""))
    if len(compact) < 2:
        return False
    # Must contain at least one Chinese character for aerospace domain
    chinese_chars = re.findall(r"[一-鿿]", compact)
    if len(chinese_chars) < 1:
        return False
    # Check if query contains at least some meaningful characters
    meaningful = re.findall(r"[一-鿿A-Za-z0-9]", compact)
    if len(meaningful) < 2:
        return False
    # Reject if > 70% is non-meaningful characters
    if len(meaningful) / len(compact) < 0.3:
        return False
    return True


def _definition_query_term(query: str) -> str:
    cleaned = re.sub(r"\s+", "", str(query or ""))
    if not cleaned:
        return ""
    match = re.search(r"^(?P<term>.+?)的定义(?:是|是什么|为)?$", cleaned)
    if match:
        return match.group("term")
    match = re.search(r"^(?P<term>.+?)是什么$", cleaned)
    if match:
        term = match.group("term")
        if not any(token in term for token in ("颁发", "条件", "要求", "步骤", "程序", "流程", "申请", "审查", "如何", "怎样", "哪个", "哪些", "怎么")):
            return term
    return ""

