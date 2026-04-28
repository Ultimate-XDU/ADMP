import concurrent.futures
import hashlib
import math
import os
import re
import threading
import warnings
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz
import numpy as np
import torch
from docx import Document
from elasticsearch import Elasticsearch, helpers
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility
from sentence_transformers import CrossEncoder, SentenceTransformer

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


BASE_DIR = Path(__file__).resolve().parent
DOC_DIR = str(BASE_DIR / "aerospace_docs")
HEALTH_REPORT_FILE = BASE_DIR / "document_health_reports.json"

MILVUS_HOST = "localhost"
MILVUS_PORT = 19530
MILVUS_COLLECTION = "aerospace_m3e_db"
ES_HOST = "http://localhost:9200"
ES_INDEX = "aerospace_kw_db"

MAX_CHUNK_SIZE = 960
MAX_CHUNK_OVERLAP = 120
MILVUS_CONTENT_MAX = 3500
EMBEDDING_MODEL = "moka-ai/m3e-base"
RERANK_MODEL = "BAAI/bge-reranker-base"
VECTOR_DIM = 768

TOP_K = 5
QA_TOP_K = 4
VEC_RECALL = 120
KW_RECALL = 120
HEADING_RECALL = 40
TABLE_RECALL = 20
ARTICLE_RECALL = 40

ARTICLE_LABEL_PATTERN = re.compile(r"第\s*([0-9A-Za-z一二三四五六七八九十百千万]+(?:\.[0-9A-Za-z一二三四五六七八九十百千万]+)*)\s*条")
SECTION_LABEL_PATTERN = re.compile(r"第\s*([0-9A-Za-z一二三四五六七八九十百千万]+(?:\.[0-9A-Za-z一二三四五六七八九十百千万]+)*)\s*节")
NUMBERED_HEADING_PATTERN = re.compile(r"^([0-9]{1,2}|[一二三四五六七八九十]{1,3})\s*[．.、]\s*([^\n。；;：:]{2,40})$")
CLAUSE_PATTERN = re.compile(r"(?:^|\n)\s*([（(][一二三四五六七八九十百千万\dA-Za-zivxIVX]+[)）]|[一二三四五六七八九十]+、)")
LIST_ITEM_PATTERN = re.compile(r"(?:^|\n)\s*([0-9]{1,3}[\.、]|[A-Za-z][\.\)])")

CLASSIFICATION_TERMS = ("类别", "分类", "分为", "分成", "包括", "类型")
DEFINITION_TERMS = ("定义", "是指", "含义", "术语", "解释")
PROCEDURE_TERMS = ("步骤", "程序", "流程", "操作", "检查", "执行", "更换", "拆装")
REQUIREMENT_TERMS = ("应当", "必须", "不得", "要求", "符合", "条件", "限制", "批准", "颁发", "申请")
NOTE_TERMS = ("注意", "提示", "警告", "注：", "说明")
EFFECTIVE_DATE_TERMS = ("施行", "实施", "生效", "开始施行", "起施行", "废止")
CERTIFICATE_VARIANT_PENALTIES = {
    "出口适航证": 0.18,
    "型号合格证": 0.22,
    "生产许可证": 0.22,
    "特许飞行证": 0.18,
    "标准适航证": 0.08,
    "特殊适航证": 0.08,
    "外国适航证认可书": 0.04,
}
INDEXED_STRUCTURAL_FIELDS = (
    "article_id",
    "article_label",
    "section_id",
    "section_label",
    "clause_id",
    "list_item_id",
    "chunk_role",
    "normalized_heading",
    "has_enumeration",
    "list_density",
    "contains_definition",
    "contains_steps",
    "contains_classification",
    "contains_requirements",
    "title_body_consistency",
    "clause_path",
    "cross_page",
    "boundary_conflict",
)


@dataclass
class IndexOperationResult:
    success: bool
    action: str
    message: str
    doc_name: str = ""
    chunk_count: int = 0
    error_code: str = ""
    health_report: Optional[dict] = None
    details: dict = field(default_factory=dict)


def _normalize_search_text(text: str) -> str:
    normalized = clean_block_text(text).lower()
    return re.sub(r"\s+", " ", normalized).strip()


CHINESE_NUMERAL_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
CHINESE_NUMERAL_UNITS = {"十": 10, "百": 100, "千": 1000, "万": 10000}


def _arabic_to_chinese_number(value: int) -> str:
    if value <= 0:
        return ""
    digits = "零一二三四五六七八九"
    units = ["", "十", "百", "千"]
    if value < 10:
        return digits[value]
    if value < 10000:
        parts: List[str] = []
        zero_pending = False
        for index, digit in enumerate(str(value)):
            digit_value = int(digit)
            unit_index = len(str(value)) - index - 1
            if digit_value == 0:
                zero_pending = bool(parts)
                continue
            if zero_pending:
                parts.append("零")
                zero_pending = False
            if not (digit_value == 1 and unit_index == 1 and not parts):
                parts.append(digits[digit_value])
            parts.append(units[unit_index])
        return "".join(parts) or digits[0]
    return str(value)


def _chinese_to_arabic_number(text: str) -> Optional[int]:
    cleaned = clean_block_text(text)
    if not cleaned:
        return None
    if cleaned.isdigit():
        return int(cleaned)
    if not re.fullmatch(r"[零〇一二两三四五六七八九十百千万]+", cleaned):
        return None

    total = 0
    section = 0
    number = 0
    for char in cleaned:
        if char in CHINESE_NUMERAL_DIGITS:
            number = CHINESE_NUMERAL_DIGITS[char]
            continue
        unit = CHINESE_NUMERAL_UNITS.get(char)
        if unit is None:
            return None
        if unit == 10000:
            section = (section + (number or 1)) * unit
            total += section
            section = 0
            number = 0
            continue
        section += (number or 1) * unit
        number = 0
    return total + section + number


def _expand_reference_id_variants(value: str) -> List[str]:
    cleaned = clean_block_text(value)
    if not cleaned:
        return []

    variants: List[str] = []
    seen = set()

    def push(item: str):
        token = clean_block_text(item)
        if not token or token in seen:
            return
        seen.add(token)
        variants.append(token)

    push(cleaned)
    if "." in cleaned:
        return variants

    if cleaned.isdigit():
        push(_arabic_to_chinese_number(int(cleaned)))
        return variants

    arabic = _chinese_to_arabic_number(cleaned)
    if arabic is not None:
        push(str(arabic))
    return variants


def _expand_article_reference_variants(article_id: str = "", article_label: str = "") -> Tuple[List[str], List[str]]:
    base_id = clean_block_text(article_id)
    if not base_id and article_label:
        extracted_id, _ = _extract_article_reference(article_label)
        base_id = extracted_id
    id_variants = _expand_reference_id_variants(base_id)
    label_variants = [f"第{variant}条" for variant in id_variants]
    if article_label:
        normalized_label = clean_block_text(article_label)
        if normalized_label and normalized_label not in label_variants:
            label_variants.append(normalized_label)
    return id_variants, label_variants


def _article_reference_matches(article_id: str, article_label: str, query_signature: dict) -> bool:
    query_variants = set(query_signature.get("article_id_variants", []))
    if not query_variants:
        return False
    candidate_variants = set(_expand_article_reference_variants(article_id, article_label)[0])
    return bool(candidate_variants & query_variants)


def _query_tokens(query: str) -> List[str]:
    normalized = _normalize_search_text(query)
    if not normalized:
        return []

    ascii_tokens = re.findall(r"[a-z0-9_.-]+", normalized)
    chinese_terms = re.findall(r"[\u4e00-\u9fff]{2,}", normalized)

    tokens: List[str] = []
    seen = set()
    for token in [normalized] + ascii_tokens + chinese_terms:
        token = token.strip()
        if token and token not in seen:
            tokens.append(token)
            seen.add(token)
        if re.fullmatch(r"[\u4e00-\u9fff]{3,}", token):
            for index in range(len(token) - 1):
                gram = token[index : index + 2]
                if gram not in seen:
                    tokens.append(gram)
                    seen.add(gram)
    return tokens


def _token_overlap_score(query: str, text: str) -> float:
    normalized_query = _normalize_search_text(query)
    normalized_text = _normalize_search_text(text)
    if not normalized_query or not normalized_text:
        return 0.0

    score = 0.0
    if normalized_query in normalized_text:
        score += 1.8

    for token in _query_tokens(normalized_query)[1:]:
        if token in normalized_text:
            score += 0.24 if len(token) >= 3 else 0.08
    return round(score, 6)


def _char_ngrams(text: str, n: int = 2) -> set:
    normalized = re.sub(r"\s+", "", _normalize_search_text(text))
    if not normalized:
        return set()
    if len(normalized) <= n:
        return {normalized}
    return {normalized[index : index + n] for index in range(len(normalized) - n + 1)}


def _ngram_similarity(query: str, text: str) -> float:
    query_grams = _char_ngrams(query)
    text_grams = _char_ngrams(text)
    if not query_grams or not text_grams:
        return 0.0
    return round(len(query_grams & text_grams) / math.sqrt(len(query_grams) * len(text_grams)), 6)


def infer_document_type(doc_name: str) -> str:
    upper = doc_name.upper()
    for candidate in ("AMM", "FIM", "TSM", "IPC", "SRM"):
        if candidate in upper:
            return candidate
    return "AMM"


def clean_block_text(text: str) -> str:
    text = str(text or "").replace("\xa0", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
    return text.strip()


def detect_heading(line: str) -> Tuple[bool, int, str]:
    s = clean_block_text(line).split("\n", 1)[0].strip()
    if not s:
        return False, 0, ""
    if re.match(r"^第\s*[0-9一二三四五六七八九十百千]+\s*章", s) or re.match(r"^[A-Z]\s*章", s):
        return True, 1, s
    if re.match(r"^附录\s*[A-Za-z\d]", s):
        return True, 1, s
    if re.match(r"^第\s*[0-9一二三四五六七八九十百千]+\s*节", s):
        return True, 2, s
    if re.match(r"^第\s*[0-9一二三四五六七八九十百千\d\.]+\s*条", s):
        return True, 2, s
    if NUMBERED_HEADING_PATTERN.match(s):
        return True, 2, s
    if re.match(r"^\d+\.\d+(?:\.\d+)*\s+[\u4e00-\u9fffA-Za-z].{1,}", s):
        return True, 2, s
    if re.match(r"^\d{2,3}-\d{2}(?:-\d{2,3})?\s+[\u4e00-\u9fffA-Za-z].{1,}", s):
        return True, 2, s
    if re.match(r"^(?:图|表)\s*\d+(?:[-–]\d+)?", s):
        return True, 2, s
    if re.match(r"^\([a-z]\)\s+", s):
        return True, 3, s
    if re.match(r"^\(\d+\)\s+", s):
        return True, 4, s
    if re.match(r"^\((?:i{1,3}|iv|v|vi{0,3})\)\s+", s):
        return True, 5, s
    return False, 0, ""


def _normalize_heading_value(text: str) -> str:
    value = clean_block_text(text).lower()
    if not value:
        return ""
    replacements = (
        ("办法条件", "颁发条件"),
        ("申请办法", "申请条件"),
        ("申请要求", "申请条件"),
        ("分类", "类别"),
        ("类型", "类别"),
        ("适用情形", "适用范围"),
        ("适用条件", "适用范围"),
        ("是什么", "定义"),
        ("含义", "定义"),
    )
    for source, target in replacements:
        value = value.replace(source, target)
    value = re.sub(r"[《》“”\"'·,，。；;：:\-—_/\\()（）\[\]\s]+", "", value)
    value = value.replace("pdf", "").replace("docx", "")
    value = value.replace("的", "")
    return value[:160]


def _extract_article_reference(text: str) -> Tuple[str, str]:
    cleaned = clean_block_text(text)
    if not cleaned:
        return "", ""
    match = ARTICLE_LABEL_PATTERN.search(cleaned)
    if not match:
        return "", ""
    article_id = re.sub(r"\s+", "", match.group(1))
    article_label = f"第{article_id}条"
    return article_id, article_label


def _extract_section_reference(text: str) -> Tuple[str, str]:
    cleaned = clean_block_text(text)
    if not cleaned:
        return "", ""
    match = SECTION_LABEL_PATTERN.search(cleaned)
    if not match:
        return "", ""
    section_id = re.sub(r"\s+", "", match.group(1))
    section_label = f"第{section_id}节"
    return section_id, section_label


def _extract_clause_id(text: str) -> str:
    cleaned = clean_block_text(text)
    if not cleaned:
        return ""
    match = CLAUSE_PATTERN.search(cleaned)
    return clean_block_text(match.group(1)) if match else ""


def _extract_list_item_id(text: str) -> str:
    cleaned = clean_block_text(text)
    if not cleaned:
        return ""
    match = LIST_ITEM_PATTERN.search(cleaned)
    return clean_block_text(match.group(1)) if match else ""


def _estimate_list_density(text: str) -> float:
    cleaned = clean_block_text(text)
    if not cleaned:
        return 0.0
    marker_count = len(CLAUSE_PATTERN.findall(cleaned)) + len(LIST_ITEM_PATTERN.findall(cleaned))
    return round(min(marker_count / max(len(cleaned) / 160.0, 1.0), 4.0), 4)


def _detect_chunk_role(text: str, source_type: str = "text") -> str:
    cleaned = clean_block_text(text)
    if not cleaned:
        return "text"
    if source_type == "table":
        return "table"
    if any(token in cleaned for token in EFFECTIVE_DATE_TERMS) and re.search(r"\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日", cleaned):
        return "effective_date"
    if source_type == "heading":
        return "heading"
    if any(token in cleaned for token in CLASSIFICATION_TERMS):
        return "classification"
    if any(token in cleaned for token in DEFINITION_TERMS):
        return "definition"
    if any(token in cleaned for token in PROCEDURE_TERMS):
        return "procedure"
    if any(token in cleaned for token in NOTE_TERMS):
        return "note"
    if any(token in cleaned for token in REQUIREMENT_TERMS):
        return "requirement"
    return "text"


def _infer_target_roles(query: str) -> List[str]:
    roles: List[str] = []
    if any(token in query for token in CLASSIFICATION_TERMS):
        roles.append("classification")
    if any(token in query for token in DEFINITION_TERMS):
        roles.append("definition")
    if any(token in query for token in PROCEDURE_TERMS):
        roles.append("procedure")
    if any(token in query for token in REQUIREMENT_TERMS):
        roles.append("requirement")
    if any(token in query for token in EFFECTIVE_DATE_TERMS) or any(token in query for token in ("什么时候", "何时", "哪天", "日期")):
        roles.append("effective_date")
    if not roles:
        roles.append("text")
    return roles


def _build_query_signature(query: str) -> dict:
    article_id, article_label = _extract_article_reference(query)
    article_id_variants, article_label_variants = _expand_article_reference_variants(article_id, article_label)
    normalized_heading = _normalize_heading_value(query)
    return {
        "article_id": article_id,
        "article_label": article_label,
        "article_id_variants": article_id_variants,
        "article_label_variants": article_label_variants,
        "normalized_heading": normalized_heading,
        "target_roles": _infer_target_roles(query),
    }


def _certificate_variant_adjustment(query: str, text: str) -> float:
    cleaned_query = clean_block_text(query)
    cleaned_text = clean_block_text(text)
    if not cleaned_query or not cleaned_text:
        return 0.0

    adjustment = 0.0
    if "适航证" in cleaned_query and "颁发" in cleaned_query and "适航证和外国适航证认可书的颁发" in cleaned_text:
        adjustment += 0.12
    for phrase, penalty in CERTIFICATE_VARIANT_PENALTIES.items():
        if phrase in cleaned_text and phrase not in cleaned_query:
            adjustment -= penalty
    return round(adjustment, 6)


def _maybe_split_inline_articles(text: str) -> str:
    cleaned = clean_block_text(text)
    if not cleaned:
        return ""
    cleaned = re.sub(
        rf"([。；;])\s*((?:{ARTICLE_LABEL_PATTERN.pattern})|(?:{SECTION_LABEL_PATTERN.pattern}))",
        r"\1\n\2",
        cleaned,
    )
    return clean_block_text(cleaned)


def _split_heading_from_body(content: str, heading_text: str) -> Tuple[str, str]:
    cleaned = clean_block_text(content)
    heading = clean_block_text(heading_text)
    if not cleaned or not heading:
        return heading, ""

    lines = [line for line in cleaned.splitlines() if line.strip()]
    if not lines:
        return heading, ""
    if clean_block_text(lines[0]) == heading:
        remainder = clean_block_text("\n".join(lines[1:]))
        if remainder and (len(remainder) >= 18 or len(lines) >= 3):
            return heading, remainder
        return heading, ""

    inline_match = re.match(rf"^\s*({re.escape(heading)})(?:[:：]\s*)?(.*)$", cleaned, flags=re.S)
    if inline_match:
        remainder = clean_block_text(inline_match.group(2))
        if remainder and len(remainder) >= 18:
            return heading, remainder
    return heading, ""


def _is_structural_boundary_line(line: str) -> bool:
    stripped = clean_block_text(line)
    if not stripped:
        return False
    if ARTICLE_LABEL_PATTERN.match(stripped) or SECTION_LABEL_PATTERN.match(stripped):
        return True
    if NUMBERED_HEADING_PATTERN.match(stripped):
        return True
    if CLAUSE_PATTERN.match(stripped) or LIST_ITEM_PATTERN.match(stripped):
        return True
    return False


def _split_structured_text(text: str) -> List[str]:
    cleaned = _maybe_split_inline_articles(text)
    if not cleaned:
        return []

    lines = [clean_block_text(line) for line in cleaned.splitlines() if clean_block_text(line)]
    if not lines:
        return []

    segments: List[str] = []
    current: List[str] = []
    for line in lines:
        is_boundary = _is_structural_boundary_line(line)
        if current and is_boundary:
            current_text = clean_block_text("\n".join(current))
            if current_text:
                segments.append(current_text)
            current = [line]
            continue
        current.append(line)

    if current:
        current_text = clean_block_text("\n".join(current))
        if current_text:
            segments.append(current_text)

    deduped: List[str] = []
    for segment in segments:
        segment = clean_block_text(segment)
        if not segment:
            continue
        if deduped and deduped[-1] == segment:
            continue
        deduped.append(segment)
    return deduped


def _split_text_block_by_headings(content: str) -> List[str]:
    cleaned = clean_block_text(content)
    if not cleaned:
        return []

    lines = [line for line in cleaned.splitlines() if clean_block_text(line)]
    if not lines:
        return []

    sections: List[str] = []
    current: List[str] = []
    for line in lines:
        stripped = clean_block_text(line)
        is_heading, _, _ = detect_heading(stripped)
        if current and is_heading:
            sections.append(clean_block_text("\n".join(current)))
            current = [stripped]
            continue
        current.append(stripped)

    if current:
        sections.append(clean_block_text("\n".join(current)))
    return [section for section in sections if section]


def _build_structural_metadata(text: str, source_type: str = "text", section_title: str = "", heading_text: str = "", parent_heading: str = "") -> dict:
    content = clean_block_text(text)
    article_id, article_label = _extract_article_reference("\n".join([section_title, heading_text, content]))
    section_id, section_label = _extract_section_reference("\n".join([section_title, heading_text, content]))
    clause_id = _extract_clause_id(content)
    list_item_id = _extract_list_item_id(content)
    role = _detect_chunk_role("\n".join([section_title, heading_text, content]), source_type=source_type)
    heading_seed = _compact_section_label(section_title or heading_text or article_label or _extract_highlight_text(content))
    normalized_heading = _normalize_heading_value(heading_seed)
    has_enumeration = 1 if (clause_id or list_item_id or any(token in content for token in CLASSIFICATION_TERMS)) else 0
    list_density = _estimate_list_density(content)
    title_seed = clean_block_text("\n".join(part for part in [section_title, heading_text, parent_heading] if part))
    title_body_consistency = 1.0 if not title_seed else min(1.0, _token_overlap_score(title_seed, content))
    return {
        "article_id": article_id,
        "article_label": article_label,
        "section_id": section_id,
        "section_label": section_label,
        "clause_id": clause_id,
        "list_item_id": list_item_id,
        "chunk_role": role,
        "normalized_heading": normalized_heading,
        "has_enumeration": has_enumeration,
        "list_density": list_density,
        "contains_definition": 1 if role == "definition" else 0,
        "contains_steps": 1 if role == "procedure" else 0,
        "contains_classification": 1 if role == "classification" else 0,
        "contains_requirements": 1 if role == "requirement" else 0,
        "title_body_consistency": round(title_body_consistency, 4),
    }


def _copy_structural_fields(row: dict) -> dict:
    return {key: row.get(key) for key in INDEXED_STRUCTURAL_FIELDS if key in row}


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(value, 12.0), -12.0)))


def analyze_query_intent(query: str) -> dict:
    text = query.strip()
    normalized = text.lower()
    query_signature = _build_query_signature(text)
    classification_like = (
        any(token in text for token in ("类别", "分类", "分为", "分成", "包括", "哪几类", "哪两类", "哪些类型"))
        and any(token in text for token in ("适航证", "合格证", "认可证", "批准标签", "证书", "特许飞行证"))
    )
    article_like = bool(
        re.search(r"第\s*[0-9一二三四五六七八九十百千]+\s*条", text)
        or re.search(r"\b\d+\.\d+(?:\.\d+)*\b", text)
        or re.search(r"\b(?:ccar|ata|amm|fim|tsm|ipc|srm)\b", normalized)
        or classification_like
    )
    fault_like = any(token in text for token in ("故障", "异常", "告警", "失效", "不工作", "排查", "原因", "报警"))
    procedure_like = any(token in text for token in ("步骤", "如何", "怎么", "流程", "更换", "检查", "操作", "procedure"))
    concept_like = any(token in text for token in ("是什么", "含义", "定义", "区别", "作用", "概念"))

    if article_like:
        query_type = "article_lookup"
        profile = {
            "rrf_k": 36,
            "kw_weight": 1.8 if not classification_like else 2.05,
            "vec_weight": 0.9,
            "article_weight": 2.35 if not classification_like else 2.6,
            "heading_weight": 1.9 if not classification_like else 2.15,
            "table_weight": 0.7,
            "expand_children": 5,
            "child_char_budget": 1100,
        }
    elif fault_like:
        query_type = "fault_diagnosis"
        profile = {"rrf_k": 70, "kw_weight": 1.05, "vec_weight": 1.45, "article_weight": 0.85, "heading_weight": 0.8, "table_weight": 0.6, "expand_children": 3, "child_char_budget": 780}
    elif procedure_like:
        query_type = "procedure"
        profile = {"rrf_k": 52, "kw_weight": 1.2, "vec_weight": 1.25, "article_weight": 1.0, "heading_weight": 1.05, "table_weight": 0.75, "expand_children": 4, "child_char_budget": 900}
    elif concept_like:
        query_type = "concept_explanation"
        profile = {"rrf_k": 58, "kw_weight": 0.95, "vec_weight": 1.3, "article_weight": 1.15, "heading_weight": 1.0, "table_weight": 0.65, "expand_children": 3, "child_char_budget": 700}
    else:
        query_type = "general"
        profile = {"rrf_k": 60, "kw_weight": 1.0, "vec_weight": 1.2, "article_weight": 0.95, "heading_weight": 1.0, "table_weight": 0.7, "expand_children": 3, "child_char_budget": 800}

    profile.update(
        {
            "query_type": query_type,
            "exact_match_bias": article_like,
            "fault_keywords": fault_like,
            "procedure_keywords": procedure_like,
            "classification_keywords": classification_like,
            "query_signature": query_signature,
            "target_roles": query_signature["target_roles"],
        }
    )
    return profile


def _build_query_aliases(query: str) -> List[str]:
    aliases: List[str] = []
    seen = set()

    def push(value: str):
        cleaned = clean_block_text(value)
        if not cleaned or cleaned in seen:
            return
        seen.add(cleaned)
        aliases.append(cleaned)

    push(query)

    replacements = (
        ("类别", "分类"),
        ("分类", "类别"),
        ("有哪些", "分为哪几类"),
        ("包括哪些", "分为哪几类"),
        ("包括什么", "分为哪几类"),
        ("是什么", "定义"),
        ("含义", "定义"),
        ("申请条件", "颁发条件"),
        ("适用条件", "适用范围"),
    )
    for source, target in replacements:
        if source in query:
            push(query.replace(source, target))

    if "适航证" in query and "办法条件" in query and "管理办法" not in query:
        push(query.replace("办法条件", "颁发条件"))

    if "颁发条件" in query:
        push(query.replace("颁发条件", "颁发"))
        push(query.replace("颁发条件", "要求"))
    if "定义" in query:
        push(query.replace("定义", "是什么"))
    if "适用范围" in query:
        push(query.replace("适用范围", "适用"))
    if any(token in query for token in EFFECTIVE_DATE_TERMS) or any(token in query for token in ("什么时候", "何时", "哪天", "日期")):
        push("本规定自 起施行")
        push("自 起施行")
        push("施行日期")
        push("开始施行")
    if any(token in query for token in ("类别", "分类", "分为", "分成")) and "适航证" in query:
        push(query.replace("民用航空器", "").replace("的", ""))
        push("适航证的类别")
        push("适航证分为哪两种类别")

    normalized_heading = _normalize_heading_value(query)
    if normalized_heading and normalized_heading != _normalize_heading_value(aliases[0] if aliases else ""):
        push(normalized_heading)

    article_id, article_label = _extract_article_reference(query)
    if article_label:
        article_id_variants, article_label_variants = _expand_article_reference_variants(article_id, article_label)
        for variant in article_label_variants:
            push(variant)
        for variant in article_id_variants:
            push(variant)

    return aliases[:8]


def _compact_section_label(text: str) -> str:
    cleaned = clean_block_text(text)
    cleaned = cleaned.replace("[SECTION]", "").replace("交通运输部规章交通运输部发布", "").strip(" ：:;；，,")
    cleaned = re.sub(r"(第\s*[0-9A-Za-z一二三四五六七八九十百千万\d\.]+\s*条)\s*", r"\1 ", cleaned)
    cleaned = re.sub(r"(第\s*[0-9一二三四五六七八九十百千万\d\.]+\s*节)\s*", r"\1 ", cleaned)
    if not cleaned:
        return ""

    for marker in ("适航证的类别", "适航证分类", "适用范围", "定义", "要求", "颁发", "限制"):
        marker_index = cleaned.find(marker)
        if 0 <= marker_index <= 48:
            return cleaned[: marker_index + len(marker)].strip(" ：:;；，,")

    article_match = re.search(r"(第?\s*[0-9A-Za-z一二三四五六七八九十百千万\d\.]+\s*条\s*[^\n。！？；]{0,20})", cleaned)
    if article_match:
        return article_match.group(1).strip(" ：:;；，,")

    section_match = re.search(r"(第\s*[0-9一二三四五六七八九十百千万\d\.]+\s*节\s*[^\n。！？；]{0,20})", cleaned)
    if section_match:
        return section_match.group(1).strip(" ：:;；，,")

    if len(cleaned) <= 72:
        return cleaned
    return cleaned[:72].rstrip(" ：:;；，,")


def _restore_article_reference(text: str, metadata: dict) -> str:
    cleaned = clean_block_text(text).strip(" ：:;；，,")
    if not cleaned:
        return ""

    prefix_source = clean_block_text(
        "\n".join(
            part
            for part in [
                metadata.get("section_title", ""),
                metadata.get("heading_text", ""),
                metadata.get("parent_heading", ""),
            ]
            if part
        )
    )
    prefix_match = re.search(r"第\s*(\d+)\.\d+\s*条", prefix_source)
    short_article_match = re.match(r"^(\d{2,3})\s*条\s*(.*)$", cleaned)
    if prefix_match and short_article_match:
        major = prefix_match.group(1)
        minor = short_article_match.group(1)
        suffix = short_article_match.group(2).strip()
        return f"第{major}.{minor} 条 {suffix}".strip(" ：:;；，,")

    return cleaned


def _split_text_candidates(text: str) -> List[str]:
    body = clean_block_text(re.sub(r"^\[SECTION\]\s*", "", text or ""))
    if not body:
        return []

    candidates: List[str] = []
    seen = set()

    def push(value: str):
        cleaned = clean_block_text(value).strip(" ：:;；，,")
        if len(cleaned) < 6 or cleaned in seen:
            return
        seen.add(cleaned)
        candidates.append(cleaned[:180])

    for line in body.splitlines():
        line = clean_block_text(line)
        if not line:
            continue
        if len(line) <= 180:
            push(line)
        else:
            fragments = re.split(r"(?<=[。！？；：:])", line)
            for fragment in fragments:
                fragment = clean_block_text(fragment)
                if 8 <= len(fragment) <= 180:
                    push(fragment)
            push(line[:120])
        if len(candidates) >= 12:
            break

    return candidates


def _derive_section_title(metadata: dict, content: str) -> str:
    for value in (
        metadata.get("section_title", ""),
        metadata.get("heading_text", ""),
        metadata.get("highlight_text", ""),
        metadata.get("article_label", ""),
        metadata.get("clause_path", ""),
    ):
        label = _compact_section_label(_restore_article_reference(value, metadata))
        if label:
            return label

    for candidate in _split_text_candidates(content):
        label = _compact_section_label(_restore_article_reference(candidate, metadata))
        if label:
            return label

    return _extract_highlight_text(content)


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
            if "标准适航证" in restored and "特殊适航证" in restored:
                score += 1.35
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


def _query_alignment_score(query: str, evidence: dict, profile: dict) -> float:
    highlight = clean_block_text(evidence.get("highlight_text", ""))
    section = clean_block_text(evidence.get("section_title", ""))
    content = clean_block_text(evidence.get("content", ""))[:420]
    combined = "\n".join(part for part in [section, highlight, content] if part)
    query_signature = profile.get("query_signature", {})

    score = 0.85 * _token_overlap_score(query, highlight) + 0.6 * _token_overlap_score(query, section) + 0.2 * _ngram_similarity(query, combined)
    score += _certificate_variant_adjustment(query, combined)
    if _article_reference_matches(evidence.get("article_id", ""), evidence.get("article_label", ""), query_signature):
        score += 0.9
    if query_signature.get("normalized_heading") and evidence.get("normalized_heading") == query_signature.get("normalized_heading"):
        score += 0.45
    if evidence.get("chunk_role") in profile.get("target_roles", []):
        score += 0.18
    if evidence.get("chunk_role") == "effective_date" and "effective_date" in profile.get("target_roles", []):
        score += 0.65
    target_doc_name = profile.get("target_doc_name", "")
    if target_doc_name:
        if evidence.get("doc_name") == target_doc_name:
            score += 0.65
        else:
            score -= 0.7
    if profile.get("classification_keywords"):
        if any(token in combined for token in ("适航证的类别", "特殊适航证的分类", "分成以下两种类别", "标准适航证", "特殊适航证")):
            score += 0.95
        if "特许飞行证" in combined:
            score -= 0.9
        if any(token in combined for token in ("类别变更", "适航检查", "重新颁发", "破损", "丢失")):
            score -= 0.95
        if any(token in combined for token in ("20 厘米", "标牌", "标识")) and "特殊适航证的分类" not in combined:
            score -= 0.45

    return round(score, 6)


def _normalize_bbox(raw_bbox, page_rect: fitz.Rect) -> Optional[dict]:
    if not raw_bbox:
        return None
    x0, y0, x1, y1 = raw_bbox
    width = max(page_rect.width, 1.0)
    height = max(page_rect.height, 1.0)
    left = max(0.0, min(1.0, x0 / width))
    top = max(0.0, min(1.0, y0 / height))
    right = max(0.0, min(1.0, x1 / width))
    bottom = max(0.0, min(1.0, y1 / height))
    return {
        "x": round(left, 6),
        "y": round(top, 6),
        "width": round(max(0.0, right - left), 6),
        "height": round(max(0.0, bottom - top), 6),
    }


def _bbox_union(boxes: List[Optional[dict]]) -> Optional[dict]:
    normalized = [box for box in boxes if box]
    if not normalized:
        return None
    x0 = min(box["x"] for box in normalized)
    y0 = min(box["y"] for box in normalized)
    x1 = max(box["x"] + box["width"] for box in normalized)
    y1 = max(box["y"] + box["height"] for box in normalized)
    return {
        "x": round(x0, 6),
        "y": round(y0, 6),
        "width": round(max(0.0, x1 - x0), 6),
        "height": round(max(0.0, y1 - y0), 6),
    }


def _bbox_overlap_ratio(block_bbox, table_bbox) -> float:
    try:
        block = fitz.Rect(block_bbox)
        table = fitz.Rect(table_bbox)
        intersection = block & table
        if intersection.is_empty:
            return 0.0
        return intersection.get_area() / max(block.get_area(), 1.0)
    except Exception:
        return 0.0


def _sort_page_blocks(text_blocks: List[dict], layout_type: str) -> List[dict]:
    if layout_type == "multi_column":
        x_centers = [((item["bbox"][0] + item["bbox"][2]) / 2) for item in text_blocks if item.get("bbox")]
        split_midpoint = (min(x_centers) + max(x_centers)) / 2 if x_centers else 0.0
        return sorted(
            text_blocks,
            key=lambda item: (
                0 if ((item["bbox"][0] + item["bbox"][2]) / 2) <= split_midpoint else 1,
                item["bbox"][1],
                item["bbox"][0],
            ),
        )
    return sorted(text_blocks, key=lambda item: (item["bbox"][1], item["bbox"][0]))


def _detect_page_layout(text_blocks: List[dict], page_rect: fitz.Rect) -> Tuple[str, float]:
    if not text_blocks:
        return "image_sparse", 0.25

    centers = [((block["bbox"][0] + block["bbox"][2]) / 2) / max(page_rect.width, 1.0) for block in text_blocks]
    left_blocks = sum(1 for center in centers if center < 0.44)
    right_blocks = sum(1 for center in centers if center > 0.56)
    total_chars = sum(len(block["text"]) for block in text_blocks)

    if total_chars < 140:
        return "image_sparse", 0.35
    if left_blocks >= 2 and right_blocks >= 2:
        return "multi_column", 0.48
    return "single_column", 0.86


def _find_table_caption(page: fitz.Page, bbox, page_num: int) -> str:
    try:
        clip_rect = fitz.Rect(bbox[0], max(0, bbox[1] - 80), bbox[2], bbox[1])
        text_above = clean_block_text(page.get_text("text", clip=clip_rect))
        if text_above:
            match = re.search('表\\s*[\\d]+[-–][\\d]+[^\\n]*', text_above)
            if match:
                return match.group().strip()
        clip_rect = fitz.Rect(bbox[0], bbox[3], bbox[2], bbox[3] + 50)
        text_below = clean_block_text(page.get_text("text", clip=clip_rect))
        if text_below:
            match = re.search('表\\s*[\\d]+[-–][\\d]+[^\\n]*', text_below)
            if match:
                return match.group().strip()
    except Exception:
        pass
    return f"表_{page_num}_{int(bbox[0])}_{int(bbox[1])}"


def _table_to_markdown(data: List[List[str]]) -> str:
    if not data:
        return ""
    header = [str(cell or "").strip() for cell in data[0]]
    rows = [[str(cell or "").strip() for cell in row] for row in data[1:]]
    markdown = ["| " + " | ".join(header) + " |", "|" + "|".join([" --- "] * len(header)) + "|"]
    for row in rows:
        markdown.append("| " + " | ".join(row) + " |")
    return "\n".join(markdown)


def _extract_page_text_blocks(page: fitz.Page) -> List[dict]:
    page_dict = page.get_text("dict")
    blocks = []
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        lines = []
        for line in block.get("lines", []):
            span_text = "".join(span.get("text", "") for span in line.get("spans", []))
            span_text = clean_block_text(span_text)
            if span_text:
                lines.append(span_text)
        text = clean_block_text("\n".join(lines))
        if not text:
            continue
        blocks.append({"text": text, "bbox": block.get("bbox", page.rect)})
    return blocks


def _extract_highlight_text(text: str, fallback: str = "") -> str:
    lines = [line.strip() for line in clean_block_text(text).splitlines() if line.strip()]
    for line in lines[:4]:
        if 8 <= len(line) <= 180:
            return line
    if fallback:
        return fallback[:180]
    return clean_block_text(text)[:180]


def _materialize_text_units(
    doc_name: str,
    content: str,
    page_number: int,
    page_count: int,
    bbox,
    layout_type: str,
    parse_confidence: float,
    is_ocr_fallback: bool,
    document_type: str,
    is_heading: bool = False,
    heading_level: int = 0,
    heading_text: str = "",
    source_type: str = "text",
) -> List[dict]:
    normalized_content = clean_block_text(content)
    if not normalized_content:
        return []

    emitted: List[dict] = []
    if source_type == "table":
        unit = {
            "doc_name": doc_name,
            "content": normalized_content,
            "page_number": page_number,
            "page_count": page_count,
            "bbox": bbox,
            "heading_level": 0,
            "is_heading": False,
            "source_type": "table",
            "layout_type": "table",
            "parse_confidence": parse_confidence,
            "is_table": True,
            "is_ocr_fallback": is_ocr_fallback,
            "section_title": heading_text,
            "highlight_text": _extract_highlight_text(normalized_content, heading_text),
            "document_type": document_type,
        }
        unit.update(_build_structural_metadata(normalized_content, source_type="table", section_title=heading_text, heading_text=heading_text))
        emitted.append(unit)
        return emitted

    if is_heading:
        heading_only, body = _split_heading_from_body(normalized_content, heading_text or normalized_content)
        heading_only = heading_only or clean_block_text(heading_text or normalized_content)
        heading_unit = {
            "doc_name": doc_name,
            "content": heading_only,
            "page_number": page_number,
            "page_count": page_count,
            "bbox": bbox,
            "heading_level": heading_level,
            "is_heading": True,
            "source_type": "heading",
            "layout_type": layout_type,
            "parse_confidence": parse_confidence,
            "is_table": False,
            "is_ocr_fallback": is_ocr_fallback,
            "section_title": heading_only,
            "highlight_text": _extract_highlight_text(heading_only, heading_only),
            "document_type": document_type,
        }
        heading_unit.update(_build_structural_metadata(heading_only, source_type="heading", section_title=heading_only, heading_text=heading_only))
        emitted.append(heading_unit)
        if body:
            for segment in _split_structured_text(body) or [body]:
                segment = clean_block_text(segment)
                if len(segment) < 2:
                    continue
                text_unit = {
                    "doc_name": doc_name,
                    "content": segment,
                    "page_number": page_number,
                    "page_count": page_count,
                    "bbox": bbox,
                    "heading_level": 0,
                    "is_heading": False,
                    "source_type": "text",
                    "layout_type": layout_type,
                    "parse_confidence": parse_confidence,
                    "is_table": False,
                    "is_ocr_fallback": is_ocr_fallback,
                    "section_title": heading_only,
                    "highlight_text": _extract_highlight_text(segment, heading_only),
                    "document_type": document_type,
                }
                text_unit.update(_build_structural_metadata(segment, source_type="text", section_title=heading_only, heading_text=heading_only))
                emitted.append(text_unit)
        return emitted

    for segment in _split_structured_text(normalized_content) or [normalized_content]:
        segment = clean_block_text(segment)
        if len(segment) < 2:
            continue
        unit = {
            "doc_name": doc_name,
            "content": segment,
            "page_number": page_number,
            "page_count": page_count,
            "bbox": bbox,
            "heading_level": 0,
            "is_heading": False,
            "source_type": "text",
            "layout_type": layout_type,
            "parse_confidence": parse_confidence,
            "is_table": False,
            "is_ocr_fallback": is_ocr_fallback,
            "section_title": "",
            "highlight_text": _extract_highlight_text(segment),
            "document_type": document_type,
        }
        unit.update(_build_structural_metadata(segment, source_type="text"))
        emitted.append(unit)
    return emitted


def semantic_split(block_text: str, embedder, max_len: int = MAX_CHUNK_SIZE, overlap_chars: int = MAX_CHUNK_OVERLAP) -> List[str]:
    text = clean_block_text(block_text)
    if len(text) <= max_len:
        return [text]

    sentences = re.split('(?<=[。！？；?!?])\\s*', text)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    if len(sentences) <= 1:
        parts = [text[i : i + max_len] for i in range(0, len(text), max(max_len - overlap_chars, 1))]
        return [part for part in parts if part]

    from sentence_transformers import util

    embeddings = embedder.encode(sentences, convert_to_tensor=True)
    adjacency = [util.cos_sim(embeddings[i], embeddings[i + 1]).item() for i in range(len(sentences) - 1)]

    def dynamic_threshold(window: List[float], base: float = 0.72) -> float:
        if len(window) < 3:
            return base
        mean_value = float(np.mean(window))
        std_value = float(np.std(window))
        return float(np.clip(mean_value - 0.3 * std_value, 0.55, 0.87))

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    overlap_cache = ""

    for index, sentence in enumerate(sentences):
        sentence_len = len(sentence)
        if not current:
            current = [sentence]
            current_len = sentence_len
            continue

        start_index = index - len(current)
        center_vector = torch.mean(embeddings[start_index:index], dim=0)
        similarity = util.cos_sim(embeddings[index], center_vector).item()
        window = adjacency[max(0, index - 4) : min(len(adjacency), index + 4)]
        threshold = dynamic_threshold(window)
        exceeds_limit = current_len + sentence_len + 1 > max_len

        if exceeds_limit or similarity < threshold:
            chunk_text = " ".join(current).strip()
            if overlap_cache:
                chunk_text = f"{overlap_cache} {chunk_text}".strip()
            chunks.append(chunk_text)
            overlap_cache = chunk_text[-overlap_chars:].strip() if overlap_chars > 0 else ""
            current = [sentence]
            current_len = sentence_len
        else:
            current.append(sentence)
            current_len += sentence_len + 1

    if current:
        chunk_text = " ".join(current).strip()
        if overlap_cache:
            chunk_text = f"{overlap_cache} {chunk_text}".strip()
        chunks.append(chunk_text)

    return [chunk for chunk in chunks if chunk]


def _chunk_text_units(text_units: List[dict], embedder, section_title: str, parent_heading: str) -> List[dict]:
    content = clean_block_text("\n".join(unit["content"] for unit in text_units if unit.get("content")))
    if not content:
        return []

    first_unit = text_units[0]
    bbox = _bbox_union([unit.get("bbox") for unit in text_units])
    parse_confidence = float(np.mean([unit.get("parse_confidence", 0.5) for unit in text_units]))
    page_numbers = {int(unit.get("page_number", 0)) for unit in text_units if unit.get("page_number")}
    article_ids = {unit.get("article_id", "") for unit in text_units if unit.get("article_id")}
    section_ids = {unit.get("section_id", "") for unit in text_units if unit.get("section_id")}
    clause_ids = {unit.get("clause_id", "") for unit in text_units if unit.get("clause_id")}
    list_item_ids = {unit.get("list_item_id", "") for unit in text_units if unit.get("list_item_id")}
    roles = [unit.get("chunk_role", "") for unit in text_units if unit.get("chunk_role") not in {"", "text"}]
    aggregated_role = roles[0] if roles else _detect_chunk_role(content)
    base_record = {
        "doc_name": first_unit["doc_name"],
        "page_number": first_unit["page_number"],
        "page_count": first_unit["page_count"],
        "bbox": bbox,
        "section_title": section_title,
        "parent_heading": parent_heading,
        "heading_text": section_title,
        "heading_level": 0,
        "is_heading": False,
        "source_type": "text",
        "layout_type": first_unit.get("layout_type", "single_column"),
        "parse_confidence": round(parse_confidence, 4),
        "is_table": False,
        "is_ocr_fallback": first_unit.get("is_ocr_fallback", False),
        "document_type": first_unit.get("document_type", "AMM"),
        "cross_page": 1 if len(page_numbers) > 1 else 0,
        "boundary_conflict": 1 if len(article_ids) > 1 or len(section_ids) > 1 else 0,
        "chunk_role": aggregated_role,
    }
    base_record.update(_build_structural_metadata(content, source_type="text", section_title=section_title, heading_text=section_title, parent_heading=parent_heading))
    if len(article_ids) == 1:
        article_id = next(iter(article_ids))
        base_record["article_id"] = article_id
        base_record["article_label"] = f"第{article_id}条"
    if len(section_ids) == 1:
        section_id = next(iter(section_ids))
        base_record["section_id"] = section_id
        base_record["section_label"] = f"第{section_id}节"
    if len(clause_ids) == 1:
        base_record["clause_id"] = next(iter(clause_ids))
    if len(list_item_ids) == 1:
        base_record["list_item_id"] = next(iter(list_item_ids))
    base_record["has_enumeration"] = 1 if clause_ids or list_item_ids or base_record.get("has_enumeration") else 0
    base_record["list_density"] = round(float(np.mean([unit.get("list_density", 0.0) for unit in text_units])) if text_units else 0.0, 4)
    base_record["contains_definition"] = 1 if any(unit.get("contains_definition") for unit in text_units) else base_record.get("contains_definition", 0)
    base_record["contains_steps"] = 1 if any(unit.get("contains_steps") for unit in text_units) else base_record.get("contains_steps", 0)
    base_record["contains_classification"] = 1 if any(unit.get("contains_classification") for unit in text_units) else base_record.get("contains_classification", 0)
    base_record["contains_requirements"] = 1 if any(unit.get("contains_requirements") for unit in text_units) else base_record.get("contains_requirements", 0)
    base_record["title_body_consistency"] = round(float(np.mean([unit.get("title_body_consistency", 0.0) for unit in text_units])) if text_units else 0.0, 4)

    texts = semantic_split(content, embedder, MAX_CHUNK_SIZE, MAX_CHUNK_OVERLAP) if len(content) > MAX_CHUNK_SIZE else [content]
    blocks = []
    for text in texts:
        structural = _build_structural_metadata(text, source_type="text", section_title=section_title, heading_text=section_title, parent_heading=parent_heading)
        blocks.append(
            {
                **base_record,
                "content": text,
                "highlight_text": _extract_highlight_text(text, section_title),
                **structural,
            }
        )
    return blocks


def _build_clause_path(heading_path: List[str], block: dict) -> str:
    parts: List[str] = []
    for title in heading_path:
        label = _compact_section_label(title)
        if label and label not in parts:
            parts.append(label)
    for value in (
        block.get("article_label", ""),
        block.get("section_label", ""),
        block.get("clause_id", ""),
        block.get("list_item_id", ""),
    ):
        value = clean_block_text(value)
        if value and not any(value == existing or value in existing or existing in value for existing in parts):
            parts.append(value)
    return " > ".join(parts[:6])


def build_chain(blocks: List[dict]) -> List[dict]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for block in blocks:
        grouped[block.get("doc_name", "")].append(block)

    chained: List[dict] = []
    for doc_name, doc_blocks in grouped.items():
        for index, block in enumerate(doc_blocks):
            seed = f"{doc_name}_{index}_{block.get('page_number', 0)}_{block.get('content', '')[:120]}"
            block["chunk_id"] = hashlib.md5(seed.encode("utf-8")).hexdigest()[:16]

        for index, block in enumerate(doc_blocks):
            block["prev_id"] = doc_blocks[index - 1]["chunk_id"] if index > 0 else None
            block["next_id"] = doc_blocks[index + 1]["chunk_id"] if index < len(doc_blocks) - 1 else None
            block["parent_id"] = None
            block["doc_order"] = index

        stack: List[dict] = []
        for block in doc_blocks:
            if block.get("is_heading"):
                level = max(int(block.get("heading_level", 0)), 1)
                while stack and int(stack[-1]["level"]) >= level:
                    stack.pop()
                if stack:
                    block["parent_id"] = stack[-1]["chunk_id"]
                    block["parent_heading"] = stack[-1]["title"]
                block["heading_path"] = [item["title"] for item in stack] + [block.get("heading_text", "")]
                stack.append({"level": level, "chunk_id": block["chunk_id"], "title": block.get("heading_text", "")})
            elif stack:
                block["parent_id"] = stack[-1]["chunk_id"]
                block["parent_heading"] = stack[-1]["title"]
                if not block.get("section_title"):
                    block["section_title"] = stack[-1]["title"]
                if not block.get("heading_text"):
                    block["heading_text"] = stack[-1]["title"]
                block["heading_path"] = [item["title"] for item in stack]
            else:
                block["heading_path"] = [block.get("heading_text", "")] if block.get("is_heading") and block.get("heading_text") else []

        for block in doc_blocks:
            block.update(
                _build_structural_metadata(
                    block.get("content", ""),
                    source_type=block.get("source_type", "text"),
                    section_title=block.get("section_title", ""),
                    heading_text=block.get("heading_text", ""),
                    parent_heading=block.get("parent_heading", ""),
                )
            )
            block["clause_path"] = _build_clause_path(block.get("heading_path", []), block)
            if block.get("is_heading"):
                block["title_body_consistency"] = 1.0

        chained.extend(doc_blocks)
    return chained


def _extract_pdf_units(file_path: str, doc_name: str) -> Tuple[List[dict], dict]:
    document = fitz.open(file_path)
    page_count = document.page_count
    units: List[dict] = []
    health = {
        "doc_name": doc_name,
        "page_count": page_count,
        "unit_count": 0,
        "chunk_count": 0,
        "table_count": 0,
        "multi_column_pages": [],
        "image_sparse_pages": [],
        "low_confidence_pages": [],
        "ocr_fallback_pages": [],
        "parse_success_rate": 1.0,
    }

    try:
        for page_number, page in enumerate(document, start=1):
            raw_text_blocks = _extract_page_text_blocks(page)
            layout_type, page_confidence = _detect_page_layout(raw_text_blocks, page.rect)
            if layout_type == "multi_column":
                health["multi_column_pages"].append(page_number)
            if layout_type == "image_sparse":
                health["image_sparse_pages"].append(page_number)
                health["ocr_fallback_pages"].append(page_number)
            if page_confidence < 0.55:
                health["low_confidence_pages"].append(page_number)

            text_blocks = _sort_page_blocks(raw_text_blocks, layout_type)
            table_regions = []
            try:
                page_tables = page.find_tables()
                for table in page_tables.tables:
                    data = table.extract()
                    markdown = _table_to_markdown(data)
                    if not markdown:
                        continue
                    title = _find_table_caption(page, table.bbox, page_number)
                    bbox = _normalize_bbox(table.bbox, page.rect)
                    units.extend(
                        _materialize_text_units(
                            doc_name=doc_name,
                            content=f"{title}\n\n{markdown}",
                            page_number=page_number,
                            page_count=page_count,
                            bbox=bbox,
                            layout_type="table",
                            parse_confidence=0.9,
                            is_ocr_fallback=False,
                            document_type=infer_document_type(doc_name),
                            heading_text=title,
                            source_type="table",
                        )
                    )
                    health["table_count"] += 1
                    table_regions.append(table.bbox)
            except Exception:
                pass

            for block in text_blocks:
                if any(_bbox_overlap_ratio(block["bbox"], table_bbox) > 0.24 for table_bbox in table_regions):
                    continue
                content = clean_block_text(block["text"])
                if len(content) < 4:
                    continue
                for section_content in _split_text_block_by_headings(content) or [content]:
                    first_line = next((line for line in section_content.splitlines() if line.strip()), section_content)
                    is_heading, level, heading_text = detect_heading(first_line)
                    units.extend(
                        _materialize_text_units(
                            doc_name=doc_name,
                            content=section_content,
                            page_number=page_number,
                            page_count=page_count,
                            bbox=_normalize_bbox(block["bbox"], page.rect),
                            layout_type=layout_type,
                            parse_confidence=0.92 if is_heading else page_confidence,
                            is_ocr_fallback=layout_type == "image_sparse",
                            document_type=infer_document_type(doc_name),
                            is_heading=is_heading,
                            heading_level=level if is_heading else 0,
                            heading_text=heading_text,
                        )
                    )
    finally:
        document.close()

    health["unit_count"] = len(units)
    if page_count > 0:
        anomaly_count = len(set(health["multi_column_pages"] + health["image_sparse_pages"]))
        health["parse_success_rate"] = round(max(0.0, 1.0 - anomaly_count / page_count), 4)
    return units, health


def _extract_docx_units(file_path: str, doc_name: str) -> Tuple[List[dict], dict]:
    document = Document(file_path)
    units: List[dict] = []
    page_count = 1
    document_type = infer_document_type(doc_name)

    for paragraph in document.paragraphs:
        content = clean_block_text(paragraph.text)
        if not content:
            continue
        is_heading, level, heading_text = detect_heading(content)
        units.extend(
            _materialize_text_units(
                doc_name=doc_name,
                content=content,
                page_number=1,
                page_count=page_count,
                bbox=None,
                layout_type="docx",
                parse_confidence=0.92,
                is_ocr_fallback=False,
                document_type=document_type,
                is_heading=is_heading,
                heading_level=level if is_heading else 0,
                heading_text=heading_text,
            )
        )

    table_count = 0
    for table in document.tables:
        data = [[clean_block_text(cell.text) for cell in row.cells] for row in table.rows]
        markdown = _table_to_markdown(data)
        if not markdown:
            continue
        table_count += 1
        title = f"表格_{table_count}"
        units.extend(
            _materialize_text_units(
                doc_name=doc_name,
                content=f"{title}\n\n{markdown}",
                page_number=1,
                page_count=page_count,
                bbox=None,
                layout_type="table",
                parse_confidence=0.88,
                is_ocr_fallback=False,
                document_type=document_type,
                heading_text=title,
                source_type="table",
            )
        )

    health = {
        "doc_name": doc_name,
        "page_count": 1,
        "unit_count": len(units),
        "chunk_count": 0,
        "table_count": table_count,
        "multi_column_pages": [],
        "image_sparse_pages": [],
        "low_confidence_pages": [],
        "ocr_fallback_pages": [],
        "parse_success_rate": 1.0,
    }
    return units, health


def extract_document_units(file_path: str) -> Tuple[List[dict], dict]:
    doc_name = os.path.basename(file_path)
    if file_path.lower().endswith(".pdf"):
        return _extract_pdf_units(file_path, doc_name)
    if file_path.lower().endswith(".docx"):
        return _extract_docx_units(file_path, doc_name)
    return [], {
        "doc_name": doc_name,
        "page_count": 0,
        "unit_count": 0,
        "chunk_count": 0,
        "table_count": 0,
        "multi_column_pages": [],
        "image_sparse_pages": [],
        "low_confidence_pages": [],
        "ocr_fallback_pages": [],
        "parse_success_rate": 0.0,
    }


def _load_document_health_reports() -> Dict[str, dict]:
    if not HEALTH_REPORT_FILE.exists():
        return {}
    try:
        payload = json.loads(HEALTH_REPORT_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("读取文档健康报告失败：%s", exc)
        return {}

    if isinstance(payload, dict):
        documents = payload.get("documents")
        if isinstance(documents, dict):
            return documents
        return payload
    return {}


def _persist_document_health_reports(reports: Dict[str, dict]):
    if not reports:
        return
    try:
        existing = _load_document_health_reports()
        existing.update(reports)
        HEALTH_REPORT_FILE.write_text(
            json.dumps({"documents": existing}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning("写入文档健康报告失败：%s", exc)


def _remove_document_health_report(doc_name: str):
    if not doc_name:
        return
    try:
        existing = _load_document_health_reports()
        if doc_name not in existing:
            return
        existing.pop(doc_name, None)
        HEALTH_REPORT_FILE.write_text(
            json.dumps({"documents": existing}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning("删除文档健康报告失败 doc_name=%s: %s", doc_name, exc)


def build_document_chunks(units: List[dict], embedder) -> List[dict]:
    if not units:
        return []

    blocks: List[dict] = []
    heading_stack: List[dict] = []
    pending_text_units: List[dict] = []
    pending_chars = 0

    def current_section() -> str:
        return heading_stack[-1]["text"] if heading_stack else ""

    def current_parent() -> str:
        return heading_stack[-2]["text"] if len(heading_stack) > 1 else ""

    def flush_pending():
        nonlocal pending_text_units, pending_chars
        if not pending_text_units:
            return
        blocks.extend(_chunk_text_units(pending_text_units, embedder, current_section(), current_parent()))
        pending_text_units = []
        pending_chars = 0

    def should_flush_text(last_unit: dict, current_unit: dict) -> bool:
        if current_unit["page_number"] != last_unit["page_number"]:
            return True
        if current_unit.get("layout_type") != last_unit.get("layout_type"):
            return True
        if pending_chars + len(current_unit["content"]) > int(MAX_CHUNK_SIZE * 1.2):
            return True

        last_article = last_unit.get("article_id", "")
        current_article = current_unit.get("article_id", "")
        if last_article and current_article and last_article != current_article:
            return True
        if bool(last_article) != bool(current_article):
            return True

        last_clause = last_unit.get("clause_id") or last_unit.get("list_item_id")
        current_clause = current_unit.get("clause_id") or current_unit.get("list_item_id")
        if last_clause and current_clause and last_clause != current_clause:
            return True

        last_role = last_unit.get("chunk_role", "text")
        current_role = current_unit.get("chunk_role", "text")
        if last_role != current_role and last_role != "text" and current_role != "text":
            return True

        if current_unit.get("boundary_conflict"):
            return True
        return False

    for unit in units:
        source_type = unit.get("source_type")
        if source_type == "heading":
            flush_pending()
            level = max(int(unit.get("heading_level", 1)), 1)
            while heading_stack and heading_stack[-1]["level"] >= level:
                heading_stack.pop()
            parent_heading = heading_stack[-1]["text"] if heading_stack else ""
            blocks.append(
                {
                    "doc_name": unit["doc_name"],
                    "content": unit["content"],
                    "page_number": unit["page_number"],
                    "page_count": unit["page_count"],
                    "bbox": unit.get("bbox"),
                    "section_title": unit["content"],
                    "parent_heading": parent_heading,
                    "heading_text": unit["content"],
                    "heading_level": level,
                    "is_heading": True,
                    "source_type": "heading",
                    "layout_type": unit.get("layout_type", "single_column"),
                    "parse_confidence": unit.get("parse_confidence", 0.9),
                    "is_table": False,
                    "is_ocr_fallback": unit.get("is_ocr_fallback", False),
                    "highlight_text": unit.get("highlight_text") or unit["content"],
                    "document_type": unit.get("document_type", "AMM"),
                    "cross_page": 0,
                    "boundary_conflict": 0,
                    **_copy_structural_fields(unit),
                }
            )
            heading_stack.append({"level": level, "text": unit["content"]})
            continue

        if source_type == "table":
            flush_pending()
            blocks.append(
                {
                    "doc_name": unit["doc_name"],
                    "content": unit["content"],
                    "page_number": unit["page_number"],
                    "page_count": unit["page_count"],
                    "bbox": unit.get("bbox"),
                    "section_title": unit.get("section_title") or current_section(),
                    "parent_heading": current_parent(),
                    "heading_text": current_section(),
                    "heading_level": 0,
                    "is_heading": False,
                    "source_type": "table",
                    "layout_type": unit.get("layout_type", "table"),
                    "parse_confidence": unit.get("parse_confidence", 0.88),
                    "is_table": True,
                    "is_ocr_fallback": unit.get("is_ocr_fallback", False),
                    "highlight_text": unit.get("highlight_text") or unit.get("section_title") or current_section(),
                    "document_type": unit.get("document_type", "AMM"),
                    "cross_page": 0,
                    "boundary_conflict": 0,
                    **_copy_structural_fields(unit),
                }
            )
            continue

        if pending_text_units:
            last_unit = pending_text_units[-1]
            if should_flush_text(last_unit, unit):
                flush_pending()

        pending_text_units.append(unit)
        pending_chars += len(unit["content"])

    flush_pending()
    return build_chain(blocks)


def save_metadata_to_es(es_client: Elasticsearch, blocks: List[dict]):
    actions = []
    for block in blocks:
        actions.append(
            {
                "_index": ES_INDEX,
                "_id": block["chunk_id"],
                "_source": {
                    "chunk_id": block["chunk_id"],
                    "content": block["content"],
                    "doc_name": block.get("doc_name", ""),
                    "document_type": block.get("document_type", "AMM"),
                    "section_title": block.get("section_title", ""),
                    "heading_text": block.get("heading_text", ""),
                    "parent_heading": block.get("parent_heading", ""),
                    "heading_level": int(block.get("heading_level", 0)),
                    "is_heading": 1 if block.get("is_heading") else 0,
                    "source_type": block.get("source_type", "text"),
                    "layout_type": block.get("layout_type", "single_column"),
                    "parse_confidence": float(block.get("parse_confidence", 0.0)),
                    "is_table": 1 if block.get("is_table") else 0,
                    "is_ocr_fallback": 1 if block.get("is_ocr_fallback") else 0,
                    "page_number": int(block.get("page_number", 1)),
                    "page_count": int(block.get("page_count", 1)),
                    "bbox": block.get("bbox"),
                    "highlight_text": block.get("highlight_text", ""),
                    "prev_id": block.get("prev_id", ""),
                    "next_id": block.get("next_id", ""),
                    "parent_id": block.get("parent_id", ""),
                    "doc_order": int(block.get("doc_order", 0)),
                    "article_id": block.get("article_id", ""),
                    "article_label": block.get("article_label", ""),
                    "section_id": block.get("section_id", ""),
                    "section_label": block.get("section_label", ""),
                    "clause_id": block.get("clause_id", ""),
                    "list_item_id": block.get("list_item_id", ""),
                    "chunk_role": block.get("chunk_role", "text"),
                    "normalized_heading": block.get("normalized_heading", ""),
                    "has_enumeration": int(block.get("has_enumeration", 0)),
                    "list_density": float(block.get("list_density", 0.0)),
                    "contains_definition": int(block.get("contains_definition", 0)),
                    "contains_steps": int(block.get("contains_steps", 0)),
                    "contains_classification": int(block.get("contains_classification", 0)),
                    "contains_requirements": int(block.get("contains_requirements", 0)),
                    "title_body_consistency": float(block.get("title_body_consistency", 0.0)),
                    "clause_path": block.get("clause_path", ""),
                    "cross_page": int(block.get("cross_page", 0)),
                    "boundary_conflict": int(block.get("boundary_conflict", 0)),
                },
            }
        )
    if actions:
        helpers.bulk(es_client, actions)
        es_client.indices.refresh(index=ES_INDEX)


def _es_get_by_id(es_client: Elasticsearch, chunk_id: str) -> Dict:
    try:
        response = es_client.get(index=ES_INDEX, id=chunk_id)
        return response["_source"]
    except Exception:
        return {}


def get_chunk_context(es_client: Elasticsearch, chunk_id: str, context_size: int = 1) -> Dict:
    current = _es_get_by_id(es_client, chunk_id)
    if not current:
        return {}

    prev_blocks = []
    prev_id = current.get("prev_id", "")
    while prev_id and len(prev_blocks) < context_size:
        row = _es_get_by_id(es_client, prev_id)
        if not row:
            break
        prev_blocks.insert(0, row)
        prev_id = row.get("prev_id", "")

    next_blocks = []
    next_id = current.get("next_id", "")
    while next_id and len(next_blocks) < context_size:
        row = _es_get_by_id(es_client, next_id)
        if not row:
            break
        next_blocks.append(row)
        next_id = row.get("next_id", "")

    parent = _es_get_by_id(es_client, current["parent_id"]) if current.get("parent_id") else None
    return {"current": current, "prev": prev_blocks, "next": next_blocks, "parent": parent}


def format_structural_context(ctx: Dict) -> str:
    if not ctx:
        return ""
    parts = []
    if ctx.get("parent"):
        parts.append(f"[PARENT] {ctx['parent'].get('content', '')}")
    for row in ctx.get("prev", []):
        parts.append(f"[PREV] {row.get('content', '')}")
    parts.append(f"[CURRENT] {ctx['current'].get('content', '')}")
    for row in ctx.get("next", []):
        parts.append(f"[NEXT] {row.get('content', '')}")
    return "\n\n".join(parts)


_embedder = None
_reranker = None
_embedder_lock = threading.Lock()
_reranker_lock = threading.Lock()


def get_embedder():
    global _embedder
    if _embedder is None:
        with _embedder_lock:
            if _embedder is None:
                print('加载嵌入模型...')
                _embedder = SentenceTransformer(EMBEDDING_MODEL)
                if torch.cuda.is_available():
                    _embedder = _embedder.to("cuda")
                    print('嵌入模型已加载到 GPU')
                else:
                    print('嵌入模型使用 CPU')
    return _embedder


def get_reranker():
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                try:
                    print("加载重排序模型...")
                    _reranker = CrossEncoder(RERANK_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
                except Exception as exc:
                    print(f"重排序模型加载失败：{exc}，将跳过重排序。")
                    _reranker = None
    return _reranker


class DualDBManager:
    def __init__(self):
        if connections.has_connection("default"):
            connections.disconnect("default")
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
        self.es_client = Elasticsearch(ES_HOST)
        self.embedder = None

    def _get_embedder(self):
        if self.embedder is None:
            self.embedder = get_embedder()
        return self.embedder

    def _es_mapping(self):
        return {
            "mappings": {
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "content": {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_smart"},
                    "doc_name": {"type": "keyword"},
                    "document_type": {"type": "keyword"},
                    "section_title": {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_smart"},
                    "heading_text": {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_smart"},
                    "parent_heading": {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_smart"},
                    "heading_level": {"type": "integer"},
                    "is_heading": {"type": "integer"},
                    "source_type": {"type": "keyword"},
                    "layout_type": {"type": "keyword"},
                    "parse_confidence": {"type": "float"},
                    "is_table": {"type": "integer"},
                    "is_ocr_fallback": {"type": "integer"},
                    "page_number": {"type": "integer"},
                    "page_count": {"type": "integer"},
                    "bbox": {
                        "properties": {
                            "x": {"type": "float"},
                            "y": {"type": "float"},
                            "width": {"type": "float"},
                            "height": {"type": "float"},
                        }
                    },
                    "highlight_text": {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_smart"},
                    "prev_id": {"type": "keyword"},
                    "next_id": {"type": "keyword"},
                    "parent_id": {"type": "keyword"},
                    "doc_order": {"type": "integer"},
                    "article_id": {"type": "keyword"},
                    "article_label": {"type": "keyword"},
                    "section_id": {"type": "keyword"},
                    "section_label": {"type": "keyword"},
                    "clause_id": {"type": "keyword"},
                    "list_item_id": {"type": "keyword"},
                    "chunk_role": {"type": "keyword"},
                    "normalized_heading": {"type": "keyword"},
                    "has_enumeration": {"type": "integer"},
                    "list_density": {"type": "float"},
                    "contains_definition": {"type": "integer"},
                    "contains_steps": {"type": "integer"},
                    "contains_classification": {"type": "integer"},
                    "contains_requirements": {"type": "integer"},
                    "title_body_consistency": {"type": "float"},
                    "clause_path": {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_smart"},
                    "cross_page": {"type": "integer"},
                    "boundary_conflict": {"type": "integer"},
                }
            }
        }

    def ensure_databases(self) -> Collection:
        if not utility.has_collection(MILVUS_COLLECTION):
            fields = [
                FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=MILVUS_CONTENT_MAX),
            ]
            schema = CollectionSchema(fields, "aerospace semantic index")
            collection = Collection(MILVUS_COLLECTION, schema)
            collection.create_index("vector", {"metric_type": "COSINE", "index_type": "IVF_FLAT", "params": {"nlist": 1024}})
        else:
            collection = Collection(MILVUS_COLLECTION)
        collection.load()
        if not self.es_client.indices.exists(index=ES_INDEX):
            self.es_client.indices.create(index=ES_INDEX, body=self._es_mapping())
        else:
            self.es_client.indices.put_mapping(index=ES_INDEX, body=self._es_mapping()["mappings"])
        return collection

    def reset_databases(self) -> Collection:
        if utility.has_collection(MILVUS_COLLECTION):
            utility.drop_collection(MILVUS_COLLECTION)
        fields = [
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=MILVUS_CONTENT_MAX),
        ]
        schema = CollectionSchema(fields, "aerospace semantic index")
        collection = Collection(MILVUS_COLLECTION, schema)
        collection.create_index("vector", {"metric_type": "COSINE", "index_type": "IVF_FLAT", "params": {"nlist": 1024}})
        if self.es_client.indices.exists(index=ES_INDEX):
            self.es_client.indices.delete(index=ES_INDEX)
        self.es_client.indices.create(index=ES_INDEX, body=self._es_mapping())
        return collection

    def _remove_doc_from_es(self, doc_name: str):
        self.es_client.delete_by_query(index=ES_INDEX, body={"query": {"term": {"doc_name": doc_name}}}, refresh=True)

    def _remove_doc_from_milvus(self, collection: Collection, doc_name: str):
        try:
            response = self.es_client.search(
                index=ES_INDEX,
                body={"query": {"term": {"doc_name": doc_name}}, "_source": ["chunk_id"], "size": 10000},
            )
            chunk_ids = [hit["_source"]["chunk_id"] for hit in response["hits"]["hits"]]
            if chunk_ids:
                quoted_chunk_ids = ", ".join(json.dumps(str(chunk_id), ensure_ascii=False) for chunk_id in chunk_ids)
                collection.delete(f"chunk_id in [{quoted_chunk_ids}]")
        except Exception as exc:
            logger.warning("从 Milvus 删除文档失败 doc_name=%s: %s", doc_name, exc)

    def _index_document_blocks(self, file_path: str) -> Tuple[List[dict], dict]:
        units, health = extract_document_units(file_path)
        if not units:
            return [], health
        blocks = build_document_chunks(units, self._get_embedder())
        health["chunk_count"] = len(blocks)
        if blocks:
            low_conf_chunks = sum(1 for block in blocks if block.get("parse_confidence", 0.0) < 0.55)
            health["low_confidence_chunk_ratio"] = round(low_conf_chunks / len(blocks), 4)
        else:
            health["low_confidence_chunk_ratio"] = 0.0
        return blocks, health

    def add_document(self, file_path: str) -> IndexOperationResult:
        file_name = os.path.basename(file_path)
        if not os.path.exists(file_path):
            return IndexOperationResult(
                success=False,
                action="add_document",
                message=f"文件不存在：{file_path}",
                doc_name=file_name,
                error_code="file_not_found",
            )

        try:
            collection = self.ensure_databases()
        except Exception as exc:
            return IndexOperationResult(
                success=False,
                action="add_document",
                message=f"数据库连接失败：{exc}",
                doc_name=file_name,
                error_code="database_unavailable",
            )

        self._remove_doc_from_milvus(collection, file_name)
        self._remove_doc_from_es(file_name)

        print(f"增量索引：{file_name}")
        blocks, health = self._index_document_blocks(file_path)
        if not blocks:
            return IndexOperationResult(
                success=False,
                action="add_document",
                message=f"文档 {file_name} 未提取到有效内容",
                doc_name=file_name,
                error_code="no_effective_content",
                health_report=health,
            )

        contents = [block["content"][:MILVUS_CONTENT_MAX] for block in blocks]
        embeddings = self._get_embedder().encode(contents, normalize_embeddings=True, show_progress_bar=True)
        collection.insert([[block["chunk_id"] for block in blocks], embeddings.tolist(), contents])
        collection.flush()
        collection.load()
        save_metadata_to_es(self.es_client, blocks)
        _persist_document_health_reports({file_name: health})
        message = f"增量索引成功！文档 {file_name} 共 {len(blocks)} 个证据块"
        return IndexOperationResult(
            success=True,
            action="add_document",
            message=message,
            doc_name=file_name,
            chunk_count=len(blocks),
            health_report=health,
        )

    def remove_document(self, doc_name: str) -> IndexOperationResult:
        try:
            collection = self.ensure_databases()
        except Exception as exc:
            return IndexOperationResult(
                success=False,
                action="remove_document",
                message=f"数据库连接失败：{exc}",
                doc_name=doc_name,
                error_code="database_unavailable",
            )
        self._remove_doc_from_milvus(collection, doc_name)
        self._remove_doc_from_es(doc_name)
        _remove_document_health_report(doc_name)
        return IndexOperationResult(
            success=True,
            action="remove_document",
            message=f"已从索引中删除文档：{doc_name}",
            doc_name=doc_name,
        )

    def build_index(self) -> IndexOperationResult:
        if not os.path.exists(DOC_DIR):
            return IndexOperationResult(
                success=False,
                action="build_index",
                message=f"目录不存在：{DOC_DIR}",
                error_code="doc_dir_missing",
            )

        files = [name for name in os.listdir(DOC_DIR) if name.lower().endswith((".pdf", ".docx"))]
        if not files:
            return IndexOperationResult(
                success=False,
                action="build_index",
                message=f"目录下没有可索引文档：{DOC_DIR}",
                error_code="no_documents",
            )

        try:
            collection = self.reset_databases()
        except Exception as exc:
            return IndexOperationResult(
                success=False,
                action="build_index",
                message=f"数据库连接失败：{exc}",
                error_code="database_unavailable",
            )

        reports = {}
        all_blocks: List[dict] = []
        for file_name in files:
            path = os.path.join(DOC_DIR, file_name)
            print(f"解析文档：{file_name}")
            blocks, health = self._index_document_blocks(path)
            reports[file_name] = health
            all_blocks.extend(blocks)

        if not all_blocks:
            return IndexOperationResult(
                success=False,
                action="build_index",
                message="未提取到任何有效内容",
                error_code="no_effective_content",
                details={"reports": reports},
            )

        contents = [block["content"][:MILVUS_CONTENT_MAX] for block in all_blocks]
        embeddings = self._get_embedder().encode(contents, normalize_embeddings=True, show_progress_bar=True)
        collection.insert([[block["chunk_id"] for block in all_blocks], embeddings.tolist(), contents])
        collection.flush()
        collection.load()
        save_metadata_to_es(self.es_client, all_blocks)
        _persist_document_health_reports(reports)
        message = f"索引构建成功！共 {len(all_blocks)} 个证据块"
        return IndexOperationResult(
            success=True,
            action="build_index",
            message=message,
            chunk_count=len(all_blocks),
            details={"reports": reports, "document_count": len(files)},
        )


class HybridSearch:
    def __init__(self):
        if connections.has_connection("default"):
            connections.disconnect("default")
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
        self.es_client = Elasticsearch(ES_HOST)
        self.milvus_coll = Collection(MILVUS_COLLECTION)
        self.milvus_coll.load()
        self.embedder = get_embedder()
        self.reranker = get_reranker()
        self._pdf_page_cache: Dict[str, dict] = {}
        self._metadata_cache: Dict[str, dict] = {}

    def _resolve_doc_path(self, doc_name: str) -> Optional[str]:
        if not doc_name:
            return None
        direct_path = os.path.join(DOC_DIR, doc_name)
        if os.path.exists(direct_path):
            return direct_path

        target_stem = Path(doc_name).stem.lower()
        target_suffix = Path(doc_name).suffix.lower()
        try:
            for file_name in os.listdir(DOC_DIR):
                file_path = os.path.join(DOC_DIR, file_name)
                if not os.path.isfile(file_path):
                    continue
                if file_name == doc_name:
                    return file_path
                if Path(file_name).stem.lower() == target_stem and (not target_suffix or Path(file_name).suffix.lower() == target_suffix):
                    return file_path
        except Exception:
            return None
        return None

    def _load_pdf_page_cache(self, file_path: str) -> dict:
        cached = self._pdf_page_cache.get(file_path)
        if cached:
            return cached

        pages = []
        with fitz.open(file_path) as document:
            for page_number, page in enumerate(document, start=1):
                text = clean_block_text(page.get_text("text"))
                pages.append(
                    {
                        "page_number": page_number,
                        "text": text,
                        "normalized_text": _normalize_search_text(text),
                    }
                )
            payload = {"page_count": document.page_count, "pages": pages}

        self._pdf_page_cache[file_path] = payload
        return payload

    def _find_pdf_by_query_reference(self, query: str) -> Optional[str]:
        query_key = re.sub(r"[^\w\u4e00-\u9fff]+", "", clean_block_text(query).lower())
        candidates: List[Tuple[int, str]] = []
        try:
            file_names = [name for name in os.listdir(DOC_DIR) if name.lower().endswith(".pdf")]
        except Exception:
            return None

        for file_name in file_names:
            stem_key = re.sub(r"[^\w\u4e00-\u9fff]+", "", Path(file_name).stem.lower())
            if stem_key and stem_key in query_key:
                candidates.append((len(stem_key), os.path.join(DOC_DIR, file_name)))
        if candidates:
            candidates.sort(reverse=True, key=lambda item: item[0])
            return candidates[0][1]

        date_match = re.search(r"(\d{4})\s*\u5e74\s*(\d{1,2})\s*\u6708\s*(\d{1,2})\s*\u65e5", query)
        if not date_match:
            return None
        normalized_date = f"{int(date_match.group(1))}\u5e74{int(date_match.group(2))}\u6708{int(date_match.group(3))}\u65e5"
        for file_name in file_names:
            file_path = os.path.join(DOC_DIR, file_name)
            try:
                with fitz.open(file_path) as document:
                    front_text = "\n".join(
                        document.load_page(index).get_text("text")
                        for index in range(min(document.page_count, 2))
                    )
            except Exception:
                continue
            if normalized_date in re.sub(r"\s+", "", front_text):
                return file_path
        return None

    def _find_doc_name_by_query_reference(self, query: str) -> str:
        query_key = re.sub(r"[^\w\u4e00-\u9fff]+", "", clean_block_text(query).lower())
        candidates: List[Tuple[int, str]] = []
        try:
            file_names = [name for name in os.listdir(DOC_DIR) if name.lower().endswith((".pdf", ".docx"))]
        except Exception:
            return ""

        for file_name in file_names:
            stem_key = re.sub(r"[^\w\u4e00-\u9fff]+", "", Path(file_name).stem.lower())
            if stem_key and stem_key in query_key:
                candidates.append((len(stem_key), file_name))
        if not candidates:
            return ""
        candidates.sort(reverse=True, key=lambda item: item[0])
        return candidates[0][1]

    def _document_applicability_evidence(self, query: str) -> Optional[dict]:
        if "\u9002\u7528\u8303\u56f4" not in query:
            return None
        if not any(marker in query for marker in ("\u4e0b\u53d1", "\u6587\u4ef6", "\u54a8\u8be2\u901a\u544a", "\u672c\u6587\u4ef6")):
            return None

        file_path = self._find_pdf_by_query_reference(query)
        if not file_path:
            return None
        cache = self._load_pdf_page_cache(file_path)
        pages = cache.get("pages", [])[:4]
        combined = "\n".join(page.get("text", "") for page in pages)
        match = re.search(
            r"(?P<label>(?:\d+|[一二三四五六七八九十]+)\s*[．\.\u3001]\s*\u9002\u7528\u8303\u56f4)\s*"
            r"(?P<body>.*?)(?=\n\s*(?:\d+|[一二三四五六七八九十]+)\s*[．\.\u3001]\s*[\u4e00-\u9fff]{2,20}|$)",
            combined,
            flags=re.S,
        )
        if not match:
            return None

        raw_section = f"{match.group('label')}\n{match.group('body')}"
        section = re.sub(r"\n\s*\u2014\s*\d+\s*\u2014\s*", "\n", raw_section)
        section = clean_block_text(section)
        label = re.sub(r"\s+", "", match.group("label"))
        page_number = next((page.get("page_number", 1) for page in pages if "\u9002\u7528\u8303\u56f4" in page.get("text", "")), 1)
        doc_name = Path(file_path).name
        stem = Path(file_path).stem
        return {
            "content": section,
            "doc_name": doc_name,
            "heading": label,
            "parent_heading": stem,
            "score": 1.0,
            "chunk_id": f"pdf-applicability:{doc_name}",
            "page_number": int(page_number),
            "page_count": int(cache.get("page_count", 1)),
            "section_title": label,
            "highlight_text": label,
            "bbox": None,
            "source_type": "pdf_text",
            "layout_type": "single_column",
            "parse_confidence": 1.0,
            "document_type": "AC",
            "is_ocr_fallback": False,
            "article_id": "",
            "article_label": "",
            "clause_path": label,
            "chunk_role": "text",
            "normalized_heading": _normalize_heading_value(label),
        }

    def _build_anchor_candidates(self, metadata: dict, content: str) -> List[str]:
        candidates: List[str] = []
        seen = set()

        def push(value: str):
            cleaned = clean_block_text(value)
            if len(cleaned) < 4:
                return
            if cleaned in seen:
                return
            seen.add(cleaned)
            candidates.append(cleaned[:180])

        for value in (
            metadata.get("highlight_text", ""),
            metadata.get("section_title", ""),
            metadata.get("heading_text", ""),
            metadata.get("parent_heading", ""),
            metadata.get("article_label", ""),
            metadata.get("clause_path", ""),
        ):
            push(value)
            restored = _restore_article_reference(value, metadata)
            if restored != value:
                push(restored)

        body = clean_block_text(re.sub(r"^\[SECTION\]\s*", "", content or ""))
        for line in body.splitlines():
            line = clean_block_text(line)
            if 8 <= len(line) <= 120:
                push(line)
                restored = _restore_article_reference(line, metadata)
                if restored != line:
                    push(restored)
            elif len(line) > 120:
                push(line[:120])
                restored = _restore_article_reference(line[:120], metadata)
                if restored != line[:120]:
                    push(restored)
            if len(candidates) >= 8:
                break

        return candidates

    def _build_focus_anchor_candidates(self, metadata: dict, content: str) -> List[str]:
        candidates: List[str] = []
        seen = set()

        def push(value: str):
            cleaned = clean_block_text(value)
            if len(cleaned) < 4 or cleaned in seen:
                return
            seen.add(cleaned)
            candidates.append(cleaned[:140])

        for value in (
            metadata.get("highlight_text", ""),
            metadata.get("section_title", ""),
            metadata.get("heading_text", ""),
            metadata.get("parent_heading", ""),
            metadata.get("article_label", ""),
            metadata.get("clause_path", ""),
        ):
            restored = _restore_article_reference(value, metadata)
            compact = _compact_section_label(restored or value)
            push(compact or restored or value)

        for candidate in _split_text_candidates(content):
            restored = _restore_article_reference(candidate, metadata)
            compact = _compact_section_label(restored or candidate)
            if compact:
                push(compact)
            if len(candidates) >= 8:
                break

        return candidates

    def _search_rects_on_page(self, page: fitz.Page, anchors: List[str]) -> Tuple[Optional[dict], str]:
        for anchor in anchors:
            anchor_variants: List[str] = []
            seen = set()
            for line in clean_block_text(anchor).splitlines():
                line = clean_block_text(line)
                if len(line) < 6:
                    continue
                normalized_line = re.sub(r"\s+", " ", line).strip()
                spaced_line = normalized_line
                spaced_line = re.sub(r"(第\s*[0-9一二三四五六七八九十百千万\d\.]+\s*章)", r"\1 ", spaced_line)
                spaced_line = re.sub(r"(第\s*[0-9A-Za-z一二三四五六七八九十百千万\d\.]+\s*条)", r"\1 ", spaced_line)
                segments = [line, normalized_line, spaced_line]
                compact_label = _compact_section_label(line)
                if compact_label:
                    segments.append(compact_label)
                    segments.append(re.sub(r"(第\s*[0-9A-Za-z一二三四五六七八九十百千万\d\.]+\s*条)\s*", r"\1 ", compact_label).strip())
                if len(line) > 80:
                    segments.extend([line[:80], line[-80:], normalized_line[:80], normalized_line[-80:]])
                chapter_match = re.search(r"第\s*[0-9一二三四五六七八九十百千万\d\.]+\s*章\s*[^\n]{2,24}", spaced_line)
                if chapter_match:
                    segments.append(chapter_match.group().strip())
                article_match = re.search(r"第\s*[0-9A-Za-z一二三四五六七八九十百千万\d\.]+\s*条\s*[^\n]{2,28}", spaced_line)
                if article_match:
                    segments.append(article_match.group().strip())
                for marker in ("适航证的类别", "适航证分类", "特殊适航证的分类", "标准适航证", "特殊适航证"):
                    marker_index = line.find(marker)
                    if marker_index != -1:
                        segments.append(marker)
                        start = max(0, marker_index - 10)
                        end = min(len(line), marker_index + len(marker) + 16)
                        segments.append(line[start:end].strip())
                for segment in segments:
                    segment = re.sub(r"[ \t]{2,}", " ", str(segment or "")).strip()
                    if len(segment) < 6:
                        continue
                    if segment not in seen:
                        anchor_variants.append(segment)
                        seen.add(segment)
            anchor_variants.sort(key=len, reverse=True)
            for variant in anchor_variants:
                try:
                    rects = page.search_for(variant)
                except Exception:
                    rects = []
                if rects:
                    bbox = _bbox_union([_normalize_bbox((rect.x0, rect.y0, rect.x1, rect.y1), page.rect) for rect in rects])
                    if bbox:
                        return bbox, variant
        return None, anchors[0] if anchors else ""

    def _repair_pdf_evidence_metadata(self, metadata: dict, content: str) -> dict:
        doc_name = metadata.get("doc_name", "")
        if not doc_name.lower().endswith(".pdf"):
            return {}

        file_path = self._resolve_doc_path(doc_name)
        if not file_path or not os.path.exists(file_path):
            return {}

        anchors = self._build_anchor_candidates(metadata, content)
        focus_anchors = self._build_focus_anchor_candidates(metadata, content) or anchors[:4]
        if not anchors:
            return {}

        try:
            page_cache = self._load_pdf_page_cache(file_path)
        except Exception:
            return {}

        preferred_page = metadata.get("page_number")
        try:
            preferred_page = int(preferred_page) if preferred_page else None
        except (TypeError, ValueError):
            preferred_page = None

        prioritized_anchors = focus_anchors + [anchor for anchor in anchors if anchor not in focus_anchors]
        normalized_anchors = [(_normalize_search_text(anchor), anchor) for anchor in prioritized_anchors]
        page_candidates: List[dict] = []

        for page_info in page_cache["pages"]:
            exact_score = 0.0
            fuzzy_score = 0.0
            exact_rank = 99
            normalized_page_text = page_info["normalized_text"]
            raw_page_text = page_info["text"]
            for index, (normalized_anchor, raw_anchor) in enumerate(normalized_anchors):
                if not normalized_anchor:
                    continue
                if normalized_anchor in normalized_page_text:
                    anchor_bonus = 11.0 - min(index, 7) * 0.7 + min(len(normalized_anchor) / 90.0, 1.4)
                    if anchor_bonus > exact_score:
                        exact_score = anchor_bonus
                        exact_rank = index
                    continue
                fuzzy_value = _token_overlap_score(raw_anchor, raw_page_text) + 0.65 * _ngram_similarity(raw_anchor, raw_page_text)
                if index >= len(focus_anchors):
                    fuzzy_value *= 0.58
                elif len(raw_anchor) > 96:
                    fuzzy_value *= 0.74
                fuzzy_score = max(fuzzy_score, fuzzy_value)

            page_candidates.append(
                {
                    "page_number": page_info["page_number"],
                    "exact_score": round(exact_score, 6),
                    "exact_rank": exact_rank,
                    "fuzzy_score": round(fuzzy_score, 6),
                    "distance": abs(page_info["page_number"] - preferred_page) if preferred_page else 0,
                }
            )

        exact_candidates = [item for item in page_candidates if item["exact_score"] > 0]
        if exact_candidates:
            exact_candidates.sort(key=lambda item: (-item["exact_score"], item["exact_rank"], item["distance"], item["page_number"]))
            ranked_candidates = exact_candidates[:6]
        else:
            fuzzy_candidates = [item for item in page_candidates if item["fuzzy_score"] >= 0.45]
            if not fuzzy_candidates:
                return {}
            fuzzy_candidates.sort(key=lambda item: (-item["fuzzy_score"], item["distance"], item["page_number"]))
            ranked_candidates = fuzzy_candidates[:4]

        if not ranked_candidates:
            return {}

        bbox = None
        highlight_text = metadata.get("highlight_text", "") or (focus_anchors[0] if focus_anchors else "")
        resolved_page = ranked_candidates[0]["page_number"]
        try:
            with fitz.open(file_path) as document:
                page_count = document.page_count
                visited_pages = set()
                ordered_pages: List[int] = []
                for candidate in ranked_candidates:
                    page_number = candidate["page_number"]
                    if page_number not in visited_pages:
                        ordered_pages.append(page_number)
                        visited_pages.add(page_number)
                    for delta in (-1, 1):
                        nearby_page = page_number + delta
                        if 1 <= nearby_page <= page_count and nearby_page not in visited_pages:
                            ordered_pages.append(nearby_page)
                            visited_pages.add(nearby_page)

                for candidate_page in ordered_pages:
                    page = document.load_page(candidate_page - 1)
                    bbox, matched_text = self._search_rects_on_page(page, focus_anchors)
                    if bbox:
                        resolved_page = candidate_page
                        if matched_text:
                            highlight_text = matched_text
                        break

                if not bbox:
                    page = document.load_page(resolved_page - 1)
                    bbox, matched_text = self._search_rects_on_page(page, anchors)
                    if matched_text:
                        highlight_text = matched_text
                page_count = document.page_count
        except Exception:
            page_count = int(page_cache.get("page_count", metadata.get("page_count", 1) or 1))

        return {
            "page_number": resolved_page,
            "page_count": max(page_count, 1),
            "bbox": bbox,
            "highlight_text": highlight_text or focus_anchors[0] if focus_anchors else anchors[0],
            "parse_confidence": max(float(metadata.get("parse_confidence", 0.0)), 0.78),
        }

    def _pdf_page_text_window(self, metadata: dict, include_next_page: bool = False, max_chars: int = 3600) -> str:
        doc_name = metadata.get("doc_name", "")
        if not doc_name.lower().endswith(".pdf"):
            return ""
        file_path = self._resolve_doc_path(doc_name)
        if not file_path or not os.path.exists(file_path):
            return ""

        try:
            page_cache = self._load_pdf_page_cache(file_path)
        except Exception:
            return ""

        try:
            page_number = int(metadata.get("page_number", 1))
        except (TypeError, ValueError):
            page_number = 1

        wanted_pages = {page_number}
        if include_next_page:
            wanted_pages.add(page_number + 1)

        parts = []
        for page_info in page_cache.get("pages", []):
            if int(page_info.get("page_number", 0)) in wanted_pages:
                text = clean_block_text(page_info.get("text", ""))
                if text:
                    parts.append(f"[第 {page_info['page_number']} 页]\n{text}")

        return clean_block_text("\n\n".join(parts))[:max_chars]

    def _vector_search(self, query_vec, top_k: int = VEC_RECALL) -> List[dict]:
        response = self.milvus_coll.search(
            data=[query_vec],
            anns_field="vector",
            param={"metric_type": "COSINE"},
            limit=top_k,
            output_fields=["chunk_id", "content"],
        )
        hits = []
        if response and response[0]:
            for hit in response[0]:
                hits.append({"chunk_id": hit.id, "content": hit.entity.get("content", ""), "score": float(hit.score), "channel": "vector"})
        return hits

    def _article_search(self, query: str, profile: dict, top_k: int = ARTICLE_RECALL) -> List[dict]:
        query_signature = profile.get("query_signature", {})
        should = []
        article_id = query_signature.get("article_id", "")
        article_label = query_signature.get("article_label", "")
        article_id_variants = query_signature.get("article_id_variants", [])
        article_label_variants = query_signature.get("article_label_variants", [])
        normalized_heading = query_signature.get("normalized_heading", "")
        target_roles = query_signature.get("target_roles", [])

        for article_id_variant in article_id_variants or ([article_id] if article_id else []):
            should.append({"term": {"article_id": {"value": article_id_variant, "boost": 8.0}}})
        for article_label_variant in article_label_variants or ([article_label] if article_label else []):
            should.append({"term": {"article_label": {"value": article_label_variant, "boost": 8.8}}})
            should.append({"match_phrase": {"section_title": {"query": article_label_variant, "boost": 6.2}}})
            should.append({"match_phrase": {"heading_text": {"query": article_label_variant, "boost": 6.8}}})

        if normalized_heading:
            should.append({"term": {"normalized_heading": {"value": normalized_heading, "boost": 7.2}}})

        for alias in _build_query_aliases(query):
            should.extend(
                [
                    {"match_phrase": {"section_title": {"query": alias, "boost": 4.5}}},
                    {"match_phrase": {"heading_text": {"query": alias, "boost": 4.9}}},
                    {"match_phrase": {"clause_path": {"query": alias, "boost": 3.8}}},
                ]
            )

        for role in target_roles:
            if role and role != "text":
                should.append({"term": {"chunk_role": {"value": role, "boost": 1.8}}})

        if not should:
            return []

        body = {"query": {"bool": {"should": should, "minimum_should_match": 1}}, "size": top_k}
        try:
            response = self.es_client.search(index=ES_INDEX, body=body)
            return [
                {
                    "chunk_id": hit["_source"]["chunk_id"],
                    "content": hit["_source"].get("content", ""),
                    "score": float(hit.get("_score", 0.0)),
                    "channel": "article",
                }
                for hit in response["hits"]["hits"]
            ]
        except Exception:
            return []

    def _keyword_search(self, query: str, profile: dict, top_k: int = KW_RECALL) -> List[dict]:
        aliases = _build_query_aliases(query)
        should = []
        for index, alias in enumerate(aliases):
            decay = max(0.58, 1.0 - 0.14 * index)
            should.extend(
                [
                    {
                        "multi_match": {
                            "query": alias,
                            "fields": [
                                "content^1.2",
                                "heading_text^2.6",
                                "section_title^2.2",
                                "parent_heading^1.3",
                                "highlight_text^1.4",
                                "doc_name^0.6",
                                "article_label^3.0",
                                "clause_path^2.0",
                            ],
                            "type": "best_fields",
                            "boost": 1.0 * decay,
                        }
                    },
                    {"match_phrase": {"content": {"query": alias, "boost": 3.5 * decay}}},
                    {"match_phrase": {"heading_text": {"query": alias, "boost": 4.2 * decay}}},
                    {"match_phrase": {"clause_path": {"query": alias, "boost": 4.4 * decay}}},
                ]
            )
            if profile.get("exact_match_bias"):
                should.append({"match_phrase": {"section_title": {"query": alias, "boost": 4.8 * decay}}})

        body = {"query": {"bool": {"should": should, "minimum_should_match": 1}}, "size": top_k}
        try:
            response = self.es_client.search(index=ES_INDEX, body=body)
            return [
                {
                    "chunk_id": hit["_source"]["chunk_id"],
                    "content": hit["_source"].get("content", ""),
                    "score": float(hit.get("_score", 0.0)),
                    "channel": "keyword",
                }
                for hit in response["hits"]["hits"]
            ]
        except Exception:
            return []

    def _heading_search(self, query: str, profile: dict, top_k: int = HEADING_RECALL) -> List[dict]:
        aliases = _build_query_aliases(query)
        should = []
        for index, alias in enumerate(aliases):
            decay = max(0.58, 1.0 - 0.14 * index)
            should.extend(
                [
                    {"match_phrase": {"heading_text": {"query": alias, "boost": 6.0 * decay}}},
                    {"match_phrase": {"section_title": {"query": alias, "boost": 5.0 * decay}}},
                    {"match_phrase": {"clause_path": {"query": alias, "boost": 4.6 * decay}}},
                    {"match": {"heading_text": {"query": alias, "boost": 2.5 * decay}}},
                ]
            )

        body = {
            "query": {
                "bool": {
                    "should": should,
                    "minimum_should_match": 1,
                }
            },
            "size": top_k,
        }
        try:
            response = self.es_client.search(index=ES_INDEX, body=body)
            return [
                {
                    "chunk_id": hit["_source"]["chunk_id"],
                    "content": hit["_source"].get("content", ""),
                    "score": float(hit.get("_score", 0.0)),
                    "channel": "heading",
                }
                for hit in response["hits"]["hits"]
            ]
        except Exception:
            return []

    def _table_search(self, query: str, top_k: int = TABLE_RECALL) -> List[dict]:
        body = {
            "query": {
                "bool": {
                    "filter": [{"term": {"source_type": "table"}}],
                    "should": [
                        {"match_phrase": {"section_title": {"query": query, "boost": 4.0}}},
                        {"match": {"content": {"query": query, "boost": 1.6}}},
                    ],
                    "minimum_should_match": 1,
                }
            },
            "size": top_k,
        }
        try:
            response = self.es_client.search(index=ES_INDEX, body=body)
            return [
                {
                    "chunk_id": hit["_source"]["chunk_id"],
                    "content": hit["_source"].get("content", ""),
                    "score": float(hit.get("_score", 0.0)),
                    "channel": "table",
                }
                for hit in response["hits"]["hits"]
            ]
        except Exception:
            return []

    def _rrf_fusion(self, recall_groups: List[Tuple[str, List[dict], float]], profile: dict) -> List[dict]:
        scores: Dict[str, float] = {}
        content_map: Dict[str, str] = {}
        channels: Dict[str, set] = defaultdict(set)
        k = int(profile.get("rrf_k", 60))

        for name, hits, weight in recall_groups:
            for rank, hit in enumerate(hits):
                chunk_id = hit["chunk_id"]
                scores[chunk_id] = scores.get(chunk_id, 0.0) + float(weight) / (k + rank + 1)
                channels[chunk_id].add(name)
                content_map.setdefault(chunk_id, hit.get("content", ""))

        fused = [
            {"chunk_id": chunk_id, "rrf_score": score, "content": content_map.get(chunk_id, ""), "channels": sorted(channels[chunk_id])}
            for chunk_id, score in scores.items()
        ]
        return sorted(fused, key=lambda item: item["rrf_score"], reverse=True)

    def _get_metadata(self, chunk_id: str) -> dict:
        if not chunk_id:
            return {}
        if chunk_id in self._metadata_cache:
            return self._metadata_cache[chunk_id]
        metadata = _es_get_by_id(self.es_client, chunk_id)
        if metadata:
            self._metadata_cache[chunk_id] = metadata
        return metadata

    def _get_metadata_batch(self, chunk_ids: List[str]) -> Dict[str, dict]:
        if not chunk_ids:
            return {}
        missing_ids = [chunk_id for chunk_id in chunk_ids if chunk_id not in self._metadata_cache]
        if missing_ids:
            try:
                response = self.es_client.mget(index=ES_INDEX, body={"ids": missing_ids})
            except Exception:
                response = {"docs": []}
            for document in response.get("docs", []):
                if document.get("found"):
                    self._metadata_cache[document["_id"]] = document["_source"]
        return {chunk_id: self._metadata_cache.get(chunk_id, {}) for chunk_id in chunk_ids if self._metadata_cache.get(chunk_id)}

    def _get_chunk_context(self, chunk_id: str, context_size: int = 1) -> Dict:
        current = self._get_metadata(chunk_id)
        if not current:
            return {}

        prev_blocks = []
        prev_id = current.get("prev_id", "")
        while prev_id and len(prev_blocks) < context_size:
            row = self._get_metadata(prev_id)
            if not row:
                break
            prev_blocks.insert(0, row)
            prev_id = row.get("prev_id", "")

        next_blocks = []
        next_id = current.get("next_id", "")
        while next_id and len(next_blocks) < context_size:
            row = self._get_metadata(next_id)
            if not row:
                break
            next_blocks.append(row)
            next_id = row.get("next_id", "")

        parent = self._get_metadata(current.get("parent_id", "")) if current.get("parent_id") else None
        return {"current": current, "prev": prev_blocks, "next": next_blocks, "parent": parent}

    def _expand_heading_children(self, current: dict, max_children: int, max_chars: int) -> List[str]:
        if not current:
            return []

        children: List[str] = []
        char_budget = 0
        heading_level = int(current.get("heading_level", 1))
        start_page = int(current.get("page_number", 1))
        parent_id = current.get("chunk_id", "")
        doc_name = current.get("doc_name", "")
        next_id = current.get("next_id", "")
        while next_id and len(children) < max_children and char_budget < max_chars:
            row = self._get_metadata(next_id)
            if not row:
                break
            if row.get("doc_name") != doc_name:
                break
            if abs(int(row.get("page_number", start_page)) - start_page) > 1:
                break
            if row.get("is_heading") and int(row.get("heading_level", 0)) <= heading_level:
                break
            if row.get("parent_id") and row.get("parent_id") != parent_id:
                break
            if current.get("article_id") and row.get("article_id") and row.get("article_id") != current.get("article_id"):
                break
            snippet = clean_block_text(row.get("content", ""))
            if not snippet:
                next_id = row.get("next_id", "")
                continue
            if char_budget + len(snippet) > max_chars:
                snippet = snippet[: max_chars - char_budget].strip()
            if snippet:
                children.append(snippet)
                char_budget += len(snippet)
            next_id = row.get("next_id", "")
        return children

    def _expand_following_context(self, current: dict, max_chunks: int, max_chars: int) -> List[str]:
        """Collect following blocks that still belong to the same structural section/article."""
        if not current:
            return []

        siblings: List[str] = []
        char_budget = 0
        start_page = int(current.get("page_number", 1))
        doc_name = current.get("doc_name", "")
        parent_id = current.get("parent_id", "")
        article_id = current.get("article_id", "")
        section_title = current.get("section_title", "") or current.get("heading_text", "")
        next_id = current.get("next_id", "")

        while next_id and len(siblings) < max_chunks and char_budget < max_chars:
            row = self._get_metadata(next_id)
            if not row:
                break
            if row.get("doc_name") != doc_name:
                break
            if abs(int(row.get("page_number", start_page)) - start_page) > 1:
                break
            if row.get("is_heading"):
                break
            if parent_id and row.get("parent_id") and row.get("parent_id") != parent_id:
                break
            if article_id and row.get("article_id") and row.get("article_id") != article_id:
                break
            if not article_id and section_title:
                row_section = row.get("section_title", "") or row.get("heading_text", "")
                if row_section and row_section != section_title:
                    break

            snippet = clean_block_text(row.get("content", ""))
            if not snippet:
                next_id = row.get("next_id", "")
                continue
            if char_budget + len(snippet) > max_chars:
                snippet = snippet[: max_chars - char_budget].strip()
            if snippet:
                siblings.append(snippet)
                char_budget += len(snippet)
            next_id = row.get("next_id", "")

        return siblings

    def _compose_rerank_text(self, metadata: dict, fallback_content: str) -> str:
        section = metadata.get("section_title", "")
        parent = metadata.get("parent_heading", "")
        clause_path = metadata.get("clause_path", "")
        article_label = metadata.get("article_label", "")
        content = metadata.get("content", fallback_content)
        return "\n".join(part for part in [section, article_label, clause_path, parent, content] if part).strip()

    def _score_candidate(self, query: str, hit: dict, metadata: dict, profile: dict) -> float:
        rrf_component = hit.get("rrf_score", 0.0) * float(profile.get("rrf_k", 60) + 1)
        bounded_rrf = min(rrf_component, 1.0)
        rerank_component = _sigmoid(float(hit.get("rerank_score", 0.0))) if "rerank_score" in hit else min(0.55 * bounded_rrf, 0.65)
        score = 0.55 * rerank_component + 0.45 * bounded_rrf

        channels = set(hit.get("channels", []))
        query_signature = profile.get("query_signature", {})
        target_doc_name = profile.get("target_doc_name", "")
        if target_doc_name:
            if metadata.get("doc_name") == target_doc_name:
                score += 0.55
            else:
                score -= 0.48
        if profile["query_type"] == "article_lookup":
            if "article" in channels:
                score += 0.24
            if "heading" in channels:
                score += 0.16
            if metadata.get("is_heading"):
                score += 0.1
            if _article_reference_matches(metadata.get("article_id", ""), metadata.get("article_label", ""), query_signature):
                score += 0.36
            elif query_signature.get("article_id_variants"):
                score -= 0.18
            if query_signature.get("normalized_heading") and metadata.get("normalized_heading") == query_signature.get("normalized_heading"):
                score += 0.28
            if profile.get("classification_keywords") and any(
                token in clean_block_text(
                    "\n".join(
                        part
                        for part in [
                            metadata.get("section_title", ""),
                            metadata.get("heading_text", ""),
                            metadata.get("content", hit.get("content", "")),
                        ]
                        if part
                    )
                )
                for token in ("类别", "分类", "分为", "分成")
            ):
                score += 0.14
        elif profile["query_type"] == "fault_diagnosis":
            if "vector" in channels:
                score += 0.1
        elif profile["query_type"] == "procedure":
            if metadata.get("heading_text"):
                score += 0.06
        if metadata.get("chunk_role") in profile.get("target_roles", []):
            score += 0.11
        if metadata.get("chunk_role") == "effective_date" and "effective_date" in profile.get("target_roles", []):
            score += 0.35
        if metadata.get("contains_definition") and "definition" in profile.get("target_roles", []):
            score += 0.08
        if metadata.get("contains_steps") and "procedure" in profile.get("target_roles", []):
            score += 0.08
        if metadata.get("contains_classification") and "classification" in profile.get("target_roles", []):
            score += 0.1
        if metadata.get("contains_requirements") and "requirement" in profile.get("target_roles", []):
            score += 0.08

        if metadata.get("source_type") == "table" and any(token in query for token in ("表", "参数", "限制", "数值")):
            score += 0.08
        score += _certificate_variant_adjustment(
            query,
            clean_block_text(
                "\n".join(
                    part
                    for part in [
                        metadata.get("section_title", ""),
                        metadata.get("heading_text", ""),
                        metadata.get("highlight_text", ""),
                    ]
                    if part
                )
            ),
        )
        if metadata.get("layout_type") == "multi_column":
            score -= 0.08
        if metadata.get("is_ocr_fallback"):
            score -= 0.06
        if metadata.get("boundary_conflict"):
            score -= 0.14
        if metadata.get("cross_page"):
            score -= 0.05
        score += 0.1 * max(min(float(metadata.get("title_body_consistency", 0.0)) - 0.45, 0.55), -0.45)
        score += 0.12 * max(min(float(metadata.get("parse_confidence", 0.0)) - 0.5, 0.5), -0.5)
        return round(score, 6)

    def _search_candidates(self, query: str) -> Tuple[List[dict], dict]:
        profile = analyze_query_intent(query)
        target_doc_name = self._find_doc_name_by_query_reference(query)
        if target_doc_name:
            profile["target_doc_name"] = target_doc_name
        query_vec = self.embedder.encode([query], normalize_embeddings=True)[0].tolist()

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            vec_future = executor.submit(self._vector_search, query_vec, VEC_RECALL)
            article_future = executor.submit(self._article_search, query, profile, ARTICLE_RECALL)
            kw_future = executor.submit(self._keyword_search, query, profile, KW_RECALL)
            heading_future = executor.submit(self._heading_search, query, profile, HEADING_RECALL)
            table_future = executor.submit(self._table_search, query, TABLE_RECALL)
            recall_groups = [
                ("vector", vec_future.result(), profile["vec_weight"]),
                ("article", article_future.result(), profile["article_weight"]),
                ("keyword", kw_future.result(), profile["kw_weight"]),
                ("heading", heading_future.result(), profile["heading_weight"]),
                ("table", table_future.result(), profile["table_weight"]),
            ]

        fused = self._rrf_fusion(recall_groups, profile)
        if not fused:
            return [], profile

        candidate_ids = [hit["chunk_id"] for hit in fused[:60]]
        metadata_map = self._get_metadata_batch(candidate_ids)
        for hit in fused[:60]:
            hit["metadata"] = metadata_map.get(hit["chunk_id"], {})

        if self.reranker:
            pairs = [
                (query, self._compose_rerank_text(hit.get("metadata", {}), hit.get("content", "")))
                for hit in fused[:50]
            ]
            scores = self.reranker.predict(pairs, batch_size=16)
            for index, hit in enumerate(fused[:50]):
                hit["rerank_score"] = float(scores[index])

        for hit in fused[:60]:
            hit["final_score"] = self._score_candidate(query, hit, hit.get("metadata", {}), profile)

        fused.sort(key=lambda item: item.get("final_score", item.get("rrf_score", 0.0)), reverse=True)
        return fused, profile

    def _build_evidence_item(self, hit: dict, query: str, profile: dict, structural_depth: int = 1, allow_pdf_repair: bool = True) -> dict:
        metadata = hit.get("metadata", {})
        if not metadata:
            metadata = self._get_metadata(hit["chunk_id"])

        content = metadata.get("content", hit.get("content", ""))
        if metadata.get("is_heading"):
            children = self._expand_heading_children(
                {**metadata, "chunk_id": hit["chunk_id"]},
                int(profile.get("expand_children", 4)),
                int(profile.get("child_char_budget", 900)),
            )
            if children:
                content = "\n".join([content] + children)
        elif structural_depth > 0:
            ctx = self._get_chunk_context(hit["chunk_id"], context_size=structural_depth)
            if ctx.get("parent"):
                section_prefix = ctx["parent"].get("content", "")
                if section_prefix and section_prefix not in content:
                    content = f"[SECTION]\n{section_prefix}\n{content}"
            following = self._expand_following_context(
                {**metadata, "chunk_id": hit["chunk_id"]},
                max(int(profile.get("expand_children", 4)) * 3, 6),
                max(int(profile.get("child_char_budget", 900)), 2200 if profile["query_type"] == "article_lookup" else 900),
            )
            if following:
                content = "\n".join([content] + following)

        if profile["query_type"] == "article_lookup":
            page_window = self._pdf_page_text_window(metadata, include_next_page=True, max_chars=3600)
            if page_window and _token_overlap_score(metadata.get("section_title", ""), page_window) >= 0.2:
                content = f"{content}\n\n[PDF页原文]\n{page_window}"

        content = clean_block_text(content)
        max_content_chars = 4200 if profile["query_type"] == "article_lookup" else 1200
        if len(content) > max_content_chars:
            content = content[:max_content_chars].rstrip() + "..."

        derived_section_title = _derive_section_title(metadata, content)
        derived_highlight_text = _select_query_aligned_highlight(query, metadata, content, fallback=derived_section_title)
        metadata = {
            **metadata,
            "section_title": derived_section_title or metadata.get("section_title", ""),
            "highlight_text": derived_highlight_text or metadata.get("highlight_text", ""),
        }

        if allow_pdf_repair and str(metadata.get("doc_name", "")).lower().endswith(".pdf"):
            repaired = self._repair_pdf_evidence_metadata(metadata, content)
            if repaired:
                metadata = {**metadata, **repaired}

        query_signature = profile.get("query_signature", {})
        if profile.get("query_type") == "article_lookup" and _article_reference_matches(
            metadata.get("article_id", ""),
            metadata.get("article_label", ""),
            query_signature,
        ):
            article_label_variants = query_signature.get("article_label_variants", [])
            current_section = _compact_section_label(metadata.get("section_title", ""))
            current_highlight = _compact_section_label(metadata.get("highlight_text", ""))
            has_section_article_match = any(variant in current_section for variant in article_label_variants)
            has_highlight_article_match = any(variant in current_highlight for variant in article_label_variants)
            preferred_article_label = next(
                (
                    _compact_section_label(candidate)
                    for candidate in (
                        metadata.get("parent_heading", ""),
                        metadata.get("article_label", ""),
                        metadata.get("heading_text", ""),
                    )
                    if any(variant in _compact_section_label(candidate) for variant in article_label_variants)
                ),
                "",
            )
            if preferred_article_label and not has_section_article_match:
                metadata["section_title"] = preferred_article_label
            if preferred_article_label and not has_highlight_article_match:
                metadata["highlight_text"] = preferred_article_label

        highlight_label = _compact_section_label(metadata.get("highlight_text", ""))
        section_label = _compact_section_label(metadata.get("section_title", ""))
        highlight_article = re.search(r"第\s*[0-9A-Za-z一二三四五六七八九十百千万\d\.]+\s*条", highlight_label)
        section_article = re.search(r"第\s*[0-9A-Za-z一二三四五六七八九十百千万\d\.]+\s*条", section_label)
        if highlight_label and (
            (highlight_article and not section_article)
            or (highlight_article and section_article and highlight_article.group() != section_article.group())
        ):
            metadata["section_title"] = highlight_label
            section_label = highlight_label

        section_overlap = _token_overlap_score(query, metadata.get("section_title", ""))
        highlight_overlap = _token_overlap_score(query, highlight_label)
        if highlight_label and (
            profile.get("classification_keywords")
            or highlight_overlap > section_overlap + 0.12
        ):
            metadata["section_title"] = highlight_label

        heading = metadata.get("section_title") or metadata.get("heading_text", "")
        return {
            "content": content,
            "doc_name": metadata.get("doc_name", ""),
            "heading": heading,
            "parent_heading": metadata.get("parent_heading", ""),
            "score": hit.get("final_score", hit.get("rrf_score", 0.0)),
            "chunk_id": hit["chunk_id"],
            "page_number": int(metadata.get("page_number", 1)),
            "page_count": int(metadata.get("page_count", 1)),
            "section_title": metadata.get("section_title") or heading,
            "highlight_text": metadata.get("highlight_text", derived_highlight_text or heading or _extract_highlight_text(content)),
            "bbox": metadata.get("bbox"),
            "source_type": metadata.get("source_type", "text"),
            "layout_type": metadata.get("layout_type", "single_column"),
            "parse_confidence": float(metadata.get("parse_confidence", 0.0)),
            "document_type": metadata.get("document_type", infer_document_type(metadata.get("doc_name", ""))),
            "is_ocr_fallback": bool(metadata.get("is_ocr_fallback")),
            "article_id": metadata.get("article_id", ""),
            "article_label": metadata.get("article_label", ""),
            "clause_path": metadata.get("clause_path", ""),
            "chunk_role": metadata.get("chunk_role", "text"),
            "normalized_heading": metadata.get("normalized_heading", ""),
        }

    def search_for_qa(self, query: str, structural_depth: int = 1, semantic_extra: int = 0) -> List[dict]:
        try:
            direct_evidence = self._document_applicability_evidence(query)
            if direct_evidence:
                return [direct_evidence]

            fused, profile = self._search_candidates(query)
            if not fused:
                return []

            results: List[Tuple[float, dict, bool]] = []
            seen_pages = set()
            article_variants = set(profile.get("query_signature", {}).get("article_id_variants", []))
            for rank, hit in enumerate(fused[:60]):
                try:
                    evidence = self._build_evidence_item(
                        hit,
                        query,
                        profile,
                        structural_depth=structural_depth,
                        allow_pdf_repair=rank < 12 or not hit.get("metadata", {}).get("bbox"),
                    )
                except Exception as exc:
                    logger.warning("构建证据对象失败 chunk_id=%s: %s", hit.get("chunk_id"), exc)
                    continue
                combined_text = clean_block_text(
                    "\n".join(
                        part
                        for part in [
                            evidence.get("section_title", ""),
                            evidence.get("highlight_text", ""),
                            evidence.get("content", ""),
                        ]
                        if part
                    )
                )
                alignment = _query_alignment_score(query, evidence, profile)
                if profile.get("classification_keywords"):
                    classification_phrase = any(
                        phrase in combined_text
                        for phrase in (
                            "适航证的类别",
                            "适航证分成以下两种类别",
                            "特殊适航证的分类",
                            "分成以下两种类别",
                            "适航证分为",
                            "特殊适航证分为",
                        )
                    ) or ("标准适航证" in combined_text and "特殊适航证" in combined_text)
                    if "类别变更" in combined_text and "分为" not in combined_text and "分类" not in combined_text:
                        classification_phrase = False
                    if "特许飞行证" in combined_text:
                        classification_phrase = False
                    if not classification_phrase or alignment < 0.55:
                        continue
                dedupe_key = (
                    evidence["doc_name"],
                    evidence["page_number"],
                    evidence.get("article_id") or evidence.get("section_title"),
                )
                if dedupe_key in seen_pages:
                    continue
                seen_pages.add(dedupe_key)
                is_exact_article = bool(article_variants) and _article_reference_matches(
                    evidence.get("article_id", ""),
                    evidence.get("article_label", ""),
                    profile.get("query_signature", {}),
                )
                article_heading_bonus = 0.0
                if is_exact_article:
                    article_label_variants = profile.get("query_signature", {}).get("article_label_variants", [])
                    if any(variant in clean_block_text(evidence.get("section_title", "")) for variant in article_label_variants):
                        article_heading_bonus += 0.18
                    if evidence.get("source_type") == "heading":
                        article_heading_bonus += 0.12
                rank_score = evidence["score"] + 0.22 * alignment + (0.42 if is_exact_article else 0.0) + article_heading_bonus
                results.append((rank_score, evidence, is_exact_article))
            if profile.get("query_type") == "article_lookup" and article_variants:
                exact_results = [item for item in results if item[2]]
                if exact_results:
                    target_doc_name = profile.get("target_doc_name", "")
                    if target_doc_name:
                        target_exact_results = [item for item in exact_results if item[1].get("doc_name") == target_doc_name]
                        results = target_exact_results or exact_results
                    else:
                        leading_doc_name = exact_results[0][1].get("doc_name", "")
                        leading_doc_results = [item for item in exact_results if item[1].get("doc_name") == leading_doc_name]
                        results = leading_doc_results or exact_results
            results.sort(key=lambda item: item[0], reverse=True)
            return [item[1] for item in results[:QA_TOP_K]]
        except Exception as exc:
            logger.exception("检索问答链路执行失败 query=%s", query)
            raise RuntimeError(f"检索执行失败：{exc}") from exc

    def search(self, query: str, structural_depth: int = 1, semantic_extra: int = 0) -> str:
        try:
            results = self.search_for_qa(query, structural_depth=structural_depth, semantic_extra=semantic_extra)
        except RuntimeError as exc:
            return str(exc)
        if not results:
            return "未找到匹配内容。"

        query_profile = analyze_query_intent(query)
        lines = [f"查询类型: {query_profile['query_type']}", ""]
        for index, item in enumerate(results, start=1):
            lines.append(
                f"结果 {index} | 相关度: {item['score']:.4f} | 页码: {item['page_number']}/{item['page_count']} | 来源: {item['doc_name']}"
            )
            if item.get("section_title"):
                lines.append(f"章节: {item['section_title']}")
            lines.append(item.get("content", ""))
            lines.append("=" * 80)
        return "\n".join(lines)


def main():
    print("=" * 60)
    print("航空文档检索命令行")
    print("=" * 60)

    while True:
        print("\n请选择操作：")
        print("  1. 构建索引")
        print("  2. 检索查询")
        print("  3. 退出")
        choice = input('\n输入选项 (1/2/3): ').strip()

        if choice == "1":
            print("\n正在构建索引，请稍候...")
            manager = DualDBManager()
            print(manager.build_index().message)
        elif choice == "2":
            query = input("\n请输入检索问题：").strip()
            if not query:
                print("查询内容不能为空。")
                continue
            engine = HybridSearch()
            print(engine.search(query))
        elif choice == "3":
            print("再见。")
            break
        else:
            print("无效选项，请重新输入。")


if __name__ == "__main__":
    main()
