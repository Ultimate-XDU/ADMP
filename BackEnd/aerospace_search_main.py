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
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
# Parallel weight materialization has caused native access violations in the
# local Windows conda environment when embedding and rerank models load together.
os.environ["HF_ENABLE_PARALLEL_LOADING"] = os.environ.get("AEROSPACE_HF_ENABLE_PARALLEL_LOADING", "false")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import fitz
import numpy as np
import torch
from docx import Document
from elasticsearch import Elasticsearch, helpers
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility
from sentence_transformers import CrossEncoder, SentenceTransformer

from aerospace_structure import (
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
EMBEDDING_MODEL = os.environ.get("AEROSPACE_EMBEDDING_MODEL", "BAAI/bge-m3")
RERANK_MODEL = os.environ.get("AEROSPACE_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_DEVICE = os.environ.get("AEROSPACE_RERANK_DEVICE", "cuda").strip().lower()
VECTOR_DIM = int(os.environ.get("AEROSPACE_VECTOR_DIM", "1024"))

TOP_K = 5
QA_TOP_K = 4
VEC_RECALL = 120
KW_RECALL = 120
HEADING_RECALL = 40
TABLE_RECALL = 20
ARTICLE_RECALL = 40

ARTICLE_LABEL_PATTERN = re.compile(r"(?:第\s*)?([0-9A-Za-z一二三四五六七八九十百千万]+(?:\.[0-9A-Za-z一二三四五六七八九十百千万]+)*)\s*条")
SECTION_LABEL_PATTERN = re.compile(r"(?:第\s*)?([0-9A-Za-z一二三四五六七八九十百千万]+(?:\.[0-9A-Za-z一二三四五六七八九十百千万]+)*)\s*节")
NUMBERED_HEADING_PATTERN = re.compile(r"^([0-9]{1,2}|[一二三四五六七八九十]{1,3})\s*[．、]\s*([^\n。；;]{2,40})$")
DOT_HEADING_PATTERN = re.compile(r"^([0-9]{1,2})\.\s*([\u4e00-\u9fffA-Za-z][^\n。；;]{1,80})$")
DECIMAL_HEADING_PATTERN = re.compile(r"^([0-9]{1,2}(?:\.[0-9]{1,2})+)\s*([\u4e00-\u9fffA-Za-z][^\n。；;]{1,80})$")
CLAUSE_PATTERN = re.compile(r"(?:^|\n)\s*([（(][一二三四五六七八九十百千万\dA-Za-zivxIVX]+[)）]|[一二三四五六七八九十]+、)")
LIST_ITEM_PATTERN = re.compile(r"(?:^|\n)\s*([0-9]{1,3}(?:、|\.(?!\d))|[A-Za-z][\.\)])")

CLASSIFICATION_TERMS = ("类别", "分类", "分为", "分成", "包括", "类型")
DEFINITION_TERMS = ("定义", "是指", "含义", "术语", "解释")
PROCEDURE_TERMS = ("步骤", "程序", "流程", "操作", "检查", "执行", "更换", "拆装")
REQUIREMENT_TERMS = ("应当", "必须", "不得", "要求", "符合", "条件", "限制", "批准", "颁发", "申请")
NOTE_TERMS = ("注意", "提示", "警告", "注：", "说明")
EFFECTIVE_DATE_TERMS = ("施行", "实施", "生效", "开始施行", "起施行", "废止")
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
    "heading_path_text",
    "scope_anchor",
    "scope_key",
    "retrieval_text",
    "cross_page",
    "boundary_conflict",
)


def _compact_match_text(text: str) -> str:
    return re.sub(r"[^\u4e00-\u9fffA-Za-z0-9.]+", "", str(text or "").lower())


def _contains_any(text: str, terms: Tuple[str, ...]) -> bool:
    return any(term and term in text for term in terms)


SPECIFICITY_MODIFIERS = ("出口", "外国", "特殊", "标准", "临时", "限制", "生产", "型号", "补充", "改装", "认可", "初级", "限用")


def _doc_alias_keys(file_name: str) -> List[str]:
    stem = Path(file_name).stem
    compact_stem = _compact_match_text(stem)
    variants = {
        compact_stem,
        _compact_match_text(re.sub(r"(适航规定|适航标准|管理规定|审定规定|合格审定规定|运行安全管理规定|运行合格审定规则)$", "", stem)),
        _compact_match_text(re.sub(r"(规定|规则|标准)$", "", stem)),
        _compact_match_text(re.sub(r"(运行合格审定|合格审定|审定|管理|适航|运行安全)", "", stem)),
    }
    aliases = [alias for alias in variants if alias and len(alias) >= 4]
    return sorted(set(aliases), key=len, reverse=True)


def _extract_title_keyphrases(query: str, subject: str = "") -> List[str]:
    compact = _compact_match_text(query)
    if not compact:
        return []
    doc_name = _resolve_doc_name_from_query_text(query)
    if doc_name:
        compact = compact.replace(_compact_match_text(Path(doc_name).stem), "")
    noise_terms = (
        "是什么", "有哪些", "哪些", "什么", "如何", "怎么", "怎样", "是否", "能否",
        "应当", "需要", "必须", "多久", "多长时间", "在什么时候", "什么时候",
        "分别", "中", "的", "了", "后",
    )
    candidates: List[str] = []
    subject_key = _compact_match_text(subject)
    if subject_key and len(subject_key) >= 2:
        candidates.append(subject_key)
    if "最大重量" in compact or "最小重量" in compact:
        candidates.append("重量限制")
    if "报告情形" in compact:
        candidates.append("报告情况")
        candidates.append("报告情况要求")
    anchor_suffixes = (
        "申请书", "申请文件", "质量系统更改", "质量系统", "更改", "转让", "责任", "职责", "内容", "用途",
        "最大重量", "最小重量", "空重", "重心", "安全系数", "火警探测系统", "构造和安装",
        "灾难性失效状态", "概率要求", "持续适航事件报告", "事件报告", "报告情形", "报告情况", "报告责任", "报告时限", "报告程序", "报告要求",
        "事件收集", "事件筛选", "每日首次巡查", "全面巡查", "道面破损", "关闭跑道",
        "最低设备清单", "仪表或者设备失效", "适用范围", "分类", "类别", "定义", "程序",
        "条件", "要求", "时限", "规定", "流程", "步骤", "文件", "机制",
    )
    for suffix in anchor_suffixes:
        start = 0
        while True:
            pos = compact.find(suffix, start)
            if pos < 0:
                break
            prefix_start = max(0, pos - 12)
            phrase = compact[prefix_start : pos + len(suffix)]
            phrase = re.sub(r"^[的第\d一二三四五六七八九十百千万两种项条类]+", "", phrase)
            phrase = re.sub(r"^(?:[^一-鿿A-Za-z0-9]*)(?:第?[一二三四五六七八九十]+类和第?[一二三四五六七八九十]+类)?", "", phrase)
            phrase = re.sub(r"的\d+种", "", phrase)
            if len(phrase) >= len(suffix) and phrase not in candidates:
                candidates.append(phrase)
            if suffix not in candidates:
                candidates.append(suffix)
            start = pos + len(suffix)
    for piece in re.split(r"(?:应当|需要|必须|如何|怎么|怎样|有什么|有哪些|哪些|多久|多长时间|在什么时候|什么时候|能否|是否|分为|包括)", compact):
        piece = piece.strip("的了后中")
        for noise in noise_terms:
            piece = piece.replace(noise, "")
        if 3 <= len(piece) <= 24:
            candidates.append(piece)
    normalized: List[str] = []
    short_anchor_terms = {"空重", "重心", "时限", "职责", "责任"}
    for candidate in candidates:
        value = _compact_match_text(candidate).strip("的了后")
        if any(token in value for token in ("如何", "怎么", "怎样", "哪些", "什么", "多久", "什么时候", "应当", "需要", "必须", "是否", "能否")):
            continue
        if len(value) < 3 and value not in short_anchor_terms:
            continue
        if value in {"规定", "要求", "条件", "内容", "用途", "情况", "项目", "办法", "方式"}:
            continue
        if value not in normalized:
            normalized.append(value)
    return sorted(normalized, key=len, reverse=True)[:8]


def _definition_term_from_query(query: str) -> str:
    compact = _compact_match_text(query)
    if not compact:
        return ""
    for suffix in ("的定义是什么", "定义是什么", "的定义", "是什么"):
        if compact.endswith(suffix) and len(compact) > len(suffix):
            term = compact[: -len(suffix)]
            if not _contains_any(term, ("颁发", "条件", "要求", "步骤", "程序", "流程", "申请", "哪些", "哪几", "多少")):
                return term
    return ""


def _definition_context_term_from_query(query: str) -> str:
    compact = _compact_match_text(query)
    if "定义" not in compact:
        return ""
    match = re.search(r"(?P<term>[A-Za-z][A-Za-z0-9-]{1,12}|[\u4e00-\u9fff]{2,16})定义(?:中|里|内)?", compact, re.I)
    if match:
        return _compact_match_text(match.group("term"))
    return ""


def _is_valid_search_query(query: str) -> bool:
    compact = re.sub(r"\s+", "", str(query or ""))
    if len(compact) < 2:
        return False
    meaningful = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", compact)
    if len(meaningful) < 2:
        return False
    if not re.search(r"[\u4e00-\u9fff]", compact):
        return False
    return len(meaningful) / max(len(compact), 1) >= 0.3


def _resolve_doc_name_from_query_text(query: str) -> str:
    compact = _compact_match_text(query)
    if not compact:
        return ""
    candidates: List[Tuple[int, str]] = []
    try:
        file_names = [name for name in os.listdir(DOC_DIR) if name.lower().endswith((".pdf", ".docx"))]
    except Exception:
        file_names = []
    for file_name in file_names:
        for alias in _doc_alias_keys(file_name):
            if alias and len(alias) >= 4 and alias in compact:
                candidates.append((len(alias), file_name))
                break
    if not candidates:
        return ""
    candidates.sort(reverse=True)
    return candidates[0][1]


def _extract_query_subject(query: str) -> str:
    compact = _compact_match_text(query)
    if not compact:
        return ""
    doc_name = _resolve_doc_name_from_query_text(query)
    if doc_name:
        compact = compact.replace(_compact_match_text(Path(doc_name).stem), "")
    definition_term = _definition_term_from_query(query)
    if definition_term:
        return definition_term
    definition_context_term = _definition_context_term_from_query(query)
    if definition_context_term:
        return definition_context_term
    stop_suffixes = (
        "的颁发条件是什么",
        "颁发条件是什么",
        "的颁发要求是什么",
        "颁发要求是什么",
        "申请人资格和文件要求是什么",
        "申请和颁发程序包括哪些步骤",
        "应在多长时间内提交",
        "在多长时间内提交",
        "多长时间内提交",
        "应在什么时间完成",
        "在什么时间完成",
        "什么时间完成",
        "按用途分为哪几类",
        "分为哪两种类别",
        "第一类和第二类分别适用于什么用途",
        "第一类和第二类适用于什么用途",
        "分别适用于什么用途",
        "分为哪几类",
        "的分类",
        "分类",
        "的类别",
        "类别",
        "包括哪些类别",
        "包括哪些",
        "适用于哪些对象",
        "适用于什么",
        "的适用范围是什么",
        "如何申请",
        "怎么申请",
        "怎样申请",
        "如何确定",
        "怎么确定",
        "怎样确定",
        "是什么",
    )
    for suffix in stop_suffixes:
        if compact.endswith(suffix) and len(compact) > len(suffix):
            return compact[: -len(suffix)].strip("的")
    subject_cut_markers = (
        "应当在什么时候", "在什么时候", "多长时间", "什么时间",
        "应当怎么", "应包括", "包括哪些", "哪些项目", "什么时候",
        "如何", "怎么", "怎样", "是否", "能否", "应当", "需要", "必须",
        "颁发", "申请", "提交", "分为", "包括", "适用", "流程", "程序",
        "制定", "处理", "完成", "确认", "满足", "报告", "巡查", "失效",
        "关闭", "阈值", "用途", "内容", "要求",
    )
    for marker in subject_cut_markers:
        index = compact.find(marker)
        if index > 1:
            candidate = compact[:index].strip("的")
            candidate = re.sub(r"第?[一二三四五六七八九十]+类(?:和第?[一二三四五六七八九十]+类)?", "", candidate)
            candidate = candidate.strip("的")
            return candidate
    terms = re.findall(r"[\u4e00-\u9fffA-Za-z0-9.]{2,}", compact)
    if not terms:
        return ""
    if len(terms) >= 2:
        for term in terms:
            if not any(token in term for token in ("如何", "怎么", "怎样", "什么", "哪些", "多少", "有没有")):
                return term
    return terms[0]


def infer_query_precision_hints(query: str) -> dict:
    """Infer retrieval constraints from query structure, not from fixed answers."""
    compact = _compact_match_text(query)
    subject = _extract_query_subject(query)
    target_doc_name = _resolve_doc_name_from_query_text(query)
    hints = {
        "target_doc_name": target_doc_name,
        "preferred_articles": [],
        "related_articles": [],
        "negative_articles": [],
        "preferred_sections": [],
        "required_terms": [],
        "forbidden_terms": [],
        "subject_terms": [subject] if subject else [],
        "intent_terms": [],
        "route": "",
        "strict_term_match": False,
        "title_keyphrases": [],
    }
    if not compact:
        return hints

    if _contains_any(compact, ("适用范围", "适用于什么", "适用于哪些", "适用于")):
        hints["route"] = "document_scope" if target_doc_name else "scope"
        hints["intent_terms"] = ["适用范围", "适用"]
        if target_doc_name:
            hints["subject_terms"] = []
    elif _definition_term_from_query(query):
        hints["route"] = "definition"
        hints["intent_terms"] = ["定义", "指", "是指"]
        hints["strict_term_match"] = True
    elif _definition_context_term_from_query(query):
        context_term = _definition_context_term_from_query(query)
        hints["route"] = "method"
        hints["subject_terms"] = [context_term]
        hints["required_terms"] = [context_term]
        hints["intent_terms"] = ["定义", "包括", "不包括", "排除"]
    elif _contains_any(compact, ("颁发条件", "颁发要求", "签发条件", "颁发需要满足", "条件是什么")):
        hints["route"] = "issuance_requirement"
        hints["intent_terms"] = ["颁发", "条件", "要求", "应当", "符合"]
    elif _contains_any(compact, ("申请", "提交", "文件", "资格")):
        hints["route"] = "application_requirement"
        hints["intent_terms"] = ["申请", "提交", "文件", "资格", "应当", "颁发"]
    elif _contains_any(compact, ("类别", "分类", "分为", "分成", "哪几类", "哪两种")) or (
        _contains_any(compact, ("第一类", "第二类", "第三类")) and _contains_any(compact, ("分别", "适用于", "用途"))
    ):
        hints["route"] = "classification"
        hints["intent_terms"] = ["类别", "分类", "分为", "分成", "包括"]
    elif _contains_any(compact, ("包括哪些", "包括什么", "哪些项目", "哪些内容", "报告情形")):
        hints["route"] = "method"
        hints["intent_terms"] = ["包括", "项目", "内容", "情形", "应当", "必须"]
    elif _contains_any(compact, ("程序", "流程", "步骤")):
        hints["route"] = "procedure"
        hints["intent_terms"] = ["程序", "流程", "步骤", "申请", "颁发"]
    elif _contains_any(compact, ("如何", "怎么", "怎样")):
        hints["route"] = "method"
        hints["intent_terms"] = ["如何", "怎么", "怎样", "确定", "方法", "应当", "必须"]
    elif _contains_any(compact, ("多长时间", "什么时间", "什么时候", "何时", "多久")):
        hints["route"] = "time_limit"
        hints["intent_terms"] = ["时间", "日内", "小时", "巡查", "报告"]
    elif _contains_any(compact, ("阈值", "超过多少", "取多少", "多少必须")):
        hints["route"] = "numeric_threshold"
        hints["intent_terms"] = ["超过", "千克", "数值", "必须", "取", "适用范围"]

    if hints["route"] in {"time_limit", "numeric_threshold", "method"}:
        focus_terms = re.findall(
            r"(最大[\u4e00-\u9fff]{1,10}(?:重量|商载|座位数)|[\u4e00-\u9fff]{2,12}(?:时间|期限|重量|重心|系数|阈值|巡查|报告))",
            compact,
        )
        for term in focus_terms:
            term = term.rstrip("阈值")
            if term and term not in hints["required_terms"]:
                hints["required_terms"].append(term)

    if hints["route"] != "document_scope":
        for term in hints["subject_terms"]:
            if term:
                hints["required_terms"].append(term)
                suffix_terms = ("适航证", "合格证", "认可证", "批准标签", "许可证", "证书", "航空器", "飞机", "旋翼航空器")
                for suffix in suffix_terms:
                    if term.endswith(suffix) and len(term) > len(suffix) + 1:
                        prefix = term[: -len(suffix)]
                        if prefix and prefix not in hints["forbidden_terms"]:
                            hints.setdefault("subject_modifiers", []).append(prefix)
    hints["title_keyphrases"] = _extract_title_keyphrases(query, subject)
    hints["compact_query"] = compact
    return hints


def _has_structural_subject_support(hints: dict, records: List[dict]) -> bool:
    modifiers = [_compact_match_text(term) for term in hints.get("subject_modifiers", []) if term]
    if not modifiers:
        return True
    haystacks = [
        _compact_match_text(
            " ".join(
                str(record.get(key) or "")
                for key in ("doc_name", "label", "title", "article_label", "body_text")
            )
        )
        for record in records
        if not record.get("is_toc")
    ]
    for modifier in modifiers:
        if not any(modifier in text for text in haystacks):
            return False
    return True


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
    text = re.sub(r"(?<=[\u4e00-\u9fff])[ \t]+(?=[\u4e00-\u9fff])", "", text)
    return text.strip()


def _truncate_for_milvus_varchar(text: str, max_bytes: int = MILVUS_CONTENT_MAX) -> str:
    value = str(text or "")
    if not value:
        return ""
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value
    return encoded[:max_bytes].decode("utf-8", errors="ignore").rstrip()


def _looks_like_regulatory_reference(line: str) -> bool:
    s = clean_block_text(line).split("\n", 1)[0].strip()
    if not s:
        return False
    reference_hits = len(re.findall(r"(?:CCAR|§|\b\d{2,3}\.\d{3,4}\b)", s, flags=re.I))
    return reference_hits >= 2 and not re.match(r"^\d+(?:\.\d+)*\s+[\u4e00-\u9fff]{2,}", s)


def _looks_like_inline_article_reference(line: str) -> bool:
    s = clean_block_text(line).split("\n", 1)[0].strip()
    match = re.match(r"^第\s*[0-9A-Za-z一二三四五六七八九十百千万]+(?:\.[0-9A-Za-z一二三四五六七八九十百千万]+)*\s*条\s*(?P<title>.*)$", s)
    if not match:
        return False
    title = clean_block_text(match.group("title"))
    if not title:
        return False
    if re.search(r"[。；;，,：:]", title):
        return True
    return bool(re.match(r"^(?:进行|接受|检查|确认|规定|所列|相关|执行|符合|申请|颁发|提交|款|中|和|及)", title))


def _numeric_heading_level(value: str) -> int:
    number = clean_block_text(value)
    if "." not in number:
        return 2
    depth = len([part for part in number.split(".") if part])
    return min(1 + depth, 6)


def _looks_like_short_decimal_heading(title: str) -> bool:
    title = clean_block_text(title)
    if not title:
        return False
    if len(title) <= 22:
        return True
    return bool(re.search(r"(要求|说明|范围|分类|程序|资料|文件|章节|原则|内容|格式|语言|定义|概述)$", title)) and len(title) <= 28


def detect_heading(line: str) -> Tuple[bool, int, str]:
    s = clean_block_text(line).split("\n", 1)[0].strip()
    if not s:
        return False, 0, ""
    if _looks_like_regulatory_reference(s):
        return False, 0, ""
    if re.match(r"^第\s*[0-9一二三四五六七八九十百千]+\s*章", s) or re.match(r"^[A-Z]\s*章", s):
        return True, 1, s
    if re.match(r"^附录\s*[A-Za-z\d]", s):
        return True, 1, s
    if re.match(r"^第\s*[0-9一二三四五六七八九十百千]+\s*节", s):
        return True, 2, s
    if re.match(r"^第\s*[0-9一二三四五六七八九十百千\d\.]+\s*条", s):
        if _looks_like_inline_article_reference(s):
            return False, 0, ""
        return True, 2, s
    if NUMBERED_HEADING_PATTERN.match(s):
        return True, 2, s
    decimal_match = DECIMAL_HEADING_PATTERN.match(s)
    if decimal_match and _looks_like_short_decimal_heading(decimal_match.group(2)):
        return True, _numeric_heading_level(decimal_match.group(1)), s
    dot_match = DOT_HEADING_PATTERN.match(s)
    if dot_match and _looks_like_short_decimal_heading(dot_match.group(2)):
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
    if match:
        article_id = re.sub(r"\s+", "", match.group(1))
        article_label = f"第{article_id}条"
        return article_id, article_label
    bare_match = re.search(r"(?<![A-Za-z0-9.])(\d{1,4}(?:\.\d{1,4})+)(?![A-Za-z0-9.])", cleaned)
    if not bare_match:
        return "", ""
    article_id = bare_match.group(1)
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
        if remainder and (len(remainder) >= 4 or len(lines) >= 2):
            return heading, remainder
        return heading, ""

    inline_match = re.match(rf"^\s*({re.escape(heading)})(?:[:：]\s*)?(.*)$", cleaned, flags=re.S)
    if inline_match:
        remainder = clean_block_text(inline_match.group(2))
        if remainder and len(remainder) >= 4:
            return heading, remainder
    return heading, ""


def _is_structural_boundary_line(line: str) -> bool:
    stripped = clean_block_text(line)
    if not stripped:
        return False
    if _looks_like_regulatory_reference(stripped):
        return False
    if ARTICLE_LABEL_PATTERN.match(stripped) or SECTION_LABEL_PATTERN.match(stripped):
        if _looks_like_inline_article_reference(stripped):
            return False
        return True
    if detect_heading(stripped)[0]:
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
    clause_id = "" if source_type == "heading" else _extract_clause_id(content)
    list_item_id = "" if source_type == "heading" else _extract_list_item_id(content)
    role = _detect_chunk_role("\n".join([section_title, heading_text, content]), source_type=source_type)
    heading_seed = _compact_section_label(section_title or heading_text or article_label or _extract_highlight_text(content))
    normalized_heading = _normalize_heading_value(heading_seed)
    has_enumeration = 0 if source_type == "heading" else (1 if (clause_id or list_item_id or any(token in content for token in CLASSIFICATION_TERMS)) else 0)
    list_density = 0.0 if source_type == "heading" else _estimate_list_density(content)
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
    target_roles = query_signature["target_roles"]
    classification_like = (
        any(token in text for token in ("类别", "分类", "分为", "分成", "包括", "哪几类", "哪两类", "哪些类型"))
        and any(token in text for token in ("适航证", "合格证", "认可证", "批准标签", "证书", "特许飞行证"))
    )
    article_like = bool(
        query_signature.get("article_id_variants")
        or re.search(r"(?:第\s*)?[0-9A-Za-z一二三四五六七八九十百千万]+(?:\.[0-9A-Za-z一二三四五六七八九十百千万]+)*\s*条", text)
        or re.search(r"(?<![A-Za-z0-9.])\d+\.\d+(?:\.\d+)*(?![A-Za-z0-9.])", text)
        or re.search(r"\b(?:ccar|ata|amm|fim|tsm|ipc|srm)\b", normalized)
        or classification_like
        or any(token in text for token in ("颁发条件", "颁发要求", "签发条件", "申请条件", "批准条件"))
    )
    fault_like = any(token in text for token in ("故障", "异常", "告警", "失效", "不工作", "排查", "原因", "报警"))
    procedure_like = any(token in text for token in ("步骤", "如何", "怎么", "流程", "更换", "检查", "操作", "procedure", "程序", "颁发程序"))
    concept_like = any(token in text for token in ("是什么", "含义", "定义", "区别", "作用", "概念"))
    structure_first = bool(
        article_like
        or "适用范围" in text
        or "effective_date" in target_roles
        or "definition" in target_roles
        or classification_like
        or re.search(r"第\s*[0-9一二三四五六七八九十百千]+\s*章", text)
        or re.search(r"第\s*[0-9一二三四五六七八九十百千]+\s*节", text)
    )

    if article_like:
        query_type = "article_lookup"
        profile = {
            "rrf_k": 36,
            "kw_weight": 1.8,
            "vec_weight": 1.4,
            "structure_weight": 2.2,
            "article_weight": 2.35,
            "heading_weight": 1.9,
            "table_weight": 0.7,
            "expand_children": 5,
            "child_char_budget": 1100,
        }
    elif fault_like:
        query_type = "fault_diagnosis"
        profile = {"rrf_k": 70, "kw_weight": 1.05, "vec_weight": 1.45, "structure_weight": 0.7, "article_weight": 0.85, "heading_weight": 0.8, "table_weight": 0.6, "expand_children": 3, "child_char_budget": 780}
    elif procedure_like:
        query_type = "procedure"
        profile = {"rrf_k": 52, "kw_weight": 1.2, "vec_weight": 1.25, "structure_weight": 0.95, "article_weight": 1.0, "heading_weight": 1.05, "table_weight": 0.75, "expand_children": 4, "child_char_budget": 900}
    elif concept_like:
        query_type = "concept_explanation"
        profile = {"rrf_k": 58, "kw_weight": 0.95, "vec_weight": 1.3, "structure_weight": 1.3, "article_weight": 1.15, "heading_weight": 1.0, "table_weight": 0.65, "expand_children": 3, "child_char_budget": 700}
    else:
        query_type = "general"
        profile = {"rrf_k": 60, "kw_weight": 1.0, "vec_weight": 1.2, "structure_weight": 1.15, "article_weight": 0.95, "heading_weight": 1.0, "table_weight": 0.7, "expand_children": 3, "child_char_budget": 800}

    profile.update(
        {
            "query_type": query_type,
            "exact_match_bias": article_like,
            "fault_keywords": fault_like,
            "procedure_keywords": procedure_like,
            "classification_keywords": classification_like,
            "query_signature": query_signature,
            "target_roles": target_roles,
            "structure_first": structure_first,
        }
    )
    precision_hints = infer_query_precision_hints(text)
    profile["precision_hints"] = precision_hints
    if precision_hints.get("target_doc_name"):
        profile["target_doc_name"] = precision_hints["target_doc_name"]
        profile["structure_first"] = True
    if precision_hints.get("preferred_articles") or precision_hints.get("preferred_sections"):
        profile["exact_match_bias"] = True
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


HYDE_SYSTEM_PROMPT = "你是一个专业的民航法规问答助手。请根据问题，生成一段准确、完整的回答摘要。你的回答将用于信息检索，请直接给出答案摘要，不要有多余的说明。回答应在150字以内，使用中文。"


def _generate_hypothetical_answer(query: str, llm_call: Optional[Callable[[str, str], str]] = None) -> str:
    """Generate a hypothetical answer for HyDE retrieval. Falls back gracefully if LLM unavailable."""
    if not llm_call:
        return ""
    user_prompt = f"问题：{query}\n\n回答摘要："
    try:
        return llm_call(HYDE_SYSTEM_PROMPT, user_prompt).strip()
    except Exception:
        return ""


def _compact_section_label(text: str) -> str:
    cleaned = clean_block_text(text)
    cleaned = cleaned.replace("[SECTION]", "").replace("交通运输部规章交通运输部发布", "").strip(" ：:;；，,")
    cleaned = re.sub(r"(第\s*[0-9A-Za-z一二三四五六七八九十百千万\d\.]+\s*条)\s*", r"\1 ", cleaned)
    cleaned = re.sub(r"(第\s*[0-9一二三四五六七八九十百千万\d\.]+\s*节)\s*", r"\1 ", cleaned)
    if not cleaned:
        return ""

    for marker in ("适用范围", "定义", "要求", "限制"):
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

    return round(score, 6)


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


ROUTE_TITLE_ACTION_TERMS = {
    "issuance_requirement": ("颁发", "签发"),
    "application_requirement": ("申请", "提交", "文件", "资格"),
    "classification": ("类别", "分类", "分为", "分成"),
    "procedure": ("程序", "流程", "步骤"),
    "method": ("方法", "确定", "制定", "处理"),
    "time_limit": ("时间", "时限", "日内", "小时"),
    "numeric_threshold": ("数值", "超过", "阈值", "重量"),
}

ACTION_VARIANT_MODIFIERS = ("重新", "再次", "补发", "更换")


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


def _is_repeating_pdf_noise(text: str) -> bool:
    compact = re.sub(r"\s+", "", clean_block_text(text))
    if not compact:
        return True
    if re.fullmatch(r"[-—]?\d{1,4}[-—]?", compact):
        return True
    return compact in {
        "交通运输部规章",
        "交通运输部发布",
        "中国民用航空局",
        "民航局",
    }


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

    layout_blocks = []
    for block in text_blocks:
        bbox = block.get("bbox") or page_rect
        y0 = float(bbox[1])
        y1 = float(bbox[3])
        if _is_repeating_pdf_noise(block.get("text", "")):
            continue
        if y1 < page_rect.height * 0.10 or y0 > page_rect.height * 0.84:
            continue
        layout_blocks.append(block)
    if not layout_blocks:
        layout_blocks = text_blocks

    centers = [((block["bbox"][0] + block["bbox"][2]) / 2) / max(page_rect.width, 1.0) for block in layout_blocks]
    left_blocks = sum(1 for center in centers if center < 0.44)
    right_blocks = sum(1 for center in centers if center > 0.56)
    total_chars = sum(len(block["text"]) for block in layout_blocks)
    width_ratios = [
        max(0.0, float(block["bbox"][2]) - float(block["bbox"][0])) / max(page_rect.width, 1.0)
        for block in layout_blocks
        if block.get("bbox")
    ]
    median_width_ratio = float(np.median(width_ratios)) if width_ratios else 0.0

    if total_chars < 140:
        return "image_sparse", 0.35
    if median_width_ratio >= 0.56:
        return "single_column", 0.86
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


def _append_segment_units(
    emitted: List[dict],
    doc_name: str,
    segment: str,
    page_number: int,
    page_count: int,
    bbox,
    layout_type: str,
    parse_confidence: float,
    is_ocr_fallback: bool,
    document_type: str,
    default_section_title: str = "",
    default_heading_text: str = "",
) -> None:
    segment = clean_block_text(segment)
    if len(segment) < 2:
        return

    is_heading, heading_level, heading_text = detect_heading(segment)
    if is_heading:
        heading_only, body = _split_heading_from_body(segment, heading_text or segment)
        heading_only = heading_only or clean_block_text(heading_text or segment)
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
            "heading_text": heading_only,
            "highlight_text": _extract_highlight_text(heading_only, heading_only),
            "document_type": document_type,
        }
        heading_unit.update(
            _build_structural_metadata(
                heading_only,
                source_type="heading",
                section_title=heading_only,
                heading_text=heading_only,
            )
        )
        emitted.append(heading_unit)

        if body:
            for child in _split_structured_text(body) or [body]:
                _append_segment_units(
                    emitted,
                    doc_name=doc_name,
                    segment=child,
                    page_number=page_number,
                    page_count=page_count,
                    bbox=bbox,
                    layout_type=layout_type,
                    parse_confidence=parse_confidence,
                    is_ocr_fallback=is_ocr_fallback,
                    document_type=document_type,
                    default_section_title=heading_only,
                    default_heading_text=heading_only,
                )
        return

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
        "section_title": default_section_title,
        "heading_text": default_heading_text,
        "highlight_text": _extract_highlight_text(segment, default_heading_text or default_section_title),
        "document_type": document_type,
    }
    unit.update(
        _build_structural_metadata(
            segment,
            source_type="text",
            section_title=default_section_title,
            heading_text=default_heading_text,
        )
    )
    emitted.append(unit)


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
        _append_segment_units(
            emitted,
            doc_name=doc_name,
            segment=normalized_content,
            page_number=page_number,
            page_count=page_count,
            bbox=bbox,
            layout_type=layout_type,
            parse_confidence=parse_confidence,
            is_ocr_fallback=is_ocr_fallback,
            document_type=document_type,
            default_section_title=heading_text,
            default_heading_text=heading_text,
        )
        return emitted

    for segment in _split_structured_text(normalized_content) or [normalized_content]:
        _append_segment_units(
            emitted,
            doc_name=doc_name,
            segment=segment,
            page_number=page_number,
            page_count=page_count,
            bbox=bbox,
            layout_type=layout_type,
            parse_confidence=parse_confidence,
            is_ocr_fallback=is_ocr_fallback,
            document_type=document_type,
        )
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

    # Contextual Retrieval: generate article/section-level context prefix for embedding
    context_prefix = None
    if article_ids or section_ids:
        parts: List[str] = []
        if article_ids:
            article_id = next(iter(article_ids))
            parts.append(f"第{article_id}条")
        if section_ids:
            section_id = next(iter(section_ids))
            parts.append(f"第{section_id}节")
        if clause_ids:
            clause_id = next(iter(clause_ids))
            parts.append(f"第{clause_id}款")
        if parts:
            context_prefix = "本文出自《" + first_unit.get("doc_name", "") + "》" + "，".join(parts) + "，"

    blocks = []
    for text in texts:
        # Inject context prefix into chunk (Contextual Retrieval)
        if context_prefix and len(text) > 60 and article_ids:
            article_id = next(iter(article_ids))
            if article_id not in text[:200]:
                text = context_prefix + text
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


def _dedupe_compact_parts(parts: List[str], limit: int = 8) -> List[str]:
    compacted: List[str] = []
    for value in parts:
        normalized = _compact_section_label(value)
        if not normalized:
            normalized = clean_block_text(value)
        if not normalized:
            continue
        if normalized in compacted:
            continue
        if any(len(normalized) <= 18 and normalized in existing for existing in compacted):
            continue
        compacted.append(normalized)
        if len(compacted) >= limit:
            break
    return compacted


def _build_heading_path_text(heading_path: List[str], block: dict) -> str:
    current_label = block.get("heading_text", "") if block.get("is_heading") else block.get("section_title", "")
    return " > ".join(_dedupe_compact_parts([*heading_path, current_label], limit=8))


def _build_scope_anchor(block: dict) -> str:
    doc_stem = Path(str(block.get("doc_name", "") or "")).stem
    heading_context = block.get("heading_path_text", "") or block.get("section_title", "") or block.get("heading_text", "")
    heading_parts = [part.strip() for part in clean_block_text(heading_context).split(">") if part.strip()]
    return " > ".join(
        _dedupe_compact_parts(
            [
                doc_stem,
                *heading_parts,
                block.get("article_label", ""),
                block.get("section_label", ""),
                block.get("clause_id", ""),
                block.get("list_item_id", ""),
                block.get("chunk_role", ""),
            ],
            limit=10,
        )
    )


def _build_scope_key(block: dict) -> str:
    raw_parts = [
        Path(str(block.get("doc_name", "") or "")).stem,
        str(block.get("section_id", "") or ""),
        str(block.get("article_id", "") or ""),
        str(block.get("clause_id", "") or ""),
        str(block.get("list_item_id", "") or ""),
        str(block.get("normalized_heading", "") or ""),
        str(block.get("chunk_role", "") or ""),
    ]
    return "::".join(part for part in raw_parts if part)


def _build_retrieval_text(block: dict) -> str:
    doc_name = Path(str(block.get("doc_name") or "")).stem
    return clean_block_text(
        "\n".join(
            part
            for part in [
                doc_name,
                block.get("scope_anchor", ""),
                block.get("heading_path_text", ""),
                block.get("section_title", ""),
                block.get("heading_text", ""),
                block.get("parent_heading", ""),
                block.get("article_label", ""),
                block.get("clause_path", ""),
                block.get("content", ""),
            ]
            if part
        )
    )[:MILVUS_CONTENT_MAX]


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
            block["heading_path_text"] = _build_heading_path_text(block.get("heading_path", []), block)
            block["scope_anchor"] = _build_scope_anchor(block)
            block["scope_key"] = _build_scope_key(block)
            block["retrieval_text"] = _build_retrieval_text(block)
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
                if _is_repeating_pdf_noise(content):
                    continue
                if len(content) < 2:
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
    total_units = len(units)
    progress_interval = max(200, total_units // 10) if total_units >= 400 else 0

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
        if pending_chars + len(current_unit["content"]) > int(MAX_CHUNK_SIZE * 1.35):
            return True

        last_article = last_unit.get("article_id", "")
        current_article = current_unit.get("article_id", "")
        if last_article and current_article and last_article != current_article:
            return True

        last_section = last_unit.get("section_id", "")
        current_section = current_unit.get("section_id", "")
        if last_section and current_section and last_section != current_section:
            return True

        # Clause/list/role tags are retrieval metadata, not hard chunk boundaries.
        # Splitting on every tag change makes PDF lines look like sentence chunks.
        if current_unit.get("boundary_conflict"):
            return True
        return False

    for index, unit in enumerate(units, start=1):
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
            if progress_interval and index % progress_interval == 0:
                print(f"    * 切片进度：{index}/{total_units} 单元，当前证据块数：{len(blocks)}")
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
            if progress_interval and index % progress_interval == 0:
                print(f"    * 切片进度：{index}/{total_units} 单元，当前证据块数：{len(blocks)}")
            continue

        if pending_text_units:
            last_unit = pending_text_units[-1]
            if should_flush_text(last_unit, unit):
                flush_pending()

        pending_text_units.append(unit)
        pending_chars += len(unit["content"])
        if progress_interval and index % progress_interval == 0:
            print(f"    * 切片进度：{index}/{total_units} 单元，当前证据块数：{len(blocks)}")

    flush_pending()
    return build_chain(blocks)


def _encode_with_progress(embedder, contents: List[str], batch_size: int = 128, stage_label: str = "向量化") -> np.ndarray:
    if not contents:
        return np.empty((0, VECTOR_DIM), dtype=np.float32)

    total = len(contents)
    total_batches = math.ceil(total / batch_size)
    arrays: List[np.ndarray] = []
    started_at = time.time()

    for batch_index, start in enumerate(range(0, total, batch_size), start=1):
        end = min(start + batch_size, total)
        batch_contents = contents[start:end]
        batch_started_at = time.time()
        batch_embeddings = embedder.encode(
            batch_contents,
            batch_size=min(64, len(batch_contents)),
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        arrays.append(np.asarray(batch_embeddings, dtype=np.float32))
        batch_elapsed = time.time() - batch_started_at
        total_elapsed = time.time() - started_at
        print(
            f"    * {stage_label}进度：{end}/{total} 块 "
            f"({end / total:.1%})，batch {batch_index}/{total_batches}，"
            f"本批 {batch_elapsed:.1f}s，累计 {total_elapsed:.1f}s"
        )

    if len(arrays) == 1:
        return arrays[0]
    return np.vstack(arrays)


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
                    "heading_path_text": block.get("heading_path_text", ""),
                    "scope_anchor": block.get("scope_anchor", ""),
                    "scope_key": block.get("scope_key", ""),
                    "retrieval_text": block.get("retrieval_text", ""),
                    "structural_node_id": block.get("structural_node_id", ""),
                    "structural_parent_id": block.get("structural_parent_id", ""),
                    "structural_record_type": block.get("structural_record_type", ""),
                    "structural_level": int(block.get("structural_level", 0)),
                    "structural_number": block.get("structural_number", ""),
                    "structural_path": block.get("structural_path", ""),
                    "structural_source": block.get("structural_source", ""),
                    "direct_child_count": int(block.get("direct_child_count", 0)),
                    "descendant_count": int(block.get("descendant_count", 0)),
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
                started_at = time.time()
                print('加载嵌入模型（本地缓存）...')
                _embedder = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
                if torch.cuda.is_available():
                    _embedder = _embedder.to("cuda")
                    print(f"嵌入模型已加载到 GPU，用时 {time.time() - started_at:.1f}s")
                else:
                    print(f"嵌入模型使用 CPU，用时 {time.time() - started_at:.1f}s")
    return _embedder


def get_reranker():
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                try:
                    started_at = time.time()
                    print("加载重排序模型（本地缓存）...")
                    rerank_device = "cuda" if RERANK_DEVICE == "cuda" and torch.cuda.is_available() else "cpu"
                    _reranker = CrossEncoder(
                        RERANK_MODEL,
                        device=rerank_device,
                        local_files_only=True,
                    )
                    print(f"重排序模型加载完成（{rerank_device}），用时 {time.time() - started_at:.1f}s")
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
                    "heading_path_text": {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_smart"},
                    "scope_anchor": {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_smart"},
                    "scope_key": {"type": "keyword"},
                    "retrieval_text": {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_smart"},
                    "structural_node_id": {"type": "keyword"},
                    "structural_parent_id": {"type": "keyword"},
                    "structural_record_type": {"type": "keyword"},
                    "structural_level": {"type": "integer"},
                    "structural_number": {"type": "keyword"},
                    "structural_path": {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_smart"},
                    "structural_source": {"type": "keyword"},
                    "direct_child_count": {"type": "integer"},
                    "descendant_count": {"type": "integer"},
                    "cross_page": {"type": "integer"},
                    "boundary_conflict": {"type": "integer"},
                }
            }
        }

    def _create_milvus_collection(self) -> Collection:
        fields = [
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=MILVUS_CONTENT_MAX),
        ]
        schema = CollectionSchema(fields, f"aerospace semantic index ({EMBEDDING_MODEL}, dim={VECTOR_DIM})")
        collection = Collection(MILVUS_COLLECTION, schema)
        collection.create_index("vector", {"metric_type": "COSINE", "index_type": "IVF_FLAT", "params": {"nlist": 1024}})
        return collection

    def _collection_vector_dim(self, collection: Collection) -> int:
        for field in collection.schema.fields:
            if field.name == "vector":
                try:
                    return int(field.params.get("dim") or 0)
                except Exception:
                    return 0
        return 0

    def ensure_databases(self) -> Collection:
        if not utility.has_collection(MILVUS_COLLECTION):
            collection = self._create_milvus_collection()
        else:
            collection = Collection(MILVUS_COLLECTION)
            current_dim = self._collection_vector_dim(collection)
            if current_dim != VECTOR_DIM:
                logger.warning(
                    "Milvus collection vector dim mismatch: current=%s expected=%s. Recreating collection.",
                    current_dim,
                    VECTOR_DIM,
                )
                utility.drop_collection(MILVUS_COLLECTION)
                collection = self._create_milvus_collection()
        collection.load()
        if not self.es_client.indices.exists(index=ES_INDEX):
            self.es_client.indices.create(index=ES_INDEX, body=self._es_mapping())
        else:
            self.es_client.indices.put_mapping(index=ES_INDEX, body=self._es_mapping()["mappings"])
        return collection

    def reset_databases(self) -> Collection:
        if utility.has_collection(MILVUS_COLLECTION):
            utility.drop_collection(MILVUS_COLLECTION)
        collection = self._create_milvus_collection()
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

    def _insert_blocks_to_milvus(self, collection: Collection, blocks: List[dict], embeddings: List[List[float]], contents: List[str], batch_size: int = 256):
        total = len(blocks)
        if not (len(embeddings) == total == len(contents)):
            raise ValueError("Milvus 插入数据长度不一致")
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_blocks = blocks[start:end]
            batch_embeddings = embeddings[start:end]
            batch_contents = contents[start:end]
            collection.insert(
                [
                    [block["chunk_id"] for block in batch_blocks],
                    batch_embeddings,
                    batch_contents,
                ]
            )

    def _index_document_blocks(self, file_path: str) -> Tuple[List[dict], dict]:
        started_at = time.time()
        units, health = extract_document_units(file_path)
        print(f"  - 文本单元提取完成：{len(units)} 个单元，用时 {time.time() - started_at:.1f}s")
        if not units:
            return [], health
        structure_started_at = time.time()
        structural_records = build_structural_records(units, os.path.basename(file_path))
        save_structural_records(os.path.basename(file_path), structural_records)
        health["structural_record_count"] = len(structural_records)
        print(f"  - 结构记录构建完成：{len(structural_records)} 条，用时 {time.time() - structure_started_at:.1f}s")
        chunk_started_at = time.time()
        print("  - 开始生成证据块...")
        blocks = build_document_chunks(units, self._get_embedder())
        print(f"  - 证据块切片完成：{len(blocks)} 个，用时 {time.time() - chunk_started_at:.1f}s")
        attach_started_at = time.time()
        print("  - 开始挂载结构标签...")
        blocks = attach_structure_to_blocks(blocks, structural_records)
        print(f"  - 结构标签挂载完成，用时 {time.time() - attach_started_at:.1f}s")
        health["chunk_count"] = len(blocks)
        print(f"  - 证据块生成完成：{len(blocks)} 个，用时 {time.time() - chunk_started_at:.1f}s")
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
        print("  - 开始解析与结构化...")
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

        contents = [_truncate_for_milvus_varchar(block.get("retrieval_text", block["content"])) for block in blocks]
        print(f"  - 开始向量化：{len(contents)} 个块")
        embedding_started_at = time.time()
        embeddings = _encode_with_progress(self._get_embedder(), contents, stage_label="向量化")
        print(f"  - 向量化完成，用时 {time.time() - embedding_started_at:.1f}s")
        milvus_started_at = time.time()
        print("  - 写入 Milvus...")
        self._insert_blocks_to_milvus(collection, blocks, embeddings.tolist(), contents)
        collection.flush()
        collection.load()
        print(f"  - Milvus 写入完成，用时 {time.time() - milvus_started_at:.1f}s")
        es_started_at = time.time()
        print("  - 写入 Elasticsearch...")
        save_metadata_to_es(self.es_client, blocks)
        print(f"  - Elasticsearch 写入完成，用时 {time.time() - es_started_at:.1f}s")
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
        remove_structural_records(doc_name)
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

        clear_structural_records()
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

        contents = [_truncate_for_milvus_varchar(block.get("retrieval_text", block["content"])) for block in all_blocks]
        embeddings = _encode_with_progress(self._get_embedder(), contents, stage_label="全量向量化")
        self._insert_blocks_to_milvus(collection, all_blocks, embeddings.tolist(), contents)
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
    def __init__(self, hyde_llm_call: Optional[Callable[[str, str], str]] = None):
        if connections.has_connection("default"):
            connections.disconnect("default")
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
        self.es_client = Elasticsearch(ES_HOST)
        self.milvus_coll = Collection(MILVUS_COLLECTION)
        self.milvus_coll.load()
        self.reranker = get_reranker()
        self.embedder = get_embedder()
        self.hyde_llm_call = hyde_llm_call
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
            for alias in _doc_alias_keys(file_name):
                if alias and len(alias) >= 5 and alias in query_key:
                    candidates.append((len(alias) * 10 - 3, file_name))
                    break
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

    def _semantic_unit_lines(self, metadata: dict, content: str) -> List[str]:
        text = clean_block_text(re.sub(r"^\[SECTION\]\s*", "", content or ""))
        if "[PDF页原文]" in text:
            text = text.split("[PDF页原文]", 1)[0]

        lines: List[str] = []
        seen = set()

        def push(value: str):
            cleaned = clean_block_text(value)
            if not cleaned:
                return
            if cleaned.startswith("[第") and cleaned.endswith("页]"):
                return
            if _is_repeating_pdf_noise(cleaned):
                return
            compact = re.sub(r"\s+", "", cleaned)
            if len(compact) < 3 and not (lines and re.search(r"[；;。，,.、：:]$", compact)):
                return
            if cleaned in seen:
                return
            seen.add(cleaned)
            lines.append(cleaned)

        for value in (
            metadata.get("heading", ""),
            metadata.get("section_title", ""),
            metadata.get("highlight_text", ""),
        ):
            for part in clean_block_text(value).splitlines():
                push(part)

        for line in text.splitlines():
            push(line)
            if len(lines) >= 80:
                break

        return lines

    def _line_search_variants(self, line: str) -> List[str]:
        cleaned = clean_block_text(line)
        if not cleaned:
            return []

        variants: List[str] = []
        seen = set()

        def push(value: str):
            item = re.sub(r"[ \t]{2,}", " ", str(value or "")).strip()
            compact_item = re.sub(r"\s+", "", item)
            is_short_continuation = len(compact_item) >= 2 and bool(re.search(r"[；;。，,.、：:]$", compact_item))
            if (len(item) < 4 and not is_short_continuation) or item in seen:
                return
            seen.add(item)
            variants.append(item)

        normalized = re.sub(r"\s+", " ", cleaned).strip()
        spaced = normalized
        spaced = re.sub(r"(第\s*[0-9一二三四五六七八九十百千万\d\.]+\s*章)", r"\1 ", spaced)
        spaced = re.sub(r"(第\s*[0-9A-Za-z一二三四五六七八九十百千万\d\.]+\s*条)", r"\1 ", spaced)

        for candidate in (cleaned, normalized, spaced, _compact_section_label(cleaned)):
            push(candidate)
        article_heading_match = re.match(
            r"^(第\s*[0-9A-Za-z一二三四五六七八九十百千万\d\.]+\s*条)(?P<title>[\u4e00-\u9fffA-Za-z].+)$",
            cleaned,
        )
        if article_heading_match:
            label = article_heading_match.group(1)
            title = article_heading_match.group("title")
            article_number = re.sub(r"^\s*第\s*|\s*条\s*$", "", label)
            push(f"{label} {title}")
            compact_label = re.sub(r"\s+", "", label)
            push(f"{compact_label} {title}")
            push(f"第 {article_number} 条 {title}")
            push(f"第 {article_number} 条  {title}")
        numeric_heading_match = re.match(r"^(\d+(?:\.\d+)+)(?P<title>[\u4e00-\u9fffA-Za-z].+)$", cleaned)
        if numeric_heading_match:
            push(f"{numeric_heading_match.group(1)} {numeric_heading_match.group('title')}")
        if len(cleaned) > 80:
            push(cleaned[:100])
            push(cleaned[-100:])
            push(normalized[:100])
            push(normalized[-100:])

        variants.sort(key=len, reverse=True)
        return variants

    def _semantic_unit_bbox_on_page(self, page: fitz.Page, metadata: dict, content: str) -> Tuple[Optional[dict], str, int, int]:
        lines = self._semantic_unit_lines(metadata, content)
        if not lines:
            return None, "", 0, 0

        matched_boxes: List[Optional[dict]] = []
        matched_lines: List[str] = []
        for line in lines:
            line_bbox = None
            matched_text = ""
            for variant in self._line_search_variants(line):
                try:
                    rects = page.search_for(variant)
                except Exception:
                    rects = []
                if not rects:
                    continue
                line_bbox = _bbox_union(
                    [_normalize_bbox((rect.x0, rect.y0, rect.x1, rect.y1), page.rect) for rect in rects]
                )
                matched_text = line
                break
            if line_bbox:
                matched_boxes.append(line_bbox)
                matched_lines.append(matched_text)

        if not matched_boxes:
            return None, "", 0, len(lines)
        if len(lines) > 1 and len(matched_boxes) == 1:
            return None, "", 1, len(lines)

        bbox = _bbox_union(matched_boxes)
        if not bbox:
            return None, "", len(matched_boxes), len(lines)

        highlight_text = clean_block_text("\n".join(matched_lines))
        return bbox, highlight_text, len(matched_boxes), len(lines)

    def _semantic_unit_bboxes_on_pages(
        self,
        document: fitz.Document,
        page_numbers: List[int],
        metadata: dict,
        content: str,
    ) -> List[dict]:
        page_hits: List[dict] = []
        seen_pages = set()
        for page_number in page_numbers:
            if page_number in seen_pages or page_number < 1 or page_number > document.page_count:
                continue
            seen_pages.add(page_number)
            page = document.load_page(page_number - 1)
            bbox, highlight_text, matched_count, total_count = self._semantic_unit_bbox_on_page(page, metadata, content)
            if not bbox:
                continue
            if total_count > 1 and matched_count < 2:
                continue
            page_hits.append(
                {
                    "page_number": page_number,
                    "bbox": bbox,
                    "highlight_text": highlight_text,
                    "matched_line_count": matched_count,
                    "semantic_line_count": total_count,
                }
            )
        page_hits.sort(key=lambda item: item["page_number"])
        return page_hits

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
        bbox_pages: List[dict] = []
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
                if bbox:
                    bbox_pages = self._semantic_unit_bboxes_on_pages(document, ordered_pages, metadata, content)
                    if bbox_pages:
                        preferred_hit = next((item for item in bbox_pages if item["page_number"] == resolved_page), bbox_pages[0])
                        resolved_page = int(preferred_hit["page_number"])
                        bbox = preferred_hit["bbox"]
                        highlight_text = clean_block_text(
                            "\n".join(item.get("highlight_text", "") for item in bbox_pages if item.get("highlight_text"))
                        ) or preferred_hit.get("highlight_text", "") or highlight_text
                page_count = document.page_count
        except Exception:
            page_count = int(page_cache.get("page_count", metadata.get("page_count", 1) or 1))

        return {
            "page_number": resolved_page,
            "page_count": max(page_count, 1),
            "bbox": bbox,
            "bbox_pages": bbox_pages,
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
        precision_hints = profile.get("precision_hints") or {}

        if precision_hints.get("target_doc_name"):
            should.append({"term": {"doc_name": {"value": precision_hints["target_doc_name"], "boost": 2.2}}})
        for preferred_article in precision_hints.get("preferred_articles", []):
            should.append({"term": {"article_id": {"value": preferred_article, "boost": 12.0}}})
            should.append({"match_phrase": {"article_label": {"query": f"第{preferred_article}条", "boost": 9.5}}})
        for related_article in precision_hints.get("related_articles", []):
            should.append({"term": {"article_id": {"value": related_article, "boost": 3.2}}})
        for preferred_section in precision_hints.get("preferred_sections", []):
            should.append({"term": {"section_id": {"value": preferred_section, "boost": 8.0}}})
            should.append({"match_phrase": {"section_title": {"query": preferred_section, "boost": 3.5}}})
        for required_term in precision_hints.get("required_terms", []):
            should.append({"match_phrase": {"retrieval_text": {"query": required_term, "boost": 3.4}}})
            should.append({"match_phrase": {"content": {"query": required_term, "boost": 2.8}}})

        for article_id_variant in article_id_variants or ([article_id] if article_id else []):
            should.append({"term": {"article_id": {"value": article_id_variant, "boost": 8.0}}})
        for article_label_variant in article_label_variants or ([article_label] if article_label else []):
            should.append({"term": {"article_label": {"value": article_label_variant, "boost": 8.8}}})
            should.append({"match_phrase": {"section_title": {"query": article_label_variant, "boost": 6.2}}})
            should.append({"match_phrase": {"heading_text": {"query": article_label_variant, "boost": 6.8}}})
            should.append({"match_phrase": {"scope_anchor": {"query": article_label_variant, "boost": 7.2}}})
            should.append({"match_phrase": {"heading_path_text": {"query": article_label_variant, "boost": 6.4}}})

        if normalized_heading:
            should.append({"term": {"normalized_heading": {"value": normalized_heading, "boost": 7.2}}})

        for alias in _build_query_aliases(query):
            should.extend(
                [
                    {"match_phrase": {"scope_anchor": {"query": alias, "boost": 5.4}}},
                    {"match_phrase": {"heading_path_text": {"query": alias, "boost": 5.0}}},
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
                                "retrieval_text^1.4",
                                "scope_anchor^3.8",
                                "heading_path_text^2.8",
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
                    {"match_phrase": {"scope_anchor": {"query": alias, "boost": 5.0 * decay}}},
                    {"match_phrase": {"heading_path_text": {"query": alias, "boost": 4.8 * decay}}},
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
                    {"match_phrase": {"scope_anchor": {"query": alias, "boost": 6.4 * decay}}},
                    {"match_phrase": {"heading_path_text": {"query": alias, "boost": 5.6 * decay}}},
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

    def _structure_search(self, query: str, profile: dict, top_k: int = HEADING_RECALL) -> List[dict]:
        aliases = _build_query_aliases(query)
        query_signature = profile.get("query_signature", {})
        precision_hints = profile.get("precision_hints") or {}
        should = []

        target_doc_name = profile.get("target_doc_name", "")
        if target_doc_name:
            should.append({"term": {"doc_name": {"value": target_doc_name, "boost": 2.8}}})
        for preferred_article in precision_hints.get("preferred_articles", []):
            should.append({"term": {"article_id": {"value": preferred_article, "boost": 13.0}}})
            should.append({"match_phrase": {"article_label": {"query": f"第{preferred_article}条", "boost": 10.0}}})
            should.append({"match_phrase": {"scope_anchor": {"query": f"第{preferred_article}条", "boost": 8.5}}})
        for related_article in precision_hints.get("related_articles", []):
            should.append({"term": {"article_id": {"value": related_article, "boost": 3.4}}})
        for preferred_section in precision_hints.get("preferred_sections", []):
            should.append({"term": {"section_id": {"value": preferred_section, "boost": 8.8}}})
            should.append({"match_phrase": {"section_title": {"query": preferred_section, "boost": 4.0}}})
        for required_term in precision_hints.get("required_terms", []):
            should.append({"match_phrase": {"scope_anchor": {"query": required_term, "boost": 5.8}}})
            should.append({"match_phrase": {"heading_path_text": {"query": required_term, "boost": 4.8}}})
            should.append({"match_phrase": {"retrieval_text": {"query": required_term, "boost": 3.8}}})

        article_id = query_signature.get("article_id", "")
        article_label = query_signature.get("article_label", "")
        for article_id_variant in query_signature.get("article_id_variants", []) or ([article_id] if article_id else []):
            should.append({"term": {"article_id": {"value": article_id_variant, "boost": 8.5}}})
        for article_label_variant in query_signature.get("article_label_variants", []) or ([article_label] if article_label else []):
            should.extend(
                [
                    {"term": {"article_label": {"value": article_label_variant, "boost": 9.0}}},
                    {"match_phrase": {"scope_anchor": {"query": article_label_variant, "boost": 7.8}}},
                    {"match_phrase": {"heading_path_text": {"query": article_label_variant, "boost": 6.8}}},
                    {"match_phrase": {"section_title": {"query": article_label_variant, "boost": 6.4}}},
                ]
            )

        normalized_heading = query_signature.get("normalized_heading", "")
        if normalized_heading:
            should.append({"term": {"normalized_heading": {"value": normalized_heading, "boost": 8.2}}})

        for role in profile.get("target_roles", []):
            if role and role != "text":
                should.append({"term": {"chunk_role": {"value": role, "boost": 2.3}}})

        for index, alias in enumerate(aliases):
            decay = max(0.58, 1.0 - 0.12 * index)
            should.extend(
                [
                    {"match_phrase": {"scope_anchor": {"query": alias, "boost": 7.0 * decay}}},
                    {"match_phrase": {"heading_path_text": {"query": alias, "boost": 6.2 * decay}}},
                    {"match_phrase": {"section_title": {"query": alias, "boost": 5.8 * decay}}},
                    {"match_phrase": {"heading_text": {"query": alias, "boost": 5.6 * decay}}},
                    {"match_phrase": {"clause_path": {"query": alias, "boost": 4.6 * decay}}},
                    {
                        "multi_match": {
                            "query": alias,
                            "fields": [
                                "scope_anchor^4.4",
                                "heading_path_text^3.8",
                                "section_title^3.5",
                                "heading_text^3.5",
                                "article_label^4.0",
                                "clause_path^2.8",
                                "highlight_text^1.4",
                                "content^0.9",
                            ],
                            "type": "best_fields",
                            "boost": 1.0 * decay,
                        }
                    },
                ]
            )

        if not should:
            return []

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
                    "channel": "structure",
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

    def _expand_following_context(self, current: dict, max_chunks: int, max_chars: int, include_headings: bool = False) -> List[str]:
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
        heading_level = int(current.get("heading_level", 0))
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
                row_level = int(row.get("heading_level", 0))
                # Break if we hit a same-level or higher-level heading (new section)
                # But include sub-headings (lower level) if include_headings is True
                if not include_headings or (heading_level > 0 and row_level <= heading_level):
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
        heading_path_text = metadata.get("heading_path_text", "")
        scope_anchor = metadata.get("scope_anchor", "")
        article_label = metadata.get("article_label", "")
        content = metadata.get("retrieval_text") or metadata.get("content", fallback_content)
        return "\n".join(part for part in [scope_anchor, heading_path_text, section, article_label, clause_path, parent, content] if part).strip()

    def _structural_record_to_evidence(self, record: dict, score: float) -> dict:
        raw_heading_path = record.get("heading_path", []) if isinstance(record.get("heading_path"), list) else []
        heading_path: List[str] = []
        for candidate in raw_heading_path:
            normalized = clean_block_text(candidate)
            if normalized and normalized not in heading_path:
                heading_path.append(normalized)
        chapter_part = clean_block_text(" ".join(part for part in [record.get("chapter_label", ""), record.get("chapter_title", "")] if part))
        section_part = clean_block_text(" ".join(part for part in [record.get("section_label", ""), record.get("section_title", "")] if part))
        label_part = clean_block_text(" ".join(part for part in [record.get("label", ""), record.get("title", "")] if part))
        for candidate in (chapter_part, section_part, label_part):
            if candidate and candidate not in heading_path:
                heading_path.append(candidate)
        heading = clean_block_text(
            " ".join(
                part
                for part in [
                    record.get("label", ""),
                    record.get("title", ""),
                ]
                if part
            )
        )
        section_title = clean_block_text(" > ".join(heading_path)) or heading or clean_block_text(record.get("section_title", ""))
        page_number = int(record.get("page_start", 1) or 1)
        page_end = int(record.get("page_end", page_number) or page_number)
        page_count = int(record.get("page_count", page_end) or page_end)
        content = clean_block_text(record.get("body_text", ""))[:4200]
        highlight_text = _structural_highlight_text(record, heading, content)
        return {
            "content": content,
            "doc_name": record.get("doc_name", ""),
            "heading": heading or clean_block_text(record.get("section_title", "")),
            "parent_heading": clean_block_text(record.get("section_title", "")),
            "score": round(float(score), 6),
            "chunk_id": record.get("record_id", ""),
            "page_number": page_number,
            "page_count": max(page_count, page_end),
            "section_title": section_title,
            "highlight_text": highlight_text,
            "bbox": None,
            "source_type": f"structural_{record.get('record_type', 'record')}",
            "layout_type": "structural",
            "parse_confidence": 0.98,
            "document_type": infer_document_type(record.get("doc_name", "")),
            "is_ocr_fallback": False,
            "article_id": record.get("article_id", ""),
            "article_label": record.get("article_label", ""),
            "clause_path": clean_block_text(" > ".join(heading_path)),
            "chunk_role": record.get("semantic_role") or record.get("record_type", "text"),
            "normalized_heading": _normalize_heading_value(section_title or heading),
            "structural_node_id": record.get("record_id", ""),
            "structural_parent_id": record.get("parent_record_id", ""),
            "structural_record_type": record.get("record_type", ""),
            "structural_level": int(record.get("level", 0) or 0),
            "structural_number": record.get("number", ""),
            "structural_path": clean_block_text(" > ".join(heading_path)),
            "direct_child_count": int(record.get("direct_child_count", 0) or 0),
            "descendant_count": int(record.get("descendant_count", 0) or 0),
        }

    def repair_evidence_location(self, evidence: dict) -> dict:
        """Resolve PDF page/highlight/bbox for evidence already selected for display."""
        if not evidence:
            return evidence
        doc_name = str(evidence.get("doc_name") or "")
        if not doc_name.lower().endswith(".pdf"):
            return evidence
        content = clean_block_text(evidence.get("content", ""))
        repaired = self._repair_pdf_evidence_metadata(evidence, content)
        if not repaired:
            return evidence
        if not repaired.get("bbox"):
            repaired = {key: value for key, value in repaired.items() if key != "highlight_text"}
        return {**evidence, **repaired}

    def _search_structural_records(self, query: str, profile: dict, top_k: int = 12) -> List[dict]:
        target_doc_name = profile.get("target_doc_name", "")
        target_doc_is_explicit = bool(profile.get("target_doc_is_explicit"))
        query_signature = profile.get("query_signature", {})
        records = load_all_structural_records()
        if not records:
            return []

        ranked: List[Tuple[float, dict]] = []
        for record in records:
            record_type = str(record.get("record_type", ""))
            semantic_role = str(record.get("semantic_role", ""))
            if record.get("is_toc") or record_type == "doc":
                continue
            if target_doc_name and target_doc_is_explicit and record.get("doc_name") != target_doc_name:
                continue
            joined_text = clean_block_text(
                "\n".join(
                    [
                        " > ".join(record.get("heading_path", []) or []),
                        str(record.get("label", "") or ""),
                        str(record.get("title", "") or ""),
                        str(record.get("article_label", "") or ""),
                        str(record.get("body_text", "") or "")[:1200],
                    ]
                )
            )
            score = _token_overlap_score(query, joined_text) + 0.9 * _ngram_similarity(query, joined_text)
            precision_hints = profile.get("precision_hints") or {}
            route = str(precision_hints.get("route") or "")
            subject_terms = [_compact_match_text(term) for term in precision_hints.get("subject_terms", []) if term]
            intent_terms = [_compact_match_text(term) for term in precision_hints.get("intent_terms", []) if term]
            title_keyphrases = [_compact_match_text(term) for term in precision_hints.get("title_keyphrases", []) if term]
            title_compact = _compact_match_text(
                " ".join(
                    str(record.get(key) or "")
                    for key in ("label", "title", "section_title", "article_label")
                )
            )
            title_subject_hits = sum(1 for term in subject_terms if term and _has_unmodified_subject(title_compact, term))
            title_intent_hits = sum(1 for term in intent_terms if term and term in title_compact)
            title_keyphrase_hits = sum(1 for term in title_keyphrases if term and term in title_compact)
            joined_keyphrase_hits = sum(1 for term in title_keyphrases if term and term in _compact_match_text(joined_text[:700]))
            if "附则" in title_compact and not any(token in _compact_match_text(query) for token in ("附则", "施行", "生效", "解释")):
                score -= 3.2
            if title_keyphrase_hits:
                score += min(7.2, 2.8 * title_keyphrase_hits)
                if record_type in {"article", "section"}:
                    score += 1.1
            elif joined_keyphrase_hits:
                score += min(1.4, 0.28 * joined_keyphrase_hits)
            if title_subject_hits:
                score += 2.2 + 0.45 * title_subject_hits
            if title_intent_hits:
                score += 1.05 + 0.25 * title_intent_hits
            if title_subject_hits and title_intent_hits:
                score += 2.2
                if record_type in {"article", "section"}:
                    score += 0.75
            if route in {"document_scope", "scope", "issuance_requirement", "application_requirement", "procedure", "method", "time_limit", "numeric_threshold"}:
                if record_type == "article":
                    score += 1.15
                elif record_type == "clause":
                    score += 0.65
                elif record_type == "item":
                    score -= 0.65
                    if subject_terms and not title_subject_hits:
                        score -= 0.85
                    if title_keyphrases and not title_keyphrase_hits:
                        score -= 0.85
                if route in {"numeric_threshold", "time_limit"} and semantic_role == "scope" and any(term in _compact_match_text(joined_text[:500]) for term in intent_terms):
                    score += 2.0
                if route == "method" and any(term in _compact_match_text(joined_text[:420]) for term in ("确定", "方法", "称重", "应当", "必须")):
                    score += 1.3
                if route == "method" and "制定" in _compact_match_text(query) and "制定" in _compact_match_text(joined_text[:700]):
                    score += 2.1
                if route in {"method", "time_limit"} and title_keyphrases:
                    if any(term and term in _compact_match_text(joined_text[:900]) for term in title_keyphrases):
                        score += 1.4
                        if route == "time_limit" and any(token in joined_text for token in ("小时", "日内", "工作日", "时限")):
                            score += 1.7
                if route == "time_limit":
                    compact_joined_head = _compact_match_text(joined_text[:1000])
                    compact_query = _compact_match_text(query)
                    acronym_terms = [term for term in re.findall(r"[a-zA-Z]{2,}\d*(?:\.\d+)?", compact_query) if len(term) >= 3]
                    if acronym_terms and any(term in compact_joined_head for term in acronym_terms) and any(token in joined_text for token in ("小时", "日内", "工作日", "时限")):
                        score += 3.2
            compact_query = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9.]+", "", query)
            compact_joined = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9.]+", "", joined_text)
            exact_anchor_hits = 0
            if len(compact_query) >= 6:
                for size in (6, 5, 4):
                    anchors = {compact_query[index : index + size] for index in range(0, max(len(compact_query) - size + 1, 0))}
                    exact_anchor_hits += sum(1 for anchor in anchors if anchor and anchor in compact_joined)
                score += min(0.75, 0.08 * exact_anchor_hits)
                if exact_anchor_hits and record_type == "item":
                    score += 0.55
                elif exact_anchor_hits and record_type == "clause":
                    score += 0.25
            if record_type in {"article", "clause", "item", "section"}:
                core_terms = [
                    term
                    for term in re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9.]{2,}", compact_query)
                    if term not in {"运输类旋翼航空器适航规定", "运输机场运行安全管理规定", "民用航空产品和零件合格审定规定"}
                ]
                matched_terms = sum(1 for term in core_terms if term in compact_joined or term.replace("要求", "的要求") in compact_joined)
                if core_terms:
                    score += 0.55 * (matched_terms / len(core_terms))
                if any(token in compact_query for token in ("包括哪些", "包括什么", "哪些部件", "哪几类", "有哪些")) and any(
                    token in compact_joined for token in ("包括下列", "下列部件", "包括以下", "分为下列", "分成")
                ):
                    score += 1.05
                if "要求" in compact_query and "的要求" in compact_joined:
                    score += 0.35
                requirement_subjects = re.findall(r"([\u4e00-\u9fffA-Za-z0-9]{2,12})要求", compact_query)
                for subject in requirement_subjects:
                    subject_variants = [subject]
                    if "中的" in subject:
                        subject_variants.append(subject.rsplit("中的", 1)[-1])
                    if "的" in subject:
                        subject_variants.append(subject.rsplit("的", 1)[-1])
                    if any(f"{variant}的要求" in compact_joined for variant in subject_variants if variant):
                        score += 1.45
                    elif any(variant and variant in compact_joined for variant in subject_variants) and "要求" in compact_joined and any(token in compact_query for token in ("由什么", "什么来", "如何规定", "怎么规定")):
                        score += 0.7
                if "规定" in compact_query and "规定" in compact_joined:
                    score += 0.16
            if target_doc_name:
                score += 0.8 if record.get("doc_name") == target_doc_name else -0.6
            if profile.get("query_type") == "article_lookup":
                if _article_reference_matches(record.get("article_id", ""), record.get("article_label", ""), query_signature):
                    score += 4.2
                    broad_article_question = any(token in query for token in ("规定了什么", "内容是什么", "有哪些内容", "是什么"))
                    if broad_article_question:
                        if record_type == "article":
                            score += 2.6
                        elif record_type in {"clause", "item"}:
                            score -= 1.1
                elif query_signature.get("article_id_variants"):
                    score -= 1.4
                if record_type in {"article", "clause", "item"}:
                    score += 0.65
            if "effective_date" in profile.get("target_roles", []) and semantic_role == "effective_date":
                score += 3.0
            elif semantic_role == "effective_date":
                score -= 0.8
            if "适用范围" in query and semantic_role == "scope":
                score += 3.0
            if profile.get("classification_keywords") and semantic_role == "classification":
                score += 1.5
            if "定义" in query and semantic_role == "definition":
                score += 1.2

            score += _precision_hint_score(record, profile, joined_text, structural=True)

            if score < 0.75:
                continue
            ranked.append((round(score, 6), record))

        ranked.sort(key=lambda item: item[0], reverse=True)
        seen = set()
        results: List[dict] = []
        for score, record in ranked:
            key = (
                record.get("doc_name", ""),
                record.get("record_type", ""),
                record.get("article_id", ""),
                record.get("semantic_role", ""),
            )
            if key in seen:
                continue
            seen.add(key)
            results.append(self._structural_record_to_evidence(record, score))
            if len(results) >= top_k:
                break
        return results

    def _infer_target_doc_from_structure(self, query: str, profile: dict) -> str:
        if profile.get("target_doc_name"):
            return str(profile["target_doc_name"])
        hints = profile.get("precision_hints") or {}
        subject_terms = [_compact_match_text(term) for term in hints.get("subject_terms", []) if term]
        title_keyphrases = [_compact_match_text(term) for term in hints.get("title_keyphrases", []) if term]
        if not subject_terms and not title_keyphrases:
            return ""
        intent_terms = [_compact_match_text(term) for term in hints.get("intent_terms", []) if term]
        route = str(hints.get("route") or "")
        scores: Dict[str, float] = defaultdict(float)
        records = load_all_structural_records()
        for record in records:
            if record.get("is_toc") or str(record.get("record_type") or "") == "doc":
                continue
            doc_name = str(record.get("doc_name") or "")
            if not doc_name:
                continue
            title_text = _compact_match_text(
                " ".join(
                    str(record.get(key) or "")
                    for key in ("label", "title", "section_title", "article_label")
                )
            )
            path_text = _compact_match_text(" > ".join(record.get("heading_path", []) or []))
            body_text = _compact_match_text(str(record.get("body_text") or "")[:900])
            combined = title_text + path_text + body_text
            subject_hits = sum(1 for term in subject_terms if term and term in combined)
            keyphrase_title_hits = sum(1 for term in title_keyphrases if term and term in (title_text + path_text))
            keyphrase_body_hits = sum(1 for term in title_keyphrases if term and term in body_text[:500])
            if subject_hits == 0 and keyphrase_title_hits == 0 and keyphrase_body_hits == 0:
                continue
            score = 0.7 * subject_hits
            if keyphrase_title_hits:
                score += 2.7 * keyphrase_title_hits
            if keyphrase_body_hits:
                score += 0.55 * keyphrase_body_hits
            if any(term and term in title_text for term in subject_terms):
                score += 3.0
            if any(term and term in path_text for term in subject_terms):
                score += 1.1
            if intent_terms:
                intent_hits = sum(1 for term in intent_terms if term and term in combined)
                score += min(2.0, 0.45 * intent_hits)
            semantic_role = str(record.get("semantic_role") or "")
            record_type = str(record.get("record_type") or "")
            if route == "definition" and semantic_role == "definition":
                score += 1.8
            elif route == "classification" and semantic_role == "classification":
                score += 1.6
            elif route in {"issuance_requirement", "application_requirement"} and semantic_role == "requirement":
                score += 1.2
            elif route == "document_scope" and semantic_role == "scope":
                score += 1.2
            if record_type in {"article", "section"}:
                score += 0.45
            try:
                score += max(0.0, 0.5 - 0.01 * max(int(record.get("page_start") or 1) - 1, 0))
            except Exception:
                pass
            scores[doc_name] += score
        if not scores:
            return ""
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if len(ranked) == 1:
            return ranked[0][0] if ranked[0][1] >= 3.0 else ""
        if ranked[0][1] >= 3.5 and ranked[0][1] >= ranked[1][1] * 1.18:
            return ranked[0][0]
        return ""

    def _score_candidate(self, query: str, hit: dict, metadata: dict, profile: dict) -> float:
        max_weight = max(
            float(profile.get("kw_weight", 1.0)),
            float(profile.get("vec_weight", 1.0)),
            float(profile.get("structure_weight", 1.0)),
            float(profile.get("article_weight", 1.0)),
            float(profile.get("heading_weight", 1.0)),
            float(profile.get("table_weight", 1.0)),
        )
        k = float(profile.get("rrf_k", 60))
        rrf_component = hit.get("rrf_score", 0.0) * (k + 1.0) / max(max_weight, 0.01)
        bounded_rrf = min(rrf_component, 1.0)
        rerank_component = _sigmoid(float(hit.get("rerank_score", 0.0))) if "rerank_score" in hit else min(0.55 * bounded_rrf, 0.65)
        score = 0.55 * rerank_component + 0.45 * bounded_rrf

        channels = set(hit.get("channels", []))
        query_signature = profile.get("query_signature", {})
        target_doc_name = profile.get("target_doc_name", "")
        scope_anchor = clean_block_text(metadata.get("scope_anchor", ""))
        heading_path_text = clean_block_text(metadata.get("heading_path_text", ""))
        structure_alignment = max(
            _token_overlap_score(query, scope_anchor),
            _token_overlap_score(query, heading_path_text),
            0.75 * _ngram_similarity(query, scope_anchor or heading_path_text),
        )
        if target_doc_name:
            if metadata.get("doc_name") == target_doc_name:
                score += 0.55
            else:
                score -= 0.48
        if "structure" in channels:
            score += 0.18 if profile.get("structure_first") else 0.08
        if profile.get("structure_first"):
            score += 0.22 * min(structure_alignment, 1.15)
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
            if query_signature.get("normalized_heading") and query_signature.get("normalized_heading") in _normalize_heading_value(scope_anchor):
                score += 0.16
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

        score += _precision_hint_score(metadata, profile, hit.get("content", ""), structural=False)

        if metadata.get("source_type") == "table" and any(token in query for token in ("表", "参数", "限制", "数值")):
            score += 0.08
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
            profile["target_doc_is_explicit"] = True
            profile.setdefault("precision_hints", {})["target_doc_name"] = target_doc_name
        else:
            inferred_doc_name = self._infer_target_doc_from_structure(query, profile)
            if inferred_doc_name:
                profile["target_doc_name"] = inferred_doc_name
                profile["target_doc_is_explicit"] = False
                profile.setdefault("precision_hints", {})["target_doc_name"] = inferred_doc_name
        query_vec = self.embedder.encode([query], normalize_embeddings=True)[0].tolist()

        # Multi-Query: use query aliases for parallel vector search
        aliases = _build_query_aliases(query)
        unique_aliases = [a for a in aliases if a != query][:4]
        alias_vecs: List[List[float]] = []
        if unique_aliases and float(profile.get("vec_weight", 1.0)) >= 1.0:
            alias_vecs = self.embedder.encode(unique_aliases, normalize_embeddings=True).tolist()

        # HyDE: generate hypothetical answer for short article_lookup queries
        hyde_vec: Optional[List[float]] = None
        hyde_answer: str = ""
        if (
            self.hyde_llm_call
            and profile["query_type"] == "article_lookup"
            and len(query) < 25
        ):
            hyde_answer = _generate_hypothetical_answer(query, self.hyde_llm_call)
            if hyde_answer:
                hyde_vec = self.embedder.encode([hyde_answer], normalize_embeddings=True)[0].tolist()

        def _run_vec_search(vec: List[float], label: str):
            results = self._vector_search(vec, VEC_RECALL)
            for hit in results:
                hit["channels"] = list(hit.get("channels", [])) + [label]
            return results

        with concurrent.futures.ThreadPoolExecutor(max_workers=7) as executor:
            vec_future = executor.submit(_run_vec_search, query_vec, "vector")
            structure_future = executor.submit(self._structure_search, query, profile, HEADING_RECALL)
            article_future = executor.submit(self._article_search, query, profile, ARTICLE_RECALL)
            kw_future = executor.submit(self._keyword_search, query, profile, KW_RECALL)
            heading_future = executor.submit(self._heading_search, query, profile, HEADING_RECALL)
            table_future = executor.submit(self._table_search, query, TABLE_RECALL)
            alias_futures = {
                label: executor.submit(_run_vec_search, vec, f"alias_{idx}")
                for idx, (label, vec) in enumerate(zip(unique_aliases, alias_vecs))
            }
            recall_groups = [
                ("vector", vec_future.result(), profile["vec_weight"]),
                ("structure", structure_future.result(), profile["structure_weight"]),
                ("article", article_future.result(), profile["article_weight"]),
                ("keyword", kw_future.result(), profile["kw_weight"]),
                ("heading", heading_future.result(), profile["heading_weight"]),
                ("table", table_future.result(), profile["table_weight"]),
            ]
            for label, future in alias_futures.items():
                recall_groups.append((label, future.result(), profile["vec_weight"] * 0.5))
            # HyDE search: run hypothetical answer embedding alongside other searches
            if hyde_vec:
                hyde_future = executor.submit(_run_vec_search, hyde_vec, "hyde")
                hyde_results = hyde_future.result()
                for hit in hyde_results:
                    hit["channels"] = list(hit.get("channels", [])) + ["hyde"]
                recall_groups.append(("hyde", hyde_results, profile["vec_weight"] * 0.7))
                # Store hyde answer in profile for downstream use
                profile["hyde_answer"] = hyde_answer

        fused = self._rrf_fusion(recall_groups, profile)
        if not fused:
            return [], profile

        candidate_ids = [hit["chunk_id"] for hit in fused[:60]]
        metadata_map = self._get_metadata_batch(candidate_ids)
        for hit in fused[:60]:
            hit["metadata"] = metadata_map.get(hit["chunk_id"], {})
        if profile.get("target_doc_name") and profile.get("target_doc_is_explicit"):
            target_doc_hits = [
                hit
                for hit in fused[:60]
                if (hit.get("metadata") or {}).get("doc_name") == profile["target_doc_name"]
            ]
            if target_doc_hits:
                fused = target_doc_hits

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
            # Try expanding from the parent heading chunk if this is a body chunk under an article heading
            parent_id = metadata.get("parent_id", "")
            if parent_id and metadata.get("article_id"):
                parent_meta = self._get_metadata(parent_id)
                if parent_meta and parent_meta.get("is_heading"):
                    heading_children = self._expand_heading_children(
                        {**parent_meta, "chunk_id": parent_id},
                        max(int(profile.get("expand_children", 4)) * 2, 8),
                        max(int(profile.get("child_char_budget", 900)), 2800),
                    )
                    if heading_children:
                        content = "\n".join([f"[SECTION]\n{parent_meta.get('content', '')}"] + heading_children)
            else:
                following = self._expand_following_context(
                    {**metadata, "chunk_id": hit["chunk_id"]},
                    max(int(profile.get("expand_children", 4)) * 3, 6),
                    max(int(profile.get("child_char_budget", 900)), 2200 if profile["query_type"] == "article_lookup" else 900),
                    include_headings=profile["query_type"] == "article_lookup",
                )
                if following:
                    content = "\n".join([content] + following)

        if profile["query_type"] == "article_lookup":
            page_window = self._pdf_page_text_window(metadata, include_next_page=True, max_chars=3600)
            if page_window and _token_overlap_score(metadata.get("section_title", ""), page_window) >= 0.2:
                content = f"{content}\n\n[PDF页原文]\n{page_window}"

        content = clean_block_text(content)
        if profile["query_type"] == "article_lookup":
            max_content_chars = 4200
        elif metadata.get("section_title") and len(content) > 1200:
            # For non-article queries with strong section matches, allow more content
            max_content_chars = 2800
        else:
            max_content_chars = 1200
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
            if not _is_valid_search_query(query):
                return []

            direct_evidence = self._document_applicability_evidence(query)
            if direct_evidence:
                return [direct_evidence]

            structural_profile = analyze_query_intent(query)
            target_doc_name = self._find_doc_name_by_query_reference(query)
            if target_doc_name:
                structural_profile["target_doc_name"] = target_doc_name
                structural_profile["target_doc_is_explicit"] = True
                structural_profile.setdefault("precision_hints", {})["target_doc_name"] = target_doc_name
            else:
                inferred_doc_name = self._infer_target_doc_from_structure(query, structural_profile)
                if inferred_doc_name:
                    structural_profile["target_doc_name"] = inferred_doc_name
                    structural_profile["target_doc_is_explicit"] = False
                    structural_profile.setdefault("precision_hints", {})["target_doc_name"] = inferred_doc_name
            structural_records = load_all_structural_records()
            precision_hints = structural_profile.get("precision_hints") or {}
            if not _has_structural_subject_support(precision_hints, structural_records):
                return []
            structural_count_answer = answer_structural_count(query, structural_records)
            if structural_count_answer and isinstance(structural_count_answer.get("sources"), list):
                return structural_count_answer["sources"][:QA_TOP_K]
            structural_classification_answer = answer_structural_classification(query, structural_records)
            if structural_classification_answer and isinstance(structural_classification_answer.get("sources"), list):
                return structural_classification_answer["sources"][:QA_TOP_K]
            structural_definition_answer = answer_structural_definition(query, structural_records)
            if structural_definition_answer and isinstance(structural_definition_answer.get("sources"), list):
                return structural_definition_answer["sources"][:QA_TOP_K]
            if (structural_profile.get("precision_hints") or {}).get("route") == "definition":
                return []

            query_signature = structural_profile.get("query_signature", {})
            if query_signature.get("article_id_variants") and structural_profile.get("target_doc_name"):
                exact_records = [
                    record
                    for record in structural_records
                    if not record.get("is_toc")
                    and str(record.get("doc_name") or "") == structural_profile["target_doc_name"]
                    and _article_reference_matches(record.get("article_id", ""), record.get("article_label", ""), query_signature)
                ]
                if not exact_records:
                    return []

            preferred_articles = {str(value) for value in precision_hints.get("preferred_articles", []) if value}
            exact_preferred_hits: List[dict] = []
            if preferred_articles and precision_hints.get("target_doc_name"):
                exact_preferred_records = [
                    record
                    for record in structural_records
                    if not record.get("is_toc")
                    and str(record.get("doc_name") or "") == precision_hints["target_doc_name"]
                    and str(record.get("article_id") or record.get("number") or "") in preferred_articles
                    and str(record.get("record_type") or "") == "article"
                ]
                exact_preferred_records.sort(key=lambda record: (int(record.get("page_start") or 1), int(record.get("order") or 0)))
                exact_preferred_hits = [self._structural_record_to_evidence(record, 20.0) for record in exact_preferred_records[: len(preferred_articles)]]

            structural_hits = self._search_structural_records(query, structural_profile, top_k=max(QA_TOP_K, 6))
            if structural_hits:
                structural_hits.sort(
                    key=lambda row: (
                        0
                        if (
                            str(row.get("structural_record_type") or "") in {"clause", "item"}
                            and str((precision_hints or {}).get("route") or "") in {"method", "classification", "application_requirement", "issuance_requirement"}
                        )
                        else 1,
                        -float(row.get("score") or 0.0),
                    )
                )
            if exact_preferred_hits:
                structural_hits = exact_preferred_hits + [
                    row
                    for row in structural_hits
                    if str(row.get("article_id") or "") not in preferred_articles
                ]
            if precision_hints.get("route") == "document_scope" and preferred_articles:
                scoped_hits = [row for row in structural_hits if str(row.get("article_id") or "") in preferred_articles]
                if scoped_hits:
                    structural_hits = scoped_hits + [
                        row for row in structural_hits if str(row.get("article_id") or "") not in preferred_articles
                    ]
            focus_routes = {"issuance_requirement", "application_requirement", "procedure", "method"}
            structural_focus_article_id = ""
            exact_article_variants = {str(value) for value in query_signature.get("article_id_variants", []) if value}
            if structural_hits and exact_article_variants:
                focused_hits = [row for row in structural_hits if str(row.get("article_id") or "") in exact_article_variants]
                if focused_hits:
                    structural_focus_article_id = str(focused_hits[0].get("article_id") or "")
                    structural_hits = focused_hits + [
                        row for row in structural_hits if str(row.get("article_id") or "") not in exact_article_variants
                    ]
            elif structural_hits and precision_hints.get("route") in focus_routes:
                top_article_id = str(structural_hits[0].get("article_id") or "")
                top_score = float(structural_hits[0].get("score") or 0.0)
                next_score = float(structural_hits[1].get("score") or 0.0) if len(structural_hits) > 1 else 0.0
                if top_article_id and (next_score <= 0.0 or top_score >= next_score * 1.25):
                    focused_hits = [row for row in structural_hits if str(row.get("article_id") or "") == top_article_id]
                    if focused_hits:
                        structural_focus_article_id = top_article_id
                        structural_hits = focused_hits + [
                            row for row in structural_hits if str(row.get("article_id") or "") != top_article_id
                        ]

            fused, profile = self._search_candidates(query)
            if not fused:
                return structural_hits[:QA_TOP_K] if structural_hits else []

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
            merged: List[dict] = []
            structure_route = str((profile.get("precision_hints") or {}).get("route") or "")
            structure_priority_routes = {
                "document_scope",
                "definition",
                "classification",
                "issuance_requirement",
                "application_requirement",
                "procedure",
                "time_limit",
                "numeric_threshold",
            }
            prefer_structural_hits = bool(
                profile.get("structure_first")
                or profile.get("query_type") in {"general", "concept_explanation"}
                or structure_route in structure_priority_routes
            )
            if structural_hits and prefer_structural_hits:
                for row in structural_hits:
                    if structural_focus_article_id and str(row.get("article_id") or "") != structural_focus_article_id:
                        continue
                    if len(merged) >= min(3 if structure_route in structure_priority_routes else 2, QA_TOP_K):
                        break
                    merged.append(row)

            def evidence_keys(row: dict) -> set:
                doc_name = row.get("doc_name", "")
                page_number = row.get("page_number", 1)
                keys = set()
                for value in (
                    row.get("chunk_id", ""),
                    row.get("structural_node_id", ""),
                    row.get("article_id", ""),
                    row.get("section_title", ""),
                ):
                    if value:
                        keys.add((doc_name, page_number, value))
                return keys or {(doc_name, page_number, "")}

            seen = set()
            for row in merged:
                seen.update(evidence_keys(row))
            for _, row, _ in results:
                if structural_focus_article_id and str(row.get("article_id") or "") != structural_focus_article_id:
                    continue
                keys = evidence_keys(row)
                if seen & keys:
                    continue
                merged.append(row)
                seen.update(keys)
                if len(merged) >= QA_TOP_K:
                    break
            if structural_hits and len(merged) < QA_TOP_K and prefer_structural_hits:
                for row in structural_hits:
                    if structural_focus_article_id and str(row.get("article_id") or "") != structural_focus_article_id:
                        continue
                    keys = evidence_keys(row)
                    if seen & keys:
                        continue
                    merged.append(row)
                    seen.update(keys)
                    if len(merged) >= QA_TOP_K:
                        break
            return merged[:QA_TOP_K]
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
            engine = HybridSearch(hyde_llm_call=None)
            print(engine.search(query))
        elif choice == "3":
            print("再见。")
            break
        else:
            print("无效选项，请重新输入。")


if __name__ == "__main__":
    main()
