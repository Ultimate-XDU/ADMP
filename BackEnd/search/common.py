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

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default



def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default




ES_REQUEST_TIMEOUT_SECONDS = max(_env_float("AEROSPACE_ES_REQUEST_TIMEOUT_SECONDS", 8.0), 0.5)

ES_QUERY_TIMEOUT_SECONDS = max(_env_float("AEROSPACE_ES_QUERY_TIMEOUT_SECONDS", 6.0), 0.5)

ES_SLOW_QUERY_SECONDS = max(_env_float("AEROSPACE_ES_SLOW_QUERY_SECONDS", 2.5), 0.1)

PDF_PAGE_CACHE_MAX_ITEMS = max(_env_int("AEROSPACE_PDF_PAGE_CACHE_MAX_ITEMS", 8), 1)

METADATA_CACHE_MAX_ITEMS = max(_env_int("AEROSPACE_METADATA_CACHE_MAX_ITEMS", 4096), 128)

BASE_DIR = Path(__file__).resolve().parent

DOC_DIR = str(BASE_DIR.parent / "aerospace_docs")


def _consolidate_legacy_doc_dirs() -> None:
    canonical_dir = Path(DOC_DIR)
    legacy_dirs = (
        BASE_DIR / "aerospace_docs",
        BASE_DIR.parent / "api" / "aerospace_docs",
        BASE_DIR.parent / "utils" / "aerospace_docs",
    )
    moved: list[str] = []
    skipped: list[str] = []
    for legacy in legacy_dirs:
        try:
            if not legacy.is_dir():
                continue
            for item in legacy.iterdir():
                if not item.is_file():
                    continue
                dest = canonical_dir / item.name
                if dest.exists():
                    skipped.append(item.name)
                    continue
                os.replace(item, dest)
                moved.append(item.name)
            if not any(legacy.iterdir()):
                legacy.rmdir()
        except OSError as exc:
            logger.warning("合并旧文档目录 %s 失败：%s", legacy, exc)
    if moved or skipped:
        logger.warning(
            "已将旧目录中的文档合并到 %s：移动 %d 个，同名跳过 %d 个；旧目录为空时已删除。",
            canonical_dir,
            len(moved),
            len(skipped),
        )


try:
    Path(DOC_DIR).mkdir(parents=True, exist_ok=True)
    _consolidate_legacy_doc_dirs()
except OSError:
    pass

HEALTH_REPORT_FILE = BASE_DIR / "document_health_reports.json"

MILVUS_HOST = "localhost"

MILVUS_PORT = 19530

MILVUS_COLLECTION = "aerospace_m3e_db"

ES_HOST = "http://localhost:9200"

ES_INDEX = "aerospace_kw_db"

_MILVUS_CONNECTION_LOCK = threading.RLock()

_MILVUS_CONNECTION_REFCOUNT = 0

MAX_CHUNK_SIZE = 960

MAX_CHUNK_OVERLAP = 120

MILVUS_CONTENT_MAX = 3500

EMBEDDING_MODEL = os.environ.get("AEROSPACE_EMBEDDING_MODEL", "BAAI/bge-m3")

RERANK_MODEL = os.environ.get("AEROSPACE_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

RERANK_DEVICE = os.environ.get("AEROSPACE_RERANK_DEVICE", "cuda").strip().lower()

VECTOR_DIM = int(os.environ.get("AEROSPACE_VECTOR_DIM", "1024"))

TOP_K = 5

QA_TOP_K = 4

VEC_RECALL = 80

KW_RECALL = 80

HEADING_RECALL = 32

TABLE_RECALL = 16

ARTICLE_RECALL = 32

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

SPECIFICITY_MODIFIERS = ("出口", "外国", "特殊", "标准", "临时", "限制", "生产", "型号", "补充", "改装", "认可", "初级", "限用")

CODE_REFERENCE_PATTERN = re.compile(
    r"\b(?:CCAR|AC|FAA|EASA|ATA|AMM|FIM|TSM|IPC|SRM)\s*[- ]?\s*[A-Za-z0-9.]+(?:[-. ]?[A-Za-z0-9]+)*\b",
    re.I,
)

CCAR_PART_PATTERN = re.compile(r"^ccar(\d{2,4})(?:r\d+)?$", re.I)

CONTRACT_INTENT_QUERY_TYPE = {
    "article_lookup": "article_lookup",
    "procedure_or_method": "procedure",
    "definition_or_explanation": "concept_explanation",
    "comparison": "concept_explanation",
}

ARTICLE_PRECISION_PROFILE = PrecisionHintProfile(
    target_doc_boost=2.2,
    preferred_article_boost=12.0,
    preferred_article_label_boost=9.5,
    related_article_boost=3.2,
    preferred_section_boost=8.0,
    preferred_section_title_boost=3.5,
    required_term_fields=(WeightedField("retrieval_text", 3.4), WeightedField("content", 2.8)),
)

ARTICLE_REFERENCE_PROFILE = ArticleReferenceProfile(
    article_id_boost=8.0,
    article_label_term_boost=8.8,
    article_label_phrase_fields=(
        WeightedField("section_title", 6.2),
        WeightedField("heading_text", 6.8),
        WeightedField("scope_anchor", 7.2),
        WeightedField("heading_path_text", 6.4),
    ),
    normalized_heading_boost=7.2,
    role_boost=1.8,
)

ARTICLE_ALIAS_PROFILE = AliasClauseProfile(
    phrase_fields=(
        WeightedField("scope_anchor", 5.4),
        WeightedField("heading_path_text", 5.0),
        WeightedField("section_title", 4.5),
        WeightedField("heading_text", 4.9),
        WeightedField("clause_path", 3.8),
    ),
    decay_step=0.0,
    decay_floor=1.0,
)

CODE_LEXICAL_PROFILE = CodeVariantProfile(
    fields=(
        WeightedField("retrieval_text", 3.8),
        WeightedField("content", 3.2),
        WeightedField("doc_name", 2.8),
    )
)

KEYWORD_ALIAS_PROFILE = AliasClauseProfile(
    multi_match_fields=(
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
    ),
    multi_match_boost=1.0,
    phrase_fields=(
        WeightedField("content", 3.5),
        WeightedField("scope_anchor", 5.0),
        WeightedField("heading_path_text", 4.8),
        WeightedField("heading_text", 4.2),
        WeightedField("clause_path", 4.4),
    ),
    decay_step=0.14,
    decay_floor=0.58,
    exact_section_boost=4.8,
)

HEADING_ALIAS_PROFILE = AliasClauseProfile(
    phrase_fields=(
        WeightedField("scope_anchor", 6.4),
        WeightedField("heading_path_text", 5.6),
        WeightedField("heading_text", 6.0),
        WeightedField("section_title", 5.0),
        WeightedField("clause_path", 4.6),
    ),
    match_fields=(WeightedField("heading_text", 2.5),),
    decay_step=0.14,
    decay_floor=0.58,
)

STRUCTURE_PRECISION_PROFILE = PrecisionHintProfile(
    target_doc_boost=2.8,
    preferred_article_boost=13.0,
    preferred_article_label_boost=10.0,
    preferred_article_scope_boost=8.5,
    related_article_boost=3.4,
    preferred_section_boost=8.8,
    preferred_section_title_boost=4.0,
    required_term_fields=(
        WeightedField("scope_anchor", 5.8),
        WeightedField("heading_path_text", 4.8),
        WeightedField("retrieval_text", 3.8),
    ),
)

STRUCTURE_REFERENCE_PROFILE = ArticleReferenceProfile(
    article_id_boost=8.5,
    article_label_term_boost=9.0,
    article_label_phrase_fields=(
        WeightedField("scope_anchor", 7.8),
        WeightedField("heading_path_text", 6.8),
        WeightedField("section_title", 6.4),
    ),
    normalized_heading_boost=8.2,
    role_boost=2.3,
)

STRUCTURE_ALIAS_PROFILE = AliasClauseProfile(
    multi_match_fields=(
        "scope_anchor^4.4",
        "heading_path_text^3.8",
        "section_title^3.5",
        "heading_text^3.5",
        "article_label^4.0",
        "clause_path^2.8",
        "highlight_text^1.4",
        "content^0.9",
    ),
    multi_match_boost=1.0,
    phrase_fields=(
        WeightedField("scope_anchor", 7.0),
        WeightedField("heading_path_text", 6.2),
        WeightedField("section_title", 5.8),
        WeightedField("heading_text", 5.6),
        WeightedField("clause_path", 4.6),
    ),
    decay_step=0.12,
    decay_floor=0.58,
)

TABLE_PRECISION_PROFILE = PrecisionHintProfile(
    target_doc_boost=2.4,
    preferred_article_boost=6.0,
    preferred_section_boost=4.0,
    preferred_section_title_boost=3.6,
    required_term_fields=(WeightedField("content", 2.4), WeightedField("retrieval_text", 2.2)),
)

TABLE_ALIAS_PROFILE = AliasClauseProfile(
    phrase_fields=(
        WeightedField("section_title", 4.0),
        WeightedField("heading_path_text", 3.6),
        WeightedField("content", 2.6),
    ),
    match_fields=(WeightedField("content", 1.6),),
    decay_step=0.1,
    decay_floor=0.68,
    limit=8,
)

TARGET_DOC_ALIAS_PROFILE = AliasClauseProfile(
    phrase_fields=(
        WeightedField("scope_anchor", 5.4),
        WeightedField("heading_path_text", 4.6),
        WeightedField("section_title", 4.0),
        WeightedField("content", 2.0),
    ),
    match_fields=(WeightedField("retrieval_text", 1.8),),
    decay_step=0.08,
    decay_floor=0.55,
    limit=12,
)

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

HYDE_SYSTEM_PROMPT = "你是一个专业的民航法规问答助手。请根据问题，生成一段准确、完整的回答摘要。你的回答将用于信息检索，请直接给出答案摘要，不要有多余的说明。回答应在150字以内，使用中文。"

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

_embedder = None

_reranker = None

_embedder_lock = threading.Lock()

_reranker_lock = threading.Lock()

def _acquire_milvus_connection() -> None:
    global _MILVUS_CONNECTION_REFCOUNT
    with _MILVUS_CONNECTION_LOCK:
        if not connections.has_connection("default"):
            connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
        _MILVUS_CONNECTION_REFCOUNT += 1


def _release_milvus_connection() -> None:
    global _MILVUS_CONNECTION_REFCOUNT
    with _MILVUS_CONNECTION_LOCK:
        _MILVUS_CONNECTION_REFCOUNT = max(_MILVUS_CONNECTION_REFCOUNT - 1, 0)
        if _MILVUS_CONNECTION_REFCOUNT == 0 and connections.has_connection("default"):
            connections.disconnect("default")


_lru_cache_lock = threading.RLock()


def _lru_cache_get(cache: OrderedDict, key: str):
    with _lru_cache_lock:
        if key not in cache:
            return None
        cache.move_to_end(key)
        return cache[key]


def _lru_cache_put(cache: OrderedDict, key: str, value, max_items: int) -> None:
    with _lru_cache_lock:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > max_items:
            cache.popitem(last=False)


@dataclass(frozen=True)
class RecallBudget:
    vector: int = VEC_RECALL
    keyword: int = KW_RECALL
    heading: int = HEADING_RECALL
    structure: int = HEADING_RECALL
    article: int = ARTICLE_RECALL
    table: int = TABLE_RECALL
    target_document: int = HEADING_RECALL
    alias_limit: int = 4
    candidate_min: int = 36
    candidate_max: int = 100
    rerank_window: int = 30


@dataclass(frozen=True)
class FusionPolicy:
    rrf_k: int = 60
    score_weight: float = 0.16
    channel_bonus: float = 0.012


@dataclass(frozen=True)
class RetrievalPolicy:
    query_type: str
    contract_intent: str
    recall: RecallBudget = field(default_factory=RecallBudget)
    fusion: FusionPolicy = field(default_factory=FusionPolicy)


# Per-intent recall budgets — tuned for speed: final output is QA_TOP_K=4,
# so oversized recall windows mostly waste rerank and evidence-building time.
RECALL_BUDGET_BY_INTENT: Dict[str, RecallBudget] = {
    "article_lookup": RecallBudget(
        vector=48, keyword=64, heading=40, structure=48, article=64,
        table=12, target_document=48, candidate_min=32, candidate_max=84, rerank_window=28,
    ),
    "count": RecallBudget(
        vector=36, keyword=48, heading=48, structure=64, article=48,
        table=8, target_document=44, alias_limit=3, candidate_min=28, candidate_max=72, rerank_window=24,
    ),
    "table": RecallBudget(
        vector=64, keyword=72, heading=36, structure=40, article=24,
        table=56, target_document=48, candidate_min=36, candidate_max=96, rerank_window=32,
    ),
    "requirement": RecallBudget(
        vector=72, keyword=80, heading=40, structure=48, article=44,
        table=16, target_document=48, candidate_min=36, candidate_max=100, rerank_window=32,
    ),
    "procedure_or_method": RecallBudget(
        vector=72, keyword=72, heading=40, structure=44, article=36,
        table=16, target_document=44, candidate_min=36, candidate_max=96, rerank_window=32,
    ),
    "definition_or_explanation": RecallBudget(
        vector=64, keyword=56, heading=44, structure=48, article=36,
        table=12, target_document=40, candidate_min=32, candidate_max=84, rerank_window=28,
    ),
    "comparison": RecallBudget(
        vector=96, keyword=80, heading=48, structure=56, article=40,
        table=16, target_document=48, alias_limit=6, candidate_min=48, candidate_max=112, rerank_window=38,
    ),
    "general": RecallBudget(),
}


def _compact_match_text(text: str) -> str:
    return compact_query_text(text)


def _contains_any(text: str, terms: Tuple[str, ...]) -> bool:
    return any(term and term in text for term in terms)


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


def _extract_highlight_text(text: str, fallback: str = "") -> str:
    lines = [line.strip() for line in clean_block_text(text).splitlines() if line.strip()]
    for line in lines[:4]:
        if 8 <= len(line) <= 180:
            return line
    if fallback:
        return fallback[:180]
    return clean_block_text(text)[:180]


def _looks_like_toc_or_index_listing(text: str) -> bool:
    cleaned = clean_block_text(text)
    if not cleaned:
        return False
    lower = cleaned.lower()
    compact = _compact_match_text(cleaned)

    if any(token in compact for token in ("tableofcontents", "figurepage", "目录")):
        return True
    if len(re.findall(r"\bsection\s+\d+\.?", lower)) >= 3:
        return True
    if len(re.findall(r"\b(?:figure|appendix|section)\s+[a-z]?\d+(?:[-.]\d+)?", lower)) >= 5:
        return True

    dash_runs = len(re.findall(r"[-‐‑‒–—_]{5,}", cleaned))
    numbered_leaders = len(
        re.findall(
            r"\b(?:[a-z]?\d{1,3}(?:[-–—.]\d{1,3})?|section\s+\d+|appendix\s+[a-z])\.?\s+.{2,90}[-‐‑‒–—_]{5,}\s*[ivx\d]{1,4}\b",
            lower,
        )
    )
    if dash_runs >= 5 and numbered_leaders >= 2:
        return True
    if dash_runs >= 9:
        return True

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    short_entry_lines = [
        line
        for line in lines
        if len(line) <= 140 and re.search(r"[-‐‑‒–—_]{4,}\s*[ivx\d]{1,4}\s*$", line.lower())
    ]
    return len(short_entry_lines) >= 3


