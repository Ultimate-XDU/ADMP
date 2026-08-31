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

from search.common import (
    CCAR_PART_PATTERN,
    CLASSIFICATION_TERMS,
    CODE_REFERENCE_PATTERN,
    CONTRACT_INTENT_QUERY_TYPE,
    DEFINITION_TERMS,
    DOC_DIR,
    EFFECTIVE_DATE_TERMS,
    FusionPolicy,
    HYDE_SYSTEM_PROMPT,
    PROCEDURE_TERMS,
    QA_TOP_K,
    RECALL_BUDGET_BY_INTENT,
    REQUIREMENT_TERMS,
    RecallBudget,
    RetrievalPolicy,
    _compact_match_text,
    _contains_any,
    _doc_alias_keys,
    _expand_article_reference_variants,
    _extract_article_reference,
    _normalize_heading_value,
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
def _contract_target_doc_name(profile: dict) -> str:
    contract = profile.get("retrieval_contract") or {}
    synced_targets = contract.get("synced_target_doc_names") or []
    if synced_targets:
        return str(synced_targets[0] or "")
    return str(profile.get("target_doc_name") or "")


def _contract_has_synced_hard_target(profile: dict) -> bool:
    contract = profile.get("retrieval_contract") or {}
    return bool(contract.get("hard_target") and contract.get("synced_target_doc_names"))


def _contract_needs_code_lexical_recall(profile: dict) -> bool:
    contract = profile.get("retrieval_contract") or {}
    return bool(contract.get("query_codes")) and not _contract_has_synced_hard_target(profile)


def _strip_contract_identity_terms(value: str, profile: dict) -> str:
    text = clean_block_text(value)
    if not text or not _contract_has_synced_hard_target(profile):
        return text

    contract = profile.get("retrieval_contract") or {}
    text = CODE_REFERENCE_PATTERN.sub(" ", text)
    for code in contract.get("query_codes", []) or []:
        for variant in code_query_variants(str(code)):
            if variant:
                text = re.sub(re.escape(variant), " ", text, flags=re.I)

    for target in contract.get("targets", []) or []:
        if not isinstance(target, dict):
            continue
        for raw in (target.get("doc_name"), Path(str(target.get("doc_name") or "")).stem, target.get("title")):
            raw_text = clean_block_text(str(raw or ""))
            if raw_text:
                text = text.replace(raw_text, " ")

    text = re.sub(r"[《》\"'“”‘’（）()【】\[\]]+", " ", text)
    text = clean_block_text(text)
    text = re.sub(r"^(?:中|中的|里|里的|关于|针对|对)\s*", "", text)
    return clean_block_text(text) or clean_block_text(value)


def _contract_query_aliases(query: str, profile: dict) -> List[str]:
    return _build_query_aliases(_strip_contract_identity_terms(query, profile))


def _contract_recall_aliases(query: str, profile: dict) -> List[str]:
    aliases = _contract_query_aliases(query, profile)
    for term in _cross_lingual_query_terms(query):
        if term and term not in aliases:
            aliases.append(term)
    if not _contract_needs_code_lexical_recall(profile):
        return aliases

    contract = profile.get("retrieval_contract") or {}
    seen = {alias for alias in aliases if alias}
    for code in contract.get("query_codes", []) or []:
        for variant in code_query_variants(str(code)):
            if variant and variant not in seen:
                seen.add(variant)
                aliases.append(variant)
    return aliases


def _clean_precision_hints_for_contract(query: str, profile: dict) -> None:
    if not _contract_has_synced_hard_target(profile):
        return
    hints = profile.get("precision_hints") or {}
    if not hints:
        return

    for key in ("required_terms", "subject_terms", "title_keyphrases"):
        cleaned_values: List[str] = []
        for raw in hints.get(key, []) or []:
            cleaned = _strip_contract_identity_terms(str(raw or ""), profile)
            if not cleaned or len(_compact_match_text(cleaned)) < 2:
                continue
            if cleaned not in cleaned_values:
                cleaned_values.append(cleaned)
        hints[key] = cleaned_values

    hints["compact_query"] = _compact_match_text(_strip_contract_identity_terms(query, profile))


def _contract_target_doc_names(profile: dict) -> List[str]:
    contract = profile.get("retrieval_contract") or {}
    contract_names = contract.get("hard_target_doc_names") if contract.get("hard_target") else contract.get("synced_target_doc_names")
    names = [str(name or "") for name in contract_names or [] if name]
    if not contract.get("allow_multi_doc") and profile.get("target_doc_name") and profile["target_doc_name"] not in names:
        names.append(str(profile["target_doc_name"]))
    return names


def _ccar_code_article_variants(profile: dict) -> List[str]:
    contract = profile.get("retrieval_contract") or {}
    variants: List[str] = []
    for code in contract.get("query_codes", []) or []:
        match = CCAR_PART_PATTERN.match(str(code or ""))
        if not match:
            continue
        part = match.group(1)
        if part not in variants:
            variants.append(part)
    return variants


def _contract_focus_terms(profile: dict) -> List[str]:
    contract = profile.get("retrieval_contract") or {}
    return [str(term or "") for term in contract.get("focus_terms", []) if term]


def _cross_lingual_query_terms(query: str) -> List[str]:
    return list(cross_lingual_query_terms(query))


def _policy_recall_budget(profile: dict) -> RecallBudget:
    contract = profile.get("retrieval_contract") or {}
    intent = str(contract.get("intent") or profile.get("query_type") or "general")
    budget = RECALL_BUDGET_BY_INTENT.get(intent) or RECALL_BUDGET_BY_INTENT["general"]
    if contract.get("hard_target") and contract.get("synced_target_doc_names"):
        return RecallBudget(
            vector=max(48, int(budget.vector * 0.7)),
            keyword=budget.keyword,
            heading=budget.heading,
            structure=budget.structure,
            article=budget.article,
            table=budget.table,
            target_document=max(budget.target_document, 80),
            alias_limit=budget.alias_limit,
            candidate_min=budget.candidate_min,
            candidate_max=min(budget.candidate_max, 140),
            rerank_window=min(budget.rerank_window, 50),
        )
    return budget


def _build_retrieval_policy(profile: dict) -> RetrievalPolicy:
    contract = profile.get("retrieval_contract") or {}
    intent = str(contract.get("intent") or "general")
    rrf_k = int(profile.get("rrf_k", 60))
    if intent == "article_lookup":
        rrf_k = min(rrf_k, 42)
    elif intent == "comparison":
        rrf_k = max(rrf_k, 68)
    return RetrievalPolicy(
        query_type=str(profile.get("query_type") or "general"),
        contract_intent=intent,
        recall=_policy_recall_budget(profile),
        fusion=FusionPolicy(rrf_k=rrf_k),
    )


def _align_profile_with_contract_intent(profile: dict, contract: RetrievalContract) -> None:
    profile["contract_intent"] = contract.intent
    mapped_type = CONTRACT_INTENT_QUERY_TYPE.get(contract.intent)
    if mapped_type and profile.get("query_type") == "general":
        profile["query_type"] = mapped_type
    if contract.intent in {"article_lookup", "count", "table"}:
        profile["structure_first"] = True


def _contract_needs_multi_target_recall(contract: RetrievalContract) -> bool:
    return contract_needs_multi_target_recall(contract)


def _contracted_sources_or_none(contract: RetrievalContract, sources: Iterable[dict], *, limit: int = QA_TOP_K) -> Optional[List[dict]]:
    contracted = apply_contract(contract, sources, keep_failed_when_no_hard_constraints=True)
    return contracted[:limit] if contracted else None


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
    if re.search(r"[A-Za-z]", query) and not any(token in compact for token in ("定义", "是指", "含义", "术语")):
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
    for term in _cross_lingual_query_terms(query):
        if term not in hints["intent_terms"]:
            hints["intent_terms"].append(term)
        if term not in hints["title_keyphrases"]:
            hints["title_keyphrases"].append(term)
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


def analyze_query_intent(query: str) -> dict:
    text = query.strip()
    normalized = text.lower()
    query_signature = _build_query_signature(text)
    target_roles = query_signature["target_roles"]

    # Shared intent as single source of truth for the flags
    decision = classify_query_intent(query)
    article_like = decision.article_like
    # Override: query_signature's article_id_variants also forces article_lookup
    if query_signature.get("article_id_variants"):
        article_like = True
    fault_like = decision.fault_like
    procedure_like = decision.procedure_like
    concept_like = decision.concept_like
    classification_like = decision.classification_like
    structure_first = decision.structure_first or bool(
        "effective_date" in target_roles
        or "definition" in target_roles
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


def _generate_hypothetical_answer(query: str, llm_call: Optional[Callable[[str, str], str]] = None) -> str:
    """Generate a hypothetical answer for HyDE retrieval. Falls back gracefully if LLM unavailable."""
    if not llm_call:
        return ""
    user_prompt = f"问题：{query}\n\n回答摘要："
    try:
        return llm_call(HYDE_SYSTEM_PROMPT, user_prompt).strip()
    except Exception:
        return ""


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

