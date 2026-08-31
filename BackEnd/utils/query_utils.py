"""Shared query text and intent utilities for aerospace retrieval.

The retrieval contract layer and search layer both need the same low-level
query normalization, bilingual recall expansion, and coarse intent decision.
Keeping those decisions here prevents the two layers from drifting.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import List, Tuple


def compact_query_text(value: str) -> str:
    """Normalize text to compact lowercase CJK/ASCII tokens."""
    normalized = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"[^\u4e00-\u9fffA-Za-z0-9.]+", "", normalized.lower())


_CROSS_LINGUAL_MAPPING: Tuple[Tuple[Tuple[str, ...], Tuple[str, ...]], ...] = (
    (("目的", "主要目的", "用途", "为了"), ("purpose",)),
    (("适用范围", "适用于", "适用"), ("scope", "applicability", "applies")),
    (("定义", "含义", "是指"), ("definition", "means")),
    (("要求", "应当", "必须"), ("requirement", "should", "must")),
    (("建议", "推荐"), ("recommendation", "recommended", "suggested")),
    (("培训", "训练"), ("training",)),
    (("复训", "定期训练"), ("recurrent training", "recurrent")),
    (("初始训练", "初始培训"), ("initial training", "initial")),
    (("错误", "差错"), ("error",)),
    (("风险",), ("risk",)),
    (("疲劳",), ("fatigue",)),
)


def cross_lingual_query_terms(query: str) -> Tuple[str, ...]:
    """Expand Chinese intent markers when the query already mixes languages."""
    text = str(query or "")
    if not re.search(r"[A-Za-z]", text):
        return ()

    compact = compact_query_text(text)
    terms: List[str] = []
    for triggers, expansions in _CROSS_LINGUAL_MAPPING:
        if any(compact_query_text(trigger) in compact for trigger in triggers):
            for term in expansions:
                if term not in terms:
                    terms.append(term)
    return tuple(terms)


_ARTICLE_LABEL_RE = re.compile(
    r"(?:第\s*)?([0-9A-Za-z一二三四五六七八九十百千万]+"
    r"(?:\.[0-9A-Za-z一二三四五六七八九十百千万]+)*)\s*条"
)
_SECTION_REF_RE = re.compile(r"第\s*[0-9一二三四五六七八九十百千万]+\s*[章节]")
_SECTION_NUM_RE = re.compile(r"(?<![A-Za-z0-9.])\d+\.\d+(?:\.\d+)*(?![A-Za-z0-9.])")

_FAULT_TOKENS = ("故障", "异常", "告警", "失效", "不工作", "排查", "原因", "报警")
_PROCEDURE_TOKENS = (
    "步骤",
    "如何",
    "怎么",
    "怎样",
    "流程",
    "更换",
    "检查",
    "操作",
    "程序",
    "procedure",
    "颁发程序",
)
_CONCEPT_TOKENS = ("是什么", "含义", "定义", "区别", "作用", "概念")
_CLASSIFICATION_TOKENS = ("类别", "分类", "分为", "分成", "哪几类", "哪两类", "哪些类型")
_CLASSIFICATION_SUBJECTS = ("适航证", "合格证", "许可证", "批准标签", "证书", "特许飞行证")
_COMPARISON_TOKENS = ("区别", "不同", "差异", "比较", "对比", "侧重点", "异同")
_COUNT_TOKENS = ("多少章", "多少节", "多少条", "共有", "有几")
_TABLE_TOKENS = ("表", "清单", "项目表", "检查表")
_DEFINITION_TOKENS = ("定义", "含义", "什么是")
_REQUIREMENT_TOKENS = ("申请", "提交", "条件", "资格", "要求", "应当", "必须")
_ARTICLE_CODE_TOKENS = ("ccar", "ata", "amm", "fim", "tsm", "ipc", "srm")
_ARTICLE_CONDITION_TOKENS = ("颁发条件", "颁发要求", "签发条件", "申请条件", "批准条件")
_CODE_REFERENCE_RE = re.compile(
    r"(?<![A-Za-z])"
    r"(?:AC|FAA|EASA|CCAR|ATA|AMM|FIM|TSM|IPC|SRM)"
    r"(?:[- ]?(?:Part[- ]?)?)?\d",
    re.I,
)
_PROCEDURE_INTENT_TOKENS = ("程序", "流程", "步骤", "如何", "怎么", "怎样")


@dataclass(frozen=True)
class QueryIntentDecision:
    """Unified query intent decision shared by retrieval layers."""

    intent: str
    query_type: str
    structure_first: bool
    article_like: bool
    fault_like: bool
    procedure_like: bool
    concept_like: bool
    classification_like: bool


def _contains_any_text(text: str, tokens: Tuple[str, ...]) -> bool:
    return any(token and token in text for token in tokens)


def _contains_any_compact(compact_text: str, tokens: Tuple[str, ...]) -> bool:
    return any(compact_query_text(token) in compact_text for token in tokens)


def classify_query_intent(query: str, *, article_label: str = "") -> QueryIntentDecision:
    """Classify a query once for both retrieval contract and search profile."""
    text = str(query or "").strip()
    normalized = text.lower()
    compact = compact_query_text(text)

    article_like = bool(
        article_label
        or _ARTICLE_LABEL_RE.search(text)
        or _SECTION_NUM_RE.search(text)
        or _CODE_REFERENCE_RE.search(text)
        or _contains_any_text(normalized, _ARTICLE_CODE_TOKENS)
        or _contains_any_text(text, _ARTICLE_CONDITION_TOKENS)
    )

    classification_like = (
        _contains_any_text(text, _CLASSIFICATION_TOKENS)
        and _contains_any_text(text, _CLASSIFICATION_SUBJECTS)
    )
    article_like = article_like or classification_like

    fault_like = _contains_any_text(text, _FAULT_TOKENS)
    procedure_like = _contains_any_text(text, _PROCEDURE_TOKENS)
    concept_like = _contains_any_text(text, _CONCEPT_TOKENS)
    structure_first = bool(
        article_like
        or "适用范围" in text
        or _SECTION_REF_RE.search(text)
        or classification_like
    )

    if _contains_any_compact(compact, _COMPARISON_TOKENS):
        intent = "comparison"
    elif article_label or article_like:
        intent = "article_lookup"
    elif _contains_any_compact(compact, _COUNT_TOKENS):
        intent = "count"
    elif _contains_any_compact(compact, _TABLE_TOKENS):
        intent = "table"
    elif _contains_any_compact(compact, _DEFINITION_TOKENS) or (
        compact.endswith("是什么") and len(compact) <= 16
    ):
        intent = "definition_or_explanation"
    elif _contains_any_compact(compact, _REQUIREMENT_TOKENS):
        intent = "requirement"
    elif _contains_any_compact(compact, _PROCEDURE_INTENT_TOKENS):
        intent = "procedure_or_method"
    else:
        intent = "general"

    if article_like:
        query_type = "article_lookup"
    elif fault_like:
        query_type = "fault_diagnosis"
    elif procedure_like:
        query_type = "procedure"
    elif concept_like:
        query_type = "concept_explanation"
    else:
        query_type = "general"

    return QueryIntentDecision(
        intent=intent,
        query_type=query_type,
        structure_first=structure_first,
        article_like=article_like,
        fault_like=fault_like,
        procedure_like=procedure_like,
        concept_like=concept_like,
        classification_like=classification_like,
    )
