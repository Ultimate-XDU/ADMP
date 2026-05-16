"""Composable Elasticsearch query builders for aerospace retrieval.

This module keeps ES bool-query construction out of individual retrieval
channels.  Channels still own their weights, but shared signals such as aliases,
precision hints, article references, target document filters, and role hints are
translated through one set of helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class WeightedField:
    field: str
    boost: float


@dataclass(frozen=True)
class AliasClauseProfile:
    phrase_fields: tuple[WeightedField, ...] = ()
    match_fields: tuple[WeightedField, ...] = ()
    multi_match_fields: tuple[str, ...] = ()
    multi_match_boost: float = 0.0
    multi_match_type: str = "best_fields"
    decay_step: float = 0.12
    decay_floor: float = 0.58
    exact_section_boost: float = 0.0
    limit: int | None = None


@dataclass(frozen=True)
class PrecisionHintProfile:
    target_doc_boost: float = 0.0
    preferred_article_boost: float = 0.0
    preferred_article_label_boost: float = 0.0
    preferred_article_scope_boost: float = 0.0
    related_article_boost: float = 0.0
    preferred_section_boost: float = 0.0
    preferred_section_title_boost: float = 0.0
    required_term_fields: tuple[WeightedField, ...] = ()


@dataclass(frozen=True)
class ArticleReferenceProfile:
    article_id_boost: float = 0.0
    article_label_term_boost: float = 0.0
    article_label_phrase_fields: tuple[WeightedField, ...] = ()
    normalized_heading_boost: float = 0.0
    role_boost: float = 0.0


@dataclass(frozen=True)
class CodeVariantProfile:
    fields: tuple[WeightedField, ...] = ()


@dataclass
class EsQueryPlan:
    should: list[dict] = field(default_factory=list)
    filters: list[dict] = field(default_factory=list)
    minimum_should_match: int = 1
    allow_empty_match_all: bool = False

    def body(self, size: int) -> dict:
        query: dict = {"bool": {}}
        if self.filters:
            query["bool"]["filter"] = self.filters
        if self.should:
            query["bool"]["should"] = self.should
            query["bool"]["minimum_should_match"] = self.minimum_should_match
        elif not self.filters:
            query = {"match_all": {}} if self.allow_empty_match_all else {"match_none": {}}
        return {"query": query, "size": size}


def unique_texts(values: Iterable[str], *, limit: int | None = None) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for raw in values:
        value = str(raw or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        output.append(value)
        if limit is not None and len(output) >= limit:
            break
    return output


def term_clause(field: str, value: str, boost: float) -> dict:
    return {"term": {field: {"value": value, "boost": boost}}}


def terms_filter(field: str, values: Sequence[str]) -> dict | None:
    cleaned = unique_texts(values)
    if not cleaned:
        return None
    return {"terms": {field: cleaned}}


def match_phrase_clause(field: str, query: str, boost: float) -> dict:
    return {"match_phrase": {field: {"query": query, "boost": boost}}}


def match_clause(field: str, query: str, boost: float) -> dict:
    return {"match": {field: {"query": query, "boost": boost}}}


def multi_match_clause(query: str, fields: Sequence[str], boost: float, match_type: str = "best_fields") -> dict:
    return {
        "multi_match": {
            "query": query,
            "fields": list(fields),
            "type": match_type,
            "boost": boost,
        }
    }


def build_alias_clauses(
    aliases: Iterable[str],
    profile: AliasClauseProfile,
    *,
    exact_match_bias: bool = False,
) -> list[dict]:
    clauses: list[dict] = []
    for index, alias in enumerate(unique_texts(aliases, limit=profile.limit)):
        decay = max(profile.decay_floor, 1.0 - profile.decay_step * index)
        if profile.multi_match_fields and profile.multi_match_boost > 0:
            clauses.append(
                multi_match_clause(
                    alias,
                    profile.multi_match_fields,
                    profile.multi_match_boost * decay,
                    profile.multi_match_type,
                )
            )
        for field in profile.phrase_fields:
            clauses.append(match_phrase_clause(field.field, alias, field.boost * decay))
        for field in profile.match_fields:
            clauses.append(match_clause(field.field, alias, field.boost * decay))
        if exact_match_bias and profile.exact_section_boost > 0:
            clauses.append(match_phrase_clause("section_title", alias, profile.exact_section_boost * decay))
    return clauses


def build_precision_hint_clauses(hints: Mapping, profile: PrecisionHintProfile) -> list[dict]:
    clauses: list[dict] = []
    target_doc_name = str(hints.get("target_doc_name") or "")
    if target_doc_name and profile.target_doc_boost > 0:
        clauses.append(term_clause("doc_name", target_doc_name, profile.target_doc_boost))
    for article in unique_texts(hints.get("preferred_articles", []) or []):
        if profile.preferred_article_boost > 0:
            clauses.append(term_clause("article_id", article, profile.preferred_article_boost))
        if profile.preferred_article_label_boost > 0:
            clauses.append(match_phrase_clause("article_label", f"第{article}条", profile.preferred_article_label_boost))
        if profile.preferred_article_scope_boost > 0:
            clauses.append(match_phrase_clause("scope_anchor", f"第{article}条", profile.preferred_article_scope_boost))
    for article in unique_texts(hints.get("related_articles", []) or []):
        if profile.related_article_boost > 0:
            clauses.append(term_clause("article_id", article, profile.related_article_boost))
    for section in unique_texts(hints.get("preferred_sections", []) or []):
        if profile.preferred_section_boost > 0:
            clauses.append(term_clause("section_id", section, profile.preferred_section_boost))
        if profile.preferred_section_title_boost > 0:
            clauses.append(match_phrase_clause("section_title", section, profile.preferred_section_title_boost))
    for required_term in unique_texts(hints.get("required_terms", []) or []):
        for field in profile.required_term_fields:
            clauses.append(match_phrase_clause(field.field, required_term, field.boost))
    return clauses


def build_article_reference_clauses(
    query_signature: Mapping,
    target_roles: Iterable[str],
    profile: ArticleReferenceProfile,
) -> list[dict]:
    clauses: list[dict] = []
    article_id = str(query_signature.get("article_id") or "")
    article_label = str(query_signature.get("article_label") or "")
    for article_id_variant in unique_texts(query_signature.get("article_id_variants", []) or ([article_id] if article_id else [])):
        if profile.article_id_boost > 0:
            clauses.append(term_clause("article_id", article_id_variant, profile.article_id_boost))
    for article_label_variant in unique_texts(query_signature.get("article_label_variants", []) or ([article_label] if article_label else [])):
        if profile.article_label_term_boost > 0:
            clauses.append(term_clause("article_label", article_label_variant, profile.article_label_term_boost))
        for field in profile.article_label_phrase_fields:
            clauses.append(match_phrase_clause(field.field, article_label_variant, field.boost))
    normalized_heading = str(query_signature.get("normalized_heading") or "")
    if normalized_heading and profile.normalized_heading_boost > 0:
        clauses.append(term_clause("normalized_heading", normalized_heading, profile.normalized_heading_boost))
    for role in unique_texts(target_roles):
        if role and role != "text" and profile.role_boost > 0:
            clauses.append(term_clause("chunk_role", role, profile.role_boost))
    return clauses


def build_code_variant_clauses(
    codes: Iterable[str],
    variants_fn: Callable[[str], Iterable[str]],
    profile: CodeVariantProfile,
) -> list[dict]:
    clauses: list[dict] = []
    for code in unique_texts(codes):
        for variant in unique_texts(variants_fn(code)):
            for field in profile.fields:
                clauses.append(match_phrase_clause(field.field, variant, field.boost))
    return clauses


def es_hit_to_channel_result(hit: Mapping, channel: str) -> dict:
    source = hit.get("_source") or {}
    return {
        "chunk_id": source["chunk_id"],
        "content": source.get("content", ""),
        "score": float(hit.get("_score", 0.0)),
        "channel": channel,
    }
