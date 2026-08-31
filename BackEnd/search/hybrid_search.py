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
    _bbox_union,
    _compact_section_label,
    _derive_section_title,
    _extract_highlight_text,
    _is_repeating_pdf_noise,
    _normalize_bbox,
    _restore_article_reference,
    _split_text_candidates,
    logger,
)
from search.evidence_scoring import (
    _front_matter_penalty,
    _has_unmodified_subject,
    _looks_like_reference_material,
    _precision_hint_score,
    _query_alignment_score,
    _row_article_number,
    _select_query_aligned_highlight,
    _structural_highlight_text,
    logger,
)
from search.index_manager import _es_get_by_id, get_embedder, get_reranker, logger
from search.query_analysis import (
    _align_profile_with_contract_intent,
    _build_retrieval_policy,
    _ccar_code_article_variants,
    _clean_precision_hints_for_contract,
    _contract_focus_terms,
    _contract_needs_code_lexical_recall,
    _contract_needs_multi_target_recall,
    _contract_query_aliases,
    _contract_recall_aliases,
    _contract_target_doc_name,
    _contract_target_doc_names,
    _contracted_sources_or_none,
    _generate_hypothetical_answer,
    _has_structural_subject_support,
    _is_valid_search_query,
    _strip_contract_identity_terms,
    analyze_query_intent,
    logger,
)
from search.retrieval_agent import (
    AgentRetrievalPlan,
    agent_enhanced_retrieval_policy,
    get_retrieval_agent,
    logger as agent_logger,
)
from search.common import (
    ARTICLE_ALIAS_PROFILE,
    ARTICLE_PRECISION_PROFILE,
    ARTICLE_RECALL,
    ARTICLE_REFERENCE_PROFILE,
    CODE_LEXICAL_PROFILE,
    DOC_DIR,
    ES_HOST,
    ES_INDEX,
    ES_QUERY_TIMEOUT_SECONDS,
    ES_REQUEST_TIMEOUT_SECONDS,
    ES_SLOW_QUERY_SECONDS,
    HEADING_ALIAS_PROFILE,
    HEADING_RECALL,
    KEYWORD_ALIAS_PROFILE,
    KW_RECALL,
    METADATA_CACHE_MAX_ITEMS,
    MILVUS_COLLECTION,
    PDF_PAGE_CACHE_MAX_ITEMS,
    QA_TOP_K,
    STRUCTURE_ALIAS_PROFILE,
    STRUCTURE_PRECISION_PROFILE,
    STRUCTURE_REFERENCE_PROFILE,
    TABLE_ALIAS_PROFILE,
    TABLE_PRECISION_PROFILE,
    TABLE_RECALL,
    TARGET_DOC_ALIAS_PROFILE,
    VEC_RECALL,
    _acquire_milvus_connection,
    _article_reference_matches,
    _compact_match_text,
    _compact_section_label,
    _doc_alias_keys,
    _extract_highlight_text,
    _lru_cache_get,
    _lru_cache_lock,
    _lru_cache_put,
    _ngram_similarity,
    _normalize_heading_value,
    _normalize_search_text,
    _release_milvus_connection,
    _sigmoid,
    _token_overlap_score,
    clean_block_text,
    infer_document_type,
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
class HybridSearch:
    def __init__(self, hyde_llm_call: Optional[Callable[[str, str], str]] = None):
        _acquire_milvus_connection()
        self._closed = False
        self.es_client = None
        try:
            self.es_client = Elasticsearch(ES_HOST)
            self.milvus_coll = Collection(MILVUS_COLLECTION)
            self.milvus_coll.load()
            self.reranker = get_reranker()
            self.embedder = get_embedder()
            self.hyde_llm_call = hyde_llm_call
            self._pdf_page_cache: OrderedDict[str, dict] = OrderedDict()
            self._metadata_cache: OrderedDict[str, dict] = OrderedDict()
            self._es_stats_lock = threading.Lock()
            self._es_channel_stats: Dict[str, dict] = defaultdict(
                lambda: {
                    "calls": 0,
                    "failures": 0,
                    "hits": 0,
                    "total_elapsed_seconds": 0.0,
                    "max_elapsed_seconds": 0.0,
                    "last_elapsed_seconds": 0.0,
                    "last_error": "",
                }
            )
        except Exception:
            self._closed = True
            try:
                if self.es_client is not None:
                    self.es_client.close()
            except Exception:
                logger.debug("Elasticsearch client close failed", exc_info=True)
            _release_milvus_connection()
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._pdf_page_cache.clear()
        self._metadata_cache.clear()
        try:
            if self.es_client is not None:
                self.es_client.close()
        except Exception:
            logger.debug("Elasticsearch client close failed", exc_info=True)
        _release_milvus_connection()

    def invalidate_caches(self) -> None:
        """Drop in-memory caches after index changes without rebuilding clients/models."""
        with _lru_cache_lock:
            self._pdf_page_cache.clear()
            self._metadata_cache.clear()

    def __enter__(self) -> "HybridSearch":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

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
        cached = _lru_cache_get(self._pdf_page_cache, file_path)
        if cached is not None:
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

        _lru_cache_put(self._pdf_page_cache, file_path, payload, PDF_PAGE_CACHE_MAX_ITEMS)
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

    def _build_retrieval_profile(self, query: str, contract: Optional[RetrievalContract] = None) -> Tuple[dict, RetrievalContract]:
        contract = contract or build_retrieval_contract(query)
        profile = analyze_query_intent(query)
        profile["retrieval_contract"] = contract.to_dict()
        _align_profile_with_contract_intent(profile, contract)
        multi_target_recall = _contract_needs_multi_target_recall(contract)
        if multi_target_recall:
            profile.pop("target_doc_name", None)
            profile.pop("target_doc_is_explicit", None)
            profile.setdefault("precision_hints", {})["target_doc_name"] = ""
        if contract.block_reason:
            profile["contract_block_reason"] = contract.block_reason
            profile["retrieval_policy"] = _build_retrieval_policy(profile)
            return profile, contract

        if contract.synced_target_doc_names and not multi_target_recall and (contract.hard_target or len(contract.synced_target_doc_names) == 1):
            target_doc_name = contract.synced_target_doc_names[0]
            profile["target_doc_name"] = target_doc_name
            profile["target_doc_is_explicit"] = bool(contract.hard_target or len(contract.synced_target_doc_names) == 1)
            profile.setdefault("precision_hints", {})["target_doc_name"] = target_doc_name
            _clean_precision_hints_for_contract(query, profile)
            if contract.hard_target:
                profile["retrieval_policy"] = _build_retrieval_policy(profile)
                return profile, contract

        target_doc_name = "" if profile.get("target_doc_name") or contract.synced_target_doc_names else self._find_doc_name_by_query_reference(query)
        if target_doc_name:
            profile["target_doc_name"] = target_doc_name
            profile["target_doc_is_explicit"] = True
            profile.setdefault("precision_hints", {})["target_doc_name"] = target_doc_name
        elif not profile.get("target_doc_name") and not contract.synced_target_doc_names:
            inferred_doc_name = self._infer_target_doc_from_code(profile) or self._infer_target_doc_from_structure(query, profile)
            if inferred_doc_name:
                profile["target_doc_name"] = inferred_doc_name
                profile["target_doc_is_explicit"] = False
                profile.setdefault("precision_hints", {})["target_doc_name"] = inferred_doc_name
        profile["retrieval_policy"] = _build_retrieval_policy(profile)
        return profile, contract

    def _infer_target_doc_from_code(self, profile: dict) -> str:
        part_variants = _ccar_code_article_variants(profile)
        if not part_variants:
            return ""
        scores: Dict[str, float] = defaultdict(float)
        for record in load_all_structural_records():
            if record.get("is_toc") or str(record.get("record_type") or "") == "doc":
                continue
            doc_name = str(record.get("doc_name") or "")
            if not doc_name:
                continue
            article_id = str(record.get("article_id") or "")
            if any(article_id == part or article_id.startswith(f"{part}.") for part in part_variants):
                scores[doc_name] += 3.0
            elif any(part and part in _compact_match_text(str(record.get("body_text") or "")[:700]) for part in part_variants):
                scores[doc_name] += 0.45
        if not scores:
            return ""
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if len(ranked) == 1 or ranked[0][1] >= ranked[1][1] * 1.35:
            return ranked[0][0]
        return ""

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

    def _record_es_channel_stats(
        self,
        channel: str,
        *,
        elapsed_seconds: float,
        hit_count: int = 0,
        error: str = "",
    ) -> None:
        with self._es_stats_lock:
            stats = self._es_channel_stats[channel]
            stats["calls"] += 1
            stats["hits"] += int(hit_count)
            stats["total_elapsed_seconds"] = round(float(stats["total_elapsed_seconds"]) + elapsed_seconds, 6)
            stats["max_elapsed_seconds"] = round(max(float(stats["max_elapsed_seconds"]), elapsed_seconds), 6)
            stats["last_elapsed_seconds"] = round(elapsed_seconds, 6)
            if error:
                stats["failures"] += 1
                stats["last_error"] = error[:500]

    def es_telemetry_snapshot(self) -> Dict[str, dict]:
        with self._es_stats_lock:
            return {channel: dict(stats) for channel, stats in self._es_channel_stats.items()}

    def _execute_es_channel_search(self, channel: str, query: str, body: dict) -> List[dict]:
        started = time.time()
        try:
            response = self.es_client.options(
                request_timeout=ES_REQUEST_TIMEOUT_SECONDS,
                retry_on_timeout=False,
            ).search(
                index=ES_INDEX,
                body=body,
                timeout=f"{math.ceil(ES_QUERY_TIMEOUT_SECONDS)}s",
                track_total_hits=False,
            )
            hits = response["hits"]["hits"]
            elapsed = time.time() - started
            self._record_es_channel_stats(channel, elapsed_seconds=elapsed, hit_count=len(hits))
            if elapsed >= ES_SLOW_QUERY_SECONDS:
                logger.warning("ES channel slow channel=%s elapsed=%.3fs hits=%d", channel, elapsed, len(hits))
            return [es_hit_to_channel_result(hit, channel) for hit in hits]
        except Exception as exc:
            elapsed = time.time() - started
            self._record_es_channel_stats(channel, elapsed_seconds=elapsed, error=f"{type(exc).__name__}: {exc}")
            logger.warning("ES channel failed channel=%s query=%s elapsed=%.3fs error=%s", channel, query, elapsed, exc)
            return []

    def _article_search(self, query: str, profile: dict, top_k: int = ARTICLE_RECALL) -> List[dict]:
        query_signature = profile.get("query_signature", {})
        precision_hints = profile.get("precision_hints") or {}
        should = []
        should.extend(build_precision_hint_clauses(precision_hints, ARTICLE_PRECISION_PROFILE))
        contract = profile.get("retrieval_contract") or {}
        if _contract_needs_code_lexical_recall(profile):
            should.extend(build_code_variant_clauses(contract.get("query_codes", []) or [], code_query_variants, CODE_LEXICAL_PROFILE))
        should.extend(build_article_reference_clauses(query_signature, query_signature.get("target_roles", []), ARTICLE_REFERENCE_PROFILE))
        should.extend(build_alias_clauses(_contract_recall_aliases(query, profile), ARTICLE_ALIAS_PROFILE))

        if not should:
            return []

        body = EsQueryPlan(should=should).body(top_k)
        return self._execute_es_channel_search("article", query, body)

    def _keyword_search(self, query: str, profile: dict, top_k: int = KW_RECALL) -> List[dict]:
        aliases = _contract_recall_aliases(query, profile)
        focus_terms = _contract_focus_terms(profile)
        for term in focus_terms:
            if term and term not in aliases:
                aliases.append(term)
        should = build_alias_clauses(
            aliases,
            KEYWORD_ALIAS_PROFILE,
            exact_match_bias=bool(profile.get("exact_match_bias")),
        )

        body = EsQueryPlan(should=should).body(top_k)
        return self._execute_es_channel_search("keyword", query, body)

    def _heading_search(self, query: str, profile: dict, top_k: int = HEADING_RECALL) -> List[dict]:
        aliases = _contract_recall_aliases(query, profile)
        should = build_alias_clauses(aliases, HEADING_ALIAS_PROFILE)
        body = EsQueryPlan(should=should).body(top_k)
        return self._execute_es_channel_search("heading", query, body)

    def _structure_search(self, query: str, profile: dict, top_k: int = HEADING_RECALL) -> List[dict]:
        aliases = _contract_recall_aliases(query, profile)
        query_signature = profile.get("query_signature", {})
        precision_hints = profile.get("precision_hints") or {}
        should = []
        should.extend(build_precision_hint_clauses(precision_hints, STRUCTURE_PRECISION_PROFILE))
        should.extend(build_article_reference_clauses(query_signature, profile.get("target_roles", []), STRUCTURE_REFERENCE_PROFILE))
        should.extend(build_alias_clauses(aliases, STRUCTURE_ALIAS_PROFILE))

        if not should:
            return []

        body = EsQueryPlan(should=should).body(top_k)
        return self._execute_es_channel_search("structure", query, body)

    def _table_search(self, query: str, profile: Optional[dict] = None, top_k: int = TABLE_RECALL) -> List[dict]:
        profile = profile or {}
        precision_hints = profile.get("precision_hints") or {}
        aliases = _contract_recall_aliases(query, profile) if profile else [query]
        should = []
        should.extend(build_precision_hint_clauses(precision_hints, TABLE_PRECISION_PROFILE))
        should.extend(build_alias_clauses(aliases, TABLE_ALIAS_PROFILE))
        filters = [{"term": {"source_type": "table"}}]
        target_filter = terms_filter("doc_name", _contract_target_doc_names(profile)) if profile else None
        if target_filter:
            filters.append(target_filter)
        body = EsQueryPlan(should=should, filters=filters).body(top_k)
        return self._execute_es_channel_search("table", query, body)

    def _target_document_search(self, query: str, profile: dict, top_k: int = HEADING_RECALL) -> List[dict]:
        target_doc_name = _contract_target_doc_name(profile)
        if not target_doc_name:
            return []

        aliases = _contract_recall_aliases(query, profile)
        focus_terms = _contract_focus_terms(profile)

        should = build_alias_clauses(aliases, TARGET_DOC_ALIAS_PROFILE)
        for focus_term in focus_terms:
            should.extend(
                [
                    match_phrase_clause("doc_name", focus_term, 6.5),
                    match_phrase_clause("heading_path_text", focus_term, 5.2),
                    match_phrase_clause("section_title", focus_term, 4.6),
                    match_phrase_clause("content", focus_term, 3.2),
                ]
            )

        body = EsQueryPlan(
            should=should,
            filters=[{"terms": {"doc_name": _contract_target_doc_names(profile)}}],
            minimum_should_match=1,
        ).body(top_k)
        return self._execute_es_channel_search("target_document", query, body)

    def _target_document_fallback_search(self, query: str, profile: dict, target_doc_name: str, top_k: int = 2) -> List[dict]:
        if not target_doc_name:
            return []

        aliases = _contract_recall_aliases(query, profile)
        focus_terms = _contract_focus_terms(profile)
        doc_aliases = _doc_alias_keys(target_doc_name)
        should = build_alias_clauses(aliases + focus_terms + doc_aliases, TARGET_DOC_ALIAS_PROFILE)
        for doc_alias in doc_aliases:
            should.extend(
                [
                    match_phrase_clause("doc_name", doc_alias, 8.0),
                    match_phrase_clause("section_title", doc_alias, 4.0),
                    match_phrase_clause("heading_path_text", doc_alias, 4.0),
                ]
            )

        body = EsQueryPlan(
            should=should,
            filters=[{"terms": {"doc_name": [target_doc_name]}}],
            minimum_should_match=1,
        ).body(top_k)
        return self._execute_es_channel_search("target_document_fallback", query, body)

    def _ensure_multi_target_source_coverage(
        self,
        query: str,
        profile: dict,
        contract: RetrievalContract,
        sources: List[dict],
        fallback_pools: Iterable[dict],
    ) -> List[dict]:
        if not (contract.allow_multi_doc or _contract_needs_multi_target_recall(contract)):
            return sources
        if not contract.synced_target_doc_names:
            return sources

        output = list(sources)
        represented_docs = {str(row.get("doc_name") or "") for row in output}
        additions: List[dict] = []
        pool = list(fallback_pools)
        for target_doc_name in contract.synced_target_doc_names:
            if target_doc_name in represented_docs:
                continue
            pool_candidates = [row for row in pool if row.get("doc_name") == target_doc_name]
            passed_pool_candidates = apply_contract(contract, pool_candidates, preserve_order=True)
            target_row = passed_pool_candidates[0] if passed_pool_candidates else None
            if target_row is None:
                fallback_hits = self._target_document_fallback_search(query, profile, target_doc_name, top_k=2)
                fallback_hits = apply_contract(contract, fallback_hits, preserve_order=True)
                target_row = fallback_hits[0] if fallback_hits else None
            if target_row is None:
                continue
            additions.append(target_row)
            represented_docs.add(target_doc_name)

        if not additions:
            return output

        final: List[dict] = []
        seen_keys = set()

        def row_key(row: dict) -> Tuple[str, str, str]:
            return (
                str(row.get("doc_name") or ""),
                str(row.get("chunk_id") or row.get("structural_node_id") or row.get("article_id") or ""),
                str(row.get("section_title") or ""),
            )

        def append_unique(row: dict) -> None:
            key = row_key(row)
            if key in seen_keys:
                return
            seen_keys.add(key)
            final.append(row)

        if output:
            append_unique(output[0])
        for row in additions:
            append_unique(row)
        for row in output[1:]:
            append_unique(row)
            if len(final) >= QA_TOP_K:
                break
        return final[:QA_TOP_K]

    def _rrf_fusion(self, recall_groups: List[Tuple[str, List[dict], float]], profile: dict) -> List[dict]:
        policy = profile.get("retrieval_policy") or _build_retrieval_policy(profile)
        scores: Dict[str, float] = {}
        content_map: Dict[str, str] = {}
        channels: Dict[str, set] = defaultdict(set)
        channel_scores: Dict[str, Dict[str, float]] = defaultdict(dict)
        k = max(int(policy.fusion.rrf_k), 1)

        for name, hits, weight in recall_groups:
            raw_scores = [float(hit.get("score") or hit.get("rrf_score") or 0.0) for hit in hits]
            min_score = min(raw_scores, default=0.0)
            max_score = max(raw_scores, default=0.0)
            span = max(max_score - min_score, 1e-9)
            for rank, hit in enumerate(hits):
                chunk_id = hit["chunk_id"]
                raw_score = float(hit.get("score") or hit.get("rrf_score") or 0.0)
                normalized_score = (raw_score - min_score) / span if max_score > min_score else (1.0 if raw_score > 0 else 0.0)
                rank_score = float(weight) / (k + rank + 1)
                score_score = float(weight) * policy.fusion.score_weight * normalized_score / k
                scores[chunk_id] = scores.get(chunk_id, 0.0) + rank_score + score_score
                channels[chunk_id].add(name)
                channel_scores[chunk_id][name] = max(channel_scores[chunk_id].get(name, 0.0), round(normalized_score, 6))
                content_map.setdefault(chunk_id, hit.get("content", ""))

        fused = [
            {
                "chunk_id": chunk_id,
                "rrf_score": score + max(len(channels[chunk_id]) - 1, 0) * policy.fusion.channel_bonus,
                "content": content_map.get(chunk_id, ""),
                "channels": sorted(channels[chunk_id]),
                "channel_scores": channel_scores.get(chunk_id, {}),
            }
            for chunk_id, score in scores.items()
        ]
        return sorted(fused, key=lambda item: item["rrf_score"], reverse=True)

    def _get_metadata(self, chunk_id: str) -> dict:
        if not chunk_id:
            return {}
        cached = _lru_cache_get(self._metadata_cache, chunk_id)
        if cached is not None:
            return cached
        metadata = _es_get_by_id(self.es_client, chunk_id)
        if metadata:
            _lru_cache_put(self._metadata_cache, chunk_id, metadata, METADATA_CACHE_MAX_ITEMS)
        return metadata

    def _get_metadata_batch(self, chunk_ids: List[str]) -> Dict[str, dict]:
        if not chunk_ids:
            return {}
        cached: Dict[str, dict] = {}
        missing_ids: List[str] = []
        seen_ids = set()
        for chunk_id in chunk_ids:
            if not chunk_id or chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)
            row = _lru_cache_get(self._metadata_cache, chunk_id)
            if row is None:
                missing_ids.append(chunk_id)
            else:
                cached[chunk_id] = row
        if missing_ids:
            try:
                response = self.es_client.mget(
                    index=ES_INDEX,
                    body={"ids": missing_ids},
                    request_timeout=ES_REQUEST_TIMEOUT_SECONDS,
                )
            except Exception:
                response = {"docs": []}
            for document in response.get("docs", []):
                if document.get("found"):
                    _lru_cache_put(
                        self._metadata_cache,
                        document["_id"],
                        document["_source"],
                        METADATA_CACHE_MAX_ITEMS,
                    )
            for chunk_id in missing_ids:
                row = _lru_cache_get(self._metadata_cache, chunk_id)
                if row is not None:
                    cached[chunk_id] = row
        return cached

    def _get_chunk_context(self, chunk_id: str, context_size: int = 1) -> Dict:
        current = self._get_metadata_batch([chunk_id]).get(chunk_id, {})
        if not current:
            return {}

        prev_blocks = []
        prev_id = current.get("prev_id", "")
        next_blocks = []
        next_id = current.get("next_id", "")
        parent_id = current.get("parent_id", "")
        parent = None
        for depth in range(max(0, context_size)):
            requested_ids = [prev_id, next_id]
            if depth == 0 and parent_id:
                requested_ids.append(parent_id)
            rows = self._get_metadata_batch(requested_ids)
            if depth == 0 and parent_id:
                parent = rows.get(parent_id)
            if prev_id:
                row = rows.get(prev_id)
                if row:
                    prev_blocks.insert(0, row)
                    prev_id = row.get("prev_id", "")
                else:
                    prev_id = ""
            if next_id:
                row = rows.get(next_id)
                if row:
                    next_blocks.append(row)
                    next_id = row.get("next_id", "")
                else:
                    next_id = ""
            if not prev_id and not next_id:
                break
        return {"current": current, "prev": prev_blocks, "next": next_blocks, "parent": parent}

    def _prefetch_evidence_context(self, hits: List[dict], structural_depth: int, following_depth: int) -> None:
        """Warm evidence metadata in bounded ``mget`` rounds."""
        seed_ids = [str(hit.get("chunk_id") or "") for hit in hits]
        rows = self._get_metadata_batch(seed_ids)
        frontier = list(rows.values())
        max_hops = max(1, min(max(structural_depth, following_depth), 12))
        for _ in range(max_hops):
            requested_ids = {
                str(row.get(field) or "")
                for row in frontier
                for field in ("prev_id", "next_id", "parent_id")
                if row.get(field)
            }
            if not requested_ids:
                return
            next_rows = self._get_metadata_batch(list(requested_ids))
            if not next_rows:
                return
            frontier = list(next_rows.values())

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
        raw_score = round(float(score), 6)
        display_score = round(normalize_structural_score(raw_score), 6)
        return {
            "content": content,
            "doc_name": record.get("doc_name", ""),
            "heading": heading or clean_block_text(record.get("section_title", "")),
            "parent_heading": clean_block_text(record.get("section_title", "")),
            "score": display_score,
            "display_score": display_score,
            "ranking_score": raw_score,
            "raw_structural_score": raw_score,
            "score_components": {"structural": raw_score, "display": display_score},
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
        target_doc_names = set(_contract_target_doc_names(profile))
        ccar_part_variants = _ccar_code_article_variants(profile)
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
            if target_doc_names and target_doc_is_explicit and record.get("doc_name") not in target_doc_names:
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
            low_evidence_penalty = _front_matter_penalty(
                {
                    "source_type": f"structural_{record_type or 'record'}",
                    "section_title": " > ".join(record.get("heading_path", []) or []),
                    "title": record.get("title", ""),
                    "label": record.get("label", ""),
                    "structural_path": " > ".join(record.get("heading_path", []) or []),
                    "page_number": record.get("page_start", 1),
                    "page_count": record.get("page_count", record.get("page_end", 0)),
                },
                joined_text,
                profile,
            )
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
                    if any(token in joined_text for token in ("小时", "分钟", "日内", "工作日", "个工作日", "时限")):
                        score += 1.3
                    if any(token in joined_text for token in ("法律责任", "罚款", "责令限期改正", "逾期不改正")):
                        score -= 1.8
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
            if record_type == "article":
                score += 0.35
            elif record_type == "item":
                score -= 0.28
            if target_doc_name:
                score += 0.8 if record.get("doc_name") == target_doc_name else -0.6
            if target_doc_names and record.get("doc_name") in target_doc_names:
                score += 1.1
            elif target_doc_names:
                score -= 0.55
            if ccar_part_variants:
                article_number = _row_article_number(record)
                article_id = str(record.get("article_id") or "")
                if article_id and any(article_id.startswith(f"{part}.") or article_id == part for part in ccar_part_variants):
                    score += 3.2
                elif article_number and any(article_number.startswith(f"{part}.") or article_number == part for part in ccar_part_variants):
                    score += 3.2
                elif any(part and part in compact_joined for part in ccar_part_variants):
                    score += 1.0
                elif "ccar" in _compact_match_text(query):
                    score -= 0.25
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
            if low_evidence_penalty:
                score -= 2.4 * low_evidence_penalty
            if _looks_like_reference_material(record, joined_text):
                score -= 1.2

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
        target_doc_names = set(_contract_target_doc_names(profile))
        if target_doc_names:
            if metadata.get("doc_name") in target_doc_names:
                score += 0.75
            else:
                score -= 0.6
        focus_terms = _contract_focus_terms(profile)
        if focus_terms:
            focus_haystack = _compact_match_text(
                " ".join(
                    str(metadata.get(key) or "")
                    for key in ("doc_name", "section_title", "heading_text", "scope_anchor", "retrieval_text", "content")
                )
            )
            focus_scores = [
                max(_token_overlap_score(term, focus_haystack), _ngram_similarity(term, focus_haystack))
                for term in focus_terms
                if term
            ]
            focus_hits = sum(1 for value in focus_scores if value >= 0.2)
            focus_coverage = sum(focus_scores) / max(len(focus_scores), 1)
            if focus_hits:
                score += min(0.85, 0.22 * focus_hits + 0.45 * focus_coverage)
            elif target_doc_names and metadata.get("doc_name") not in target_doc_names:
                score -= 0.35
            elif focus_terms:
                score -= 0.16
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
        content = metadata.get("retrieval_text") or metadata.get("content", hit.get("content", ""))
        if _looks_like_reference_material(metadata, content):
            score -= 0.72
        score -= _front_matter_penalty(metadata, content, profile)
        score += 0.1 * max(min(float(metadata.get("title_body_consistency", 0.0)) - 0.45, 0.55), -0.45)
        score += 0.12 * max(min(float(metadata.get("parse_confidence", 0.0)) - 0.5, 0.5), -0.5)
        return round(score, 6)

    def _apply_target_document_candidate_policy(
        self,
        fused: List[dict],
        profile: dict,
        contract,
        recall,
        candidate_window: int,
    ) -> Tuple[List[dict], int]:
        """Filter *fused* to the explicit target document when applicable.

        When the target document is explicit, metadata is loaded in a window
        that may be *larger* than ``candidate_window`` so that target-doc hits
        beyond the normal window are still discoverable.  The returned
        ``candidate_window`` is adjusted upward when metadata was expanded so
        that downstream scoring covers all metadata-loaded hits.

        Returns ``(filtered_fused, adjusted_candidate_window)``.
        """
        if not profile.get("target_doc_name") or not profile.get("target_doc_is_explicit"):
            return fused, candidate_window

        # Scan a wider window for metadata if the target doc might be outside
        # the normal candidate window.
        expanded_window = min(
            max(candidate_window, recall.candidate_max, recall.target_document * 3),
            len(fused),
        )
        if expanded_window > candidate_window:
            extra_ids = [hit["chunk_id"] for hit in fused[candidate_window:expanded_window]
                         if "metadata" not in hit]
            if extra_ids:
                extra_map = self._get_metadata_batch(extra_ids)
                for hit in fused[candidate_window:expanded_window]:
                    if "metadata" not in hit:
                        hit["metadata"] = extra_map.get(hit["chunk_id"], {})

        scan_end = expanded_window
        target_doc_hits = [
            hit
            for hit in fused[:scan_end]
            if (hit.get("metadata") or {}).get("doc_name") == profile["target_doc_name"]
        ]

        if target_doc_hits:
            # Ensure downstream scoring covers the expanded window if we grew it.
            return target_doc_hits, max(candidate_window, min(len(target_doc_hits), expanded_window))

        # Target doc not found even in expanded window.
        if contract.hard_target:
            return [], 0

        # Non-hard explicit target: keep original fused unchanged.
        return fused, candidate_window

    def _search_candidates(self, query: str) -> Tuple[List[dict], dict]:
        profile, contract = self._build_retrieval_profile(query)
        if contract.block_reason:
            return [], profile

        # Store original query for agent use
        profile["original_query"] = query

        # Build base policy, then enhance with retrieval agent (LLM decides how to search)
        base_policy = profile.get("retrieval_policy") or _build_retrieval_policy(profile)
        try:
            policy, agent_plan = agent_enhanced_retrieval_policy(
                profile,
                lambda p: base_policy,
                agent=get_retrieval_agent(),
            )
        except Exception as agent_exc:
            logger.debug("Retrieval agent enhancement failed, using base policy: %s", agent_exc)
            policy = base_policy
            agent_plan = AgentRetrievalPlan(used_agent=False)
            profile["agent_plan"] = agent_plan.to_dict()

        recall = policy.recall
        retrieval_query = _strip_contract_identity_terms(query, profile)
        query_vec = self.embedder.encode([retrieval_query], normalize_embeddings=True)[0].tolist()

        # Determine which channels are enabled (agent can disable some for speed)
        agent_enabled_channels = profile.get("agent_enabled_channels")
        def _channel_enabled(channel: str) -> bool:
            if agent_enabled_channels is None:
                return True
            return channel in agent_enabled_channels

        # Multi-Query: use query aliases for parallel vector search
        aliases = _contract_query_aliases(query, profile)

        # Optional: knowledge graph query expansion
        # Disabled by default, controlled via ENABLE_KNOWLEDGE_GRAPH env var
        # Graph expansion failures never affect retrieval results
        try:
            from knowledge.graph import GRAPH_ENABLED, graph_expand_query, get_global_graph
            if GRAPH_ENABLED:
                graph = get_global_graph()
                if graph.entity_count > 0:
                    graph_expansions = graph_expand_query(query, graph, max_expansions=3)
                    if graph_expansions:
                        logger.info("Knowledge graph expanded query with %d terms: %s",
                                    len(graph_expansions), graph_expansions[:3])
                        aliases = list(aliases) + graph_expansions
                        profile["graph_expansions"] = graph_expansions
        except Exception as graph_exc:
            logger.debug("Knowledge graph query expansion failed (using original query): %s", graph_exc)

        unique_aliases = [a for a in aliases if a != retrieval_query][: recall.alias_limit]
        alias_vecs: List[List[float]] = []
        if unique_aliases and float(profile.get("vec_weight", 1.0)) >= 1.0:
            alias_vecs = self.embedder.encode(unique_aliases, normalize_embeddings=True).tolist()

        # HyDE's remote call runs beside the normal recall paths below rather
        # than delaying their first result.
        agent_use_hyde = profile.get("agent_use_hyde")
        hyde_condition = (
            self.hyde_llm_call
            and (
                agent_use_hyde is True
                or (
                    agent_use_hyde is None
                    and profile["query_type"] == "article_lookup"
                    and len(query) < 25
                )
            )
        )
        def _run_vec_search(vec: List[float], label: str):
            results = self._vector_search(vec, recall.vector)
            for hit in results:
                hit["channels"] = list(hit.get("channels", [])) + [label]
            return results

        def _run_hyde_search() -> Tuple[str, List[dict]]:
            hyde_answer = _generate_hypothetical_answer(query, self.hyde_llm_call)
            if not hyde_answer:
                return "", []
            hyde_vec = self.embedder.encode([hyde_answer], normalize_embeddings=True)[0].tolist()
            return hyde_answer, _run_vec_search(hyde_vec, "hyde")

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            # Submit only enabled channels (agent can disable channels for speed)
            vec_future = executor.submit(_run_vec_search, query_vec, "vector") if _channel_enabled("vector") else None
            structure_future = executor.submit(self._structure_search, query, profile, recall.structure) if _channel_enabled("structure") else None
            article_future = executor.submit(self._article_search, query, profile, recall.article) if _channel_enabled("article") else None
            kw_future = executor.submit(self._keyword_search, query, profile, recall.keyword) if _channel_enabled("keyword") else None
            heading_future = executor.submit(self._heading_search, query, profile, recall.heading) if _channel_enabled("heading") else None
            table_future = executor.submit(self._table_search, query, profile, recall.table) if _channel_enabled("table") else None
            target_doc_future = executor.submit(self._target_document_search, query, profile, recall.target_document) if _channel_enabled("target_document") else None
            hyde_future = executor.submit(_run_hyde_search) if hyde_condition and _channel_enabled("vector") else None

            # Alias vector searches only if vector channel is enabled
            alias_futures = {}
            if _channel_enabled("vector"):
                alias_futures = {
                    label: executor.submit(_run_vec_search, vec, f"alias_{idx}")
                    for idx, (label, vec) in enumerate(zip(unique_aliases, alias_vecs))
                }

            recall_groups = []
            if vec_future is not None:
                recall_groups.append(("vector", vec_future.result(), profile["vec_weight"]))
            if structure_future is not None:
                recall_groups.append(("structure", structure_future.result(), profile["structure_weight"]))
            if article_future is not None:
                recall_groups.append(("article", article_future.result(), profile["article_weight"]))
            if kw_future is not None:
                recall_groups.append(("keyword", kw_future.result(), profile["kw_weight"]))
            if heading_future is not None:
                recall_groups.append(("heading", heading_future.result(), profile["heading_weight"]))
            if table_future is not None:
                recall_groups.append(("table", table_future.result(), profile["table_weight"]))
            if target_doc_future is not None:
                recall_groups.append(("target_document", target_doc_future.result(), max(profile["kw_weight"], profile["heading_weight"], 1.0) * 1.15))

            for label, future in alias_futures.items():
                recall_groups.append((label, future.result(), profile["vec_weight"] * 0.5))
            if hyde_future is not None:
                hyde_answer, hyde_results = hyde_future.result()
                if hyde_results:
                    recall_groups.append(("hyde", hyde_results, profile["vec_weight"] * 0.7))
                    profile["hyde_answer"] = hyde_answer

        fused = self._rrf_fusion(recall_groups, profile)
        if not fused:
            return [], profile

        candidate_window = max(recall.candidate_min, min(recall.candidate_max, len(fused)))
        candidate_ids = [hit["chunk_id"] for hit in fused[:candidate_window]]
        metadata_map = self._get_metadata_batch(candidate_ids)
        for hit in fused[:candidate_window]:
            hit["metadata"] = metadata_map.get(hit["chunk_id"], {})
        fused, candidate_window = self._apply_target_document_candidate_policy(
            fused, profile, contract, recall, candidate_window,
        )

        # Rerank: skip if agent explicitly disables it (for speed)
        agent_use_rerank = profile.get("agent_use_rerank")
        if self.reranker and agent_use_rerank is not False:
            pairs = [
                (query, self._compose_rerank_text(hit.get("metadata", {}), hit.get("content", "")))
                for hit in fused[: recall.rerank_window]
            ]
            scores = self.reranker.predict(pairs, batch_size=16)
            for index, hit in enumerate(fused[: recall.rerank_window]):
                hit["rerank_score"] = float(scores[index])

        # Only score the top window needed downstream (evidence building uses fused[:30]);
        # remaining candidates keep rrf_score as fallback and sort below scored ones.
        score_window = min(candidate_window, max(recall.rerank_window + 10, 40))
        for hit in fused[:score_window]:
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
                    include_headings=profile["query_type"] in {"article_lookup", "procedure"},
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
        ranking_score = round(float(hit.get("final_score", hit.get("rrf_score", 0.0)) or 0.0), 6)
        display_score = round(calibrate_ranking_score(ranking_score), 6)
        score_components = {
            "ranking": ranking_score,
            "rrf": round(float(hit.get("rrf_score", 0.0) or 0.0), 6),
            "display": display_score,
        }
        if "rerank_score" in hit:
            score_components["rerank"] = round(float(hit.get("rerank_score", 0.0) or 0.0), 6)
        return {
            "content": content,
            "doc_name": metadata.get("doc_name", ""),
            "heading": heading,
            "parent_heading": metadata.get("parent_heading", ""),
            "score": display_score,
            "display_score": display_score,
            "ranking_score": ranking_score,
            "rrf_score": score_components["rrf"],
            "rerank_score": score_components.get("rerank", 0.0),
            "score_components": score_components,
            "channel_scores": hit.get("channel_scores", {}),
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

            contract = build_retrieval_contract(query)
            if contract.block_reason:
                return []

            direct_evidence = self._document_applicability_evidence(query)
            if direct_evidence:
                contracted_direct = _contracted_sources_or_none(contract, [direct_evidence])
                if contracted_direct:
                    return contracted_direct

            structural_profile, contract = self._build_retrieval_profile(query, contract)
            structural_records = load_all_structural_records()
            precision_hints = structural_profile.get("precision_hints") or {}
            if not _has_structural_subject_support(precision_hints, structural_records):
                structural_supported = False
            else:
                structural_supported = True
            structural_count_answer = answer_structural_count(query, structural_records)
            if structural_supported and structural_count_answer and isinstance(structural_count_answer.get("sources"), list):
                contracted_structural = _contracted_sources_or_none(contract, structural_count_answer["sources"])
                if contracted_structural:
                    return contracted_structural
            structural_classification_answer = answer_structural_classification(query, structural_records)
            if structural_supported and structural_classification_answer and isinstance(structural_classification_answer.get("sources"), list):
                contracted_structural = _contracted_sources_or_none(contract, structural_classification_answer["sources"])
                if contracted_structural:
                    return contracted_structural
            structural_definition_answer = answer_structural_definition(query, structural_records)
            if structural_supported and structural_definition_answer and isinstance(structural_definition_answer.get("sources"), list):
                contracted_structural = _contracted_sources_or_none(contract, structural_definition_answer["sources"])
                if contracted_structural:
                    return contracted_structural
            query_signature = structural_profile.get("query_signature", {})
            if structural_supported and query_signature.get("article_id_variants") and structural_profile.get("target_doc_name"):
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
            if structural_supported and preferred_articles and precision_hints.get("target_doc_name"):
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

            structural_hits = self._search_structural_records(query, structural_profile, top_k=max(QA_TOP_K, 6)) if structural_supported else []
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

            self._prefetch_evidence_context(
                fused[:30],
                structural_depth=structural_depth,
                following_depth=max(int(profile.get("expand_children", 4)) * 3, 6),
            )

            results: List[Tuple[float, dict, bool]] = []
            seen_pages = set()
            _pre_seen_chunks = set()
            article_variants = set(profile.get("query_signature", {}).get("article_id_variants", []))
            for rank, hit in enumerate(fused[:30]):
                # Lightweight pre-dedup by chunk_id to avoid expensive evidence building for duplicates
                _cid = hit.get("chunk_id", "")
                if _cid and _cid in _pre_seen_chunks:
                    continue
                if _cid:
                    _pre_seen_chunks.add(_cid)
                try:
                    evidence = self._build_evidence_item(
                        hit,
                        query,
                        profile,
                        structural_depth=structural_depth,
                        allow_pdf_repair=rank < 6 or not hit.get("metadata", {}).get("bbox"),
                    )
                except Exception as exc:
                    logger.warning("构建证据对象失败 chunk_id=%s: %s", hit.get("chunk_id"), exc)
                    continue
                alignment = _query_alignment_score(_strip_contract_identity_terms(query, profile), evidence, profile)
                evidence_content = evidence.get("content", "")
                _fmp = _front_matter_penalty(evidence, evidence_content, profile)
                evidence["_front_matter_penalty_cached"] = _fmp
                evidence_quality = -_fmp
                if _looks_like_reference_material(evidence, evidence_content):
                    evidence_quality -= 0.72
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
                evidence_rank_score = float(evidence.get("ranking_score", evidence.get("score", 0.0)) or 0.0)
                rank_score = evidence_rank_score + 0.34 * alignment + 0.42 * evidence_quality + (0.42 if is_exact_article else 0.0) + article_heading_bonus
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
            if profile.get("query_type") != "article_lookup":
                body_results = [
                    item
                    for item in results
                    if item[1].get("_front_matter_penalty_cached", 0.0) < 0.45
                ]
                if len(body_results) >= min(QA_TOP_K, 2):
                    results = body_results
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
                or structure_route in structure_priority_routes
            )
            if structural_hits and prefer_structural_hits:
                for row in structural_hits:
                    if structural_focus_article_id and str(row.get("article_id") or "") != structural_focus_article_id:
                        continue
                    if len(merged) >= min(3 if structure_route in structure_priority_routes else 2, QA_TOP_K):
                        break
                    merged.append(row)

            if (contract.allow_multi_doc or _contract_needs_multi_target_recall(contract)) and contract.synced_target_doc_names:
                represented_docs = {row.get("doc_name", "") for row in merged}
                for target_doc_name in contract.synced_target_doc_names:
                    if target_doc_name in represented_docs:
                        continue
                    target_row = next((row for row in structural_hits if row.get("doc_name") == target_doc_name), None)
                    if target_row is None:
                        target_row = next((row for _, row, _ in results if row.get("doc_name") == target_doc_name), None)
                    if target_row is None:
                        continue
                    merged.append(target_row)
                    represented_docs.add(target_doc_name)
                    if len(merged) >= QA_TOP_K:
                        break

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
            contracted = apply_contract(contract, merged, keep_failed_when_no_hard_constraints=True)
            fallback_pools = list(merged) + structural_hits + [row for _, row, _ in results]
            contracted = self._ensure_multi_target_source_coverage(query, profile, contract, contracted, fallback_pools)
            return contracted[:QA_TOP_K]
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

