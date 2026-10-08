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
    HEALTH_REPORT_FILE,
    MAX_CHUNK_OVERLAP,
    MAX_CHUNK_SIZE,
    MILVUS_CONTENT_MAX,
    _build_structural_metadata,
    _compact_match_text,
    _compact_section_label,
    _detect_chunk_role,
    _extract_highlight_text,
    _looks_like_toc_or_index_listing,
    _split_heading_from_body,
    _split_structured_text,
    _split_text_block_by_headings,
    clean_block_text,
    detect_heading,
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
def _normalize_soft_hyphen_linebreaks(text: str) -> str:
    value = str(text or "").replace("\u00ad", "")
    value = re.sub(r"([A-Za-z])-\s+([A-Za-z])", r"\1\2", value)
    return clean_block_text(value)


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
    body = _normalize_soft_hyphen_linebreaks(re.sub(r"^\[SECTION\]\s*", "", text or ""))
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

    heading_patterns = (
        r"\b(?:PART|CHAPTER|SECTION|APPENDIX)\s+[A-Z0-9IVXLC]+(?:[-.]\s*|\s+)[A-Z][A-Z0-9 /&'(),-]{3,90}",
        r"\b(?:Part|Chapter|Section|Appendix)\s+[A-Z0-9IVXLC]+(?:[-.]\s*|\s+)[A-Z][A-Za-z0-9 /&'(),-]{3,90}",
        r"\b[A-Z][A-Z0-9 /&'(),-]{6,70}\b(?=\s+[A-Z][a-z])",
        r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,8}\.",
    )
    for pattern in heading_patterns:
        for match in re.finditer(pattern, body):
            push(match.group(0).strip(" .:-"))
            if len(candidates) >= 12:
                break
        if len(candidates) >= 12:
            break

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


def _section_title_candidate_score(value: str, metadata: dict, content: str) -> float:
    label = _compact_section_label(_restore_article_reference(value, metadata))
    if not label:
        return -10.0

    compact = _compact_match_text(label)
    if not compact:
        return -10.0

    score = 0.0
    if label == clean_block_text(metadata.get("section_title", "")):
        score += 0.55
    if label == clean_block_text(metadata.get("heading_text", "")):
        score += 0.45
    if re.search(r"第\s*[0-9A-Za-z一二三四五六七八九十百千万.]+\s*[章节条]", label):
        score += 1.3
    if re.search(r"\b(?:part|chapter|section|appendix)\s+[a-z0-9ivxlc]+", label, re.I):
        score += 1.0
    if re.match(r"^[0-9A-Za-z一二三四五六七八九十百千万]+(?:[-.、．]\d+)*[.、．)]?\s+\S+", label):
        score += 0.55
    if 14 <= len(label) <= 120:
        score += 0.35
    elif len(label) < 8:
        score -= 1.6
    elif len(label) > 150:
        score -= 0.45

    if _looks_like_toc_or_index_listing(label):
        score -= 2.0
    if re.search(r"[-‐‑‒–—_]{4,}", label):
        score -= 0.65
    if re.search(r"[A-Za-z]{1,8}\s*[\u00ad-]$", label):
        score -= 1.4
    if re.match(r"^[a-z]{1,3}\s+[a-z]{1,8}[\u00ad-]?$", label):
        score -= 1.7
    if compact in _compact_match_text(Path(str(metadata.get("doc_name") or "")).stem):
        score -= 1.0
    if label and label[0].islower() and not re.search(r"\b(?:part|chapter|section|appendix)\b", label, re.I):
        score -= 0.45

    content_compact = _compact_match_text(content[:900])
    if compact and compact in content_compact:
        score += 0.15
    return round(score, 6)


def _derive_section_title(metadata: dict, content: str) -> str:
    scored: List[Tuple[float, str]] = []
    for value in (
        metadata.get("section_title", ""),
        metadata.get("heading_text", ""),
        metadata.get("highlight_text", ""),
        metadata.get("article_label", ""),
        metadata.get("clause_path", ""),
    ):
        label = _compact_section_label(_restore_article_reference(value, metadata))
        if label:
            scored.append((_section_title_candidate_score(label, metadata, content), label))

    for candidate in _split_text_candidates(content):
        label = _compact_section_label(_restore_article_reference(candidate, metadata))
        if label:
            scored.append((_section_title_candidate_score(label, metadata, content), label))

    if scored:
        scored.sort(key=lambda item: item[0], reverse=True)
        if scored[0][0] >= 0.2:
            return scored[0][1]

    return _extract_highlight_text(content)


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
    # Try MinerU first for enhanced layout analysis and table recognition
    # Falls back to PyMuPDF if MinerU is unavailable or fails
    from search.document_parser_mineru import parse_pdf_with_mineru, is_mineru_available
    if is_mineru_available():
        mineru_result = parse_pdf_with_mineru(file_path, doc_name)
        if mineru_result is not None:
            units, health = mineru_result
            logger.info("Parsed %s with MinerU: %d units, %d tables, confidence=%.2f",
                        doc_name, len(units), health.get("table_count", 0), health.get("mineru_confidence", 0))
            return units, health
        logger.info("MinerU unavailable or failed for %s, using PyMuPDF parser", doc_name)

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
        "parser_used": "pymupdf",
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

