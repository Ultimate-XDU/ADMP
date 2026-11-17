"""Pilot adapter for the frozen work/corpus manifest and units file."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

from knowledge.graph_rag.config import corpus_dir


def _read_jsonl(path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


@lru_cache(maxsize=1)
def manifest_rows() -> tuple[dict, ...]:
    path = corpus_dir() / "manifest.json"
    if not path.exists():
        return tuple()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return tuple(payload)


@lru_cache(maxsize=1)
def units() -> tuple[dict, ...]:
    return tuple(_read_jsonl(corpus_dir() / "units.jsonl"))


def manifest_by_id() -> dict[str, dict]:
    return {row["doc_id"]: row for row in manifest_rows()}


def corpus_ids() -> list[str]:
    return [row["doc_id"] for row in manifest_rows()]


def available_count(doc_id: str) -> int:
    return sum(1 for unit in units() if unit.get("doc_id") == doc_id and str(unit.get("text") or "").strip())


# Pilot builds may sample a long document. The cap is a length rule, not a document id.
PILOT_LONG_DOCUMENT_UNITS = 12

_BIBLIOGRAPHY_HEADINGS = {"参考文献", "参考资料", "references", "bibliography", "reference"}


def _bibliography_unit(unit: dict) -> bool:
    """True when the unit is a reference list rather than a technical passage."""
    section = str(unit.get("section") or "").strip().lower()
    if section in _BIBLIOGRAPHY_HEADINGS:
        return True
    for line in str(unit.get("text") or "").splitlines():
        first = line.strip().lower()
        if not first:
            continue
        return first in _BIBLIOGRAPHY_HEADINGS or first.startswith("参考文献")
    return False


def choose_units(rows: list[dict], scope: str) -> list[dict]:
    """Pick units for one document.

    ``all`` keeps every non-empty unit. ``smoke`` keeps two. ``pilot`` drops
    bibliography units and, only when a document is longer than the cap,
    samples that document. Sampling is reported separately and is not full
    graph coverage.
    """
    body = [unit for unit in rows if str(unit.get("text") or "").strip()]
    if scope == "pilot":
        body = [unit for unit in body if not _bibliography_unit(unit)]
    if scope == "smoke":
        return body[:2]
    if scope != "pilot":
        return body
    if len(body) <= PILOT_LONG_DOCUMENT_UNITS:
        return body
    ordered = sorted(body, key=lambda unit: int(unit.get("page") or 0))
    pool = [unit for unit in ordered if len(str(unit.get("text") or "")) >= 120] or ordered
    stride = max(1, len(pool) // PILOT_LONG_DOCUMENT_UNITS)
    return pool[::stride][:PILOT_LONG_DOCUMENT_UNITS]


def select_units(doc_id: str, scope: str) -> list[dict]:
    rows = [unit for unit in units() if unit.get("doc_id") == doc_id]
    return choose_units(rows, scope)


def selection_report(doc_id: str, scope: str) -> dict:
    rows = [unit for unit in units() if unit.get("doc_id") == doc_id and str(unit.get("text") or "").strip()]
    selected = select_units(doc_id, scope)
    available = len(rows)
    kept = len(selected)
    return {
        "scope": scope,
        "available": available,
        "selected": kept,
        "sampled": kept < available,
        "fullText": scope == "all" and available > 0 and kept == available,
    }


def describe_coverage(scope: str, available: int, selected: int, extracted: int) -> dict:
    """``fullGraph`` is true only when this scope stored and extracted every parsed unit."""
    full_text = scope == "all" and available > 0 and selected >= available
    full_graph = full_text and extracted >= available
    return {
        "scope": scope,
        "availableUnits": int(available),
        "selectedUnits": int(selected),
        "extractedUnits": int(extracted),
        "fullText": bool(full_text),
        "fullGraph": bool(full_graph),
        "label": "full" if full_graph else "partial",
        "note": (
            "all parsed units were extracted"
            if full_graph
            else "partial graph coverage; not a full knowledge graph"
        ),
    }


def unit_to_chunk(unit: dict) -> dict:
    meta = manifest_by_id().get(unit.get("doc_id"), {})
    page = unit.get("page")
    try:
        page_number = int(page)
    except (TypeError, ValueError):
        page_number = 1
    if page_number < 1:
        page_number = 1
    section = str(unit.get("section") or "")
    title = str(meta.get("title") or unit.get("doc_id") or "")
    try:
        page_count = int(meta.get("page_count") or meta.get("section_count") or 1)
    except (TypeError, ValueError):
        page_count = 1
    file_name = str(meta.get("file") or "")
    return {
        "chunk_id": str(unit.get("unit_id") or ""),
        "doc_id": str(unit.get("doc_id") or ""),
        "content": str(unit.get("text") or ""),
        "doc_name": title,
        "page_number": page_number,
        "page_count": max(page_count, 1),
        "section_title": section,
        "heading": section or title,
        "parent_heading": title,
        "source_url": str(unit.get("citation_url") or meta.get("source_url") or ""),
        "source_type": str(unit.get("source_type") or "text"),
        "highlight_text": "",
        "publisher": str(meta.get("publisher") or ""),
        "language": str(meta.get("language") or ""),
        "format": str(meta.get("format") or ""),
        "file_stem": Path(file_name).stem if file_name else "",
    }


def chunks_for(doc_id: str, scope: str) -> list[dict]:
    return [unit_to_chunk(unit) for unit in select_units(doc_id, scope) if unit.get("unit_id")]


def is_corpus(doc_id: str) -> bool:
    return doc_id in manifest_by_id()


def clear_cache() -> None:
    manifest_rows.cache_clear()
    units.cache_clear()
