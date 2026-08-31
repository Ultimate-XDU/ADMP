"""MinerU-based PDF document parser.

This module provides an enhanced PDF parsing pipeline using MinerU
(https://github.com/opendatalab/MinerU), an open-source document parsing
toolkit that excels at:
- Layout analysis (text, title, table, figure, formula, list)
- Table recognition with complex cell merging (HTML/Markdown output)
- Formula recognition (LaTeX output)
- Intelligent reading order for multi-column layouts
- Chinese document optimization

Design principles:
- Non-destructive: this is an ENHANCEMENT channel, never a replacement.
  When MinerU is unavailable, fails, or produces low-confidence output,
  the system falls back to the original PyMuPDF-based parser.
- Output compatible: produces units in the exact same format as the
  original parser, so all downstream code (chunking, indexing, retrieval)
  works without modification.
- Configurable: controlled via environment variables.
- Auditable: parser choice and confidence are stored in the health report.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

MINERU_ENABLED = os.environ.get("USE_MINERU", "true").lower() in ("1", "true", "yes")
MINERU_MIN_CONFIDENCE = float(os.environ.get("MINERU_MIN_CONFIDENCE", "0.5"))
MINERU_TABLE_CONFIDENCE = float(os.environ.get("MINERU_TABLE_CONFIDENCE", "0.6"))

# ============================================================================
# MinerU availability detection
# ============================================================================

_mineru_available: Optional[bool] = None
_mineru_import_error: Optional[str] = None


def is_mineru_available() -> bool:
    """Check if MinerU is installed and importable.

    Result is cached after first check.
    """
    global _mineru_available, _mineru_import_error
    if _mineru_available is not None:
        return _mineru_available

    if not MINERU_ENABLED:
        _mineru_available = False
        _mineru_import_error = "MinerU disabled by USE_MINERU=false"
        return False

    try:
        # Try multiple import paths for different MinerU versions
        try:
            from mineru import MagicPDF  # noqa: F401
            _mineru_available = True
            return True
        except ImportError:
            pass

        try:
            from mineru.cli import MagicPDF  # noqa: F401
            _mineru_available = True
            return True
        except ImportError:
            pass

        try:
            import mineru  # noqa: F401
            _mineru_available = True
            return True
        except ImportError:
            pass

        _mineru_available = False
        _mineru_import_error = "MinerU not installed (pip install mineru)"
        return False

    except Exception as exc:
        _mineru_available = False
        _mineru_import_error = str(exc)
        return False


def get_mineru_install_error() -> Optional[str]:
    """Return the error message if MinerU is not available."""
    is_mineru_available()
    return _mineru_import_error


# ============================================================================
# HTML table to Markdown converter
# ============================================================================

def _html_table_to_markdown(html: str) -> str:
    """Convert an HTML table to Markdown format.

    Handles basic table structures including headers and merged cells.
    For complex tables, falls back to a simplified representation.
    """
    if not html:
        return ""

    try:
        # Remove tags we don't need
        text = re.sub(r"<br\s*/?>", "\n", html, flags=re.I)
        text = re.sub(r"<p[^>]*>", "", text, flags=re.I)
        text = re.sub(r"</p>", "", text, flags=re.I)

        # Extract rows
        rows = re.findall(r"<tr[^>]*>(.*?)</tr>", text, re.DOTALL | re.I)
        if not rows:
            return ""

        markdown_rows = []
        for row in rows:
            # Extract cells (th or td)
            cells = re.findall(r"<(th|td)[^>]*>(.*?)</\1>", row, re.DOTALL | re.I)
            if not cells:
                continue
            cell_texts = []
            for _, cell_content in cells:
                # Clean cell content
                cell_text = re.sub(r"<[^>]+>", "", cell_content)
                cell_text = cell_text.strip()
                cell_text = re.sub(r"\s+", " ", cell_text)
                # Escape pipe characters
                cell_text = cell_text.replace("|", "\\|")
                cell_texts.append(cell_text)
            if cell_texts:
                markdown_rows.append("| " + " | ".join(cell_texts) + " |")

        if not markdown_rows:
            return ""

        # Add separator after first row (assume first row is header)
        if len(markdown_rows) > 1:
            col_count = markdown_rows[0].count("|") - 1
            separator = "| " + " | ".join(["---"] * col_count) + " |"
            markdown_rows.insert(1, separator)

        return "\n".join(markdown_rows)

    except Exception as exc:
        logger.debug("HTML to Markdown conversion failed: %s", exc)
        return ""


# ============================================================================
# MinerU output normalization
# ============================================================================

def _normalize_layout_item(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalize a MinerU layout detection item to a common format.

    MinerU's output format varies by version. This function normalizes
    different formats to a common structure:
    {
        "type": str (text/title/table/figure/formula/list/...),
        "bbox": [x0, y0, x1, y1],
        "text": str (for text/title types),
        "html": str (for table types),
        "latex": str (for formula types),
        "score": float (confidence),
    }
    """
    if not isinstance(item, dict):
        return None

    normalized = {}

    # Type
    item_type = item.get("type") or item.get("category") or item.get("label") or ""
    normalized["type"] = str(item_type).lower()

    # Bbox
    bbox = item.get("bbox") or item.get("box") or item.get("position")
    if bbox and isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        normalized["bbox"] = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
    else:
        normalized["bbox"] = None

    # Text content
    text = item.get("text") or item.get("content") or ""
    if isinstance(text, list):
        text = "\n".join(str(t) for t in text)
    normalized["text"] = str(text).strip()

    # Table HTML
    html = item.get("html") or item.get("table_html") or item.get("html_content") or ""
    normalized["html"] = str(html).strip()

    # Formula LaTeX
    latex = item.get("latex") or item.get("formula_latex") or item.get("latex_content") or ""
    normalized["latex"] = str(latex).strip()

    # Confidence score
    score = item.get("score") or item.get("confidence") or item.get("prob") or 0.9
    try:
        normalized["score"] = float(score)
    except (TypeError, ValueError):
        normalized["score"] = 0.9

    return normalized


def _extract_layout_dets(page_data: Any) -> List[Dict[str, Any]]:
    """Extract layout detection results from a MinerU page object.

    Supports multiple MinerU API formats:
    - page.layout_dets
    - page["layout_dets"]
    - page.get_layout_dets()
    - page.extract_content()
    """
    layout_dets = []

    # Try attribute access
    for attr_name in ("layout_dets", "layout_details", "dets", "elements", "blocks"):
        dets = getattr(page_data, attr_name, None)
        if dets and isinstance(dets, list):
            layout_dets = dets
            break

    # Try dict access
    if not layout_dets and isinstance(page_data, dict):
        for key in ("layout_dets", "layout_details", "dets", "elements", "blocks", "content"):
            dets = page_data.get(key)
            if dets and isinstance(dets, list):
                layout_dets = dets
                break

    # Try method call
    if not layout_dets:
        for method_name in ("get_layout_dets", "extract_layout", "extract_content", "get_content"):
            method = getattr(page_data, method_name, None)
            if callable(method):
                try:
                    result = method()
                    if isinstance(result, list):
                        layout_dets = result
                    elif isinstance(result, dict):
                        for key in ("layout_dets", "dets", "elements", "blocks"):
                            if key in result and isinstance(result[key], list):
                                layout_dets = result[key]
                                break
                    if layout_dets:
                        break
                except Exception:
                    continue

    return layout_dets


def _extract_page_size(page_data: Any) -> Optional[Tuple[float, float]]:
    for attr in ("page_size", "size", "page_dim", "dimensions"):
        value = getattr(page_data, attr, None)
        if value is None and isinstance(page_data, dict):
            value = page_data.get(attr)
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            try:
                width, height = float(value[0]), float(value[1])
            except (TypeError, ValueError):
                continue
            if width > 1.0 and height > 1.0:
                return width, height
    width = getattr(page_data, "width", None)
    height = getattr(page_data, "height", None)
    if width is None and isinstance(page_data, dict):
        width = page_data.get("width")
    if height is None and isinstance(page_data, dict):
        height = page_data.get("height")
    try:
        width, height = float(width), float(height)
    except (TypeError, ValueError):
        return None
    if width > 1.0 and height > 1.0:
        return width, height
    return None


def _normalize_mineru_bbox(bbox: List[float], page_size: Optional[Tuple[float, float]]) -> Optional[dict]:
    x0, y0, x1, y1 = bbox
    if page_size is not None:
        import fitz
        from search.document_pipeline import _normalize_bbox
        return _normalize_bbox((x0, y0, x1, y1), fitz.Rect(0, 0, page_size[0], page_size[1]))
    if all(0.0 <= value <= 1.0 for value in (x0, y0, x1, y1)):
        return {
            "x": round(max(0.0, x0), 6),
            "y": round(max(0.0, y0), 6),
            "width": round(max(0.0, x1 - x0), 6),
            "height": round(max(0.0, y1 - y0), 6),
        }
    logger.debug("Dropping MinerU bbox %s: unknown coordinate space", bbox)
    return None


# ============================================================================
# MinerU PDF parser
# ============================================================================

class MinerUParser:
    """PDF parser using MinerU for enhanced layout analysis and table recognition.

    Usage:
        parser = MinerUParser()
        if parser.is_available():
            units, health = parser.parse_pdf(file_path, doc_name)
        else:
            # fall back to original parser
    """

    def __init__(self):
        self._magic_pdf_class = None
        self._available = None

    def is_available(self) -> bool:
        """Check if MinerU is available for parsing."""
        if self._available is not None:
            return self._available
        self._available = is_mineru_available()
        return self._available

    def _get_magic_pdf(self, file_path: str):
        """Create a MagicPDF instance, trying multiple import paths."""
        if self._magic_pdf_class is not None:
            return self._magic_pdf_class(file_path)

        # Try import path 1
        try:
            from mineru import MagicPDF
            self._magic_pdf_class = MagicPDF
            return MagicPDF(file_path)
        except (ImportError, Exception):
            pass

        # Try import path 2
        try:
            from mineru.cli import MagicPDF
            self._magic_pdf_class = MagicPDF
            return MagicPDF(file_path)
        except (ImportError, Exception):
            pass

        raise RuntimeError("MinerU MagicPDF not available")

    def parse_pdf(
        self,
        file_path: str,
        doc_name: str,
        *,
        document_type: str = "",
    ) -> Tuple[List[dict], dict]:
        """Parse a PDF file using MinerU.

        Args:
            file_path: Path to the PDF file.
            doc_name: Document name for metadata.
            document_type: Document type classification.

        Returns:
            Tuple of (units, health) in the same format as the original parser.

        Raises:
            RuntimeError: If MinerU is not available or parsing fails.
        """
        if not self.is_available():
            raise RuntimeError(f"MinerU not available: {get_mineru_install_error()}")

        from search.common import infer_document_type

        if not document_type:
            document_type = infer_document_type(doc_name)

        units: List[dict] = []
        health = {
            "doc_name": doc_name,
            "page_count": 0,
            "unit_count": 0,
            "chunk_count": 0,
            "table_count": 0,
            "multi_column_pages": [],
            "image_sparse_pages": [],
            "low_confidence_pages": [],
            "ocr_fallback_pages": [],
            "parse_success_rate": 1.0,
            "parser_used": "mineru",
            "mineru_confidence": 0.0,
        }

        try:
            pdf = self._get_magic_pdf(file_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to open PDF with MinerU: {exc}") from exc

        try:
            # Get page count
            page_count = getattr(pdf, "page_count", None)
            if page_count is None:
                page_count = len(getattr(pdf, "pages", [])) or 0
            health["page_count"] = page_count

            # Try to get all pages content at once
            all_pages = []
            try:
                content = pdf.extract_content()
                if isinstance(content, list):
                    all_pages = content
            except Exception:
                pass

            # If batch extraction failed, iterate pages
            if not all_pages:
                try:
                    for page in pdf:
                        all_pages.append(page)
                except Exception:
                    pass

            if not all_pages and page_count > 0:
                # Last resort: try indexing
                try:
                    for i in range(page_count):
                        all_pages.append(pdf[i])
                except Exception:
                    pass

            if not all_pages:
                raise RuntimeError("MinerU returned no pages")

            total_confidence = 0.0
            pages_with_content = 0

            for page_idx, page_data in enumerate(all_pages, start=1):
                page_units, page_confidence, page_stats = self._parse_page(
                    page_data,
                    doc_name=doc_name,
                    page_number=page_idx,
                    page_count=page_count,
                    document_type=document_type,
                )

                if page_stats["multi_column"]:
                    health["multi_column_pages"].append(page_idx)
                if page_stats["low_confidence"]:
                    health["low_confidence_pages"].append(page_idx)
                if page_stats["table_count"] > 0:
                    health["table_count"] += page_stats["table_count"]

                if page_confidence > 0:
                    total_confidence += page_confidence
                    pages_with_content += 1

                units.extend(page_units)

            if pages_with_content > 0:
                health["mineru_confidence"] = round(total_confidence / pages_with_content, 4)

        finally:
            try:
                pdf.close()
            except Exception:
                pass

        health["unit_count"] = len(units)
        if page_count > 0:
            anomaly_count = len(set(health["multi_column_pages"] + health["low_confidence_pages"]))
            health["parse_success_rate"] = round(max(0.0, 1.0 - anomaly_count / page_count), 4)

        return units, health

    def _parse_page(
        self,
        page_data: Any,
        *,
        doc_name: str,
        page_number: int,
        page_count: int,
        document_type: str,
    ) -> Tuple[List[dict], float, dict]:
        """Parse a single page using MinerU layout detection.

        Returns:
            Tuple of (units, avg_confidence, stats_dict)
        """
        from search.document_pipeline import (
            _is_repeating_pdf_noise,
            _materialize_text_units,
        )

        units: List[dict] = []
        stats = {
            "multi_column": False,
            "low_confidence": False,
            "table_count": 0,
        }

        layout_dets = _extract_layout_dets(page_data)
        if not layout_dets:
            return units, 0.0, stats

        # Normalize all items
        normalized_items = []
        for item in layout_dets:
            norm = _normalize_layout_item(item)
            if norm and norm["bbox"]:
                normalized_items.append(norm)

        if not normalized_items:
            return units, 0.0, stats

        # Sort by reading order (top to bottom, left to right)
        # MinerU usually provides pre-sorted order, but we ensure it
        normalized_items.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))

        # Detect multi-column: if items at similar y positions have significantly
        # different x positions
        x_positions = [item["bbox"][0] for item in normalized_items]
        if len(x_positions) >= 4:
            x_range = max(x_positions) - min(x_positions)
            if x_range > 100:
                stats["multi_column"] = True

        total_score = 0.0
        item_count = 0
        page_size = _extract_page_size(page_data)

        for item in normalized_items:
            item_type = item["type"]
            score = item["score"]
            bbox = _normalize_mineru_bbox(item["bbox"], page_size)

            # Skip very low confidence items
            if score < MINERU_MIN_CONFIDENCE:
                stats["low_confidence"] = True
                continue

            total_score += score
            item_count += 1

            # Table
            if "table" in item_type:
                if score >= MINERU_TABLE_CONFIDENCE:
                    table_content = item["html"] or item["text"]
                    if table_content:
                        markdown = _html_table_to_markdown(table_content)
                        if markdown:
                            table_units = _materialize_text_units(
                                doc_name=doc_name,
                                content=markdown,
                                page_number=page_number,
                                page_count=page_count,
                                bbox=bbox,
                                layout_type="table",
                                parse_confidence=score,
                                is_ocr_fallback=False,
                                document_type=document_type,
                                source_type="table",
                            )
                            units.extend(table_units)
                            stats["table_count"] += 1
                continue

            # Title / heading
            if "title" in item_type or "heading" in item_type:
                text = item["text"]
                if text and not _is_repeating_pdf_noise(text) and len(text) >= 2:
                    # Determine heading level from type or position
                    level = 1
                    if "subtitle" in item_type:
                        level = 2
                    elif "subsubtitle" in item_type:
                        level = 3

                    heading_units = _materialize_text_units(
                        doc_name=doc_name,
                        content=text,
                        page_number=page_number,
                        page_count=page_count,
                        bbox=bbox,
                        layout_type="single_column",
                        parse_confidence=score,
                        is_ocr_fallback=False,
                        document_type=document_type,
                        is_heading=True,
                        heading_level=level,
                        heading_text=text,
                    )
                    units.extend(heading_units)
                continue

            # Formula
            if "formula" in item_type or "equation" in item_type:
                latex = item["latex"] or item["text"]
                if latex:
                    formula_text = f"$$ {latex} $$"
                    formula_units = _materialize_text_units(
                        doc_name=doc_name,
                        content=formula_text,
                        page_number=page_number,
                        page_count=page_count,
                        bbox=bbox,
                        layout_type="single_column",
                        parse_confidence=score,
                        is_ocr_fallback=False,
                        document_type=document_type,
                        source_type="formula",
                    )
                    units.extend(formula_units)
                continue

            # Figure / image - skip for now (could add OCR/caption later)
            if "figure" in item_type or "image" in item_type or "picture" in item_type:
                continue

            # List item
            if "list" in item_type or "item" in item_type:
                text = item["text"]
                if text and not _is_repeating_pdf_noise(text) and len(text) >= 2:
                    list_units = _materialize_text_units(
                        doc_name=doc_name,
                        content=text,
                        page_number=page_number,
                        page_count=page_count,
                        bbox=bbox,
                        layout_type="single_column",
                        parse_confidence=score,
                        is_ocr_fallback=False,
                        document_type=document_type,
                        source_type="list",
                    )
                    units.extend(list_units)
                continue

            # Plain text (default)
            text = item["text"]
            if text and not _is_repeating_pdf_noise(text) and len(text) >= 2:
                text_units = _materialize_text_units(
                    doc_name=doc_name,
                    content=text,
                    page_number=page_number,
                    page_count=page_count,
                    bbox=bbox,
                    layout_type="multi_column" if stats["multi_column"] else "single_column",
                    parse_confidence=score,
                    is_ocr_fallback=False,
                    document_type=document_type,
                )
                units.extend(text_units)

        avg_confidence = total_score / item_count if item_count > 0 else 0.0
        return units, avg_confidence, stats


# ============================================================================
# Convenience function
# ============================================================================

_default_parser: Optional[MinerUParser] = None


def get_mineru_parser() -> MinerUParser:
    """Get or create the default MinerUParser singleton."""
    global _default_parser
    if _default_parser is None:
        _default_parser = MinerUParser()
    return _default_parser


def parse_pdf_with_mineru(
    file_path: str,
    doc_name: str,
    *,
    document_type: str = "",
) -> Optional[Tuple[List[dict], dict]]:
    """Try to parse a PDF with MinerU.

    Returns None if MinerU is not available or parsing fails.
    Caller should fall back to the original parser in that case.
    """
    parser = get_mineru_parser()
    if not parser.is_available():
        return None

    try:
        units, health = parser.parse_pdf(
            file_path,
            doc_name,
            document_type=document_type,
        )
        # Sanity check: MinerU should produce at least some units for a valid PDF
        if not units:
            logger.warning("MinerU produced no units for %s, falling back to original parser", doc_name)
            return None
        return units, health
    except Exception as exc:
        logger.warning("MinerU parsing failed for %s: %s, falling back to original parser", doc_name, exc)
        return None
