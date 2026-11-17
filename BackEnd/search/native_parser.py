"""Native parsers for HTML, Markdown, text, CSV, JSON, XML, PPTX, and XLSX.

Units keep the visible body and a locator. Nothing is converted to PDF or DOCX.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from search.common import (
    _build_structural_metadata,
    _extract_highlight_text,
    clean_block_text,
    infer_document_type,
)

FORMAT_BY_SUFFIX = {
    ".pdf": "PDF",
    ".docx": "DOCX",
    ".pptx": "PPTX",
    ".xlsx": "XLSX",
    ".html": "HTML",
    ".htm": "HTML",
    ".md": "MD",
    ".markdown": "MD",
    ".txt": "TXT",
    ".csv": "CSV",
    ".json": "JSON",
    ".xml": "XML",
}

_RECORD_FIELDS = (
    "source_url",
    "document_title",
    "publisher",
    "published_at",
    "original_format",
    "document_key",
)


def document_type_for_name(file_name: str) -> str:
    return FORMAT_BY_SUFFIX.get(Path(str(file_name or "")).suffix.lower(), "")


def file_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize_document_key(
    explicit: str = "",
    source_url: str = "",
    publisher: str = "",
    published_at: str = "",
    content_sha256: str = "",
) -> str:
    """One identity at entry. An explicit key wins; otherwise source coordinates plus the content hash."""
    chosen = str(explicit or "").strip()
    if chosen:
        return chosen
    source_url = str(source_url or "").strip()
    publisher = str(publisher or "").strip()
    published_at = str(published_at or "").strip()
    content_sha256 = str(content_sha256 or "").strip()
    if not (source_url or publisher or published_at or content_sha256):
        return ""
    identity = "\n".join((source_url, publisher, published_at, content_sha256))
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]


def metadata_from_record(record: dict | None, file_name: str = "") -> dict:
    record = record or {}
    fallback = document_type_for_name(file_name or record.get("storageFileName") or record.get("fileName") or "")
    original = str(record.get("originalFormat") or "").strip() or fallback
    return {
        "source_url": str(record.get("sourceUrl") or "").strip(),
        "publisher": str(record.get("publisher") or "").strip(),
        "document_title": str(record.get("title") or "").strip(),
        "published_at": str(record.get("publishedAt") or "").strip(),
        "original_format": original,
        "document_key": str(record.get("documentKey") or "").strip(),
    }


def apply_record_metadata(units: list[dict], metadata: dict | None, fallback_format: str, native_title: str = "") -> list[dict]:
    metadata = metadata or {}
    title = str(metadata.get("document_title") or "").strip() or str(native_title or "").strip()
    for unit in units:
        for key in ("source_url", "publisher", "published_at", "document_key"):
            incoming = str(metadata.get(key) or "").strip()
            if incoming:
                unit[key] = incoming
            else:
                unit.setdefault(key, "")
        unit["original_format"] = str(metadata.get("original_format") or unit.get("original_format") or fallback_format or "")
        if title:
            unit["document_title"] = title
        else:
            unit.setdefault("document_title", "")
        unit.setdefault("section_path", [])
        unit.setdefault("locator", {"label": fallback_format or "document", "kind": (fallback_format or "document").lower()})
        if unit.get("role"):
            unit["chunk_role"] = unit["role"]
    return units


def provenance_from(block: dict) -> dict:
    return {
        "source_url": block.get("source_url") or "",
        "document_title": block.get("document_title") or "",
        "publisher": block.get("publisher") or "",
        "published_at": block.get("published_at") or "",
        "original_format": block.get("original_format") or "",
        "document_key": block.get("document_key") or "",
        "section_path": list(block.get("section_path") or []),
        "locator": dict(block.get("locator") or {}),
        "role": block.get("role") or "",
    }


def locators_can_merge(left: dict, right: dict) -> bool:
    if list(left.get("section_path") or []) != list(right.get("section_path") or []):
        return False
    a = left.get("locator") or {}
    b = right.get("locator") or {}
    if not a or not b:
        return True
    if a.get("kind") != b.get("kind"):
        return False
    kind = a.get("kind")
    if kind in {"json", "xml", "csv", "xlsx"}:
        return False
    if kind in {"pdf", "pptx"}:
        return int(a.get("page") or 0) == int(b.get("page") or 0)
    if kind == "html":
        return (a.get("anchor") or "") == (b.get("anchor") or "") and (a.get("path") or "") == (b.get("path") or "")
    if kind in {"txt", "md", "docx"}:
        return (a.get("path") or "") == (b.get("path") or "")
    return (a.get("label") or "") == (b.get("label") or "")


def merge_locators(locators: list[dict]) -> dict:
    rows = [dict(item) for item in locators if item]
    if not rows:
        return {}
    if len(rows) == 1 or any(item.get("kind") != rows[0].get("kind") for item in rows):
        return rows[0]
    kind = rows[0].get("kind")
    if kind not in {"txt", "md", "docx"}:
        return rows[0]
    start = min(int(item.get("rowStart") or 0) for item in rows)
    end = max(int(item.get("rowEnd") or 0) for item in rows)
    merged = dict(rows[0])
    merged["rowStart"] = start
    merged["rowEnd"] = end
    if kind == "docx":
        merged["label"] = f"段落 {start}" if start == end else f"段落 {start}-{end}"
    else:
        span = f"第{start}行" if start == end else f"第{start}-{end}行"
        heading = str(merged.get("path") or "")
        merged["label"] = f"{span} · {heading}" if heading else span
    return merged


def merge_provenance(blocks: list[dict]) -> dict:
    if not blocks:
        return provenance_from({})
    merged = provenance_from(blocks[0])
    for block in blocks[1:]:
        for key in _RECORD_FIELDS:
            if not merged.get(key) and block.get(key):
                merged[key] = block.get(key) or ""
    paths = [tuple(block.get("section_path") or []) for block in blocks]
    if len(set(paths)) == 1:
        merged["section_path"] = list(paths[0])
    merged["locator"] = merge_locators([block.get("locator") or {} for block in blocks])
    roles = {block.get("role") for block in blocks if block.get("role")}
    merged["role"] = next(iter(roles)) if len(roles) == 1 else ""
    return merged


def _local(tag) -> str:
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1]


def _line_label(start: int, end: int, heading: str = "") -> str:
    span = f"第{start}行" if start == end else f"第{start}-{end}行"
    return f"{span} · {heading}" if heading else span


def _make_unit(
    doc_name: str,
    content: str,
    *,
    original_format: str,
    kind: str,
    role: str,
    source_type: str,
    locator: dict,
    section_path: list[str] | None = None,
    page_number: int = 0,
    page_count: int = 0,
    heading_level: int = 0,
    section_title: str = "",
    raw: bool = False,
) -> dict | None:
    text = content if raw else clean_block_text(content)
    if not str(text or "").strip():
        return None
    section_path = list(section_path or [])
    heading_text = section_path[-1] if section_path else section_title
    if role == "heading":
        heading_text = text
        section_title = text
    structural = _build_structural_metadata(
        text,
        source_type=source_type,
        section_title=section_title,
        heading_text=heading_text,
    )
    unit = {
        "doc_name": doc_name,
        "content": text,
        "page_number": page_number,
        "page_count": page_count,
        "bbox": None,
        "heading_level": heading_level,
        "is_heading": role == "heading",
        "source_type": source_type,
        "layout_type": kind,
        "parse_confidence": 0.98,
        "is_table": source_type == "table",
        "is_ocr_fallback": False,
        "section_title": section_title,
        "heading_text": heading_text,
        "highlight_text": _extract_highlight_text(text, section_title or heading_text),
        "document_type": infer_document_type(doc_name),
        "original_format": original_format,
        "section_path": section_path,
        "locator": locator,
        "role": role,
        "source_url": "",
        "publisher": "",
        "document_title": "",
        "published_at": "",
        "document_key": "",
    }
    unit.update(structural)
    unit["role"] = role
    unit["chunk_role"] = role
    return unit


def _health(doc_name: str, units: list[dict], page_count: int = 0) -> dict:
    return {
        "doc_name": doc_name,
        "page_count": page_count,
        "unit_count": len(units),
        "chunk_count": 0,
        "table_count": sum(1 for unit in units if unit.get("source_type") == "table"),
        "multi_column_pages": [],
        "image_sparse_pages": [],
        "low_confidence_pages": [],
        "ocr_fallback_pages": [],
        "parse_success_rate": 1.0 if units else 0.0,
        "parser_used": "native",
        "native_title": "",
    }


def _read_text(file_path: str) -> str:
    return Path(file_path).read_text(encoding="utf-8-sig")


def _emit(units: list[dict], unit: dict | None) -> None:
    if unit is not None:
        units.append(unit)


def _parse_html(file_path: str, doc_name: str) -> tuple[list[dict], dict]:
    from bs4 import BeautifulSoup, Comment, NavigableString, Tag

    soup = BeautifulSoup(Path(file_path).read_bytes(), "html.parser")
    for tag in soup.find_all(["script", "style", "noscript"]):
        tag.decompose()
    for comment in list(soup.find_all(string=lambda text: isinstance(text, Comment))):
        comment.extract()
    title = clean_block_text(soup.title.get_text(" ", strip=True)) if soup.title else ""
    units: list[dict] = []
    stack: list[dict] = []

    def visible(node) -> str:
        if isinstance(node, NavigableString):
            return str(node)
        if not isinstance(node, Tag):
            return ""
        if node.name == "br":
            return "\n"
        if node.name == "img":
            return str(node.get("alt") or "")
        parts = [visible(child) for child in node.children]
        text = "".join(parts)
        if node.name == "a":
            href = str(node.get("href") or "")
            if href and href not in text:
                text = f"{text} ({href})" if text.strip() else href
        return text

    def anchor_of(node: Tag) -> str:
        current = node
        while isinstance(current, Tag):
            if current.get("id"):
                return str(current.get("id"))
            if current.name == "a" and current.get("name"):
                return str(current.get("name"))
            current = current.parent
        for item in reversed(stack):
            if item.get("anchor"):
                return item["anchor"]
        return ""

    def locator(node: Tag, path: list[str]) -> dict:
        anchor = anchor_of(node)
        label_path = " / ".join(path) if path else "HTML"
        label = f"{label_path} #{anchor}" if anchor else label_path
        return {
            "label": label,
            "kind": "html",
            "anchor": anchor,
            "path": label_path if path else "",
            "page": 0,
        }

    def add(node: Tag, content: str, role: str, source_type: str, level: int = 0) -> None:
        path = [item["text"] for item in stack]
        if role == "heading":
            path = path + [clean_block_text(content)]
        _emit(
            units,
            _make_unit(
                doc_name,
                content,
                original_format="HTML",
                kind="html",
                role=role,
                source_type=source_type,
                locator=locator(node, path),
                section_path=path,
                heading_level=level,
                section_title=path[-1] if path else "",
                raw=role == "code",
            ),
        )

    def walk(node) -> None:
        if isinstance(node, NavigableString) or not isinstance(node, Tag):
            return
        name = (node.name or "").lower()
        if name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            level = int(name[1])
            text = visible(node)
            while stack and stack[-1]["level"] >= level:
                stack.pop()
            add(node, text, "heading", "heading", level)
            stack.append({"level": level, "text": clean_block_text(text), "anchor": str(node.get("id") or "")})
            return
        if name == "table":
            rows = []
            for tr in node.find_all("tr"):
                cells = [clean_block_text(visible(cell)) for cell in tr.find_all(["th", "td"])]
                if any(cells):
                    rows.append(" | ".join(cells))
            add(node, "\n".join(rows), "table", "table")
            return
        if name == "figure":
            add(node, visible(node), "figure", "figure")
            return
        if name in {"p", "pre", "blockquote", "li", "figcaption", "dd", "dt"}:
            role = "code" if name == "pre" else "paragraph"
            source_type = "code" if name == "pre" else "text"
            add(node, visible(node), role, source_type)
            return
        direct = clean_block_text("".join(str(child) for child in node.children if isinstance(child, NavigableString)))
        if direct:
            add(node, direct, "paragraph", "text")
        for child in node.children:
            if isinstance(child, Tag):
                walk(child)

    main = soup.find("main") or soup.find(attrs={"role": "main"})
    nested = main.find("article") if isinstance(main, Tag) else None
    if isinstance(nested, Tag):
        outside_title = nested.find(["header", "h1"]) is None and main.find(["header", "h1"]) is not None
        root = main if outside_title else nested
    elif isinstance(main, Tag):
        root = main
    else:
        root = soup.find("article") or soup.body or soup
    for tag in list(root.find_all(["nav", "footer"])):
        tag.decompose()
    for form in list(root.find_all("form")):
        form.unwrap()
    for control in list(root.find_all(["input", "button", "select", "textarea"])):
        if not clean_block_text(control.get_text(" ", strip=True)):
            control.decompose()
    for role in ("navigation", "contentinfo"):
        for tag in list(root.find_all(attrs={"role": role})):
            tag.decompose()
    walk(root)
    health = _health(doc_name, units, 0)
    health["native_title"] = title
    return units, health


def _parse_markdown(file_path: str, doc_name: str) -> tuple[list[dict], dict]:
    lines = _read_text(file_path).splitlines()
    units: list[dict] = []
    stack: list[tuple[int, str]] = []
    index = 0

    def path() -> list[str]:
        return [title for _level, title in stack]

    def add(content: str, role: str, source_type: str, start: int, end: int, level: int = 0, raw: bool = False) -> None:
        section = path()
        heading = section[-1] if section else ""
        _emit(
            units,
            _make_unit(
                doc_name,
                content,
                original_format="MD",
                kind="md",
                role=role,
                source_type=source_type,
                locator={
                    "label": _line_label(start, end, heading),
                    "kind": "md",
                    "rowStart": start,
                    "rowEnd": end,
                    "path": heading,
                    "page": 0,
                },
                section_path=section,
                heading_level=level,
                section_title=heading,
                raw=raw,
            ),
        )

    while index < len(lines):
        line = lines[index]
        fence = re.match(r"^(```+|~~~+)", line.strip())
        if fence:
            marker = fence.group(1)[0]
            content_start = index + 2
            index += 1
            body = []
            while index < len(lines) and not lines[index].strip().startswith(marker * 3):
                body.append(lines[index])
                index += 1
            content_end = content_start + len(body) - 1 if body else content_start
            if index < len(lines):
                index += 1
            add("\n".join(body), "code", "code", content_start, content_end, raw=True)
            continue
        heading = re.match(r"^(#{1,6})\s+(.*?)\s*#*\s*$", line)
        if heading:
            level = len(heading.group(1))
            title = heading.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            number = index + 1
            add(title, "heading", "heading", number, number, level)
            index += 1
            continue
        if "|" in line and index + 1 < len(lines) and re.match(r"^\s*\|?\s*:?-{3,}", lines[index + 1]):
            start = index + 1
            rows = []
            while index < len(lines) and "|" in lines[index]:
                rows.append(lines[index])
                index += 1
            add("\n".join(rows), "table", "table", start, index)
            continue
        if not line.strip():
            index += 1
            continue
        start = index + 1
        body = []
        while index < len(lines) and lines[index].strip() and not re.match(r"^(#{1,6})\s+", lines[index]) and not re.match(r"^(```+|~~~+)", lines[index].strip()):
            if body and re.match(r"^\s*(={3,}|-{3,})\s*$", lines[index]):
                break
            body.append(lines[index])
            index += 1
        if body and index < len(lines) and re.match(r"^\s*(={3,}|-{3,})\s*$", lines[index]):
            marker = lines[index].strip()
            level = 1 if marker.startswith("=") else 2
            title = " ".join(part.strip() for part in body)
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            add(title, "heading", "heading", start, start + len(body) - 1, level)
            index += 1
            continue
        add("\n".join(body), "paragraph", "text", start, start + len(body) - 1)
    return units, _health(doc_name, units, 0)


def _parse_txt(file_path: str, doc_name: str) -> tuple[list[dict], dict]:
    lines = _read_text(file_path).splitlines()
    units: list[dict] = []
    buffer: list[tuple[int, str]] = []

    def flush() -> None:
        if not buffer:
            return
        start = buffer[0][0]
        end = buffer[-1][0]
        _emit(
            units,
            _make_unit(
                doc_name,
                "\n".join(text for _number, text in buffer),
                original_format="TXT",
                kind="txt",
                role="paragraph",
                source_type="text",
                locator={
                    "label": _line_label(start, end),
                    "kind": "txt",
                    "rowStart": start,
                    "rowEnd": end,
                    "path": "",
                    "page": 0,
                },
                section_path=[],
                raw=True,
            ),
        )
        buffer.clear()

    for number, line in enumerate(lines, start=1):
        if line.strip():
            buffer.append((number, line))
        else:
            flush()
    flush()
    return units, _health(doc_name, units, 0)


class _LineReader:
    def __init__(self, text: str):
        self.lines = text.splitlines(keepends=True)
        self.offset = 0
        self.line_no = 1

    def __iter__(self):
        return self

    def __next__(self):
        if self.offset >= len(self.lines):
            raise StopIteration
        line = self.lines[self.offset]
        self.offset += 1
        self.line_no += line.count("\n") or 1
        return line


def _parse_csv(file_path: str, doc_name: str) -> tuple[list[dict], dict]:
    text = _read_text(file_path)
    reader_input = _LineReader(text)
    reader = csv.reader(reader_input)
    units: list[dict] = []
    header: list[str] = []
    record_no = 0
    while True:
        start_line = reader_input.line_no
        try:
            row = next(reader)
        except StopIteration:
            break
        end_line = max(start_line, reader_input.line_no - 1)
        if not any(cell.strip() for cell in row):
            continue
        record_no += 1
        if record_no == 1:
            header = [cell.strip() for cell in row]
        lines = []
        for index, cell in enumerate(row):
            name = header[index] if index < len(header) and header[index] else f"column{index + 1}"
            lines.append(f"{name}={cell}")
        label = f"第{record_no}行"
        if header:
            label += " · " + ",".join(header)
        if end_line != start_line:
            label += f"（文件第{start_line}-{end_line}行）"
        _emit(
            units,
            _make_unit(
                doc_name,
                "\n".join(lines),
                original_format="CSV",
                kind="csv",
                role="row",
                source_type="table",
                locator={
                    "label": label,
                    "kind": "csv",
                    "rowStart": record_no,
                    "rowEnd": record_no,
                    "lineStart": start_line,
                    "lineEnd": end_line,
                    "path": ",".join(header),
                    "header": list(header),
                    "page": 0,
                },
                section_path=list(header),
                raw=True,
            ),
        )
    return units, _health(doc_name, units, 0)


def _json_path(segments: list[tuple[str, object]]) -> str:
    path = "$"
    for kind, value in segments:
        if kind == "index":
            path += f"[{value}]"
            continue
        text = str(value)
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", text):
            path += f".{text}"
        else:
            path += "[" + json.dumps(text, ensure_ascii=False) + "]"
    return path


def _parse_json(file_path: str, doc_name: str) -> tuple[list[dict], dict]:
    payload = json.loads(_read_text(file_path), parse_int=str, parse_float=str)
    units: list[dict] = []

    def walk(value, pointer: str, segments: list[tuple[str, object]], section: list[str]) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                escaped = str(key).replace("~", "~0").replace("/", "~1")
                walk(child, f"{pointer}/{escaped}", segments + [("key", key)], section + [str(key)])
            return
        if isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{pointer}/{index}", segments + [("index", index)], section + [str(index)])
            return
        if isinstance(value, bool):
            text = "true" if value else "false"
        elif value is None:
            text = "null"
        else:
            text = str(value)
        path = _json_path(segments)
        _emit(
            units,
            _make_unit(
                doc_name,
                text,
                original_format="JSON",
                kind="json",
                role="value",
                source_type="text",
                locator={
                    "label": path,
                    "kind": "json",
                    "path": path,
                    "jsonPointer": pointer or "/",
                    "page": 0,
                },
                section_path=section,
                raw=True,
            ),
        )

    walk(payload, "", [], [])
    return units, _health(doc_name, units, 0)


def _parse_xml(file_path: str, doc_name: str) -> tuple[list[dict], dict]:
    root = ET.fromstring(Path(file_path).read_bytes())
    units: list[dict] = []

    def walk(elem, path: str, section: list[str]) -> None:
        attrib = {_local(key): value for key, value in elem.attrib.items()}
        parts = [f"{key}={value}" for key, value in attrib.items()]
        direct = (elem.text or "").strip()
        if direct:
            parts.append(direct)
        for child in list(elem):
            tail = (child.tail or "").strip()
            if tail:
                parts.append(tail)
        content = "\n".join(part for part in parts if part)
        if content:
            _emit(
                units,
                _make_unit(
                    doc_name,
                    content,
                    original_format="XML",
                    kind="xml",
                    role="value",
                    source_type="text",
                    locator={"label": path, "kind": "xml", "path": path, "page": 0},
                    section_path=section,
                    raw=True,
                ),
            )
        counts: dict[str, int] = {}
        for child in list(elem):
            tag = _local(child.tag)
            counts[tag] = counts.get(tag, 0) + 1
            child_path = f"{path}/{tag}[{counts[tag]}]"
            walk(child, child_path, section + [tag])

    walk(root, "/" + _local(root.tag), [_local(root.tag)])
    return units, _health(doc_name, units, 0)


def _parent_map(root) -> dict:
    return {child: parent for parent in root.iter() for child in list(parent)}


def _inside(node, parents: dict, names: set[str]) -> bool:
    current = parents.get(node)
    while current is not None:
        if _local(current.tag) in names:
            return True
        current = parents.get(current)
    return False


def _xml_text(node) -> str:
    parts = []
    for child in node.iter():
        name = _local(child.tag)
        if name == "t" and child.text:
            parts.append(child.text)
        elif name == "br":
            parts.append("\n")
    return "".join(parts).strip()


def _parse_pptx(file_path: str, doc_name: str) -> tuple[list[dict], dict]:
    units: list[dict] = []
    with zipfile.ZipFile(file_path) as archive:
        slides = []
        for name in archive.namelist():
            match = re.fullmatch(r"ppt/slides/slide(\d+)\.xml", name)
            if match:
                slides.append((int(match.group(1)), name))
        slides.sort()
        total = len(slides)
        for number, name in slides:
            root = ET.fromstring(archive.read(name))
            parents = _parent_map(root)
            title = ""
            for shape in root.iter():
                if _local(shape.tag) != "sp" or _inside(shape, parents, {"tbl"}):
                    continue
                is_title = any(_local(node.tag) == "ph" and node.get("type") in {"title", "ctrTitle"} for node in shape.iter())
                paragraphs = [node for node in shape.iter() if _local(node.tag) == "p"]
                for offset, paragraph in enumerate(paragraphs):
                    text = _xml_text(paragraph)
                    if not text:
                        continue
                    heading = is_title and offset == 0
                    if heading:
                        title = text
                    section = [title] if title else []
                    _emit(
                        units,
                        _make_unit(
                            doc_name,
                            text,
                            original_format="PPTX",
                            kind="pptx",
                            role="heading" if heading else "paragraph",
                            source_type="heading" if heading else "text",
                            locator={"label": f"第{number}张幻灯片", "kind": "pptx", "page": number, "path": title},
                            section_path=section,
                            page_number=number,
                            page_count=total,
                            heading_level=1 if heading else 0,
                            section_title=title,
                        ),
                    )
            for table in root.iter():
                if _local(table.tag) != "tbl":
                    continue
                rows = []
                for row in list(table):
                    if _local(row.tag) != "tr":
                        continue
                    cells = []
                    for cell in list(row):
                        if _local(cell.tag) == "tc":
                            cells.append(_xml_text(cell))
                    if any(cells):
                        rows.append(" | ".join(cells))
                _emit(
                    units,
                    _make_unit(
                        doc_name,
                        "\n".join(rows),
                        original_format="PPTX",
                        kind="pptx",
                        role="table",
                        source_type="table",
                        locator={"label": f"第{number}张幻灯片", "kind": "pptx", "page": number, "path": title},
                        section_path=[title] if title else [],
                        page_number=number,
                        page_count=total,
                        section_title=title,
                    ),
                )
            for picture in root.iter():
                if _local(picture.tag) != "pic":
                    continue
                name_text = ""
                descr = ""
                for node in picture.iter():
                    if _local(node.tag) == "cNvPr":
                        name_text = node.get("name") or ""
                        descr = node.get("descr") or ""
                        break
                _emit(
                    units,
                    _make_unit(
                        doc_name,
                        "\n".join(part for part in (name_text, descr) if part),
                        original_format="PPTX",
                        kind="pptx",
                        role="figure",
                        source_type="figure",
                        locator={"label": f"第{number}张幻灯片", "kind": "pptx", "page": number, "path": title},
                        section_path=[title] if title else [],
                        page_number=number,
                        page_count=total,
                        section_title=title,
                    ),
                )
    return units, _health(doc_name, units, len(slides) if units else 0)


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    values = []
    for item in root.iter():
        if _local(item.tag) != "si":
            continue
        values.append("".join(node.text or "" for node in item.iter() if _local(node.tag) == "t"))
    return values


def _column_row(ref: str) -> tuple[int, int]:
    letters = "".join(char for char in ref if char.isalpha())
    digits = "".join(char for char in ref if char.isdigit())
    number = 0
    for char in letters:
        number = number * 26 + (ord(char.upper()) - 64)
    return number, int(digits or "0")


def _column_letter(number: int) -> str:
    letters = ""
    while number:
        number, remainder = divmod(number - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


def _cell_text(cell, shared: list[str]) -> str:
    kind = cell.get("t") or ""
    if kind == "inlineStr":
        return "".join(node.text or "" for node in cell.iter() if _local(node.tag) == "t")
    raw = ""
    for child in list(cell):
        if _local(child.tag) == "v" and child.text is not None:
            raw = child.text
            break
    if kind == "s" and raw != "":
        return shared[int(raw)]
    return raw


def _xlsx_targets(archive: zipfile.ZipFile) -> list[tuple[str, str]]:
    try:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    except KeyError:
        names = sorted(name for name in archive.namelist() if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", name))
        return [(Path(name).stem, name) for name in names]
    targets = {}
    for rel in rels.iter():
        if _local(rel.tag) == "Relationship":
            targets[rel.get("Id")] = rel.get("Target") or ""
    sheets = []
    for sheet in workbook.iter():
        if _local(sheet.tag) != "sheet":
            continue
        relation = ""
        for attr, value in sheet.attrib.items():
            if attr == "id" or attr.endswith("}id"):
                relation = value
        target = targets.get(relation, "")
        if target.startswith("/"):
            path = target.lstrip("/")
        elif target.startswith("xl/"):
            path = target
        else:
            path = "xl/" + target
        sheets.append((sheet.get("name") or "Sheet", path))
    return sheets


def _parse_xlsx(file_path: str, doc_name: str) -> tuple[list[dict], dict]:
    units: list[dict] = []
    with zipfile.ZipFile(file_path) as archive:
        shared = _shared_strings(archive)
        for sheet_name, path in _xlsx_targets(archive):
            root = ET.fromstring(archive.read(path))
            parsed_rows = []
            for row in root.iter():
                if _local(row.tag) != "row":
                    continue
                cells = {}
                row_number = int(row.get("r") or 0)
                for cell in list(row):
                    if _local(cell.tag) != "c":
                        continue
                    column, ref_row = _column_row(cell.get("r") or "")
                    if ref_row:
                        row_number = ref_row
                    text = _cell_text(cell, shared)
                    if text != "":
                        cells[column] = text
                if cells and row_number:
                    parsed_rows.append((row_number, cells))
            header = parsed_rows[0][1] if len(parsed_rows) >= 2 else {}
            for row_number, cells in parsed_rows:
                columns = sorted(cells)
                start, end = columns[0], columns[-1]
                cell_range = _column_letter(start) + str(row_number)
                if start != end:
                    cell_range += ":" + _column_letter(end) + str(row_number)
                header_values = [header.get(column, "") for column in range(start, end + 1)] if header and row_number != parsed_rows[0][0] else []
                lines = []
                for column in columns:
                    name = header.get(column) if header and row_number != parsed_rows[0][0] and header.get(column) else _column_letter(column)
                    lines.append(f"{name}={cells[column]}")
                sheet_path = f"{sheet_name}!{cell_range}"
                _emit(
                    units,
                    _make_unit(
                        doc_name,
                        "\n".join(lines),
                        original_format="XLSX",
                        kind="xlsx",
                        role="row",
                        source_type="table",
                        locator={
                            "label": sheet_path,
                            "kind": "xlsx",
                            "rowStart": row_number,
                            "rowEnd": row_number,
                            "path": sheet_path,
                            "header": [value for value in header_values if value],
                            "page": 0,
                        },
                        section_path=[sheet_name],
                        raw=True,
                    ),
                )
    return units, _health(doc_name, units, 0)


_PARSERS = {
    ".html": _parse_html,
    ".htm": _parse_html,
    ".md": _parse_markdown,
    ".markdown": _parse_markdown,
    ".txt": _parse_txt,
    ".csv": _parse_csv,
    ".json": _parse_json,
    ".xml": _parse_xml,
    ".pptx": _parse_pptx,
    ".xlsx": _parse_xlsx,
}


def extract_native_units(file_path: str, doc_name: str, metadata: dict | None = None) -> tuple[list[dict], dict]:
    suffix = Path(file_path).suffix.lower()
    parser = _PARSERS.get(suffix)
    if parser is None:
        return [], _health(doc_name, [])
    units, health = parser(file_path, doc_name)
    apply_record_metadata(units, metadata, FORMAT_BY_SUFFIX.get(suffix, ""), health.get("native_title") or "")
    return units, health
