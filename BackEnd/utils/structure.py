from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


BASE_DIR = Path(__file__).resolve().parent
STRUCTURAL_RECORD_DIR = BASE_DIR / "structural_records"
STRUCTURAL_RECORD_DIR.mkdir(parents=True, exist_ok=True)

CN_NUM = "零一二三四五六七八九十百千万两"
CHAPTER_PATTERN = re.compile(rf"^(?P<label>第\s*[0-9{CN_NUM}]+\s*章|[A-Z]\s*章)\s*(?P<title>.+)?$")
SECTION_PATTERN = re.compile(rf"^(?P<label>第\s*[0-9A-Za-z{CN_NUM}]+(?:\.[0-9A-Za-z{CN_NUM}]+)*\s*节)\s*(?P<title>.+)?$")
ARTICLE_PATTERN = re.compile(rf"^(?P<label>第\s*(?P<number>[0-9A-Za-z{CN_NUM}]+(?:\.[0-9A-Za-z{CN_NUM}]+)*)\s*条)\s*(?P<title>.+)?$")
NUMERIC_HEADING_PATTERN = re.compile(r"^(?P<number>\d+(?:\.\d+)*)(?:\s*[．、]\s*|\s+)(?P<title>[\u4e00-\u9fffA-Za-z][^\n。；;]{1,80})$")
MARKER_PATTERN = re.compile(
    r"^\s*(?P<label>[（(][一二三四五六七八九十百千万\dA-Za-zivxIVXⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅰⅱⅲⅳⅴⅵⅶⅷⅸⅹ]+[)）]|[一二三四五六七八九十]+、|[0-9]{1,3}(?:、|\.(?!\d))|[A-Za-z][\.\)])\s*(?P<body>.*)$"
)
TOC_DOT_PATTERN = re.compile(r"[.·。．]{3,}\s*-?\s*\d+\s*-?\s*$")
PAGE_MARK_PATTERN = re.compile(r"^\s*(?:[-—]\s*)?\d+\s*(?:[-—])?\s*$")
ROMAN_TOKENS = {"ⅰ", "ⅱ", "ⅲ", "ⅳ", "ⅴ", "ⅵ", "ⅶ", "ⅷ", "ⅸ", "ⅹ", "Ⅰ", "Ⅱ", "Ⅲ", "Ⅳ", "Ⅴ", "Ⅵ", "Ⅶ", "Ⅷ", "Ⅸ", "Ⅹ"}


def _clean_text(text: str) -> str:
    value = str(text or "").replace("\xa0", " ")
    value = value.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    value = re.sub(r"\r\n?", "\n", value)
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    value = re.sub(r"(?<=[\u4e00-\u9fff])[ \t]+(?=[\u4e00-\u9fff])", "", value)
    return value.strip()


def _json_utf8_safe(value):
    if isinstance(value, str):
        return value.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    if isinstance(value, list):
        return [_json_utf8_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_utf8_safe(item) for key, item in value.items()}
    return value


def _compact_text(text: str) -> str:
    return re.sub(r"\s+", "", _clean_text(text))


def _safe_slug(text: str) -> str:
    compact = _compact_text(text)
    if not compact:
        return "unknown"
    compact = re.sub(r"[^\w\u4e00-\u9fff.-]+", "-", compact)
    return compact[:96] or "unknown"


def _record_cache_path(doc_name: str) -> Path:
    return STRUCTURAL_RECORD_DIR / f"{Path(doc_name).stem}.json"


def _strip_toc_leader(line: str) -> str:
    cleaned = _clean_text(line)
    cleaned = re.sub(r"[.·。．]{3,}\s*-?\s*\d+\s*-?\s*$", "", cleaned).strip()
    return cleaned


def _is_toc_line(line: str, page_number: int) -> bool:
    cleaned = _clean_text(line)
    if not cleaned:
        return False
    return bool(TOC_DOT_PATTERN.search(cleaned)) or ("目录" in cleaned and page_number <= 5)


def _is_noise_line(line: str) -> bool:
    cleaned = _clean_text(line)
    compact = re.sub(r"\s+", "", cleaned)
    if not cleaned:
        return True
    if PAGE_MARK_PATTERN.match(cleaned):
        return True
    if compact in {"交通运输部规章", "交通运输部发布", "中国民用航空局", "民航局"}:
        return True
    if len(cleaned) <= 2 and cleaned in {"中", "国", "民", "用", "航", "空", "局"}:
        return True
    return False


def _normalize_heading_title(title: str) -> str:
    cleaned = _clean_text(title)
    cleaned = re.sub(r"\s+", "", cleaned)
    return cleaned


def _numeric_parent(number: str) -> str:
    parts = [part for part in str(number or "").split(".") if part]
    return ".".join(parts[:-1]) if len(parts) > 1 else ""


def _numeric_level(number: str) -> int:
    return len([part for part in str(number or "").split(".") if part]) or 1


def _looks_like_body_reference(number: str, title: str) -> bool:
    cleaned_title = _normalize_heading_title(title)
    if not number:
        return False
    if re.fullmatch(rf"[{CN_NUM}]+", str(number or "")):
        return False
    if cleaned_title.startswith(("进行", "接受", "检查", "确认", "规定", "所列", "相关", "执行", "符合", "申请", "颁发", "提交", "款", "中", "和", "及")):
        return True
    if re.search(r"[。；;，,：:]", cleaned_title):
        return True
    if "." not in str(number):
        if any(token in cleaned_title[:32] for token in ("）", ")", "，", "、", "；", "毫升", "千帕", "英寸", "转／分")):
            return True
        return False
    if re.match(r"^(条|款|项|目|规定|所述|要求)", cleaned_title):
        return True
    if "所述" in cleaned_title[:12]:
        return True
    return False


def _looks_like_detached_heading_title(line: str) -> bool:
    cleaned = _clean_text(line)
    if not cleaned:
        return False
    compact = _compact_text(cleaned)
    if len(compact) > 34:
        return False
    if ARTICLE_PATTERN.match(cleaned) or CHAPTER_PATTERN.match(cleaned) or SECTION_PATTERN.match(cleaned):
        return False
    if _match_marker(cleaned):
        return False
    if re.search(r"[。；;，,：:]$", cleaned):
        return False
    if re.search(r"(应当|必须|不得|可以|按照|符合|申请|颁发|提交|进行|包括).{4,}", cleaned):
        return False
    return bool(re.search(r"[\u4e00-\u9fffA-Za-z]", cleaned))


def _marker_inner(label: str) -> str:
    cleaned = _clean_text(label)
    if (cleaned.startswith(("(", "（")) and cleaned.endswith((")", "）"))) or cleaned.endswith((".", "、", ")")):
        cleaned = cleaned.strip("()（）")
        cleaned = cleaned.rstrip(".、)")
    return cleaned.strip()


def _marker_family(label: str) -> str:
    inner = _marker_inner(label)
    if not inner:
        return "unknown"
    if any(token in inner for token in ROMAN_TOKENS):
        return "roman"
    if re.fullmatch(r"[ivxIVX]+", inner):
        return "roman"
    if re.fullmatch(r"\d{1,3}", inner):
        return "digit"
    if re.fullmatch(r"[A-Za-z]", inner):
        return "alpha"
    if re.fullmatch(rf"[{CN_NUM}]+", inner):
        return "chinese"
    return "unknown"


def _match_marker(line: str) -> Optional[Tuple[str, str, str]]:
    match = MARKER_PATTERN.match(_clean_text(line))
    if not match:
        return None
    label = match.group("label")
    body = match.group("body")
    family = _marker_family(label)
    if family == "unknown":
        return None
    return label, body, family


def build_document_profile(units: Iterable[dict]) -> DocumentProfile:
    ordered_units = list(units)
    article_pages: List[int] = []
    for unit in ordered_units:
        page_number = int(unit.get("page_number") or 1)
        for raw_line in _clean_text(unit.get("content", "")).splitlines():
            if ARTICLE_PATTERN.match(_clean_text(raw_line)):
                article_pages.append(page_number)
                break

    body_start_page = min(article_pages) if article_pages else 1
    marker_counts: Dict[str, int] = {}
    marker_transitions: Dict[str, int] = {}
    previous_family = ""

    for unit in ordered_units:
        page_number = int(unit.get("page_number") or 1)
        for line, source in _line_events(unit, body_start_page):
            if source == "toc":
                continue
            if parse_heading(line, unit, source):
                previous_family = ""
                continue
            marker = _match_marker(line)
            if not marker:
                continue
            _, _, family = marker
            marker_counts[family] = marker_counts.get(family, 0) + 1
            if previous_family and previous_family != family:
                key = f"{previous_family}>{family}"
                marker_transitions[key] = marker_transitions.get(key, 0) + 1
            previous_family = family

    marker_family_rank = {"alpha": 1, "chinese": 1, "digit": 1, "roman": 2}
    if marker_transitions.get("alpha>digit", 0) or marker_transitions.get("chinese>digit", 0):
        marker_family_rank["digit"] = 2
    if marker_transitions.get("digit>roman", 0):
        marker_family_rank["roman"] = marker_family_rank.get("digit", 1) + 1

    return DocumentProfile(
        body_start_page=body_start_page,
        marker_family_rank=marker_family_rank,
        marker_counts=marker_counts,
        marker_transitions=marker_transitions,
    )


def parse_structure_marker(line: str, profile: DocumentProfile, active_markers: List[Tuple[int, StructuralRecord]]) -> Optional[StructureMarker]:
    matched = _match_marker(line)
    if not matched:
        return None
    label, body, family = matched
    rank = int(profile.marker_family_rank.get(family, 1) or 1)
    if family == "digit" and active_markers and any(rank <= 1 for rank, _ in active_markers):
        rank = max(rank, 2)
    if family == "roman" and active_markers:
        rank = max(rank, 2)
    record_type = "clause" if rank <= 1 else "item"
    return StructureMarker(label=label, body=body, family=family, rank=rank, record_type=record_type)


def _semantic_role(label: str, title: str, parent_title: str = "", body_text: str = "") -> str:
    joined = _clean_text(" ".join(part for part in [label, title, parent_title, body_text[:360]] if part))
    if "适用范围" in joined:
        return "scope"
    if "施行" in joined and any(token in joined for token in ("时间", "日期", "起施行")):
        return "effective_date"
    if "定义" in parent_title or title == "定义" or re.search(r"^\d+(?:\.\d+)?\s*定义$", joined):
        return "definition"
    if any(token in joined for token in ("分类", "类别", "分为", "包括")):
        return "classification"
    if any(token in joined for token in ("要求", "应当", "必须", "不得")):
        return "requirement"
    return "text"


@dataclass
class DocumentProfile:
    body_start_page: int = 1
    marker_family_rank: Dict[str, int] = field(default_factory=dict)
    marker_counts: Dict[str, int] = field(default_factory=dict)
    marker_transitions: Dict[str, int] = field(default_factory=dict)


@dataclass
class StructureMarker:
    label: str
    body: str
    family: str
    rank: int
    record_type: str
    confidence: float = 0.86


@dataclass
class ParsedHeading:
    kind: str
    label: str
    title: str
    number: str
    level: int
    source: str = "body"
    confidence: float = 1.0


@dataclass
class StructuralRecord:
    record_id: str
    doc_name: str
    record_type: str
    label: str = ""
    title: str = ""
    number: str = ""
    level: int = 0
    semantic_role: str = "text"
    chapter_label: str = ""
    chapter_title: str = ""
    section_label: str = ""
    section_title: str = ""
    article_id: str = ""
    article_label: str = ""
    clause_id: str = ""
    item_id: str = ""
    heading_path: List[str] = field(default_factory=list)
    path_numbers: List[str] = field(default_factory=list)
    body_text: str = ""
    page_start: int = 1
    page_end: int = 1
    page_count: int = 1
    parent_record_id: str = ""
    child_record_ids: List[str] = field(default_factory=list)
    previous_sibling_id: str = ""
    next_sibling_id: str = ""
    direct_child_count: int = 0
    descendant_count: int = 0
    source: str = "body"
    is_toc: bool = False
    confidence: float = 1.0
    order: int = 0
    source_unit_count: int = 0

    def display_title(self) -> str:
        return _clean_text(" ".join(part for part in [self.label, self.title] if part)) or self.title or self.label

    def to_dict(self) -> Dict:
        return asdict(self)


def parse_heading(line: str, unit: Optional[dict] = None, source: str = "body") -> Optional[Dict]:
    unit = unit or {}
    raw = _strip_toc_leader(line) if source == "toc" else _clean_text(line)
    if not raw or _is_noise_line(raw):
        return None

    article_match = ARTICLE_PATTERN.match(raw)
    if article_match:
        number = article_match.group("number")
        label = re.sub(r"\s+", "", article_match.group("label"))
        title = _normalize_heading_title(article_match.group("title") or "")
        if _looks_like_body_reference(number, title):
            return None
        return ParsedHeading("article", label, title, number, 30, source, 0.96). __dict__

    chapter_match = CHAPTER_PATTERN.match(raw)
    if chapter_match:
        label = re.sub(r"\s+", "", chapter_match.group("label"))
        title = _normalize_heading_title(chapter_match.group("title") or "")
        return ParsedHeading("chapter", label, title, label, 10, source, 0.94).__dict__

    section_match = SECTION_PATTERN.match(raw)
    if section_match:
        label = re.sub(r"\s+", "", section_match.group("label"))
        title = _normalize_heading_title(section_match.group("title") or "")
        return ParsedHeading("section", label, title, label, 20, source, 0.94).__dict__

    numeric_match = NUMERIC_HEADING_PATTERN.match(raw)
    if numeric_match:
        number = numeric_match.group("number")
        title = _normalize_heading_title(numeric_match.group("title"))
        if _looks_like_body_reference(number, title):
            return None
        if len(title) > 42 and _numeric_level(number) > 1:
            return None
        level = _numeric_level(number)
        return ParsedHeading("section", number, title, number, level, source, 0.9).__dict__

    return None


def _record_id(doc_name: str, record_type: str, number: str, label: str, title: str, source: str = "body") -> str:
    key = number or label or title or record_type
    return f"{doc_name}::{record_type}::{_safe_slug(key)}::{source}"


def _make_node(doc_name: str, parsed: Dict, page_number: int, page_count: int, order: int) -> StructuralRecord:
    label = parsed.get("label", "")
    title = parsed.get("title", "")
    number = parsed.get("number", "")
    source = parsed.get("source", "body")
    kind = parsed.get("kind", "section")
    record_type = "toc" if source == "toc" else kind
    if kind == "article":
        article_id = number
        article_label = label
    else:
        article_id = ""
        article_label = ""
    record_id = _record_id(doc_name, record_type, number, label, title, source)
    if kind == "section" and not re.match(r"^\d+(?:\.\d+)*$", str(number or "")):
        record_id = f"{doc_name}::{record_type}::{_safe_slug(label or title)}::{page_number}::{source}"
    return StructuralRecord(
        record_id=record_id,
        doc_name=doc_name,
        record_type=record_type,
        label=label,
        title=title,
        number=number,
        level=int(parsed.get("level", 0) or 0),
        semantic_role=_semantic_role(label, title),
        article_id=article_id,
        article_label=article_label,
        body_text=_clean_text(" ".join(part for part in [label, title] if part)),
        page_start=page_number,
        page_end=page_number,
        page_count=page_count,
        source=source,
        is_toc=source == "toc",
        confidence=float(parsed.get("confidence", 1.0) or 1.0),
        order=order,
        source_unit_count=1,
    )


def _append_body(record: StructuralRecord, text: str, page_number: int) -> None:
    cleaned = _clean_text(text)
    if not cleaned:
        return
    if cleaned not in record.body_text:
        record.body_text = f"{record.body_text}\n{cleaned}".strip() if record.body_text else cleaned
    record.page_end = max(record.page_end, page_number)
    record.source_unit_count += 1


def _line_events(unit: dict, body_start_page: int = 1) -> List[Tuple[str, str]]:
    content = _clean_text(unit.get("content", ""))
    if not content:
        return []
    page_number = int(unit.get("page_number") or 1)
    page_is_toc = (body_start_page > 1 and page_number < body_start_page) or (
        page_number <= 5 and ("目录" in content or len(TOC_DOT_PATTERN.findall(content)) >= 2)
    )
    events: List[Tuple[str, str]] = []
    for raw_line in content.splitlines():
        line = _clean_text(raw_line)
        if not line or _is_noise_line(line):
            continue
        source = "toc" if page_is_toc or _is_toc_line(line, page_number) else "body"
        events.append((line, source))
    if not events and content:
        events.append((content, "body"))
    return events


def _choose_parent(record: StructuralRecord, active: List[StructuralRecord], by_number: Dict[str, StructuralRecord]) -> str:
    if record.is_toc:
        return ""
    if record.record_type == "section" and re.match(r"^\d+(?:\.\d+)*$", record.number):
        parent_number = _numeric_parent(record.number)
        if parent_number and parent_number in by_number:
            return by_number[parent_number].record_id
    if record.record_type == "article":
        for candidate in reversed(active):
            if not candidate.is_toc and candidate.record_type in {"section", "chapter"}:
                return candidate.record_id
    if record.record_type == "section":
        for candidate in reversed(active):
            if not candidate.is_toc and candidate.level < record.level:
                return candidate.record_id
    if record.record_type == "clause":
        for candidate in reversed(active):
            if candidate.record_type == "article":
                return candidate.record_id
    if record.record_type == "item":
        for candidate in reversed(active):
            if candidate.record_type in {"clause", "article", "section"}:
                return candidate.record_id
    return ""


def _merge_or_add(records: Dict[str, StructuralRecord], record: StructuralRecord) -> StructuralRecord:
    existing = records.get(record.record_id)
    if existing is None:
        records[record.record_id] = record
        return record
    if existing.is_toc and not record.is_toc:
        record.heading_path = existing.heading_path or record.heading_path
        records[record.record_id] = record
        return record
    _append_body(existing, record.body_text, record.page_start)
    existing.page_start = min(existing.page_start, record.page_start)
    existing.page_end = max(existing.page_end, record.page_end)
    existing.confidence = max(existing.confidence, record.confidence)
    return existing


def _infer_missing_numeric_parents(records: Dict[str, StructuralRecord], doc_name: str, page_count: int) -> None:
    body_numeric = {
        record.number: record
        for record in records.values()
        if record.record_type == "section" and not record.is_toc and re.match(r"^\d+(?:\.\d+)*$", record.number)
    }
    toc_numeric = {
        record.number: record
        for record in records.values()
        if record.is_toc and re.match(r"^\d+(?:\.\d+)*$", record.number)
    }
    order_seed = max((record.order for record in records.values()), default=0) + 1
    for number in sorted(list(body_numeric.keys()), key=lambda value: (_numeric_level(value), value)):
        parent_number = _numeric_parent(number)
        while parent_number and parent_number not in body_numeric:
            toc = toc_numeric.get(parent_number)
            title = toc.title if toc else ""
            label = parent_number
            inferred = StructuralRecord(
                record_id=_record_id(doc_name, "section", parent_number, label, title, "inferred"),
                doc_name=doc_name,
                record_type="section",
                label=label,
                title=title,
                number=parent_number,
                level=_numeric_level(parent_number),
                semantic_role=_semantic_role(label, title),
                body_text=_clean_text(" ".join(part for part in [label, title] if part)),
                page_start=body_numeric[number].page_start,
                page_end=body_numeric[number].page_start,
                page_count=page_count,
                source="inferred",
                is_toc=False,
                confidence=0.58 if toc else 0.42,
                order=order_seed,
                source_unit_count=0,
            )
            order_seed += 1
            records[inferred.record_id] = inferred
            body_numeric[parent_number] = inferred
            parent_number = _numeric_parent(parent_number)


def _finalize_tree(records: Dict[str, StructuralRecord]) -> List[StructuralRecord]:
    body_records = [record for record in records.values() if not record.is_toc]
    body_records.sort(key=lambda item: (item.page_start, item.order, item.level, item.record_id))

    by_number = {
        record.number: record
        for record in body_records
        if record.record_type == "section" and re.match(r"^\d+(?:\.\d+)*$", record.number)
    }
    by_id = {record.record_id: record for record in body_records}

    active: List[StructuralRecord] = []
    for record in body_records:
        if record.record_type in {"doc", "toc"}:
            continue
        if record.record_type == "section" and re.match(r"^\d+(?:\.\d+)*$", record.number):
            parent_number = _numeric_parent(record.number)
            record.parent_record_id = by_number[parent_number].record_id if parent_number in by_number else ""
        elif record.parent_record_id and record.parent_record_id in by_id:
            pass
        else:
            while active and active[-1].level >= record.level and record.record_type in {"chapter", "section", "article"}:
                active.pop()
            parent_id = _choose_parent(record, active, by_number)
            if parent_id:
                record.parent_record_id = parent_id
        if record.record_type in {"chapter", "section", "article"}:
            while active and active[-1].level >= record.level:
                active.pop()
            active.append(record)

    for record in body_records:
        record.child_record_ids = []
    for record in body_records:
        parent = by_id.get(record.parent_record_id)
        if parent and record.record_id not in parent.child_record_ids:
            parent.child_record_ids.append(record.record_id)

    for parent in body_records:
        children = [by_id[child_id] for child_id in parent.child_record_ids if child_id in by_id]
        children.sort(key=lambda item: (item.page_start, item.order, item.number))
        parent.child_record_ids = [child.record_id for child in children]
        parent.direct_child_count = len(children)
        for index, child in enumerate(children):
            child.previous_sibling_id = children[index - 1].record_id if index > 0 else ""
            child.next_sibling_id = children[index + 1].record_id if index < len(children) - 1 else ""

    for record in body_records:
        if record.record_type != "article":
            continue
        children = [by_id[child_id] for child_id in record.child_record_ids if child_id in by_id]
        clause_children = [child for child in children if child.record_type == "clause" and child.body_text]
        if len(clause_children) < 2:
            continue
        title = record.display_title()
        original_body = _clean_text(record.body_text)
        intro_text = ""
        first_child_text = _clean_text(clause_children[0].body_text)
        if original_body and first_child_text:
            first_child_index = original_body.find(first_child_text[: min(len(first_child_text), 48)])
            if first_child_index > 0:
                intro_text = _clean_text(original_body[:first_child_index])
                if title and intro_text.startswith(title):
                    intro_text = intro_text[len(title) :].strip()
        rebuilt_parts = [title] if title else []
        if intro_text:
            rebuilt_parts.append(intro_text)
        rebuilt_parts.extend(child.body_text for child in clause_children)
        record.body_text = _clean_text("\n".join(rebuilt_parts))
        record.page_end = max([record.page_end, *[child.page_end for child in clause_children]])

    def descendant_count(record: StructuralRecord) -> int:
        total = 0
        for child_id in record.child_record_ids:
            child = by_id.get(child_id)
            if not child:
                continue
            total += 1 + descendant_count(child)
        return total

    for record in body_records:
        record.descendant_count = descendant_count(record)
        path: List[str] = []
        path_numbers: List[str] = []
        cursor = record
        seen = set()
        while cursor and cursor.record_id not in seen:
            seen.add(cursor.record_id)
            label = cursor.display_title()
            if label:
                path.append(label)
            if cursor.number:
                path_numbers.append(cursor.number)
            cursor = by_id.get(cursor.parent_record_id)
        record.heading_path = list(reversed(path))
        record.path_numbers = list(reversed(path_numbers))
        if record.record_type == "article":
            record.article_label = record.label
            record.article_id = record.number
        parent = by_id.get(record.parent_record_id)
        cursor = parent
        seen_ancestors = set()
        while cursor and cursor.record_id not in seen_ancestors:
            seen_ancestors.add(cursor.record_id)
            if cursor.record_type == "chapter" and not record.chapter_label:
                record.chapter_label = cursor.label
                record.chapter_title = cursor.title
            if cursor.record_type == "section" and not record.section_label:
                record.section_label = cursor.label
                record.section_title = cursor.title
            if cursor.record_type == "article" and not record.article_id:
                record.article_label = cursor.label
                record.article_id = cursor.number
            cursor = by_id.get(cursor.parent_record_id)
        record.semantic_role = _semantic_role(record.label, record.title, parent.title if parent else "", record.body_text)

    toc_records = [record for record in records.values() if record.is_toc]
    all_records = body_records + sorted(toc_records, key=lambda item: (item.page_start, item.order, item.record_id))
    for index, record in enumerate(all_records):
        record.order = index
    return all_records


def build_structural_records(units: Iterable[dict], doc_name: Optional[str] = None) -> List[Dict]:
    ordered_units = list(units)
    if not ordered_units:
        return []

    resolved_doc_name = doc_name or str(ordered_units[0].get("doc_name") or "")
    page_count = int(ordered_units[0].get("page_count") or 1)
    profile = build_document_profile(ordered_units)
    body_start_page = profile.body_start_page
    records: Dict[str, StructuralRecord] = {}
    current_node: Optional[StructuralRecord] = None
    marker_stack: List[Tuple[int, StructuralRecord]] = []
    order = 0

    root = StructuralRecord(
        record_id=_record_id(resolved_doc_name, "doc", "", resolved_doc_name, "", "body"),
        doc_name=resolved_doc_name,
        record_type="doc",
        label=Path(resolved_doc_name).stem,
        title="",
        number="",
        level=0,
        body_text=Path(resolved_doc_name).stem,
        page_start=1,
        page_end=page_count,
        page_count=page_count,
        order=order,
        confidence=1.0,
    )
    records[root.record_id] = root
    order += 1

    for unit in ordered_units:
        page_number = int(unit.get("page_number") or 1)
        for line, source in _line_events(unit, body_start_page):
            parsed = parse_heading(line, unit, source)
            if parsed:
                node = _make_node(resolved_doc_name, parsed, page_number, page_count, order)
                order += 1
                node.parent_record_id = root.record_id if not node.is_toc and node.record_type in {"chapter", "section"} and node.level <= 1 else ""
                node = _merge_or_add(records, node)
                if not node.is_toc:
                    current_node = node
                    marker_stack = []
                continue

            if source == "toc":
                continue
            if current_node is None:
                current_node = root

            if (
                current_node.record_type in {"chapter", "section", "article"}
                and not current_node.title
                and _compact_text(current_node.body_text) == _compact_text(current_node.label)
                and _looks_like_detached_heading_title(line)
            ):
                current_node.title = _normalize_heading_title(line)
                current_node.body_text = _clean_text(" ".join(part for part in [current_node.label, current_node.title] if part))
                current_node.semantic_role = _semantic_role(current_node.label, current_node.title)
                continue

            marker = parse_structure_marker(line, profile, marker_stack)
            if marker and current_node.record_type != "doc":
                while marker_stack and marker_stack[-1][0] >= marker.rank:
                    marker_stack.pop()
                parent = marker_stack[-1][1] if marker_stack else current_node
                record = StructuralRecord(
                    record_id=_record_id(resolved_doc_name, marker.record_type, "", f"{parent.record_id}:{marker.label}", "", "body"),
                    doc_name=resolved_doc_name,
                    record_type=marker.record_type,
                    label=marker.label,
                    title="",
                    number="",
                    level=current_node.level + marker.rank,
                    semantic_role=_semantic_role(marker.label, marker.body, current_node.title, line),
                    body_text=_clean_text(line),
                    page_start=page_number,
                    page_end=page_number,
                    page_count=page_count,
                    parent_record_id=parent.record_id,
                    confidence=marker.confidence,
                    order=order,
                    source_unit_count=1,
                )
                order += 1
                record = _merge_or_add(records, record)
                marker_stack.append((marker.rank, record))
                _append_body(current_node, line, page_number)
                for _, ancestor in marker_stack[:-1]:
                    _append_body(ancestor, line, page_number)
                continue

            _append_body(current_node, line, page_number)
            for _, ancestor in marker_stack:
                _append_body(ancestor, line, page_number)

    _infer_missing_numeric_parents(records, resolved_doc_name, page_count)
    finalized = _finalize_tree(records)
    return [record.to_dict() for record in finalized]


def _record_text(record: Dict) -> str:
    return _clean_text(
        "\n".join(
            str(record.get(key) or "")
            for key in ("label", "title", "section_title", "article_label", "body_text")
        )
    )


def _overlap_score(query: str, text: str) -> float:
    q = _compact_text(query)
    t = _compact_text(text)
    if not q or not t:
        return 0.0
    score = 0.0
    for size in (2, 3, 4):
        qgrams = {q[i : i + size] for i in range(max(len(q) - size + 1, 0))}
        if not qgrams:
            continue
        hits = sum(1 for gram in qgrams if gram in t)
        score += hits / len(qgrams)
    if q in t or t in q:
        score += 1.0
    return score


def _extract_definition_term(query: str) -> str:
    cleaned = _clean_text(query)
    # 优先匹配显式定义格式："X的定义"
    for pattern in (
        r"^(?P<term>.+?)的定义(?:是|是什么|为|是什么)?",
    ):
        match = re.search(pattern, cleaned)
        if match:
            return _compact_text(match.group("term"))
    # "X是什么" 仅在短且不含颁发/条件/要求/步骤等实施性关键词的查询时才视为定义
    match = re.search(r"^(?P<term>.+?)是什么$", cleaned)
    if match:
        term = _compact_text(match.group("term"))
        # 实施性关键词：说明这不是查术语定义，而是查程序/条件/步骤
        implementation_kw = ("颁发", "条件", "要求", "步骤", "程序", "流程", "申请", "审查", "如何", "怎样", "哪个", "哪些", "怎么")
        if term and len(term) <= 10 and not any(kw in term for kw in implementation_kw):
            return term
    return ""


def _extract_classification_subject(query: str) -> str:
    cleaned = _compact_text(query)
    if not cleaned:
        return ""
    question_markers = ("有哪些", "有哪几", "哪几", "哪些", "是什么", "分别是什么", "多少")
    classification_markers = ("分类", "类别", "分为", "分成")
    if not any(marker in cleaned for marker in classification_markers):
        if not any(marker in cleaned for marker in ("第一类", "第二类", "第三类")):
            return ""

    candidates: List[str] = []

    for marker in ("的分类", "分类", "的类别", "类别"):
        index = cleaned.find(marker)
        if index > 0:
            candidates.append(cleaned[:index])

    for marker in ("分为", "分成"):
        index = cleaned.find(marker)
        if index > 0:
            prefix = cleaned[:index]
            if "按" in prefix:
                prefix = prefix.split("按", 1)[0]
            candidates.append(prefix)

    for marker in ("有哪些", "有哪几", "包括哪些"):
        index = cleaned.find(marker)
        if index > 0 and any(word in cleaned[index:] for word in ("分类", "类别", "类")):
            candidates.append(cleaned[:index])

    class_pattern = re.sub(r"第?[一二三四五六七八九十]+类(?:和第?[一二三四五六七八九十]+类)?", "", cleaned)
    if class_pattern != cleaned and len(class_pattern) >= 2:
        candidates.append(class_pattern)

    normalized: List[str] = []
    for candidate in candidates:
        value = candidate.strip("的")
        for marker in question_markers:
            value = value.replace(marker, "")
        value = re.sub(r"第?[一二三四五六七八九十]+类(?:和第?[一二三四五六七八九十]+类)?", "", value)
        value = value.strip("的")
        if len(value) >= 2 and value not in normalized:
            normalized.append(value)
    if not normalized:
        return ""
    return sorted(normalized, key=len)[0]


def _has_unmodified_subject(text: str, subject: str, modifiers: Tuple[str, ...] = ("出口", "外国", "特殊", "标准", "临时", "限制", "生产", "型号", "特许")) -> bool:
    compact_text = _compact_text(text)
    compact_subject = _compact_text(subject)
    if not compact_text or not compact_subject:
        return False
    start = 0
    while True:
        index = compact_text.find(compact_subject, start)
        if index < 0:
            return False
        prefix = compact_text[max(0, index - 4) : index]
        if not any(prefix.endswith(modifier) for modifier in modifiers):
            return True
        start = index + len(compact_subject)


def find_best_structure_record(query: str, records: Iterable[Dict], roles: Optional[Iterable[str]] = None) -> Optional[Dict]:
    role_set = set(roles or [])
    best: Tuple[float, Optional[Dict]] = (0.0, None)
    for record in records:
        if record.get("is_toc"):
            continue
        if role_set and record.get("semantic_role") not in role_set and record.get("record_type") not in role_set:
            continue
        text = _record_text(record)
        score = _overlap_score(query, text)
        if record.get("source") == "body":
            score += 0.15
        if record.get("source") == "inferred":
            score -= 0.05
        if score > best[0]:
            best = (score, record)
    return best[1]


def _children(records: List[Dict], parent_id: str) -> List[Dict]:
    rows = [record for record in records if record.get("parent_record_id") == parent_id and not record.get("is_toc")]
    return sorted(rows, key=lambda item: (int(item.get("page_start") or 1), int(item.get("order") or 0), str(item.get("number") or "")))


def _matched_doc_names_from_query(query: str, records: List[Dict]) -> List[str]:
    compact_query = _compact_text(query)
    matched: List[Tuple[int, str]] = []
    seen = set()
    for record in records:
        doc_name = str(record.get("doc_name") or "")
        if not doc_name or doc_name in seen:
            continue
        seen.add(doc_name)
        stem = _compact_text(Path(doc_name).stem)
        if stem and stem in compact_query:
            matched.append((len(stem), doc_name))
    matched.sort(reverse=True)
    return [name for _, name in matched]


def answer_structural_count(query: str, records: Iterable[Dict]) -> Optional[Dict]:
    if not any(token in query for token in ("共有几", "有几", "多少条", "多少章", "多少节", "几个", "几项", "几章", "几节")):
        return None
    all_records = list(records)
    compact_query = _compact_text(query)
    matched_doc_names = set(_matched_doc_names_from_query(query, all_records))
    record_list = [
        record
        for record in all_records
        if not matched_doc_names or str(record.get("doc_name") or "") in matched_doc_names
    ]
    doc_level_count_kind = ""
    if matched_doc_names and not re.search(r"第\s*[0-9A-Za-z一二三四五六七八九十百千万两]+\s*(?:章|节)", query):
        if any(token in compact_query for token in ("一共有多少章", "共有多少章", "总共有多少章", "多少章", "几章")):
            doc_level_count_kind = "chapter"
        elif any(token in compact_query for token in ("一共有多少节", "共有多少节", "总共有多少节", "多少节", "几节")):
            doc_level_count_kind = "section"
        elif any(token in compact_query for token in ("一共有多少条", "共有多少条", "总共有多少条", "多少条", "几条")):
            doc_level_count_kind = "article"
    doc_level_count = bool(doc_level_count_kind)
    if doc_level_count:
        doc_name = sorted(matched_doc_names)[0]
        count_type = {"chapter": "chapter", "section": "section", "article": "article"}[doc_level_count_kind]
        rows = [
            record
            for record in record_list
            if record.get("record_type") == count_type and not record.get("is_toc") and str(record.get("doc_name") or "") == doc_name
        ]
        if rows:
            rows = sorted(rows, key=lambda row: (int(row.get("page_start") or 1), int(row.get("order") or 0)))
            doc_record = next(
                (
                    record
                    for record in record_list
                    if record.get("record_type") == "doc" and str(record.get("doc_name") or "") == doc_name
                ),
                rows[0],
            )
            labels = [
                _clean_text(" ".join(part for part in [str(row.get("label") or ""), str(row.get("title") or "")] if part))
                for row in rows
                if row.get("label") or row.get("title")
            ]
            target_name = Path(doc_name).stem
            child_text = f"{labels[0]}至{labels[-1]}" if len(labels) > 4 else "、".join(labels)
            unit = {"chapter": "章", "section": "节", "article": "条"}[doc_level_count_kind]
            content = f"{target_name} 的{unit}：{child_text}"
            source = {
                "content": content,
                "doc_name": doc_name,
                "heading": target_name,
                "parent_heading": "",
                "score": 1.0,
                "chunk_id": f"structure-count:{doc_record.get('record_id', doc_name)}",
                "page_number": int(rows[0].get("page_start") or 1),
                "page_count": int(doc_record.get("page_count") or rows[-1].get("page_count") or 1),
                "section_title": target_name,
                "highlight_text": _clean_text(str(rows[0].get("label") or target_name)),
                "bbox": None,
                "source_type": "structural_count",
                "layout_type": "structural",
                "parse_confidence": 1.0,
                "document_type": "REG",
                "is_ocr_fallback": False,
            }
            return {
                "answer": f"{target_name} 共有 {len(rows)} {unit}。\n\n{content}",
                "sources": [source],
                "confidence": 1.0,
                "abstainReason": "",
                "evidenceIds": [source["chunk_id"]],
            }

    exact_targets = []
    for record in record_list:
        if record.get("is_toc") or int(record.get("direct_child_count") or 0) <= 0:
            continue
        label = _compact_text(str(record.get("label") or ""))
        title = _compact_text(str(record.get("title") or ""))
        display = _compact_text("".join([label, title]))
        # Match if the record's label or display text appears in the query
        if label and label in compact_query:
            exact_targets.append(record)
        elif title and title in compact_query:
            exact_targets.append(record)
        elif display and display in compact_query:
            exact_targets.append(record)
    if exact_targets:
        # Prefer the match whose label is MOST SPECIFICALLY referenced in the query
        # (shortest label that still matches = most precise chapter/section reference)
        target = sorted(
            exact_targets,
            key=lambda item: (
                # Prefer record whose label is directly in query (not just substring of a longer match)
                0 if _compact_text(str(item.get("label") or "")) in compact_query else 1,
                int(item.get("page_start") or 1),
                int(item.get("order") or 0),
            ),
        )[0]
    else:
        target = find_best_structure_record(query, record_list)
    if not target:
        return None
    children = _children(record_list, str(target.get("record_id") or ""))
    if not children:
        return None
    article_children = [row for row in children if row.get("record_type") == "article"]
    section_children = [row for row in children if row.get("record_type") == "section"]
    clause_children = [row for row in children if row.get("record_type") in {"clause", "item"}]

    # If direct children are sections (not articles), recursively collect all descendant articles
    if not article_children and section_children:
        by_id = {r.get("record_id", ""): r for r in record_list}
        all_descendant_articles = []
        def _collect_articles(node_id):
            for child in _children(record_list, node_id):
                if child.get("record_type") == "article":
                    all_descendant_articles.append(child)
                elif child.get("direct_child_count", 0) > 0:
                    _collect_articles(child.get("record_id", ""))
        _collect_articles(str(target.get("record_id") or ""))
        if all_descendant_articles:
            article_children = sorted(all_descendant_articles, key=lambda r: (int(r.get("page_start") or 1), int(r.get("order") or 0)))

    counted = article_children or section_children or clause_children or children
    if article_children:
        labels = [_clean_text(str(row.get("label") or "")) for row in counted]
    else:
        labels = [
            _clean_text(" ".join(part for part in [str(row.get("label") or ""), str(row.get("title") or "")] if part))
            for row in counted
        ]
    labels = [label for label in labels if label]
    target_name = _clean_text(" ".join(part for part in [str(target.get("label") or ""), str(target.get("title") or "")] if part))
    unit = "条" if article_children else ("个直接小节" if section_children else "项")
    first_child_anchor = target_name
    if counted:
        first_child_anchor = _clean_text(
            " ".join(part for part in [str(counted[0].get("label") or ""), str(counted[0].get("title") or "")] if part)
        ) or target_name
    if article_children and len(labels) > 4:
        child_text = f"{labels[0]}至{labels[-1]}"
    else:
        child_text = "、".join(labels)
    content = f"{target_name} 的直接子节点：{child_text}"
    source = {
        "content": content,
        "doc_name": target.get("doc_name", ""),
        "heading": target_name,
        "parent_heading": " > ".join(target.get("heading_path", []) or []),
        "score": 1.0,
        "chunk_id": f"structure-count:{target.get('record_id', '')}",
        "page_number": int(target.get("page_start") or 1),
        "page_count": int(target.get("page_count") or 1),
        "section_title": target_name,
        "highlight_text": first_child_anchor,
        "bbox": None,
        "source_type": "structural_count",
        "layout_type": "structural",
        "parse_confidence": float(target.get("confidence") or 1.0),
        "document_type": "REG",
        "is_ocr_fallback": False,
    }
    # Calibrate confidence: higher when articles found, lower for clause/item fallback
    count_confidence = 1.0 if article_children else (0.85 if section_children else 0.70)
    return {
        "answer": f"{target_name} 共有 {len(counted)} {unit}。\n\n{content}",
        "sources": [source],
        "confidence": count_confidence,
        "abstainReason": "",
        "evidenceIds": [source["chunk_id"]],
    }


def _first_child_title(record: Dict) -> str:
    body = _clean_text(str(record.get("body_text") or ""))
    label = str(record.get("label") or "")
    if label:
        body = re.sub(rf"^{re.escape(label)}\s*", "", body).strip()
    first_line = next((line.strip() for line in body.splitlines() if line.strip()), "")
    first_line = re.sub(r"^[：:\s]+", "", first_line).strip()
    if len(first_line) > 36:
        first_line = first_line[:36].rstrip()
    return first_line


def _join_cn(items: List[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return "、".join(items[:-1]) + "和" + items[-1]


def _dedupe_labels(labels: List[str]) -> List[str]:
    result: List[str] = []
    for label in labels:
        cleaned = _clean_text(label).strip(" ：:；;，,。")
        if cleaned and cleaned not in result:
            result.append(cleaned)
    return result


def _looks_like_category_label(label: str, subject: str) -> bool:
    compact = _compact_text(label)
    compact_subject = _compact_text(subject)
    if not compact or len(compact) > 24:
        return False
    if compact in {f"{compact_subject}分类", f"{compact_subject}的分类", f"{compact_subject}类别", f"{compact_subject}的类别"}:
        return False
    non_category_actions = ("要求", "标牌", "标识", "不得", "下列", "条件", "程序", "申请", "颁发", "取得", "从事", "使用", "设置")
    if any(token in compact for token in non_category_actions):
        return False
    return True


def _extract_explicit_classification_labels(text: str, subject: str) -> List[str]:
    cleaned = _clean_text(text)
    if not cleaned:
        return []
    compact_subject = _compact_text(subject)
    patterns = [
        rf"{re.escape(subject)}\s*(?:包括|分为|分成|分作)\s*(?P<labels>[^。；;\n]{{2,120}})",
        rf"{re.escape(compact_subject)}(?:包括|分为|分成|分作)(?P<labels>[^。；;\n]{{2,120}})",
    ]
    for pattern in patterns:
        match = re.search(pattern, cleaned)
        if not match:
            continue
        raw = match.group("labels")
        raw = re.sub(r"(?:等|两种|二种|几类|类别|分类)$", "", raw).strip(" ：:，,。；;")
        parts = re.split(r"[、，,和及/]|以及", raw)
        labels = _dedupe_labels([part.strip(" ：:，,。；;") for part in parts])
        labels = [label for label in labels if _looks_like_category_label(label, subject)]
        if len(labels) >= 2:
            return labels
    return []


def answer_structural_classification(query: str, records: Iterable[Dict]) -> Optional[Dict]:
    if not any(token in query for token in ("分类", "类别", "分为", "分成", "哪几类", "哪两种", "哪三类")):
        return None
    subject = _extract_classification_subject(query)
    if not subject:
        return None

    all_records = list(records)
    matched_doc_names = set(_matched_doc_names_from_query(query, all_records))
    record_list = [
        record
        for record in all_records
        if not matched_doc_names or str(record.get("doc_name") or "") in matched_doc_names
    ]

    ranked: List[Tuple[float, Dict]] = []
    for record in record_list:
        if record.get("is_toc"):
            continue
        record_type = str(record.get("record_type") or "")
        if record_type not in {"article", "section", "clause"}:
            continue
        if str(record.get("semantic_role") or "") != "classification":
            continue
        title = _clean_text(" ".join(str(record.get(key) or "") for key in ("label", "title", "article_label")))
        body = _clean_text(str(record.get("body_text") or "")[:700])
        title_compact = _compact_text(title)
        body_compact = _compact_text(body)
        explicit_titles = _extract_explicit_classification_labels(str(record.get("body_text") or ""), subject)
        has_subject_title = _has_unmodified_subject(title, subject)
        has_subject_body = _has_unmodified_subject(body[:160], subject)
        has_intent_title = any(token in title_compact for token in ("分类", "类别", "包括", "分为", "分成"))
        has_intent_body = any(token in body_compact[:160] for token in ("分类", "类别", "分为", "分成"))
        score = 0.0
        if explicit_titles:
            score += 12.0
        if has_subject_title:
            score += 7.0
        if has_subject_body:
            score += 2.0
        if has_subject_title and has_intent_title:
            score += 5.0
        elif has_subject_body and has_intent_body:
            score += 2.5
        if record_type == "article":
            score += 1.4
        elif record_type == "section":
            score += 0.8
        else:
            score -= 0.3
        if not has_subject_title and not has_subject_body:
            continue
        if score > 0:
            ranked.append((score, record))

    if not ranked:
        return None
    ranked.sort(key=lambda item: (-item[0], int(item[1].get("page_start") or 1), int(item[1].get("order") or 0)))
    target = ranked[0][1]
    children = _children(record_list, str(target.get("record_id") or ""))
    classification_children = []
    for child in children:
        if child.get("is_toc") or child.get("record_type") not in {"clause", "item"}:
            continue
        title = _first_child_title(child)
        if str(child.get("semantic_role") or "") == "classification" or _looks_like_category_label(title, subject):
            classification_children.append(child)

    explicit_titles = _extract_explicit_classification_labels(str(target.get("body_text") or ""), subject)
    child_titles = [_first_child_title(child) for child in classification_children]
    child_titles = [title for title in _dedupe_labels(child_titles) if _looks_like_category_label(title, subject)]
    target_title = _clean_text(" ".join(part for part in [str(target.get("label") or ""), str(target.get("title") or "")] if part))
    content = _clean_text(str(target.get("body_text") or ""))
    if classification_children:
        content = _clean_text(
            "\n".join(
                [content]
                + [
                    _clean_text(str(child.get("body_text") or ""))
                    for child in classification_children
                    if _clean_text(str(child.get("body_text") or "")) not in content
                ]
            )
        )
    if not content:
        return None

    if explicit_titles:
        answer = f"{subject}分为{_join_cn(explicit_titles)}。"
    elif child_titles:
        answer = f"{subject}分为{_join_cn(child_titles)}。"
    else:
        first_sentence = re.split(r"(?<=[。；;])", content, maxsplit=1)[0].strip() or content[:180]
        answer = first_sentence

    source = {
        "content": content,
        "doc_name": target.get("doc_name", ""),
        "heading": target_title,
        "parent_heading": " > ".join(target.get("heading_path", []) or []),
        "score": 1.0,
        "chunk_id": f"structure-classification:{target.get('record_id', '')}",
        "page_number": int(target.get("page_start") or 1),
        "page_count": int(target.get("page_count") or 1),
        "section_title": target_title,
        "highlight_text": target_title,
        "bbox": None,
        "source_type": "structural_classification",
        "layout_type": "structural",
        "parse_confidence": float(target.get("confidence") or 1.0),
        "document_type": "REG",
        "is_ocr_fallback": False,
        "article_id": target.get("article_id", ""),
        "article_label": target.get("article_label", ""),
        "clause_path": " > ".join(target.get("heading_path", []) or []),
        "chunk_role": "classification",
        "structural_node_id": target.get("record_id", ""),
        "structural_parent_id": target.get("parent_record_id", ""),
        "structural_record_type": target.get("record_type", ""),
    }
    return {
        "answer": answer,
        "sources": [source],
        "confidence": 1.0,
        "abstainReason": "",
        "evidenceIds": [source["chunk_id"]],
    }


def _resolve_doc_name_for_definition(query: str) -> str:
    """Resolve target document name from query to avoid cross-document bleed."""
    compact = _compact_text(query)
    if not compact:
        return ""
    best_len = 0
    best_name = ""
    for file_path in (BASE_DIR.parent / "aerospace_docs").glob("*"):
        if file_path.suffix.lower() not in {".pdf", ".docx"}:
            continue
        stem_key = _compact_text(file_path.stem)
        if stem_key and stem_key in compact and len(stem_key) > best_len:
            best_len = len(stem_key)
            best_name = file_path.name
    return best_name


def _definition_term_aliases(term: str, records: List[Dict]) -> List[str]:
    aliases = [term]
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9-]{1,12}", term or ""):
        return aliases
    paren_pattern = re.compile(rf"(?P<cn>[\u4e00-\u9fff]{{2,24}})\s*[（(]\s*{re.escape(term)}\s*[)）]", re.I)
    for record in records:
        text = _clean_text(str(record.get("body_text") or ""))
        for match in paren_pattern.finditer(text):
            alias = _compact_text(match.group("cn"))
            if alias and alias not in aliases:
                aliases.append(alias)
    return aliases[:4]


def _definition_anchor_score(record: Dict, aliases: List[str]) -> float:
    body = _compact_text(str(record.get("body_text") or "")[:420])
    title = _compact_text("".join(str(record.get(key) or "") for key in ("label", "title")))
    record_type = str(record.get("record_type") or "")
    score = 0.0
    if "定义" in title:
        score += 1.2
    if record_type in {"clause", "item"}:
        score += 0.4
    elif record_type == "article":
        score += 0.2
    if any(token in body[:120] for token in ("不包括在", "不属于", "除", "以外")) and "定义" in body[:180]:
        score -= 4.0
    for alias in aliases:
        if not alias:
            continue
        stripped_label = re.sub(r"^[（(][一二三四五六七八九十百千万\dA-Za-z]+[)）]、?", "", str(record.get("label") or "")).strip()
        if _compact_text(stripped_label) == alias or _compact_text(str(record.get("title") or "")) == alias:
            score += 5.5
        pos = body.find(alias)
        if pos < 0:
            continue
        after = body[pos + len(alias) : pos + len(alias) + 8]
        before = body[max(0, pos - 18) : pos]
        if pos <= 40 and any(marker in after for marker in ("是指", "指", "系指", "定义为", "是")):
            score += 8.0
        elif any(marker in after for marker in ("是指", "系指", "定义为")):
            score += 5.0
        elif "定义" in title and pos <= 120:
            score += 2.2
        elif any(marker in before for marker in ("称为", "简称")):
            score += 1.0
        else:
            score += 0.2
    return score


def answer_structural_definition(query: str, records: Iterable[Dict]) -> Optional[Dict]:
    if "定义" not in query and not query.endswith("是什么"):
        return None
    term = _extract_definition_term(query)
    if not term:
        return None
    record_list = list(records)
    term_aliases = _definition_term_aliases(term, record_list)

    target_doc_name = _resolve_doc_name_for_definition(query)

    definition_records = [
        record
        for record in record_list
        if not record.get("is_toc") and record.get("semantic_role") == "definition"
        and (not target_doc_name or str(record.get("doc_name") or "") == target_doc_name)
    ]

    # If no definitions found in target doc, fall back to all docs
    if not definition_records and target_doc_name:
        definition_records = [
            record
            for record in record_list
            if not record.get("is_toc") and record.get("semantic_role") == "definition"
        ]

    target = None
    if term:
        # First: exact match on title/label
        exact_matches = [
            record
            for record in definition_records
            if _compact_text(str(record.get("title") or record.get("label") or "")) == term
        ]
        if exact_matches:
            target = sorted(exact_matches, key=lambda item: (int(item.get("page_start") or 1), int(item.get("order") or 0)))[0]
    if target is None and term:
        # Second: fuzzy match - term appears as substring in title/label or body_text
        fuzzy_matches = []
        for record in definition_records:
            record_type = str(record.get("record_type") or "")
            label = str(record.get("label") or "")
            title = str(record.get("title") or "")
            compact_title = _compact_text(title)
            compact_label = _compact_text(label)
            # Strip common prefixes like "(八)、" to get the core term
            stripped_label = re.sub(r"^[（(][一二三四五六七八九十百千万\dA-Za-z]+[)）]、?", "", label).strip()
            compact_stripped = _compact_text(stripped_label)
            body = str(record.get("body_text") or "")
            compact_body = _compact_text(body[:180])
            quality = _definition_anchor_score(record, term_aliases)
            if compact_stripped and compact_stripped in term_aliases:
                quality += 1.0
            elif any(alias in compact_title or alias in compact_stripped for alias in term_aliases):
                quality += 0.8
            elif not any(alias in compact_body for alias in term_aliases):
                continue
            if quality >= 2.0:
                fuzzy_matches.append((quality, int(record.get("page_start") or 1), int(record.get("order") or 0), record))
        if fuzzy_matches:
            fuzzy_matches.sort(key=lambda x: (-x[0], x[1], x[2]))
            target = fuzzy_matches[0][3]
    if target is None:
        return None
    if not target:
        return None
    if term:
        title_haystack = _compact_text(" ".join(str(target.get(key) or "") for key in ("label", "title")))
        body_haystack = _compact_text(str(target.get("body_text") or "")[:160])
        if not any(alias in title_haystack or alias in body_haystack for alias in term_aliases):
            return None
        if _definition_anchor_score(target, term_aliases) < 2.0:
            return None
    title = _clean_text(" ".join(part for part in [str(target.get("label") or ""), str(target.get("title") or "")] if part))
    body = _clean_text(str(target.get("body_text") or ""))
    body = re.sub(rf"^{re.escape(str(target.get('label') or ''))}\s*", "", body).strip()
    body = re.sub(rf"^{re.escape(str(target.get('title') or ''))}\s*", "", body).strip()
    if not body:
        return None
    first_sentence = re.split(r"(?<=[。；;])", body, maxsplit=1)[0].strip() or body
    source = {
        "content": body,
        "doc_name": target.get("doc_name", ""),
        "heading": title,
        "parent_heading": " > ".join(target.get("heading_path", []) or []),
        "score": 1.0,
        "chunk_id": f"structure-definition:{target.get('record_id', '')}",
        "page_number": int(target.get("page_start") or 1),
        "page_count": int(target.get("page_count") or 1),
        "section_title": title,
        "highlight_text": title,
        "bbox": None,
        "source_type": "structural_definition",
        "layout_type": "structural",
        "parse_confidence": float(target.get("confidence") or 1.0),
        "document_type": "REG",
        "is_ocr_fallback": False,
    }
    body_term_match = re.search(r"^(?:[（(][一二三四五六七八九十百千万\dA-Za-z]+[)）]|[一二三四五六七八九十]+、|\d+[、.])?\s*(?P<term>[^：:，,。；;\n]{2,24})\s*[：:]\s*指", body)
    term = body_term_match.group("term").strip() if body_term_match else str(target.get("title") or target.get("label") or "该术语")
    return {
        "answer": f"{term}的定义是：{first_sentence}",
        "sources": [source],
        "confidence": 1.0,
        "abstainReason": "",
        "evidenceIds": [source["chunk_id"]],
    }


def attach_structure_to_blocks(blocks: List[dict], records: List[Dict]) -> List[dict]:
    if not blocks or not records:
        return blocks
    candidates = [record for record in records if not record.get("is_toc")]
    if not candidates:
        return blocks

    by_article: Dict[str, List[Dict]] = {}
    page_candidates: Dict[int, List[Dict]] = {}
    for record in candidates:
        article_id = _clean_text(str(record.get("article_id") or record.get("number") or ""))
        if article_id:
            by_article.setdefault(article_id, []).append(record)
        page_start = int(record.get("page_start") or 1)
        page_end = int(record.get("page_end") or page_start)
        for page_number in range(max(1, page_start - 1), page_end + 2):
            page_candidates.setdefault(page_number, []).append(record)

    def candidate_pool(block: dict) -> List[Dict]:
        article_id = _clean_text(str(block.get("article_id") or ""))
        page_number = int(block.get("page_number") or 1)
        if article_id and article_id in by_article:
            pool = list(by_article[article_id])
        else:
            pool = list(page_candidates.get(page_number) or [])

        markers = [
            _clean_text(str(block.get("list_item_id") or "")),
            _clean_text(str(block.get("clause_id") or "")),
        ]
        markers = [marker for marker in markers if marker]
        if markers and pool:
            marker_filtered = [
                record
                for record in pool
                if _clean_text(str(record.get("label") or record.get("number") or "")) in markers
            ]
            if marker_filtered:
                pool = marker_filtered

        if not pool:
            pool = list(page_candidates.get(page_number) or candidates[:160])
        if len(pool) > 180:
            pool = sorted(
                pool,
                key=lambda record: (
                    abs(int(record.get("page_start") or page_number) - page_number),
                    0 if record.get("source") == "body" else 1,
                    int(record.get("order") or 0),
                ),
            )[:180]
        return pool

    total = len(blocks)
    progress_interval = max(100, total // 5) if total >= 300 else 0
    for index, block in enumerate(blocks, start=1):
        query = _clean_text(
            "\n".join(
                str(block.get(key) or "")
                for key in ("heading_path_text", "section_title", "heading_text", "content")
            )
        )
        best = find_best_structure_record(query, candidate_pool(block))
        if not best:
            if progress_interval and index % progress_interval == 0:
                print(f"    * 结构挂载进度：{index}/{total} 块")
            continue
        block["structural_node_id"] = best.get("record_id", "")
        block["structural_parent_id"] = best.get("parent_record_id", "")
        block["structural_record_type"] = best.get("record_type", "")
        block["structural_level"] = int(best.get("level") or 0)
        block["structural_number"] = best.get("number", "")
        block["structural_path"] = " > ".join(best.get("heading_path", []) or [])
        block["structural_source"] = best.get("source", "")
        block["direct_child_count"] = int(best.get("direct_child_count") or 0)
        block["descendant_count"] = int(best.get("descendant_count") or 0)
        if block.get("retrieval_text"):
            block["retrieval_text"] = _clean_text(
                "\n".join(
                    part
                    for part in [
                        block.get("structural_path", ""),
                        block.get("retrieval_text", ""),
                    ]
                    if part
                )
            )
        if progress_interval and index % progress_interval == 0:
            print(f"    * 结构挂载进度：{index}/{total} 块")
    return blocks


def save_structural_records(doc_name: str, records: List[Dict]) -> Path:
    path = _record_cache_path(doc_name)
    path.write_text(json.dumps(_json_utf8_safe(records), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_structural_records(doc_name: str) -> List[Dict]:
    path = _record_cache_path(doc_name)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return payload if isinstance(payload, list) else []


def load_all_structural_records(doc_names: Optional[Iterable[str]] = None) -> List[Dict]:
    records: List[Dict] = []
    wanted = {_record_cache_path(doc_name).name for doc_name in doc_names or []}
    for path in STRUCTURAL_RECORD_DIR.glob("*.json"):
        if wanted and path.name not in wanted:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, list):
            records.extend(payload)
    return records


def remove_structural_records(doc_name: str) -> None:
    path = _record_cache_path(doc_name)
    if path.exists():
        path.unlink()


def clear_structural_records() -> None:
    for path in STRUCTURAL_RECORD_DIR.glob("*.json"):
        path.unlink()
