from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote, urljoin

import fitz
import requests
from bs4 import BeautifulSoup
from docx import Document as DocxDocument

os.environ.setdefault("AEROSPACE_SKIP_UPLOAD_DIR_SYNC", "1")

from api.aerospace_api import ClassifyRequest, classify_document
from aerospace_search_main import DOC_DIR
from utils.state_store import load_state, update_state


BASE_DIR = Path(__file__).resolve().parent
DOC_DIR_PATH = Path(DOC_DIR)
REPORT_FILE = BASE_DIR / "public_reference_import_report.json"

SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://www.caac.gov.cn/XXGK/XXGK/",
    }
)

CAAC_SEARCH_URL = "https://app.caac.gov.cn/was5/web/search"

KB_LAWS = 103
KB_CONVENTIONS = 104
KB_MAINTENANCE = 106

KB_NAMES = {
    KB_LAWS: "法律法规",
    KB_CONVENTIONS: "国际公约",
    KB_MAINTENANCE: "维修手册与技术文档",
}

LAW_KEEP_TOKENS = (
    "民用航空法",
    "飞行基本规则",
    "民用机场管理条例",
    "通用航空飞行管制条例",
    "无人驾驶航空器飞行管理暂行条例",
    "民用航空器国籍登记条例",
    "民用航空器权利登记条例",
    "民用航空安全保卫条例",
    "搜寻援救民用航空器规定",
    "国内航空运输承运人赔偿责任限额规定",
    "行政法规",
)

LAW_SKIP_TOKENS = (
    "修改",
    "废止",
    "决定",
    "通知",
    "意见",
    "分工方案",
    "劳动法",
    "会计法",
    "对外贸易法",
    "计量法",
)

CONVENTION_KEEP_TOKENS = ("公约", "议定书", "协定", "华沙", "蒙特利尔", "东京", "海牙", "开普敦")
CONVENTION_SKIP_TOKENS = ("简介",)

FAA_TECHNICAL_DOCS = [
    {
        "title": "FAA-H-8083-30B Aviation Maintenance Technician Handbook General",
        "url": "https://www.faa.gov/regulations_policies/handbooks_manuals/aviation/amtg_handbook.pdf",
    },
    {
        "title": "FAA-H-8083-31B Aviation Maintenance Technician Handbook Airframe",
        "url": "https://www.faa.gov/regulations_policies/handbooks_manuals/aviation/FAA-H-8083-31B_Aviation_Maintenance_Technician_Handbook.pdf",
    },
    {
        "title": "FAA-H-8083-32B Aviation Maintenance Technician Handbook Powerplant",
        "url": "https://www.faa.gov/regulations_policies/handbooks_manuals/aviation/amt_powerplant_handbook.pdf",
    },
    {
        "title": "AC 43.13-1B Acceptable Methods Techniques and Practices Aircraft Inspection and Repair",
        "url": "https://www.faa.gov/documentlibrary/media/advisory_circular/ac_43.13-1b_w-chg1.pdf",
    },
    {
        "title": "AC 43.13-2B Acceptable Methods Techniques and Practices Aircraft Alterations",
        "url": "https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC%2043.13-2B.pdf",
    },
    {
        "title": "AC 120-16G Air Carrier Maintenance Programs",
        "url": "https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_120-16G.pdf",
    },
    {
        "title": "AC 20-106 Aircraft Inspection for the General Aviation Aircraft Owner",
        "url": "https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_20-106.pdf",
    },
    {
        "title": "AC 145-9A Guide for Developing and Evaluating Repair Station and Quality Control Manuals",
        "url": "https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_145-9A.pdf",
    },
    {
        "title": "FAA Human Factors Guide for Aviation Maintenance and Inspection",
        "url": "https://www.faa.gov/sites/faa.gov/files/about/initiatives/maintenance_hf/training_tools/HF_Guide.pdf",
    },
]


@dataclass
class ImportItem:
    title: str
    url: str
    expected_kb_id: int
    source_name: str
    page_url: str = ""
    classified_kb_id: int = 0
    classification_reason: str = ""


def _configure_console() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _fetch(url: str, **kwargs) -> requests.Response:
    response = SESSION.get(url, timeout=45, **kwargs)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or response.encoding or "utf-8"
    return response


def _normalize_text(text: str) -> str:
    value = str(text or "").replace("\xa0", " ")
    value = value.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    value = re.sub(r"\r\n?", "\n", value)
    value = re.sub(r"[ \t]+\n", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    value = re.sub(r"[ \t]{2,}", " ", value)
    return value.strip()


def _safe_file_stem(title: str) -> str:
    value = re.sub(r"[\\/:*?\"<>|]+", "_", _normalize_text(title))
    value = re.sub(r"\s+", "", value)
    return value[:120] or f"public_reference_{int(time.time())}"


def _format_file_size(size: int) -> str:
    return f"{size / (1024 * 1024):.1f} MB"


def _default_state() -> dict:
    return {"knowledgeBases": [], "documents": [], "kbIdCounter": 200, "docIdCounter": 4000}


def _load_state() -> dict:
    return load_state() or _default_state()


def _extract_field(text: str, label: str) -> str:
    pattern = rf"{re.escape(label)}[：:\s]*([^\n]+)"
    match = re.search(pattern, text)
    return _normalize_text(match.group(1)) if match else ""


def _list_caac_items(fl: str, kb_id: int, source_name: str, max_pages: int) -> list[ImportItem]:
    items: list[ImportItem] = []
    seen: set[tuple[str, str]] = set()
    for page in range(1, max_pages + 1):
        response = _fetch(CAAC_SEARCH_URL, params={"page": page, "channelid": "211383", "fl": fl})
        soup = BeautifulSoup(response.text, "lxml")
        page_items: list[ImportItem] = []
        for anchor in soup.find_all("a", href=True):
            title = _normalize_text(anchor.get_text(" ", strip=True))
            href = urljoin(response.url, anchor["href"])
            if not title or "/XXGK/XXGK/" not in href or not re.search(r"/t\d+_\d+\.html$", href):
                continue
            key = (title, href)
            if key in seen:
                continue
            seen.add(key)
            page_items.append(ImportItem(title=title, url=href, expected_kb_id=kb_id, source_name=source_name))
        if not page_items:
            break
        items.extend(page_items)
    return items


def _law_allowed(item: ImportItem) -> tuple[bool, str]:
    if "/FLFG/" not in item.url:
        return False, "not_caac_law_url"
    if any(token in item.title for token in LAW_SKIP_TOKENS):
        if not any(token in item.title for token in ("民用航空法", "飞行基本规则")):
            return False, "amendment_or_unrelated_law"
    if any(token in item.title for token in LAW_KEEP_TOKENS):
        return True, "caac_law_or_administrative_regulation"
    return False, "title_not_in_law_whitelist"


def _convention_allowed(item: ImportItem) -> tuple[bool, str]:
    if "/GJGY/" not in item.url:
        return False, "not_caac_convention_url"
    if any(token in item.title for token in CONVENTION_SKIP_TOKENS):
        return False, "convention_intro_not_primary_text"
    if any(token in item.title for token in CONVENTION_KEEP_TOKENS):
        return True, "caac_international_convention"
    return False, "title_not_convention"


def _parse_caac_detail(item: ImportItem) -> tuple[dict, list[str], list[str]]:
    response = _fetch(item.url)
    soup = BeautifulSoup(response.text, "lxml")
    full_text = _normalize_text(soup.get_text("\n", strip=True))
    title_node = soup.find("h1")
    title = _normalize_text(title_node.get_text(" ", strip=True)) if title_node else item.title
    metadata = {
        "title": title or item.title,
        "source_url": item.url,
        "source_name": item.source_name,
        "category": KB_NAMES[item.classified_kb_id or item.expected_kb_id],
        "office": _extract_field(full_text, "办文单位"),
        "date": _extract_field(full_text, "发文日期") or _extract_field(full_text, "成文日期"),
        "doc_no": _extract_field(full_text, "文号"),
        "validity": _extract_field(full_text, "有效性"),
    }
    content_root = (
        soup.select_one(".TRS_Editor")
        or soup.select_one(".Custom_UnionStyle")
        or soup.select_one(".article-content")
        or soup.select_one("#zoom")
        or soup.select_one(".content")
    )
    if content_root:
        paragraphs = [_normalize_text(p.get_text(" ", strip=True)) for p in content_root.find_all(["p", "div", "span"])]
    else:
        lines = full_text.splitlines()
        start = 0
        for index, line in enumerate(lines):
            if line == "分享到：" or line == metadata["title"]:
                start = index + 1
        stop_terms = {"关于我们", "站点地图", "ICP备案编号", "移动客户端", "国务院 互联网+督查"}
        paragraphs = []
        for line in lines[start:]:
            if line in stop_terms:
                break
            if line in {"大", "中", "小", "打印本页", "分享到：", "来源：中国民航局", "字体："}:
                continue
            paragraphs.append(_normalize_text(line))
    paragraphs = [p for p in paragraphs if p and p != metadata["title"]]
    attachments = []
    for anchor in soup.find_all("a", href=True):
        href = urljoin(response.url, anchor["href"])
        if re.search(r"\.(pdf|docx?|wps)(?:$|\?)", href, re.I):
            attachments.append(href)
    return metadata, paragraphs, attachments


def _write_docx(metadata: dict, paragraphs: Iterable[str], target_path: Path) -> None:
    document = DocxDocument()
    document.add_heading(metadata["title"], level=1)
    for key, value in [
        ("来源", metadata.get("source_name", "")),
        ("原文链接", metadata.get("source_url", "")),
        ("知识库分类", metadata.get("category", "")),
        ("办文单位", metadata.get("office", "")),
        ("发文日期", metadata.get("date", "")),
        ("文号", metadata.get("doc_no", "")),
        ("有效性", metadata.get("validity", "")),
    ]:
        if value:
            document.add_paragraph(f"{key}：{value}")
    document.add_paragraph("")
    for paragraph in paragraphs:
        cleaned = _normalize_text(paragraph)
        if cleaned:
            document.add_paragraph(cleaned)
    document.save(target_path)


def _download_file(url: str, title: str, suffix: str = "") -> Path:
    raw_name = Path(unquote(url.split("?", 1)[0])).name
    detected_suffix = Path(raw_name).suffix.lower()
    target_suffix = suffix or detected_suffix or ".pdf"
    target_path = DOC_DIR_PATH / f"{_safe_file_stem(title)}{target_suffix}"
    if target_path.exists() and target_path.stat().st_size > 0:
        return target_path
    response = _fetch(url, stream=True)
    temp_path = target_path.with_suffix(f"{target_path.suffix}.downloading")
    with temp_path.open("wb") as output:
        for chunk in response.iter_content(chunk_size=1024 * 512):
            if chunk:
                output.write(chunk)
    temp_path.replace(target_path)
    return target_path


def _choose_primary_attachment(attachments: list[str]) -> str:
    pdfs = [url for url in attachments if Path(url.split("?", 1)[0]).suffix.lower() == ".pdf"]
    return pdfs[0] if pdfs else (attachments[0] if attachments else "")


def _pdf_quality(path: Path) -> dict:
    try:
        with fitz.open(path) as document:
            page_count = document.page_count
            sample_pages = min(page_count, 8)
            text = "\n".join(document.load_page(index).get_text("text") for index in range(sample_pages))
        visible_chars = len(re.sub(r"\s+", "", text))
        return {"ok": page_count >= 2 and visible_chars >= 800, "page_count": page_count, "sample_chars": visible_chars}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _docx_quality(path: Path) -> dict:
    try:
        document = DocxDocument(path)
        paragraphs = [p.text.strip() for p in document.paragraphs if p.text.strip()]
        chars = sum(len(re.sub(r"\s+", "", paragraph)) for paragraph in paragraphs)
        return {"ok": len(paragraphs) >= 15 and chars >= 1000, "paragraph_count": len(paragraphs), "chars": chars}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _upsert_document_state(file_path: Path, item: ImportItem, metadata: dict, status: str = "READY") -> None:
    storage_name = file_path.name

    def mutate(payload: dict) -> bool:
        if not payload:
            payload.update(_default_state())
        documents = payload.setdefault("documents", [])
        existing = next((doc for doc in documents if doc.get("storageFileName") == storage_name), None)
        if existing is None:
            payload["docIdCounter"] = int(payload.get("docIdCounter", 4000)) + 1
            existing = {"id": payload["docIdCounter"]}
            documents.insert(0, existing)
        existing.update(
            {
                "fileName": file_path.stem,
                "knowledgeBaseId": item.classified_kb_id,
                "type": file_path.suffix.lstrip(".").upper(),
                "size": _format_file_size(file_path.stat().st_size),
                "chunkCount": int(existing.get("chunkCount") or 0),
                "status": status,
                "uploadedAt": time.strftime("%Y-%m-%d %H:%M", time.localtime(file_path.stat().st_mtime)),
                "summary": f"公开资料导入：{KB_NAMES[item.classified_kb_id]}，来源={item.source_name}，分类依据={item.classification_reason}",
                "storageFileName": storage_name,
                "sourceUrl": item.page_url or item.url,
            }
        )
        return True

    update_state(mutate)


def collect_candidates(max_pages: int) -> tuple[list[ImportItem], list[dict]]:
    audit: list[dict] = []
    accepted: list[ImportItem] = []
    for item in _list_caac_items("12", KB_LAWS, "中国民用航空局-法律法规", max_pages):
        ok, reason = _law_allowed(item)
        audit.append({"title": item.title, "url": item.url, "expected_kb_id": item.expected_kb_id, "accepted": ok, "reason": reason})
        if ok:
            accepted.append(item)
    for item in _list_caac_items("17", KB_CONVENTIONS, "中国民用航空局-国际公约", max_pages):
        ok, reason = _convention_allowed(item)
        audit.append({"title": item.title, "url": item.url, "expected_kb_id": item.expected_kb_id, "accepted": ok, "reason": reason})
        if ok:
            accepted.append(item)
    for item in FAA_TECHNICAL_DOCS:
        accepted.append(ImportItem(title=item["title"], url=item["url"], expected_kb_id=KB_MAINTENANCE, source_name="FAA official public technical document"))
        audit.append({"title": item["title"], "url": item["url"], "expected_kb_id": KB_MAINTENANCE, "accepted": True, "reason": "faa_public_maintenance_or_technical_pdf"})
    return accepted, audit


def _classify_item(item: ImportItem) -> tuple[bool, dict]:
    response = classify_document(ClassifyRequest(fileName=item.title))
    suggested_id = int(response.suggestedKbId or 0)
    item.classified_kb_id = suggested_id
    item.classification_reason = response.reason or ""
    allowed = suggested_id == item.expected_kb_id and suggested_id in {KB_LAWS, KB_CONVENTIONS, KB_MAINTENANCE}
    return allowed, {
        "title": item.title,
        "expected_kb_id": item.expected_kb_id,
        "classified_kb_id": suggested_id,
        "classified_kb_name": response.suggestedKbName,
        "classification_reason": response.reason,
        "accepted": allowed,
    }


def import_documents(limit: int, max_pages: int, dry_run: bool = False) -> list[dict]:
    DOC_DIR_PATH.mkdir(parents=True, exist_ok=True)
    candidates, audit = collect_candidates(max_pages)
    state = _load_state()
    seen_storage = {
        Path(str(document.get("storageFileName") or "")).stem
        for document in state.get("documents", [])
        if document.get("storageFileName")
    }
    report: list[dict] = [{"audit": audit}]
    imported = 0
    for index, item in enumerate(candidates, start=1):
        if imported >= limit:
            break
        if _safe_file_stem(item.title) in seen_storage:
            report.append({"title": item.title, "expected_kb_id": item.expected_kb_id, "skipped": "already_exists"})
            continue
        classification_ok, classification = _classify_item(item)
        report.append({"classification": classification})
        if not classification_ok:
            print(
                f"[{index}/{len(candidates)}] classification skip: expected KB{item.expected_kb_id}, "
                f"got KB{classification['classified_kb_id']} {item.title}"
            )
            continue

        print(f"[{index}/{len(candidates)}] KB{item.classified_kb_id} {KB_NAMES[item.classified_kb_id]}: {item.title}")
        try:
            metadata = {"title": item.title, "source_url": item.url, "source_name": item.source_name, "category": KB_NAMES[item.classified_kb_id]}
            if item.classified_kb_id in {KB_LAWS, KB_CONVENTIONS}:
                metadata, paragraphs, attachments = _parse_caac_detail(item)
                primary = _choose_primary_attachment(attachments)
                if primary:
                    file_path = _download_file(primary, metadata["title"], ".pdf")
                    item.page_url = item.url
                    print(f"  downloaded attachment: {file_path.name}")
                else:
                    file_path = DOC_DIR_PATH / f"{_safe_file_stem(metadata['title'])}.docx"
                    _write_docx(metadata, paragraphs, file_path)
                    print(f"  wrote docx: {file_path.name}")
            else:
                file_path = _download_file(item.url, item.title, ".pdf")
                print(f"  downloaded pdf: {file_path.name}")

            quality = _pdf_quality(file_path) if file_path.suffix.lower() == ".pdf" else _docx_quality(file_path)
            if not quality.get("ok") and item.classified_kb_id in {KB_LAWS, KB_CONVENTIONS}:
                fallback_path = DOC_DIR_PATH / f"{_safe_file_stem(metadata['title'])}.docx"
                if not fallback_path.exists():
                    _write_docx(metadata, paragraphs, fallback_path)
                fallback_quality = _docx_quality(fallback_path)
                if fallback_quality.get("ok"):
                    print(f"  pdf quality low, using docx body: {fallback_path.name}")
                    file_path = fallback_path
                    quality = fallback_quality
            if not quality.get("ok"):
                report.append({"title": item.title, "kb_id": item.classified_kb_id, "file": file_path.name, "error": "quality_skip", "quality": quality})
                print(f"  quality skip: {quality}")
                continue
            if not dry_run:
                _upsert_document_state(file_path, item, metadata)
            imported += 1
            report.append(
                {
                    "title": metadata.get("title") or item.title,
                    "kb_id": item.classified_kb_id,
                    "kb_name": KB_NAMES[item.classified_kb_id],
                    "file": file_path.name,
                    "type": file_path.suffix.lstrip(".").upper(),
                    "source_url": item.page_url or item.url,
                    "classification_reason": item.classification_reason,
                    "quality": quality,
                }
            )
        except Exception as exc:
            report.append({"title": item.title, "kb_id": item.classified_kb_id or item.expected_kb_id, "url": item.url, "error": f"{type(exc).__name__}: {exc}"})
            print(f"  FAILED: {type(exc).__name__}: {exc}")
    REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    _configure_console()
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--max-pages", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    report = import_documents(args.limit, args.max_pages, dry_run=args.dry_run)
    ok = [item for item in report[1:] if item.get("file") and not item.get("error")]
    errors = [item for item in report[1:] if item.get("error")]
    print(f"Imported: {len(ok)}")
    print(f"Errors/skips: {len(errors)}")
    print(f"Report: {REPORT_FILE}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
