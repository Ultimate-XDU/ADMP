from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote, urljoin

import requests
from bs4 import BeautifulSoup
from docx import Document as DocxDocument
import fitz

from aerospace_search_main import DOC_DIR, DualDBManager


BASE_DIR = Path(__file__).resolve().parent
DOC_DIR_PATH = Path(DOC_DIR)
STATE_FILE = BASE_DIR / "management_state.json"
IMPORT_REPORT = BASE_DIR / "caac_import_report.json"

SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://app.caac.gov.cn/XXGK/XXGK/index_172.html?fl=58",
    }
)


CHANNELS = [
    {
        "name": "民航规章",
        "kb_id": 101,
        "params": {"page": 1, "channelid": "269689"},
        "base_url": "https://app.caac.gov.cn/was5/web/search",
    },
    {
        "name": "规范性文件",
        "kb_id": 102,
        "params": {"page": 1, "channelid": "238066", "fl": "14"},
        "base_url": "https://app.caac.gov.cn/was5/web/search",
    },
    {
        "name": "法律法规",
        "kb_id": 103,
        "params": {"page": 1, "channelid": "211383", "fl": "12"},
        "base_url": "https://app.caac.gov.cn/was5/web/search",
    },
]


@dataclass
class CaacItem:
    title: str
    url: str
    category: str
    kb_id: int
    date: str = ""
    doc_no: str = ""
    ccar_no: str = ""


def _configure_console() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _fetch(url: str, **kwargs) -> requests.Response:
    response = SESSION.get(url, timeout=30, **kwargs)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or response.encoding or "utf-8"
    return response


def _safe_file_stem(title: str) -> str:
    value = re.sub(r"[\\/:*?\"<>|]+", "_", str(title or "").strip())
    value = re.sub(r"\s+", "", value)
    return value[:120] or f"CAAC文档_{int(time.time())}"


def _normalize_text(text: str) -> str:
    value = str(text or "").replace("\xa0", " ")
    value = re.sub(r"\r\n?", "\n", value)
    value = re.sub(r"[ \t]+\n", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    value = re.sub(r"[ \t]{2,}", " ", value)
    return value.strip()


def _extract_field(text: str, label: str) -> str:
    pattern = rf"{re.escape(label)}[：:\s]*([^\n]+)"
    match = re.search(pattern, text)
    return _normalize_text(match.group(1)) if match else ""


def list_items(channel: dict, max_pages: int = 10) -> list[CaacItem]:
    items: list[CaacItem] = []
    seen = set()
    for page in range(1, max_pages + 1):
        params = dict(channel["params"])
        params["page"] = page
        response = _fetch(channel["base_url"], params=params)
        soup = BeautifulSoup(response.text, "lxml")
        page_links = []
        for anchor in soup.find_all("a", href=True):
            title = _normalize_text(anchor.get_text(" ", strip=True))
            href = urljoin(response.url, anchor["href"])
            if not title or "/XXGK/XXGK/" not in href or not re.search(r"/t\d+_\d+\.html$", href):
                continue
            key = (title, href)
            if key in seen:
                continue
            seen.add(key)
            page_links.append(CaacItem(title=title, url=href, category=channel["name"], kb_id=int(channel["kb_id"])))
        if not page_links:
            break
        items.extend(page_links)
    return items


def parse_detail(item: CaacItem) -> tuple[dict, list[str]]:
    response = _fetch(item.url)
    soup = BeautifulSoup(response.text, "lxml")
    full_text = _normalize_text(soup.get_text("\n", strip=True))

    title_node = soup.find("h1")
    title = _normalize_text(title_node.get_text(" ", strip=True)) if title_node else item.title
    title = title or item.title

    metadata = {
        "title": title,
        "source_url": item.url,
        "category": item.category,
        "date": _extract_field(full_text, "发文日期") or _extract_field(full_text, "成文日期") or item.date,
        "doc_no": _extract_field(full_text, "文号") or item.doc_no,
        "ccar_no": _extract_field(full_text, "部号") or item.ccar_no,
        "office": _extract_field(full_text, "办文单位"),
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
        paragraphs = [p for p in paragraphs if p and p != title]
    else:
        paragraphs = []

    if len("\n".join(paragraphs)) < 500:
        lines = full_text.splitlines()
        try:
            start = next(i for i, line in enumerate(lines) if line.strip() == title)
        except StopIteration:
            start = 0
        stop_terms = {"关于我们", "站点地图", "ICP备案编号", "移动客户端", "国务院 互联网+督查"}
        body_lines = []
        for line in lines[start + 1 :]:
            if line in stop_terms:
                break
            if line in {"大", "中", "小", "打印本页", "分享到：", "来源：中国民航局", "字体："}:
                continue
            body_lines.append(line)
        paragraphs = [_normalize_text(line) for line in body_lines if _normalize_text(line)]

    attachments = []
    for anchor in soup.find_all("a", href=True):
        href = urljoin(response.url, anchor["href"])
        if re.search(r"\.(pdf|docx?|wps)(?:$|\?)", href, re.I):
            attachments.append(href)
    return metadata, paragraphs, attachments


def write_docx(metadata: dict, paragraphs: Iterable[str], target_path: Path) -> None:
    doc = DocxDocument()
    doc.add_heading(metadata["title"], level=1)
    meta_lines = [
        ("来源", "中国民用航空局信息公开"),
        ("原文链接", metadata.get("source_url", "")),
        ("主题分类", metadata.get("category", "")),
        ("办文单位", metadata.get("office", "")),
        ("发文日期", metadata.get("date", "")),
        ("文号", metadata.get("doc_no", "")),
        ("部号", metadata.get("ccar_no", "")),
        ("有效性", metadata.get("validity", "")),
    ]
    for key, value in meta_lines:
        if value:
            doc.add_paragraph(f"{key}：{value}")
    doc.add_paragraph("")
    for paragraph in paragraphs:
        cleaned = _normalize_text(paragraph)
        if cleaned:
            doc.add_paragraph(cleaned)
    doc.save(target_path)


def _attachment_file_name(url: str, fallback_title: str) -> str:
    raw_name = Path(unquote(url.split("?", 1)[0])).name
    suffix = Path(raw_name).suffix.lower()
    if suffix not in {".pdf", ".doc", ".docx"}:
        suffix = ".pdf"
    stem = _safe_file_stem(fallback_title)
    return f"{stem}{suffix}"


def _download_attachment(url: str, fallback_title: str) -> Path:
    target_path = DOC_DIR_PATH / _attachment_file_name(url, fallback_title)
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


def _pdf_quality(path: Path) -> dict:
    try:
        with fitz.open(path) as document:
            page_count = document.page_count
            sample_pages = min(page_count, 8)
            text = "\n".join(document.load_page(index).get_text("text") for index in range(sample_pages))
        chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
        return {
            "ok": page_count >= 2 and chinese_chars >= 800,
            "page_count": page_count,
            "sample_chinese_chars": chinese_chars,
        }
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _docx_quality(path: Path) -> dict:
    try:
        doc = DocxDocument(path)
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        chinese_chars = sum(len(re.findall(r"[\u4e00-\u9fff]", p)) for p in paragraphs)
        return {
            "ok": len(paragraphs) >= 20 and chinese_chars >= 1200,
            "paragraph_count": len(paragraphs),
            "chinese_chars": chinese_chars,
        }
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _choose_primary_attachment(attachments: list[str], title: str) -> str:
    if not attachments:
        return ""
    title_key = _safe_file_stem(title)
    scored = []
    for url in attachments:
        name = _safe_file_stem(Path(unquote(url.split("?", 1)[0])).stem)
        suffix = Path(unquote(url.split("?", 1)[0])).suffix.lower()
        score = 0
        if suffix == ".pdf":
            score += 5
        elif suffix in {".doc", ".docx"}:
            score += 4
        if title_key and (title_key in name or name in title_key):
            score += 8
        if any(token in name for token in ("申请表", "表格", "样式", "附件")) and title_key not in name:
            score -= 5
        scored.append((score, url))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def _format_file_size(size: int) -> str:
    return f"{size / (1024 * 1024):.1f} MB"


def _load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {"knowledgeBases": [], "documents": [], "kbIdCounter": 200, "docIdCounter": 4000}


def _save_state(payload: dict) -> None:
    temp = STATE_FILE.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(STATE_FILE)


def upsert_document_state(file_path: Path, item: CaacItem, metadata: dict) -> None:
    payload = _load_state()
    documents = payload.setdefault("documents", [])
    storage_name = file_path.name
    display_name = file_path.stem
    existing = next((doc for doc in documents if doc.get("storageFileName") == storage_name), None)
    if existing is None:
        payload["docIdCounter"] = int(payload.get("docIdCounter", 4000)) + 1
        existing = {"id": payload["docIdCounter"]}
        documents.insert(0, existing)
    existing.update(
        {
            "fileName": display_name,
            "knowledgeBaseId": item.kb_id,
            "type": file_path.suffix.lstrip(".").upper(),
            "size": _format_file_size(file_path.stat().st_size),
            "chunkCount": int(existing.get("chunkCount") or 0),
            "status": "READY",
            "uploadedAt": time.strftime("%Y-%m-%d %H:%M", time.localtime(file_path.stat().st_mtime)),
            "summary": f"CAAC导入：{metadata.get('category', item.category)} {metadata.get('date', '')}".strip(),
            "storageFileName": storage_name,
            "sourceUrl": item.url,
        }
    )
    _save_state(payload)


def collect_candidates(limit: int, max_pages: int) -> list[CaacItem]:
    candidates: list[CaacItem] = []
    seen_titles = {path.stem for path in DOC_DIR_PATH.glob("*") if path.is_file()}
    seen_urls = set()
    for channel in CHANNELS:
        for item in list_items(channel, max_pages=max_pages):
            if item.url in seen_urls:
                continue
            seen_urls.add(item.url)
            if _safe_file_stem(item.title) in seen_titles:
                continue
            candidates.append(item)
            if len(candidates) >= limit:
                return candidates
    return candidates


def _materialize_item(item: CaacItem) -> dict:
    metadata, paragraphs, attachments = parse_detail(item)
    title = metadata.get("title") or item.title
    primary_attachment = _choose_primary_attachment(attachments, title)
    body_chars = len("\n".join(paragraphs))
    should_use_attachment = bool(primary_attachment) and (Path(primary_attachment.split("?", 1)[0]).suffix.lower() == ".pdf" or body_chars < 1200)
    if should_use_attachment:
        file_path = _download_attachment(primary_attachment, title)
        print(f"  downloaded: {file_path.name} ({file_path.stat().st_size} bytes)")
    else:
        file_path = DOC_DIR_PATH / f"{_safe_file_stem(title)}.docx"
        if file_path.exists():
            print(f"  skip existing: {file_path.name}")
        else:
            write_docx(metadata, paragraphs, file_path)
            print(f"  wrote: {file_path.name} ({file_path.stat().st_size} bytes)")
    quality = _pdf_quality(file_path) if file_path.suffix.lower() == ".pdf" else _docx_quality(file_path)
    return {
        "metadata": metadata,
        "paragraphs": paragraphs,
        "attachments": attachments,
        "file_path": file_path,
        "quality": quality,
    }


def import_documents(limit: int, max_pages: int, index: bool, target_pdf: int = 0, target_docx: int = 0) -> list[dict]:
    DOC_DIR_PATH.mkdir(parents=True, exist_ok=True)
    search_limit = max(limit * 4, target_pdf + target_docx + 40) if (target_pdf or target_docx) else limit
    candidates = collect_candidates(search_limit, max_pages)
    report: list[dict] = []
    manager = DualDBManager() if index else None
    pdf_count = 0
    docx_count = 0

    for idx, item in enumerate(candidates, start=1):
        if target_pdf or target_docx:
            if pdf_count >= target_pdf and docx_count >= target_docx:
                break
        print(f"[{idx}/{len(candidates)}] {item.category}: {item.title}")
        try:
            materialized = _materialize_item(item)
            metadata = materialized["metadata"]
            attachments = materialized["attachments"]
            file_path = materialized["file_path"]
            quality = materialized["quality"]
            suffix = file_path.suffix.lower()
            if not quality.get("ok"):
                print(f"  quality skip: {quality}")
                continue
            if target_pdf or target_docx:
                if suffix == ".pdf" and pdf_count >= target_pdf:
                    print("  quota skip: pdf target already reached")
                    continue
                if suffix in {".doc", ".docx"} and docx_count >= target_docx:
                    print("  quota skip: docx target already reached")
                    continue
                if suffix not in {".pdf", ".doc", ".docx"}:
                    print(f"  suffix skip: {suffix}")
                    continue
            upsert_document_state(file_path, item, metadata)
            chunk_count = 0
            if manager is not None:
                result = manager.add_document(str(file_path))
                if result.success:
                    chunk_count = int(result.chunk_count or 0)
                    payload = _load_state()
                    for doc in payload.get("documents", []):
                        if doc.get("storageFileName") == file_path.name:
                            doc["status"] = "SYNCED"
                            doc["chunkCount"] = chunk_count
                            doc["summary"] = f"CAAC导入并索引成功：{time.strftime('%Y-%m-%d %H:%M:%S')}"
                            break
                    _save_state(payload)
                    print(f"  indexed chunks: {chunk_count}")
                else:
                    print(f"  index failed: {result.message}")
            report.append(
                {
                    "title": metadata.get("title") or item.title,
                    "category": item.category,
                    "kb_id": item.kb_id,
                    "url": item.url,
                    "file": file_path.name,
                    "type": file_path.suffix.lstrip(".").upper(),
                    "quality": quality,
                    "attachments": attachments,
                    "chunk_count": chunk_count,
                }
            )
            if suffix == ".pdf":
                pdf_count += 1
            elif suffix in {".doc", ".docx"}:
                docx_count += 1
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            report.append({"title": item.title, "category": item.category, "url": item.url, "error": f"{type(exc).__name__}: {exc}"})

    IMPORT_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    _configure_console()
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--max-pages", type=int, default=8)
    parser.add_argument("--index", action="store_true")
    parser.add_argument("--target-pdf", type=int, default=0)
    parser.add_argument("--target-docx", type=int, default=0)
    args = parser.parse_args()
    report = import_documents(args.limit, args.max_pages, args.index, target_pdf=args.target_pdf, target_docx=args.target_docx)
    ok = [item for item in report if not item.get("error")]
    print(f"Imported metadata/docx: {len(ok)}/{len(report)}")
    print(f"Report: {IMPORT_REPORT}")
    return 0 if len(ok) == len(report) else 1


if __name__ == "__main__":
    raise SystemExit(main())
