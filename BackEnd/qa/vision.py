"""Page-image context for sources already retrieved by the text chain.

This renders at most two distinct PDF pages. Callers pass the text evidence
prompt; the page images use those same evidence numbers. It does not search
a visual index.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional

import fitz

from search.common import DOC_DIR

from .prompt import BASE_SYSTEM_PROMPT, call_qwen_vision

_VISION_SYSTEM = "只根据给出的页面图像回答问题。依据用证据编号标注，例如 [1]。编号只能来自用户给出的列表。看不清就说明看不清。"
_MAX_PAGES = 2


def render_pdf_page(doc_name: str, page: int) -> bytes:
    path = DOC_DIR / Path(str(doc_name or "")).name
    if path.suffix.lower() != ".pdf" or not path.is_file():
        raise FileNotFoundError(path.name)
    document = fitz.open(path)
    try:
        if page < 1 or page > document.page_count:
            raise IndexError(page)
        pix = document.load_page(page - 1).get_pixmap(matrix=fitz.Matrix(110 / 72, 110 / 72), alpha=False)
        return pix.tobytes("png")
    finally:
        document.close()


def visual_context_for_sources(
    question: str,
    sources: List[dict],
    complete: Optional[Callable[[str, List[bytes]], str]] = None,
    user_prompt: str = "",
) -> Dict:
    pages: List[Dict] = []
    images: List[bytes] = []
    rendered = set()
    for index, source in enumerate(sources):
        if len(pages) >= _MAX_PAGES:
            break
        doc_name = source["doc_name"]
        page = source["page_number"]
        if not doc_name.lower().endswith(".pdf") or page < 1:
            continue
        identity = (doc_name, page)
        if identity in rendered:
            continue
        rendered.add(identity)
        try:
            images.append(render_pdf_page(doc_name, page))
        except Exception as exc:
            return {"used": False, "pages": [], "reason": f"{type(exc).__name__}: {exc}", "answer": ""}
        pages.append({"sourceIndex": index, "page": page})
    if not pages:
        return {"used": False, "pages": [], "reason": "no_pdf_page", "answer": ""}

    labels = "\n".join(f"[{item['sourceIndex'] + 1}] 第 {item['page']} 页" for item in pages)
    if user_prompt:
        prompt = (
            f"{user_prompt.rstrip()}\n\n"
            "【页面图像】\n"
            "这些页面图像使用上面的证据编号。结合文字证据和页面图像回答。\n"
            f"{labels}"
        )
        system_prompt = BASE_SYSTEM_PROMPT
    else:
        prompt = f"问题：{question}\n页面对应的证据编号：\n{labels}"
        system_prompt = _VISION_SYSTEM
    caller = complete or (lambda text, blobs: call_qwen_vision(system_prompt, text, blobs))
    try:
        answer = str(caller(prompt, images) or "").strip()
    except Exception as exc:
        return {"used": False, "pages": pages, "reason": f"{type(exc).__name__}: {exc}", "answer": ""}
    if not answer:
        return {"used": False, "pages": pages, "reason": "empty_vision_answer", "answer": ""}
    return {"used": True, "pages": pages, "reason": "", "answer": answer}
