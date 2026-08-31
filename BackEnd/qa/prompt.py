import logging
import math
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz
from openai import OpenAI

from utils.evidence_scores import ensure_evidence_score_fields, evidence_display_score, evidence_ranking_score
from utils.structure import answer_structural_classification, answer_structural_count, answer_structural_definition, load_all_structural_records
from aerospace_search_main import HybridSearch, analyze_query_intent, infer_query_precision_hints
from utils.retrieval_contract import RetrievalContract, apply_contract, build_retrieval_contract

from search.query_analysis import infer_query_precision_hints, logger
from search.common import DOC_DIR, logger

from ._shared import _is_structural_evidence

QWEN_API_KEY = os.environ.get("QWEN_API_KEY", "")
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL = "qwen-turbo"

BASE_SYSTEM_PROMPT = """你是一个专门服务于航空工程、维修放行与适航审定场景的专业技术问答助手。
你的工作目标不是“整理格式”，而是基于证据直接解决用户问题。

请严格遵守以下规则：

1. 角色定位
- 始终以资深适航工程师、维修工程师或航线排故工程师的专业口吻回答。
- 默认提问者为专业航空从业人员，直接使用专业术语和缩略语，如 CCAR、AMM、FIM、SB、MEL、CDL。

2. 绝对基于证据
- 所有结论、步骤、限制条件都必须来自给定证据。
- 严禁基于常识补充未在证据中出现的技术事实、条款、阈值或判断。
- 如果证据不足、证据不完整，或无法支撑明确结论，直接说明“当前证据不足，无法给出可靠结论”。
- 如果多条证据存在差异，客观列出差异，不替用户做最终工程决策。

3. 输出目标
- 优先回答“用户真正想解决的问题”，不要机械复述问题，不要为了凑格式写成填表式内容。
- 先给明确结论，再补充必要依据、适用条件、限制或步骤。
- 若一句话就能回答清楚，不要强行展开为三段或五段。
- 若问题本质是“分类、定义、适用范围、是否允许、依据是什么”，直接给出判断或归类结果，再补充必要条款信息。

4. 语气要求
- 语言必须客观、克制、精炼、无感情色彩。
- 不要出现“好的”“根据提供的信息”“以下是为您整理的结果”等客套话。
- 不要使用“我认为”“我觉得”“大概”“可能是”这类主观表达；除非原文证据本身使用了不确定措辞。

5. 格式要求
- 使用 Markdown，但格式服务于表达，不要机械套模板。
- 除“故障排查”和“程序步骤”类问题外，优先使用短段落或少量要点，不要固定输出“1.结论 2.证据要点 3.补充说明”。
- 对故障排查类问题，严格使用“现象、可能原因、排查步骤、注意事项”结构。
- 对程序/操作类问题，按“适用条件、操作步骤、注意事项”组织。

6. 合规提示
- 任何涉及飞机具体排故操作、适航放行、维修实施、法规符合性判定的回答，末尾必须附加：
  “注：以上信息仅供参考，最终操作与合规判定需以现行有效的官方技术出版物为准。”
"""


def _extract_pdf_page_text(doc_name: str, page_number: int) -> str:
    safe_name = Path(str(doc_name or "")).name
    if not safe_name.lower().endswith(".pdf") or page_number < 1:
        return ""

    file_path = DOC_DIR / safe_name
    if not file_path.exists():
        return ""

    try:
        document = fitz.open(file_path)
        try:
            if page_number > document.page_count:
                return ""
            return document.load_page(page_number - 1).get_text("text").strip()
        finally:
            document.close()
    except Exception as exc:
        logger.debug("Failed to extract PDF page text for %s page %s: %s", doc_name, page_number, exc)
        return ""


def _enrich_pdf_page_context(row: Dict, include_next_page: bool = False) -> Dict:
    doc_name = str(row.get("doc_name") or "")
    page_number = int(row.get("page_number") or 1)
    page_text = _extract_pdf_page_text(doc_name, page_number)
    if include_next_page:
        next_text = _extract_pdf_page_text(doc_name, page_number + 1)
        if next_text:
            page_text = f"{page_text}\n\n[下一页]\n{next_text}" if page_text else next_text
    if not page_text:
        return row

    enriched = dict(row)
    original = str(row.get("content") or "").strip()
    if original and original in page_text:
        enriched["content"] = page_text[:6000]
    else:
        enriched["content"] = f"{original}\n\n[PDF页原文]\n{page_text}"[:6000] if original else page_text[:6000]
    return enriched


def build_context(search_results: List[Dict]) -> str:
    if not search_results:
        return "【参考证据】\n暂无可用证据。"

    parts = ["【参考证据】"]
    for index, row in enumerate(search_results, start=1):
        doc_name = row.get("doc_name", "未知文档")
        section = row.get("section_title") or row.get("heading") or row.get("parent_heading") or "未标注章节"
        page_number = row.get("page_number", 1)
        source_type = row.get("source_type", "text")
        score = evidence_display_score(row)
        parse_confidence = float(row.get("parse_confidence", 0.0))
        parts.append(
            f"--- 证据 {index} | 文档: {doc_name} | 章节: {section} | 页码: {page_number} | 类型: {source_type} | 相关度: {score:.4f} | 解析置信度: {parse_confidence:.2f} ---"
        )
        parts.append(row.get("content", ""))
        parts.append("")
    return "\n".join(parts)


def prepare_prompt_evidence(search_results: List[Dict], query_type: str) -> List[Dict]:
    include_next_page = query_type == "article_lookup"
    prepared: List[Dict] = []
    for row in search_results:
        if _is_structural_evidence(row):
            prepared.append(row)
        else:
            prepared.append(_enrich_pdf_page_context(row, include_next_page=include_next_page))
    return prepared


def _query_template(query_type: str) -> str:
    if query_type == "fault_diagnosis":
        return (
            "输出要求：\n"
            "- 使用 Markdown。\n"
            "- 严格按“现象、可能原因、排查步骤、注意事项”组织。\n"
            "- 先给直接结论，再展开排查逻辑。\n"
        )
    if query_type == "procedure":
        return (
            "输出要求：\n"
            "- 使用 Markdown。\n"
            "- 按“适用条件、操作步骤、注意事项”组织。\n"
            "- 操作步骤应简洁、可执行，不要写空泛说明。\n"
        )
    if query_type == "article_lookup":
        return (
            "输出要求：\n"
            "- 先直接回答用户问题。\n"
            "- 对分类、定义、适用范围、是否允许这类问题，优先用 1 段结论 + 少量要点说明，不要机械编号。\n"
            "- 仅在确有必要时补充适用条件、限制条款或例外情况。\n"
        )
    return (
        "输出要求：\n"
        "- 先直接回答问题，再补充关键依据。\n"
        "- 优先使用自然段或少量要点，不要写成固定表格式回答。\n"
        "- 如果存在限制条件、例外或适用边界，再单独说明。\n"
    )


def build_history_context(history: Optional[List[Dict[str, Any]]] = None) -> str:
    if not history:
        return ""

    recent = history[-8:]
    lines = ["【对话上下文】"]
    for message in recent:
        role = str(message.get("role") or "").strip()
        content = str(message.get("content") or "").strip()
        if not content:
            continue
        label = "用户" if role == "user" else "助手"
        lines.append(f"{label}: {content[:900]}")
    if len(lines) == 1:
        return ""
    lines.append("仅将对话上下文用于理解追问、指代和纠正题意，不要把对话内容当作文档证据。")
    return "\n".join(lines)


def _normalize_doc_lookup_text(value: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", str(value or "").lower())


def _resolve_doc_name_from_text(text: str) -> str:
    lookup = _normalize_doc_lookup_text(text)
    if not lookup:
        return ""
    candidates = []
    for file_path in DOC_DIR.glob("*"):
        if file_path.suffix.lower() not in {".pdf", ".docx"}:
            continue
        stem_key = _normalize_doc_lookup_text(file_path.stem)
        name_key = _normalize_doc_lookup_text(file_path.name)
        if (stem_key and stem_key in lookup) or (name_key and name_key in lookup):
            candidates.append((len(stem_key), file_path.name))
    if not candidates:
        return ""
    candidates.sort(reverse=True, key=lambda item: item[0])
    return candidates[0][1]


def _payload_source_doc_name(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    sources = payload.get("sources")
    if not isinstance(sources, list):
        return ""
    for source in sources:
        if not isinstance(source, dict):
            continue
        doc_name = str(source.get("doc_name") or "").strip()
        if doc_name:
            return Path(doc_name).name
    return ""


def _extract_history_focus_doc(history: Optional[List[Dict[str, Any]]] = None) -> str:
    if not history:
        return ""

    # Only look at USER messages for document focus, not assistant responses
    # (assistant responses may reference multiple documents)
    for message in reversed(history):
        if str(message.get("role") or "").strip() != "user":
            continue

        doc_name = _payload_source_doc_name(message.get("payload"))
        if doc_name:
            return doc_name

        content = str(message.get("content") or "")
        doc_name = _resolve_doc_name_from_text(content)
        if doc_name:
            return doc_name

        quoted_titles = re.findall(r"《([^》]{2,80})》", content)
        for title in quoted_titles:
            doc_name = _resolve_doc_name_from_text(title)
            if doc_name:
                return doc_name
    return ""


def _is_followup_query(query: str) -> bool:
    compact = query.strip()
    if not compact:
        return False
    precision_hints = infer_query_precision_hints(compact)
    if precision_hints.get("route") and (precision_hints.get("subject_terms") or precision_hints.get("target_doc_name")):
        return False
    if re.search(r"第\s*[0-9一二三四五六七八九十百千万]+\s*条", compact):
        return True
    return any(marker in compact for marker in ["它", "该", "这个", "这条", "上述", "继续", "还有", "上一", "前面", "其中"])


def _build_contextual_search_query(query: str, history: Optional[List[Dict[str, Any]]] = None) -> str:
    if not history:
        return query

    compact_query = query.strip()
    focus_doc_name = _extract_history_focus_doc(history)
    explicit_doc_name = _resolve_doc_name_from_text(compact_query)
    if focus_doc_name and not explicit_doc_name and _is_followup_query(compact_query):
        return "\n".join([Path(focus_doc_name).stem, compact_query])

    needs_context = _is_followup_query(compact_query)
    if not needs_context:
        return query

    # Only use the most recent user turn for context to avoid polluting search with old queries
    recent_user_turns = [
        str(message.get("content") or "").strip()
        for message in history
        if str(message.get("role") or "").strip() == "user" and str(message.get("content") or "").strip()
    ][-1:]
    if not recent_user_turns:
        return query

    return "\n".join([*recent_user_turns, compact_query])


def build_user_prompt(query: str, context: str, query_type: str, history: Optional[List[Dict[str, Any]]] = None) -> str:
    history_context = build_history_context(history)
    return (
        f"{context}\n\n"
        f"{history_context}\n\n"
        f"【问题类型】{query_type}\n"
        f"【用户问题】{query}\n\n"
        f"{_query_template(query_type)}\n"
        "附加要求：\n"
        "- 不要复述题目，不要写客套话。\n"
        "- 不要为了对齐格式而把回答写成僵硬的“填表式”结构。\n"
        "- 如果证据足够，直接给出能解决问题的答案；再补充必要依据。\n"
        "- 若证据不足，请直接拒答，并指出还缺少哪类证据。\n"
    )


def call_qwen(system_prompt: str, user_prompt: str) -> str:
    if not QWEN_API_KEY:
        raise RuntimeError("未设置 QWEN_API_KEY，当前无法调用大模型生成答案。")

    client = OpenAI(api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)
    response = client.chat.completions.create(
        model=QWEN_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=1600,
        timeout=45.0,
    )
    return response.choices[0].message.content or ""

