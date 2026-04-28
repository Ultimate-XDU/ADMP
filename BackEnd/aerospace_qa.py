import logging
import math
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz
from openai import OpenAI

from aerospace_search_main import HybridSearch, analyze_query_intent


_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())

QWEN_API_KEY = os.environ.get("QWEN_API_KEY", "")
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL = "qwen-turbo"
logger = logging.getLogger(__name__)
DOC_DIR = Path(__file__).parent / "aerospace_docs"


@lru_cache(maxsize=256)
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


def build_context(search_results: List[Dict]) -> str:
    if not search_results:
        return "【参考证据】\n暂无可用证据。"

    parts = ["【参考证据】"]
    for index, row in enumerate(search_results, start=1):
        doc_name = row.get("doc_name", "未知文档")
        section = row.get("section_title") or row.get("heading") or row.get("parent_heading") or "未标注章节"
        page_number = row.get("page_number", 1)
        source_type = row.get("source_type", "text")
        score = float(row.get("score", 0.0))
        parse_confidence = float(row.get("parse_confidence", 0.0))
        parts.append(
            f"--- 证据 {index} | 文档: {doc_name} | 章节: {section} | 页码: {page_number} | 类型: {source_type} | 相关度: {score:.4f} | 解析置信度: {parse_confidence:.2f} ---"
        )
        parts.append(row.get("content", ""))
        parts.append("")
    return "\n".join(parts)


def prepare_prompt_evidence(search_results: List[Dict], query_type: str) -> List[Dict]:
    include_next_page = query_type == "article_lookup"
    return [_enrich_pdf_page_context(row, include_next_page=include_next_page) for row in search_results]


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

    for message in reversed(history):
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
    if len(compact) <= 36:
        return True
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

    recent_user_turns = [
        str(message.get("content") or "").strip()
        for message in history
        if str(message.get("role") or "").strip() == "user" and str(message.get("content") or "").strip()
    ][-3:]
    if not recent_user_turns:
        return query

    needs_context = _is_followup_query(compact_query)
    if not needs_context:
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


def _estimate_confidence(search_results: List[Dict]) -> float:
    if not search_results:
        return 0.0
    top_score = float(search_results[0].get("score", 0.0))
    score_component = 1.0 / (1.0 + math.exp(-max(min(top_score, 10.0), -10.0)))
    parse_component = sum(float(row.get("parse_confidence", 0.0)) for row in search_results[:3]) / max(min(len(search_results), 3), 1)
    diversity_component = min(len(search_results) / 3.0, 1.0)
    confidence = 0.55 * score_component + 0.25 * parse_component + 0.20 * diversity_component
    return round(confidence, 4)


def _select_prompt_evidence(search_results: List[Dict], query_type: str) -> List[Dict]:
    if not search_results:
        return []

    selected = []
    seen_keys = set()
    max_items = 4 if query_type in {"fault_diagnosis", "procedure"} else 3

    for row in search_results:
        key = (row.get("doc_name", ""), row.get("page_number", 1), row.get("section_title") or row.get("heading") or "")
        if key in seen_keys:
            continue
        seen_keys.add(key)
        selected.append(row)
        if len(selected) >= max_items:
            break
    return selected


def _abstain_message(query_type: str) -> str:
    if query_type == "fault_diagnosis":
        return "当前证据不足，无法给出可靠的故障排查结论。建议补充故障现象、相关章节或对应机型手册后再查询。"
    if query_type == "article_lookup":
        return "当前证据不足，无法确认对应条款或适用条件。建议补充更明确的条款编号、章节号或文档名称。"
    if query_type == "procedure":
        return "当前证据不足，无法给出可靠的操作步骤。建议补充具体部件、章节或维护任务编号。"
    return "当前证据不足，无法给出可靠结论。建议补充更具体的问题描述或相关章节信息。"


def _llm_unavailable_message(search_results: List[Dict], query_type: str) -> str:
    if not search_results:
        return _abstain_message(query_type)

    top = search_results[0]
    doc_name = top.get("doc_name", "检索文档")
    page_number = top.get("page_number", 1)
    section = top.get("section_title") or top.get("heading") or top.get("parent_heading") or "相关章节"
    return f"已检索到相关证据，但当前生成服务不可用。可先查看《{doc_name}》第 {page_number} 页的“{section}”原文。"


def _search_error_message() -> str:
    return "检索服务暂时不可用，当前无法返回可靠答案。请稍后重试并检查 Elasticsearch、Milvus 与模型服务状态。"


def ask(query: str, engine: HybridSearch = None) -> str:
    payload = ask_structured(query, engine=engine)
    answer = payload.get("answer", "")
    sources = payload.get("sources", [])
    lines = ["=" * 60, f"问题：{query}", "=" * 60, "", answer, "", "-" * 60, "参考来源："]
    for index, row in enumerate(sources, start=1):
        section = row.get("section_title") or row.get("heading") or row.get("parent_heading") or "未标注章节"
        lines.append(f"[{index}] {row.get('doc_name', '未知文档')} | 第 {row.get('page_number', 1)} 页 | {section}")
    lines.append("=" * 60)
    return "\n".join(lines)


_ARTICLE_COUNT_MARKERS = (
    "\u4e00\u5171\u6709\u51e0\u6761",
    "\u5171\u6709\u51e0\u6761",
    "\u603b\u5171\u51e0\u6761",
    "\u6709\u591a\u5c11\u6761",
    "\u5171\u591a\u5c11\u6761",
    "\u51e0\u6761",
    "\u591a\u5c11\u6761",
)

_CN_NUMERAL_VALUES = {
    "\u96f6": 0,
    "\u3007": 0,
    "\u4e00": 1,
    "\u4e8c": 2,
    "\u4e24": 2,
    "\u4e09": 3,
    "\u56db": 4,
    "\u4e94": 5,
    "\u516d": 6,
    "\u4e03": 7,
    "\u516b": 8,
    "\u4e5d": 9,
}


def _normalize_lookup_text(value: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", str(value or "").lower())


def _chinese_article_number_to_int(value: str) -> Optional[int]:
    token = re.sub(r"\s+", "", str(value or ""))
    if not token:
        return None
    if token.isdigit():
        return int(token)
    if all(ch in _CN_NUMERAL_VALUES for ch in token):
        total = 0
        for ch in token:
            total = total * 10 + _CN_NUMERAL_VALUES[ch]
        return total
    if "\u5341" in token:
        left, _, right = token.partition("\u5341")
        tens = _CN_NUMERAL_VALUES.get(left, 1) if left else 1
        ones = _CN_NUMERAL_VALUES.get(right, 0) if right else 0
        return tens * 10 + ones
    return None


def _find_pdf_named_in_query(query: str) -> Optional[Path]:
    query_key = _normalize_lookup_text(query)
    candidates = []
    for file_path in DOC_DIR.glob("*.pdf"):
        stem_key = _normalize_lookup_text(file_path.stem)
        if stem_key and stem_key in query_key:
            candidates.append((len(stem_key), file_path))
    if not candidates:
        return None
    candidates.sort(reverse=True, key=lambda item: item[0])
    return candidates[0][1]


def _find_pdf_by_query_reference(query: str) -> Optional[Path]:
    named = _find_pdf_named_in_query(query)
    if named:
        return named

    date_match = re.search(r"(\d{4})\s*\u5e74\s*(\d{1,2})\s*\u6708\s*(\d{1,2})\s*\u65e5", query)
    if not date_match:
        return None
    normalized_date = f"{int(date_match.group(1))}\u5e74{int(date_match.group(2))}\u6708{int(date_match.group(3))}\u65e5"

    for file_path in DOC_DIR.glob("*.pdf"):
        try:
            document = fitz.open(file_path)
            try:
                front_text = "\n".join(
                    document.load_page(index).get_text("text")
                    for index in range(min(document.page_count, 2))
                )
            finally:
                document.close()
        except Exception:
            continue
        if normalized_date in re.sub(r"\s+", "", front_text):
            return file_path
    return None


@lru_cache(maxsize=128)
def _count_pdf_articles(file_name: str) -> Optional[Dict]:
    file_path = DOC_DIR / Path(file_name).name
    if not file_path.exists() or file_path.suffix.lower() != ".pdf":
        return None

    articles = []
    try:
        document = fitz.open(file_path)
        try:
            for page_index in range(document.page_count):
                page_text = document.load_page(page_index).get_text("text")
                for match in re.finditer(r"\u7b2c\s*([0-9\u96f6\u3007\u4e00\u4e8c\u4e24\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u767e\u5343]+)\s*\u6761", page_text):
                    number = _chinese_article_number_to_int(match.group(1))
                    if number is None or number <= 0 or number > 300:
                        continue
                    start = max(0, match.start() - 80)
                    end = min(len(page_text), match.start() + 360)
                    articles.append(
                        {
                            "number": number,
                            "label": match.group(0),
                            "page_number": page_index + 1,
                            "excerpt": re.sub(r"\s+", " ", page_text[start:end]).strip(),
                        }
                    )
        finally:
            page_count = document.page_count
            document.close()
    except Exception as exc:
        logger.debug("Failed to count PDF articles for %s: %s", file_name, exc)
        return None

    unique = {}
    for item in articles:
        unique.setdefault(item["number"], item)
    if not unique:
        return None

    numbers = sorted(unique)
    max_number = numbers[-1]
    is_consecutive = numbers == list(range(1, max_number + 1))
    count = max_number if is_consecutive else len(numbers)
    last = unique[max_number]
    labels = "\u3001".join(unique[number]["label"] for number in numbers)
    content = (
        f"\u4ece PDF \u539f\u6587\u8bc6\u522b\u5230\u7684\u6761\u6b3e\u7f16\u53f7\uff1a{labels}\u3002\n"
        f"\u6700\u540e\u4e00\u6761\u4e3a {last['label']}\uff1a{last['excerpt']}"
    )
    return {
        "count": count,
        "last": last,
        "content": content,
        "page_count": page_count,
        "numbers": numbers,
    }


def _try_answer_article_count(query: str, query_type: str) -> Optional[Dict]:
    if not any(marker in query for marker in _ARTICLE_COUNT_MARKERS):
        return None
    file_path = _find_pdf_by_query_reference(query)
    if not file_path:
        return None
    counted = _count_pdf_articles(file_path.name)
    if not counted:
        return None

    stem = file_path.stem
    count = int(counted["count"])
    last = counted["last"]
    source = {
        "content": counted["content"],
        "doc_name": file_path.name,
        "heading": stem,
        "parent_heading": "",
        "score": 1.0,
        "chunk_id": f"pdf-article-count:{file_path.name}",
        "page_number": int(last.get("page_number", 1)),
        "page_count": int(counted.get("page_count", 1)),
        "section_title": stem,
        "highlight_text": str(last.get("label", "")),
        "bbox": None,
        "source_type": "pdf_text",
        "layout_type": "single_column",
        "parse_confidence": 1.0,
        "document_type": "REG",
        "is_ocr_fallback": False,
    }
    answer = (
        f"\u300a{stem}\u300b\u5171\u6709 {count} \u6761\u3002\n\n"
        f"\u4f9d\u636e PDF \u539f\u6587\u7684\u6761\u6b3e\u7f16\u53f7\u7edf\u8ba1\uff0c"
        f"\u8be5\u89c4\u5b9a\u4ece\u7b2c\u4e00\u6761\u7f16\u81f3{last.get('label', '')}\u3002"
    )
    return {
        "answer": answer,
        "sources": [source],
        "confidence": 1.0,
        "abstainReason": "",
        "evidenceIds": [source["chunk_id"]],
        "queryType": query_type,
    }


@lru_cache(maxsize=128)
def _extract_pdf_applicability_section(file_name: str) -> Optional[Dict]:
    file_path = DOC_DIR / Path(file_name).name
    if not file_path.exists() or file_path.suffix.lower() != ".pdf":
        return None

    try:
        document = fitz.open(file_path)
        try:
            page_count = document.page_count
            pages = []
            for page_index in range(min(page_count, 4)):
                pages.append((page_index + 1, document.load_page(page_index).get_text("text")))
        finally:
            document.close()
    except Exception as exc:
        logger.debug("Failed to extract applicability section for %s: %s", file_name, exc)
        return None

    combined = "\n".join(text for _, text in pages)
    pattern = re.compile(
        r"(?P<label>(?:\d+|[一二三四五六七八九十]+)\s*[．\.\u3001]\s*\u9002\u7528\u8303\u56f4)\s*"
        r"(?P<body>.*?)(?=\n\s*(?:\d+|[一二三四五六七八九十]+)\s*[．\.\u3001]\s*[\u4e00-\u9fff]{2,20}|$)",
        re.S,
    )
    match = pattern.search(combined)
    if not match:
        return None

    label = re.sub(r"\s+", "", match.group("label"))
    raw_section = f"{match.group('label')}\n{match.group('body')}"
    section = re.sub(r"\n\s*\u2014\s*\d+\s*\u2014\s*", "\n", raw_section)
    section = re.sub(r"[ \t]+", " ", section)
    section = re.sub(r"\n{3,}", "\n\n", section).strip()
    compact_section = re.sub(r"\s+", "", section)

    page_number = 1
    for current_page, text in pages:
        if "\u9002\u7528\u8303\u56f4" in text:
            page_number = current_page
            break

    return {
        "label": label,
        "section": section,
        "compact_section": compact_section,
        "page_number": page_number,
        "page_count": page_count,
    }


def _try_answer_document_applicability(query: str, query_type: str) -> Optional[Dict]:
    if "\u9002\u7528\u8303\u56f4" not in query:
        return None
    if not any(marker in query for marker in ("\u4e0b\u53d1", "\u6587\u4ef6", "\u54a8\u8be2\u901a\u544a", "\u672c\u6587\u4ef6")):
        return None

    file_path = _find_pdf_by_query_reference(query)
    if not file_path:
        return None
    extracted = _extract_pdf_applicability_section(file_path.name)
    if not extracted:
        return None

    stem = file_path.stem
    section = extracted["section"]
    answer_body = re.sub(r"^\s*(?:\d+|[一二三四五六七八九十]+)\s*[．\.\u3001]\s*\u9002\u7528\u8303\u56f4\s*", "", section, flags=re.S).strip()
    source = {
        "content": section,
        "doc_name": file_path.name,
        "heading": extracted["label"],
        "parent_heading": stem,
        "score": 1.0,
        "chunk_id": f"pdf-applicability:{file_path.name}",
        "page_number": int(extracted["page_number"]),
        "page_count": int(extracted["page_count"]),
        "section_title": extracted["label"],
        "highlight_text": extracted["label"],
        "bbox": None,
        "source_type": "pdf_text",
        "layout_type": "single_column",
        "parse_confidence": 1.0,
        "document_type": "AC",
        "is_ocr_fallback": False,
    }
    return {
        "answer": f"\u300a{stem}\u300b\u7684\u9002\u7528\u8303\u56f4\u4e3a\uff1a\n\n{answer_body}",
        "sources": [source],
        "confidence": 1.0,
        "abstainReason": "",
        "evidenceIds": [source["chunk_id"]],
        "queryType": query_type,
    }


def ask_structured(query: str, engine: HybridSearch = None, history: Optional[List[Dict[str, Any]]] = None) -> Dict:

    query_profile = analyze_query_intent(query)
    query_type = query_profile["query_type"]
    applicability_answer = _try_answer_document_applicability(query, query_type)
    if applicability_answer:
        return applicability_answer

    direct_answer = _try_answer_article_count(query, query_type)
    if direct_answer:
        return direct_answer

    if engine is None:
        engine = HybridSearch()
    try:
        search_results = engine.search_for_qa(_build_contextual_search_query(query, history))
    except Exception as exc:
        logger.exception("问答检索失败 query=%s", query)
        return {
            "answer": _search_error_message(),
            "sources": [],
            "confidence": 0.0,
            "abstainReason": "search_error",
            "evidenceIds": [],
            "queryType": query_type,
        }
    evidence_ids = [row.get("chunk_id", "") for row in search_results if row.get("chunk_id")]
    confidence = _estimate_confidence(search_results)

    if not search_results:
        return {
            "answer": _abstain_message(query_type),
            "sources": [],
            "confidence": 0.0,
            "abstainReason": "no_evidence",
            "evidenceIds": [],
            "queryType": query_type,
        }

    if confidence < 0.46:
        return {
            "answer": _abstain_message(query_type),
            "sources": search_results,
            "confidence": confidence,
            "abstainReason": "low_confidence",
            "evidenceIds": evidence_ids,
            "queryType": query_type,
        }

    prompt_results = prepare_prompt_evidence(_select_prompt_evidence(search_results, query_type), query_type)
    context = build_context(prompt_results)
    user_prompt = build_user_prompt(query, context, query_type, history=history)
    try:
        answer = call_qwen(BASE_SYSTEM_PROMPT, user_prompt).strip()
    except Exception as exc:
        logger.warning("调用大模型失败 query=%s: %s", query, exc)
        return {
            "answer": _llm_unavailable_message(prompt_results or search_results, query_type),
            "sources": search_results,
            "confidence": confidence,
            "abstainReason": "llm_unavailable",
            "evidenceIds": evidence_ids,
            "queryType": query_type,
        }
    if not answer:
        answer = _abstain_message(query_type)

    return {
        "answer": answer,
        "sources": search_results,
        "confidence": confidence,
        "abstainReason": "",
        "evidenceIds": evidence_ids,
        "queryType": query_type,
    }


def main():
    while True:
        query = input("\n请输入问题（直接回车退出）: ").strip()
        if not query:
            break
        print(ask(query))


if __name__ == "__main__":
    main()
