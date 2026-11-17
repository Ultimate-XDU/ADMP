"""Retrieval spine: one hybrid search, then at most one evidence-conditioned search.

Hop 2 asks the existing local model for a single missing-information query
anchored on first-hop chunks. A model failure may use the evidence-term
fallback, and the trace says which one ran. Empty first-hop evidence never
calls the planner. Single mode never searches twice.
"""

from __future__ import annotations

import json
import re
from typing import Callable, Dict, List, Optional, Tuple

from search.common import QA_TOP_K

_MULTI_MARKERS = ("以及", "分别", "同时", "对比", "比较", "区别", "差异", "另一方面", "并且")
_EVIDENCE_KEYS = ("caption", "section_title", "article_label", "heading")
_TOKEN_PATTERN = re.compile(r"[A-Za-z]{2,}-\d[\w.-]*|\d+(?:\.\d+)?(?:\s*)(?:米|公里|节|英尺|分钟|℃|度|kt)")
_PLANNER_SYSTEM = (
    "你是检索规划器。根据原问题和第一跳证据，提出恰好一个后续检索问句，"
    "用来查找原问题还缺的事实。问句必须使用证据中的一个具体实体或事实作为桥梁，"
    "不要只复述章节标题，也不要回答原问题。"
    "如果证据已经足够，query 置空。"
    "只输出一行 JSON：{\"query\":\"...\",\"basisChunkIds\":[\"...\"]}"
)

Planner = Callable[[str, List[dict]], Optional[Tuple[str, List[str]]]]


def resolve_retrieval_mode(query: str, requested: str) -> str:
    if requested == "single":
        return "single"
    if requested == "multi":
        return "multi"
    if any(marker in query for marker in _MULTI_MARKERS):
        return "multi"
    return "single"


def evidence_follow_query(question: str, rows: List[dict]) -> str:
    """Heuristic hop-2 query. Empty when the evidence adds no new term."""
    compact_question = re.sub(r"\s+", "", question)
    picked: List[str] = []
    for row in rows[:3]:
        for key in _EVIDENCE_KEYS:
            text = str(row.get(key) or "").strip()
            if not text or len(text) > 40:
                continue
            if re.sub(r"\s+", "", text) in compact_question:
                continue
            if text not in picked:
                picked.append(text)
        for token in _TOKEN_PATTERN.findall(str(row.get("content") or "")):
            token = token.strip()
            if token and token not in question and token not in picked:
                picked.append(token)
        if len(picked) >= 4:
            break
    if not picked:
        return ""
    return f"{question} {' '.join(picked[:4])}"[:400]


def parse_planner_output(text: str, allowed_ids: List[str]) -> Optional[Tuple[str, List[str]]]:
    """Validate planner text once. None means the text is not a usable plan."""
    raw = str(text or "").strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    query = str(payload.get("query") or "").strip()
    basis_raw = payload.get("basisChunkIds")
    if not isinstance(basis_raw, list):
        return None
    allowed = set(allowed_ids)
    basis: List[str] = []
    for item in basis_raw:
        chunk_id = str(item or "").strip()
        if chunk_id in allowed and chunk_id not in basis:
            basis.append(chunk_id)
    if not query:
        return "", []
    if not basis:
        return None
    return query[:400], basis


def _evidence_brief(rows: List[dict]) -> Tuple[List[str], str]:
    allowed: List[str] = []
    lines: List[str] = []
    for row in rows[:3]:
        chunk_id = str(row.get("chunk_id") or "").strip()
        if not chunk_id:
            continue
        allowed.append(chunk_id)
        content = re.sub(r"\s+", " ", str(row.get("content") or ""))[:360]
        title = str(row.get("section_title") or row.get("heading") or "")
        lines.append(f"[{chunk_id}] {title} {content}".strip())
    return allowed, "\n".join(lines)


def _llm_plan(question: str, rows: List[dict]) -> Optional[Tuple[str, List[str]]]:
    from qa.prompt import call_qwen

    allowed, brief = _evidence_brief(rows)
    if not allowed or not brief:
        return None
    user = f"原问题：{question[:500]}\n\n第一跳证据：\n{brief}"
    return parse_planner_output(call_qwen(_PLANNER_SYSTEM, user), allowed)


def _heuristic_plan(question: str, rows: List[dict]) -> Tuple[str, List[str]]:
    follow = evidence_follow_query(question, rows)
    if not follow:
        return "", []
    basis = [str(row.get("chunk_id") or "") for row in rows[:3] if row.get("chunk_id")]
    return follow, basis


def merge_hop_results(
    hop1: List[dict],
    hop2: List[dict],
    limit: int = QA_TOP_K,
    basis_ids: Optional[List[str]] = None,
) -> List[dict]:
    """Keep cited hop-1 basis rows, then hop-2 evidence, and fill to the limit.

    Callers that omit basis ids keep the previous split: a leading hop-1
    prefix, then new hop-2 rows, then the rest of hop 1.
    """
    if not hop2:
        return list(hop1[:limit])

    merged: List[dict] = []
    seen = set()

    def add(row: dict) -> None:
        key = row.get("chunk_id") or (
            row.get("doc_name"),
            row.get("page_number"),
            row.get("section_title"),
        )
        if key in seen:
            return
        seen.add(key)
        merged.append(row)

    if basis_ids:
        by_id = {}
        for row in hop1:
            chunk_id = row.get("chunk_id")
            if chunk_id and chunk_id not in by_id:
                by_id[chunk_id] = row
        for chunk_id in basis_ids:
            row = by_id.get(chunk_id)
            if row is None:
                continue
            add(row)
            if len(merged) >= limit:
                return merged
        tail = hop1
    else:
        head = max(1, limit // 2)
        for row in hop1[:head]:
            add(row)
        tail = hop1[head:]

    for row in hop2:
        add(row)
        if len(merged) >= limit:
            return merged
    for row in tail:
        add(row)
        if len(merged) >= limit:
            break
    return merged


def _trace(steps: List[dict], mode: str, planner: str, reason: str) -> Dict:
    return {"steps": steps, "mode": mode, "planner": planner, "reason": reason}


def retrieve(
    engine,
    query: str,
    mode: str = "auto",
    planner: Optional[Planner] = None,
) -> Tuple[List[dict], Dict]:
    """Run hybrid search once. In multi mode, run one follow-up search at most."""
    resolved = resolve_retrieval_mode(query, mode)
    hop1 = engine.search_for_qa(query)
    steps = [{"hop": 1, "query": query, "resultCount": len(hop1)}]
    if resolved != "multi":
        return hop1, _trace(steps, "single", "none", "single")
    if not hop1:
        return hop1, _trace(steps, "multi", "none", "empty_evidence")

    planner_fn = planner if planner is not None else _llm_plan
    failure = ""
    try:
        planned = planner_fn(query, hop1)
    except Exception as exc:
        planned = None
        failure = "timeout" if isinstance(exc, TimeoutError) or "timeout" in str(exc).lower() else "model_error"

    if planned:
        follow, basis = planned
        if follow:
            hop2 = engine.search_for_qa(follow)
            steps.append({"hop": 2, "query": follow, "resultCount": len(hop2), "basisChunkIds": list(basis)})
            return merge_hop_results(hop1, hop2, basis_ids=basis), _trace(steps, "multi", "llm", "follow_up")
        return hop1, _trace(steps, "multi", "llm", "no_gap")

    follow, basis = _heuristic_plan(query, hop1)
    reason = failure or "no_useful_subquery"
    if not follow:
        return hop1, _trace(steps, "multi", "heuristic_fallback", reason)
    hop2 = engine.search_for_qa(follow)
    steps.append({"hop": 2, "query": follow, "resultCount": len(hop2), "basisChunkIds": basis})
    return merge_hop_results(hop1, hop2, basis_ids=basis), _trace(steps, "multi", "heuristic_fallback", reason)
