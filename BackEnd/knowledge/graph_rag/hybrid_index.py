"""Windowed hybrid retrieval over stored source text.

Dense and lexical lists are fused by reciprocal rank. Raw embedding scores
and rerank logits are not mixed into that score. Windows keep the original
unit id and exact character offsets. This module does not write the LightRAG
graph.
"""

from __future__ import annotations

import math
import os
import re
from typing import Callable, Optional

from knowledge.graph_rag.config import (
    ANSWER_CHAR_BUDGET,
    ANSWER_PASSAGE_CHARS,
    MAX_TOOL_CALLS,
    tool_call_budget,
)
from knowledge.graph_rag.graph_ops import lexical_score

RRF_K = 60
WINDOW_CHARS = 700
WINDOW_OVERLAP = 80
DEFAULT_CANDIDATES = 8
DEFAULT_RERANK_POOL = 16

_WORD = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*", re.I)
_CJK = re.compile(r"[\u4e00-\u9fff]")
_LATIN_LETTER = re.compile(r"[A-Za-z]")
_BLANK = re.compile(r"\n[ \t]*\n")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_CLAUSE = re.compile(
    r"(?:,?\s+and\s+|,?\s+versus\s+|,?\s+vs\.?\s+|\s+as well as\s+|以及|并且|还有|；|;)",
    re.I,
)
_CN_AND = re.compile(r"和|与|及")
_SPACE = re.compile(r"\s+")


def index_tokens(text: str) -> list[str]:
    """Tokens for BM25. Identifiers, decimals, and CJK bigrams stay intact."""
    raw = str(text or "").casefold()
    found: list[str] = []
    for match in _WORD.finditer(raw):
        token = match.group(0)
        if len(token) >= 2:
            found.append(token)
        if any(mark in token for mark in "-."):
            for piece in re.split(r"[-.]", token):
                if len(piece) >= 2 and piece != token:
                    found.append(piece)
    chars = _CJK.findall(raw)
    found.extend("".join(chars[index : index + 2]) for index in range(len(chars) - 1))
    return found


def dominant_script(text: str) -> str:
    cjk = len(_CJK.findall(str(text or "")))
    latin = len(_LATIN_LETTER.findall(str(text or "")))
    if cjk >= 2 and latin >= 4:
        return "mixed"
    if cjk >= 2 and cjk > latin:
        return "cjk"
    if latin >= 4 and latin > cjk:
        return "latin"
    return "mixed"


def _compact(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", str(text or "").casefold(), flags=re.U)


def _contains_alias(query: str, alias: str) -> bool:
    needle = _compact(alias)
    if len(needle) < 4:
        return False
    return needle in _compact(query)


def split_spans(text: str, size: int = WINDOW_CHARS, overlap: int = WINDOW_OVERLAP) -> list[tuple[int, int]]:
    """Exact slices of ``text``. Headings, rows, numbers, and formulas are not rewritten."""
    body = str(text or "")
    if not body.strip():
        return []
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for match in _BLANK.finditer(body):
        if body[cursor : match.start()].strip():
            ranges.append((cursor, match.start()))
        cursor = match.end()
    if body[cursor:].strip():
        ranges.append((cursor, len(body)))
    spans: list[tuple[int, int]] = []
    for start, end in ranges:
        spans.extend(_slice_range(body, start, end, max(size, 80), max(overlap, 0)))
    return spans or ([(0, len(body))] if body.strip() else [])


def _avoid_number_cut(text: str, start: int, cut: int) -> int:
    window = text[start : min(len(text), cut + 8)]
    for match in _NUMBER.finditer(window):
        abs_start = start + match.start()
        abs_end = start + match.end()
        if abs_start < cut < abs_end and abs_start > start:
            return abs_start
    return cut


def _slice_range(text: str, start: int, end: int, size: int, overlap: int) -> list[tuple[int, int]]:
    if end - start <= size:
        return [(start, end)] if text[start:end].strip() else []
    spans: list[tuple[int, int]] = []
    cursor = start
    guard = 0
    while cursor < end and guard <= len(text) + 2:
        guard += 1
        limit = min(end, cursor + size)
        cut = limit
        if limit < end:
            region = text[cursor:limit]
            newline = region.rfind("\n", max(size // 2, 1))
            if newline > 0:
                cut = cursor + newline
            cut = _avoid_number_cut(text, cursor, cut)
            if cut <= cursor:
                cut = limit
        if text[cursor:cut].strip():
            spans.append((cursor, cut))
        if cut >= end:
            break
        nxt = cut - overlap if overlap else cut
        if nxt <= cursor:
            nxt = cut
        cursor = nxt
    return spans


def window_rows(chunks: dict[str, dict] | list[dict]) -> list[dict]:
    """One row per window. ``text`` is an exact slice of the stored unit."""
    if isinstance(chunks, dict):
        ordered = [chunks[key] for key in sorted(chunks)]
    else:
        ordered = list(chunks)
    rows: list[dict] = []
    for chunk in ordered:
        content = str(chunk.get("content") or "")
        unit_id = str(chunk.get("chunk_id") or "")
        if not unit_id or not content.strip():
            continue
        for start, end in split_spans(content):
            piece = content[start:end]
            if not piece.strip():
                continue
            rows.append(
                {
                    "unit_id": unit_id,
                    "doc_id": str(chunk.get("doc_id") or ""),
                    "doc_name": str(chunk.get("doc_name") or ""),
                    "publisher": str(chunk.get("publisher") or ""),
                    "language": str(chunk.get("language") or ""),
                    "section": str(chunk.get("section_title") or chunk.get("heading") or ""),
                    "page": int(chunk["page_number"]) if chunk.get("page_number") is not None else 1,
                    "source_url": str(chunk.get("source_url") or ""),
                    "start": start,
                    "end": end,
                    "text": piece,
                }
            )
    return rows


def row_key(row: dict) -> str:
    return f"{row.get('unit_id')}@{int(row.get('start') or 0)}:{int(row.get('end') or 0)}"


def rrf_order(rank_lists: list[list[str]], k: int = RRF_K) -> list[tuple[str, float, dict]]:
    """Fuse 1-based rank lists. Score is the sum of ``1 / (k + rank)`` only."""
    k = max(int(k), 1)
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    for list_index, ranked in enumerate(rank_lists):
        label = str(list_index)
        seen = set()
        for rank, key in enumerate(ranked, start=1):
            if not key or key in seen:
                continue
            seen.add(key)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            ranks.setdefault(key, {})[label] = rank
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [(key, score, ranks.get(key) or {}) for key, score in ordered]


def bm25_order(query: str, rows: list[dict], k1: float = 1.2, b: float = 0.75) -> list[str]:
    query_tokens = index_tokens(query)
    if not query_tokens or not rows:
        return []
    documents = [index_tokens(str(row.get("text") or "")) for row in rows]
    df: dict[str, int] = {}
    for document in documents:
        for token in set(document):
            df[token] = df.get(token, 0) + 1
    count = len(documents)
    average = sum(len(document) for document in documents) / max(count, 1)
    query_counts: dict[str, int] = {}
    for token in query_tokens:
        query_counts[token] = query_counts.get(token, 0) + 1
    scored: list[tuple[float, int]] = []
    for index, document in enumerate(documents):
        if not document:
            continue
        tf: dict[str, int] = {}
        for token in document:
            tf[token] = tf.get(token, 0) + 1
        score = 0.0
        for token, query_freq in query_counts.items():
            freq = tf.get(token, 0)
            if not freq:
                continue
            doc_freq = df.get(token, 0)
            idf = math.log(1.0 + (count - doc_freq + 0.5) / (doc_freq + 0.5))
            denom = freq + k1 * (1.0 - b + b * len(document) / max(average, 1.0))
            score += idf * (freq * (k1 + 1.0)) / denom * (1.0 + math.log(query_freq))
        if score > 0:
            scored.append((score, index))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [row_key(rows[index]) for _score, index in scored]


def _split_clause(text: str) -> list[str]:
    parts = [part.strip(" ，,。.?？:：") for part in _CLAUSE.split(text)]
    refined: list[str] = []
    for part in parts:
        bits = _CN_AND.split(part)
        cleaned = [bit.strip(" ，,。.?？") for bit in bits if bit.strip()]
        if len(cleaned) > 1 and all(len(_compact(bit)) >= 2 for bit in cleaned):
            refined.extend(cleaned)
        elif _compact(part):
            refined.append(part.strip())
    return [part for part in refined if len(_compact(part)) >= 2]


def question_facets(question: str, documents: Optional[list[dict]] = None) -> list[str]:
    """Split a question into aspects from its wording and named catalog titles.

    No publisher or benchmark name is special-cased. Titles come from ``documents``.
    """
    text = _SPACE.sub(" ", str(question or "")).strip()
    if not text:
        return []
    facets = _split_clause(text) or [text]
    named = []
    for document in documents or []:
        title = str(document.get("title") or document.get("name") or "").strip()
        doc_id = str(document.get("id") or document.get("doc_id") or "").strip()
        for alias in (title, doc_id):
            if alias and _contains_alias(text, alias) and alias not in named:
                named.append(alias)
    if len(named) >= 2:
        for alias in named:
            if not any(alias == facet or _contains_alias(facet, alias) for facet in facets if facet != text):
                if alias not in facets:
                    facets.append(alias)
    deduped: list[str] = []
    seen = set()
    for facet in facets:
        key = _compact(facet)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(facet)
    return deduped or [text]


def budget_for(state: dict) -> int:
    override = state.get("retrieval_budget") or {}
    if override.get("tool_calls"):
        try:
            return max(1, min(int(override["tool_calls"]), 8))
        except (TypeError, ValueError):
            return tool_call_budget(1)
    facets = question_facets(str(state.get("question") or ""), _documents_from_state(state))
    return tool_call_budget(len(facets))


def budget_note(state: dict) -> str:
    count = len(question_facets(str(state.get("question") or ""), _documents_from_state(state)))
    limit = budget_for(state)
    if os.environ.get("GRAPH_MAX_TOOL_CALLS", "").strip():
        return f"tool budget {limit}, set by GRAPH_MAX_TOOL_CALLS; base is {MAX_TOOL_CALLS}"
    if limit != MAX_TOOL_CALLS:
        return f"tool budget {limit} for {count} aspects; base is {MAX_TOOL_CALLS}, cap is {MAX_TOOL_CALLS + 2}"
    return f"tool budget {limit}; base is {MAX_TOOL_CALLS}"


def _documents_from_state(state: dict) -> list[dict]:
    documents = []
    seen = set()
    for observation in state.get("observations") or []:
        for document in observation.get("documents") or []:
            doc_id = str(document.get("id") or "")
            if doc_id and doc_id not in seen:
                seen.add(doc_id)
                documents.append(document)
    return documents


def covers(facet: str, text: str) -> bool:
    if not str(facet or "").strip() or not str(text or "").strip():
        return False
    if lexical_score(facet, text) >= 0.34:
        return True
    compact = _compact(facet)
    return len(compact) >= 4 and compact in _compact(text)


def missing_aspects(state: dict) -> list[str]:
    """Aspects with no supporting passage text yet. Unknown text does not count as missing."""
    facets = question_facets(str(state.get("question") or ""), _documents_from_state(state))
    if len(facets) <= 1:
        return []
    texts: list[str] = []
    for row in state.get("evidence") or []:
        content = str(row.get("content") or "")
        if content:
            texts.append(content)
    for observation in state.get("observations") or []:
        for span in observation.get("spans") or []:
            piece = str(span.get("text") or span.get("excerpt") or "")
            if piece:
                texts.append(piece)
        for excerpt in observation.get("excerpts") or []:
            piece = str(excerpt.get("excerpt") or "")
            if piece:
                texts.append(piece)
    if not texts:
        return []
    return [facet for facet in facets if not any(covers(facet, text) for text in texts)]


def script_compatible(query: str, text: str, *, forced: bool = False) -> bool:
    """Report shared script or a long identifier. Retrieval does not use this as a gate."""
    if forced:
        return True
    query_script = dominant_script(query)
    text_script = dominant_script(text)
    if query_script not in {"latin", "cjk"} or text_script in {query_script, "mixed"}:
        return True
    overlap = set(index_tokens(query)) & set(index_tokens(text))
    return any(len(token) >= 4 or _CJK.search(token) for token in overlap)


def named_doc_ids(query: str, chunks: dict[str, dict]) -> Optional[list[str]]:
    """Document ids whose title, id, file stem, or publisher actually appears in the query."""
    found: list[str] = []
    publishers: dict[str, list[str]] = {}
    for chunk in chunks.values():
        doc_id = str(chunk.get("doc_id") or "")
        if not doc_id:
            continue
        aliases = [
            doc_id,
            str(chunk.get("doc_name") or ""),
            str(chunk.get("parent_heading") or ""),
            str(chunk.get("file_stem") or ""),
        ]
        if doc_id not in found and any(alias and _contains_alias(query, alias) for alias in aliases):
            found.append(doc_id)
        publisher = str(chunk.get("publisher") or "").strip()
        if len(publisher) >= 4:
            publishers.setdefault(publisher.casefold(), []).append(doc_id)
    # Whole-word publisher match uses the catalog value, not a fixed agency list.
    for publisher, doc_ids in publishers.items():
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(publisher)}(?![A-Za-z0-9])", str(query or "").casefold()):
            for doc_id in doc_ids:
                if doc_id not in found:
                    found.append(doc_id)
    return found or None


def _alias_order(query: str, rows: list[dict]) -> list[str]:
    matched: list[tuple[int, str]] = []
    for row in rows:
        aliases = [str(row.get("section") or ""), str(row.get("doc_name") or ""), str(row.get("unit_id") or "")]
        best = 0
        for alias in aliases:
            if alias and _contains_alias(query, alias):
                best = max(best, len(_compact(alias)))
        if best:
            matched.append((best, row_key(row)))
    matched.sort(key=lambda item: (-item[0], item[1]))
    return [key for _score, key in matched]


def _allowed_chunks(snapshot: dict, doc_ids: Optional[list[str]], query: str) -> tuple[dict[str, dict], Optional[list[str]], bool]:
    chunks = snapshot.get("chunks") or {}
    user_filter = doc_ids is not None
    allowed = list(doc_ids) if user_filter else named_doc_ids(query, chunks)
    if allowed is None:
        return dict(chunks), None, False
    allowed_set = set(allowed)
    filtered = {key: chunk for key, chunk in chunks.items() if chunk.get("doc_id") in allowed_set}
    return filtered, allowed, user_filter


def _dense_order(query: str, rows: list[dict], snapshot: dict, encode: Optional[Callable] = None) -> list[str]:
    if not rows:
        return []
    keys = [row_key(row) for row in rows]
    matrix = snapshot.get("window_matrix")
    stored_ids = list(snapshot.get("window_ids") or [])
    # window_ids[i] is matrix row i, so a filtered subset still reuses those rows.
    if matrix is not None and stored_ids:
        position: dict[str, int] = {}
        for index, stored in enumerate(stored_ids):
            position.setdefault(stored, index)
        picked = [(key, position[key]) for key in keys if key in position]
        if picked:
            try:
                from knowledge.graph_rag.runtime import encode_texts

                vector = encode_texts([query])[0]
                scores = matrix[[index for _key, index in picked]] @ vector
                order = sorted(range(len(picked)), key=lambda index: -float(scores[index]))
                return [picked[index][0] for index in order if float(scores[index]) > 0]
            except Exception:
                return []
    legacy = snapshot.get("passage_matrix")
    if legacy is not None or snapshot.get("window_matrix") is not None:
        try:
            from knowledge.graph_rag.runtime import encode_texts

            encode_texts([query])
        except Exception:
            pass
    if encode is None:
        return []
    try:
        vectors = encode([str(row.get("text") or "") for row in rows], query)
    except Exception:
        return []
    if vectors is None:
        return []
    try:
        order = sorted(range(len(keys)), key=lambda index: -float(vectors[index]))
        return [keys[index] for index in order if float(vectors[index]) > 0]
    except Exception:
        return []


def _channel_pool(rank_lists: list[list[str]], limit: int) -> list[str]:
    """Take the next new key from each nonempty channel until ``limit``."""
    channels = [ranked for ranked in rank_lists if ranked]
    pool: list[str] = []
    seen: set[str] = set()
    cursors = [0] * len(channels)
    while len(pool) < limit and channels:
        advanced = False
        for index, ranked in enumerate(channels):
            cursor = cursors[index]
            while cursor < len(ranked) and ranked[cursor] in seen:
                cursor += 1
            cursors[index] = cursor
            if cursor >= len(ranked):
                continue
            key = ranked[cursor]
            cursors[index] = cursor + 1
            seen.add(key)
            pool.append(key)
            advanced = True
            if len(pool) >= limit:
                return pool
        if not advanced:
            break
    return pool


def retrieve_spans(
    snapshot: dict,
    query: str,
    doc_ids: Optional[list[str]] = None,
    *,
    limit: int = DEFAULT_CANDIDATES,
    rerank_pool: int = DEFAULT_RERANK_POOL,
    encode: Optional[Callable] = None,
    rerank: Optional[Callable] = None,
) -> list[dict]:
    """Rank windows of every allowed stored unit. The cache key is the query plus the filter."""
    query = str(query or "").strip()
    if not query:
        return []
    if doc_ids is not None and len(doc_ids) == 0:
        doc_ids = None
    limit = max(1, int(limit))
    cache = snapshot.setdefault("_span_cache", {})
    cache_key = (query, tuple(doc_ids) if doc_ids is not None else None, limit, int(rerank_pool), rerank is not None)
    cached = cache.get(cache_key)
    if cached is not None:
        return [dict(row) for row in cached]
    chunks, _, _ = _allowed_chunks(snapshot, doc_ids, query)
    rows = window_rows(chunks)
    if not rows:
        cache[cache_key] = []
        return []
    by_key = {row_key(row): row for row in rows}
    lexical = bm25_order(query, rows)
    dense = _dense_order(query, rows, snapshot, encode)
    alias = _alias_order(query, rows)
    lists = [ranked for ranked in (lexical, dense, alias) if ranked]
    if not lists:
        cache[cache_key] = []
        return []
    fused = rrf_order(lists)
    pool_keys = _channel_pool([lexical, dense, alias], max(int(rerank_pool), int(limit)))
    rerank_list: list[str] = []
    paired = [(key, by_key[key]["text"]) for key in pool_keys if key in by_key]
    if paired and (rerank is not None or os.environ.get("GRAPH_DISABLE_VECTORS") != "1"):
        keys = [key for key, _text in paired]
        texts = [text for _key, text in paired]
        scores = None
        try:
            if rerank is not None:
                scores = rerank(query, texts)
            else:
                from knowledge.graph_rag.runtime import rerank_scores

                scores = rerank_scores(query, texts)
        except Exception:
            scores = None
        if scores and len(scores) == len(keys):
            order = sorted(range(len(keys)), key=lambda index: (-float(scores[index]), index))
            rerank_list = [keys[index] for index in order]
    if rerank_list:
        # Reciprocal rank of the reranker order. The logit is not a confidence.
        chosen = [
            (key, 1.0 / (RRF_K + rank)) for rank, key in enumerate(rerank_list[:limit], start=1)
        ]
    else:
        chosen = [(key, score) for key, score, _ranks in fused[:limit]]
    hits = []
    for key, score in chosen:
        row = by_key.get(key)
        if row is None:
            continue
        hit = dict(row)
        hit["score"] = float(score)
        named_ranks = {}
        # Each nonempty channel keeps its own name; rerank is an extra list.
        channel_lists = [
            (label, ranked)
            for label, ranked in (("lexical", lexical), ("dense", dense), ("alias", alias))
            if ranked
        ]
        if rerank_list:
            channel_lists.append(("rerank", rerank_list))
        for label, ranked in channel_lists:
            position = _rank_of(key, ranked)
            if position:
                named_ranks[label] = position
        hit["ranks"] = named_ranks
        content = str(chunks.get(row["unit_id"], {}).get("content") or row["text"])
        if content[hit["start"] : hit["end"]] != hit["text"]:
            # Refuse a span that drifted from the stored unit.
            continue
        hits.append(hit)
    cache[cache_key] = [dict(row) for row in hits]
    return [dict(row) for row in hits]


def _rank_of(key: str, ranked: list[str]) -> int:
    for index, item in enumerate(ranked, start=1):
        if item == key:
            return index
    return 0


def retrieval_limits(snapshot: Optional[dict] = None, state: Optional[dict] = None) -> dict:
    budget = {}
    if state:
        budget.update(state.get("retrieval_budget") or {})
    if snapshot:
        budget.update(snapshot.get("retrieval_budget") or {})
    return {
        "candidates": max(1, int(budget.get("candidates") or DEFAULT_CANDIDATES)),
        "context_chars": max(200, int(budget.get("context_chars") or ANSWER_CHAR_BUDGET)),
        "passage_chars": max(80, int(budget.get("passage_chars") or ANSWER_PASSAGE_CHARS)),
        "rerank_pool": max(1, int(budget.get("rerank_pool") or DEFAULT_RERANK_POOL)),
    }


def unit_ranking(spans: list[dict]) -> list[tuple[str, float]]:
    ranked = []
    seen = set()
    for span in spans:
        unit_id = str(span.get("unit_id") or "")
        if not unit_id or unit_id in seen:
            continue
        seen.add(unit_id)
        ranked.append((unit_id, float(span.get("score") or 0.0)))
    return ranked


def catalog_brief(snapshot: dict, doc_ids: Optional[list[str]]) -> list[dict]:
    allowed = None if doc_ids is None else set(doc_ids)
    documents: dict[str, dict] = {}
    for chunk in (snapshot.get("chunks") or {}).values():
        doc_id = str(chunk.get("doc_id") or "")
        if not doc_id or (allowed is not None and doc_id not in allowed):
            continue
        row = documents.setdefault(
            doc_id,
            {
                "id": doc_id,
                "title": str(chunk.get("document_title") or chunk.get("doc_name") or doc_id),
                "publisher": str(chunk.get("publisher") or ""),
                "language": str(chunk.get("language") or ""),
                "sections": [],
            },
        )
        section = str(chunk.get("section_title") or "")
        if section and section not in row["sections"] and len(row["sections"]) < 12:
            row["sections"].append(section)
    return list(documents.values())


def compatible_bridge_labels(question: str, labels: list[str]) -> list[str]:
    return [str(label) for label in labels if str(label or "").strip()]


def bridge_keep(query: str, chunk: dict, *, user_filter: bool, named: Optional[set[str]] = None) -> bool:
    # docIds are applied by the caller. Script is not a reason to drop a stored passage.
    del query, user_filter, named
    return isinstance(chunk, dict) and bool(chunk)


def pack_context(
    question: str,
    sources: list[dict],
    *,
    context_chars: int = ANSWER_CHAR_BUDGET,
    passage_chars: int = ANSWER_PASSAGE_CHARS,
    facets: Optional[list[str]] = None,
    matched_spans: Optional[list] = None,
) -> list[dict]:
    """Choose a window per aspect. Offsets refer to the original source content.

    Real retrieval windows in ``matched_spans`` are packed first, in the given
    order. ``passage_chars`` limits each window; ``context_chars`` limits the
    whole pack. A source that has a matched window is not given a lexical one.
    """
    facets = list(facets or question_facets(question))
    if not facets:
        facets = [question] if str(question or "").strip() else []
    chosen: list[dict] = []
    used = 0
    seen = set()
    by_id: dict[str, dict] = {}
    for source in sources:
        chunk_id = str(source.get("chunk_id") or "")
        if chunk_id and chunk_id not in by_id:
            by_id[chunk_id] = source
    matched_units: set[str] = set()

    def add(source: dict, start: int, end: int, facet: str, score: float) -> bool:
        nonlocal used
        content = str(source.get("content") or "")
        if start < 0 or end > len(content) or start >= end:
            return False
        text = content[start:end]
        if not text.strip():
            return False
        key = (str(source.get("chunk_id") or ""), start, end)
        if key in seen:
            return False
        if chosen and used + len(text) > context_chars:
            return False
        seen.add(key)
        chosen.append(
            {
                "unit_id": str(source.get("chunk_id") or ""),
                "doc_id": str(source.get("doc_id") or ""),
                "doc_name": str(source.get("doc_name") or ""),
                "section": str(source.get("section_title") or source.get("heading") or ""),
                "page": int(source["page_number"]) if source.get("page_number") is not None else 1,
                "start": start,
                "end": end,
                "text": text,
                "facet": facet,
                "score": float(score),
            }
        )
        used += len(text)
        return True

    def best_window(source: dict, facet: str) -> Optional[tuple[int, int, float]]:
        content = str(source.get("content") or "")
        if not content:
            return None
        best = None
        best_score = 0.0
        for start, end in split_spans(content, passage_chars, 80):
            score = lexical_score(facet, content[start:end])
            compact = _compact(facet)
            if len(compact) >= 4 and compact in _compact(content[start:end]):
                score += 0.35
            if score > best_score:
                best = (start, end, score)
                best_score = score
        return best

    for span in matched_spans or []:
        unit_id = span["unit_id"]
        source = by_id.get(unit_id)
        if source is None:
            continue
        matched_units.add(unit_id)
        start = span["start"]
        end = span["end"]
        if end - start > passage_chars:
            end = start + passage_chars
        add(source, start, end, "", float(span.get("score") or 0.0))

    for facet in facets:
        winner = None
        winner_source = None
        for source in sources:
            if str(source.get("chunk_id") or "") in matched_units:
                continue
            found = best_window(source, facet)
            if found and (winner is None or found[2] > winner[2]):
                winner = found
                winner_source = source
        if winner and winner_source and winner[2] > 0:
            add(winner_source, winner[0], winner[1], facet, winner[2])
    if not chosen:
        for source in sources:
            if str(source.get("chunk_id") or "") in matched_units:
                continue
            content = str(source.get("content") or "")
            if not content:
                continue
            probe = question or content[:40]
            found = best_window(source, probe)
            if found:
                add(source, found[0], found[1], probe, found[2])
            if used >= context_chars:
                break
    return chosen


def aspect_support(question: str, spans: list[dict], facets: Optional[list[str]] = None) -> tuple[list[str], list[str]]:
    facets = list(facets or question_facets(question))
    if not facets:
        return [], []
    supported = []
    missing = []
    for facet in facets:
        texts = [str(span.get("text") or "") for span in spans]
        if any(covers(facet, text) for text in texts):
            supported.append(facet)
        else:
            missing.append(facet)
    return supported, missing


def coverage_status(answer: str, question: str, spans: list[dict], citation_count: int, abstain_text: str) -> str:
    """Cited-answer status. The answer text is not rewritten here.

    aspect_support and covers stay planner hints for missing facets. Lexical
    overlap is not entailment: it must not drop or downgrade a cited answer
    on nonempty source text. ``question`` stays in the signature so callers
    keep their argument order; this status does not score facets.

    insufficient_evidence: no nonempty span, no remaining words, or only the
    abstention sentence.
    uncited_answer: words remain and there is no citation.
    partial_evidence: the abstention sentence plus other cited words, on real spans.
    answered: real spans, a valid citation, and no abstention sentence.
    """
    abstained = abstain_text in (answer or "")
    remainder = answer or ""
    if abstain_text:
        remainder = remainder.replace(abstain_text, "")
    remainder = re.sub(r"\[\d+\]", "", remainder)
    remainder = re.sub(r"[\s\W_]+", "", remainder)
    if citation_count <= 0:
        return "insufficient_evidence" if abstained or not remainder else "uncited_answer"
    if not remainder:
        return "insufficient_evidence"
    has_context = any(str(span.get("text") or "").strip() for span in (spans or []))
    if not has_context:
        return "insufficient_evidence"
    if abstained and abstain_text:
        return "partial_evidence"
    return "answered"
