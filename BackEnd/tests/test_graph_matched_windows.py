"""Matched retrieval windows must reach generation unchanged."""
from __future__ import annotations

import copy
from unittest.mock import patch

from knowledge.graph_rag.config import ANSWER_CHAR_BUDGET, ANSWER_PASSAGE_CHARS
from knowledge.graph_rag.hybrid_index import pack_context
from knowledge.graph_rag.service import _answer_from_evidence, citations_from_answer


def _source(chunk_id: str, content: str, *, page: int = 0, doc_name: str = "manual") -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": f"doc-{chunk_id}",
        "doc_name": doc_name,
        "section_title": "body",
        "page_number": page,
        "content": content,
        "score": 1.0,
    }


def _span(chunk_id: str, content: str, start: int, end: int, *, score: float = 1.0) -> dict:
    return {
        "unit_id": chunk_id,
        "doc_id": f"doc-{chunk_id}",
        "start": start,
        "end": end,
        "text": content[start:end],
        "page": 0,
        "ranks": {"dense": 1},
        "score": score,
    }


def _state(question: str, sources: list[dict], spans_by_obs: list[list[dict]]) -> dict:
    evidence = [dict(row) for row in sources]
    observations = []
    for index, spans in enumerate(spans_by_obs):
        observations.append(
            {
                "action": "document_retrieve",
                "query": question if index == 0 else f"{question} {index}",
                "sourceIds": [span["unit_id"] for span in spans],
                "spans": spans,
                "tool_calls": index + 1,
            }
        )
    return {
        "question": question,
        "evidence": evidence,
        "observations": observations,
        "graph_parts": [],
        "tool_calls": len(observations),
    }


def test_late_english_window_beats_chinese_title_overlap():
    title = "航空器起飞限制 起飞限制 起飞限制"
    filler = "。" + ("起飞限制说明" * 200)
    prefix = title + filler
    evidence = "takeoff is prohibited when the tailwind exceeds the published limit"
    start = len(prefix)
    end = start + len(evidence)
    content = prefix + evidence + (" tail " * 800)
    assert start > 900
    assert content[start:end] == evidence
    assert len(content) > ANSWER_CHAR_BUDGET
    source = _source("c1", content, page=0, doc_name="起飞限制")
    span = _span("c1", content, start, end)
    question = "起飞限制 tailwind exceeds the published limit"
    packed = pack_context(
        question,
        [source],
        context_chars=ANSWER_CHAR_BUDGET,
        passage_chars=ANSWER_PASSAGE_CHARS,
        matched_spans=[span],
    )
    assert [(row["start"], row["end"], row["text"], row["page"]) for row in packed] == [
        (start, end, evidence, 0)
    ]
    state = _state(question, [source], [[span]])
    with patch("knowledge.graph_rag.runtime.ollama_chat", return_value="禁止起飞 [1]") as chat:
        result = _answer_from_evidence(state)
    prompt = chat.call_args.args[0]
    assert f"[{start}:{end}] {evidence}" in prompt
    assert f"[0:{ANSWER_PASSAGE_CHARS}]" not in prompt
    assert [(row["start"], row["end"], row["text"]) for row in result["context_spans"]] == [
        (start, end, evidence)
    ]
    assert result["sources"][0]["content"] == content
    assert result["sources"][0]["page_number"] == 0
    assert state["evidence"][0]["content"] == content


def test_two_matched_windows_keep_order_and_drop_duplicate():
    content = ("A" * 100) + "FIRST-WINDOW-ALPHA" + ("B" * 400) + "SECOND-WINDOW-BETA" + ("C" * 200)
    first = _span("c1", content, 100, 118, score=0.9)
    second = _span("c1", content, 518, 536, score=0.4)
    duplicate = dict(second)
    later = _span("c1", content, 100, 118, score=0.1)
    source = _source("c1", content, page=0)
    question = "alpha beta windows"
    packed = pack_context(
        question,
        [source],
        context_chars=ANSWER_CHAR_BUDGET,
        passage_chars=ANSWER_PASSAGE_CHARS,
        matched_spans=[first, second, duplicate],
    )
    assert [(row["start"], row["end"], row["text"], row["page"]) for row in packed] == [
        (100, 118, "FIRST-WINDOW-ALPHA", 0),
        (518, 536, "SECOND-WINDOW-BETA", 0),
    ]
    state = _state(question, [source], [[first, second], [duplicate, later]])
    with patch("knowledge.graph_rag.runtime.ollama_chat", return_value="both [1]") as chat:
        result = _answer_from_evidence(state)
    body = chat.call_args.args[0]
    assert body.index("[100:118] FIRST-WINDOW-ALPHA") < body.index("[518:536] SECOND-WINDOW-BETA")
    assert body.count("FIRST-WINDOW-ALPHA") == 1
    assert body.count("SECOND-WINDOW-BETA") == 1
    sent = [(row["start"], row["end"], row["text"]) for row in result["context_spans"]]
    assert sent == [(100, 118, "FIRST-WINDOW-ALPHA"), (518, 536, "SECOND-WINDOW-BETA")]
    assert result["sources"][0]["content"] == content
    assert state["evidence"][0]["content"] == content


def test_no_matched_spans_keeps_multifacet_windows():
    content = (
        "ZQALPHA is the first condition.\n"
        + ("pad token.\n" * 200)
        + "ZQBETA is the second condition."
    )
    question = "ZQALPHA and ZQBETA"
    source = _source("mix", content, page=0)
    packed = pack_context(
        question,
        [source],
        context_chars=ANSWER_CHAR_BUDGET,
        passage_chars=ANSWER_PASSAGE_CHARS,
    )
    assert [(row["start"], row["end"], row["text"], row["page"]) for row in packed] == [
        (0, 889, content[0:889], 0),
        (1623, 2263, content[1623:2263], 0),
    ]
    assert all(len(row["text"]) <= ANSWER_PASSAGE_CHARS for row in packed)
    assert "ZQALPHA" in packed[0]["text"]
    assert "ZQBETA" in packed[1]["text"]


def test_fallback_respects_global_budget_and_keeps_raw_sources():
    first = (
        "ZQALPHA is the first condition.\n"
        + ("pad token.\n" * 200)
        + "ZQBETA is the second condition."
    )
    question = "ZQALPHA and ZQBETA"
    sources = [_source("mix", first, page=0, doc_name="mixdoc")]
    for index in range(5):
        marker = f"F{index}-"
        text = marker + ("q" * (ANSWER_PASSAGE_CHARS - len(marker)))
        assert len(text) == ANSWER_PASSAGE_CHARS
        sources.append(_source(f"f{index}", text, page=0, doc_name=f"doc{index}"))
    state = {
        "question": question,
        "evidence": [dict(row) for row in sources],
        "observations": [
            {
                "action": "document_retrieve",
                "query": question,
                "sourceIds": [row["chunk_id"] for row in sources],
                "spans": [],
                "tool_calls": 1,
            }
        ],
        "graph_parts": [],
        "tool_calls": 1,
    }
    with patch("knowledge.graph_rag.runtime.ollama_chat", return_value="见 [1]") as chat, patch(
        "knowledge.graph_rag.runtime.rerank_scores"
    ) as rerank:
        result = _answer_from_evidence(state)
    rerank.assert_not_called()
    prompt = chat.call_args.args[0]
    sent = result["context_spans"]
    mix_rows = [row for row in sent if row["unit_id"] == "mix"]
    other = [row for row in sent if row["unit_id"] != "mix"]
    assert [(row["start"], row["end"], row["text"]) for row in mix_rows] == [
        (0, 889, first[0:889]),
        (1623, 2263, first[1623:2263]),
    ]
    assert [row["unit_id"] for row in other] == ["f0", "f1", "f2", "f3"]
    assert [(row["start"], row["end"], row["text"]) for row in other] == [
        (0, ANSWER_PASSAGE_CHARS, sources[index]["content"]) for index in range(1, 5)
    ]
    assert sum(len(row["text"]) for row in sent) <= ANSWER_CHAR_BUDGET
    assert all(len(row["text"]) <= ANSWER_PASSAGE_CHARS for row in sent)
    assert "[1] 文档：mixdoc" in prompt
    assert "[2] 文档：doc0" in prompt
    assert "[3] 文档：doc1" in prompt
    assert "[4] 文档：doc2" in prompt
    assert "[5] 文档：doc3" in prompt
    assert "[6] 文档：doc4" not in prompt
    assert f"[0:889] {first[0:889]}" in prompt
    assert f"[1623:2263] {first[1623:2263]}" in prompt
    for index in range(1, 5):
        assert sources[index]["content"] in prompt
    assert sources[-1]["content"] not in prompt
    assert [row["chunk_id"] for row in result["sources"]] == [row["chunk_id"] for row in sources]
    assert [row["content"] for row in result["sources"]] == [row["content"] for row in sources]


def test_sparse_citation_maps_to_original_index():
    contents = ["A" * 5500, "B" * 300, "C" * 100]
    sources = [
        _source("a", contents[0], doc_name="docA"),
        _source("b", contents[1], doc_name="docB"),
        _source("c", contents[2], doc_name="docC"),
    ]
    question = "请仅根据编号给出结论"
    first_spans = [
        _span("a", contents[0], index * 700, (index + 1) * 700) for index in range(7)
    ]
    second_spans = [
        _span("a", contents[0], 4900, 5500),
        _span("b", contents[1], 0, 300),
        _span("c", contents[2], 0, 100),
    ]
    assert [len(item) for item in contents] == [5500, 300, 100]
    assert len(first_spans) + len(second_spans) == 10
    state = _state(question, sources, [first_spans, second_spans])
    before = copy.deepcopy(state)
    answer = "见 [3]"
    with patch("knowledge.graph_rag.runtime.ollama_chat", return_value=answer) as chat:
        result = _answer_from_evidence(state)
    assert state == before
    prompt = chat.call_args.args[0]
    kept = first_spans + [second_spans[0], second_spans[2]]
    expected = [(span["unit_id"], span["start"], span["end"], span["text"]) for span in kept]
    assert [
        (row["unit_id"], row["start"], row["end"], row["text"]) for row in result["context_spans"]
    ] == expected
    assert sum(len(row["text"]) for row in result["context_spans"]) <= ANSWER_CHAR_BUDGET
    assert all(end - start <= ANSWER_PASSAGE_CHARS for _unit, start, end, _text in expected)
    cursor = -1
    for _unit, start, end, text in expected:
        found = prompt.index(f"[{start}:{end}] {text}")
        assert found > cursor
        cursor = found
    assert prompt.index("[1] 文档：docA") < prompt.index("[3] 文档：docC")
    assert prompt.index("[3] 文档：docC") < prompt.index(contents[2])
    assert "[2] 文档：docB" not in prompt
    assert contents[1] not in prompt
    assert [(row["chunk_id"], row["doc_name"], row["content"]) for row in result["sources"]] == [
        ("a", "docA", contents[0]),
        ("b", "docB", contents[1]),
        ("c", "docC", contents[2]),
    ]
    citations = citations_from_answer(answer, result["sources"])
    assert citations == [{"id": 3, "sourceIndex": 2, "chunkId": "c"}]
    cited = result["sources"][citations[0]["sourceIndex"]]
    assert cited["chunk_id"] == "c"
    assert cited["content"] == contents[2]
