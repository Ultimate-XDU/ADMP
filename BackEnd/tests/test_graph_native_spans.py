"""Native documents may store page_number 0. Serializers must keep that 0."""

from __future__ import annotations

from knowledge.graph_rag.eval_retrieval import _candidate, _candidate_from_source, _source_view
from knowledge.graph_rag.hybrid_index import pack_context, window_rows
from knowledge.graph_rag.service import _passage_excerpts, _public_spans

CONTENT = "Graph native span page zero stays visible."
CHUNK_ID = "html-native:0"


def _chunk(page: object = 0, *, include_page: bool = True) -> dict:
    row = {
        "chunk_id": CHUNK_ID,
        "doc_id": "html-native",
        "doc_name": "Native note",
        "section_title": "Overview",
        "content": CONTENT,
    }
    if include_page:
        row["page_number"] = page
    return row


def _row_key(row: dict, id_field: str = "unit_id") -> str:
    return f"{row[id_field]}@{row['start']}:{row['end']}"


def _without_page(rows: list[dict]) -> list[dict]:
    copied = []
    for row in rows:
        item = dict(row)
        item.pop("page", None)
        copied.append(item)
    return copied


def test_page_zero_survives_window_pack_public_and_eval_serializers() -> None:
    zero = _chunk(0)
    absent = _chunk(include_page=False)
    explicit_none = _chunk(None)
    numbered = _chunk(2)

    rows_zero = window_rows([zero])
    rows_absent = window_rows([absent])
    rows_none = window_rows([explicit_none])
    rows_numbered = window_rows([numbered])
    assert rows_zero
    assert _without_page(rows_zero) == _without_page(rows_absent) == _without_page(rows_none)
    assert _without_page(rows_zero) == _without_page(rows_numbered)
    assert [_row_key(row) for row in rows_zero] == [_row_key(row) for row in rows_absent]
    assert [_row_key(row) for row in rows_zero] == [_row_key(row) for row in rows_none]
    assert [row["page"] for row in rows_zero] == [0] * len(rows_zero)
    assert [row["page"] for row in rows_absent] == [1] * len(rows_absent)
    assert [row["page"] for row in rows_none] == [1] * len(rows_none)
    assert [row["page"] for row in rows_numbered] == [2] * len(rows_numbered)
    for row in rows_zero:
        assert row["unit_id"] == CHUNK_ID
        assert row["text"] == CONTENT[row["start"]:row["end"]]
        assert row["text"].strip()

    packed_zero = pack_context(CONTENT, [zero], facets=[CONTENT])
    packed_absent = pack_context(CONTENT, [absent], facets=[CONTENT])
    packed_none = pack_context(CONTENT, [explicit_none], facets=[CONTENT])
    packed_numbered = pack_context(CONTENT, [numbered], facets=[CONTENT])
    assert packed_zero
    assert _without_page(packed_zero) == _without_page(packed_absent) == _without_page(packed_none)
    assert _without_page(packed_zero) == _without_page(packed_numbered)
    assert packed_zero[0]["page"] == 0
    assert packed_absent[0]["page"] == 1
    assert packed_none[0]["page"] == 1
    assert packed_numbered[0]["page"] == 2
    assert packed_zero[0]["text"] == CONTENT[packed_zero[0]["start"]:packed_zero[0]["end"]]
    assert packed_zero[0]["unit_id"] == CHUNK_ID
    assert _row_key(packed_zero[0]) == _row_key(packed_absent[0]) == _row_key(packed_none[0])

    public_rows = _public_spans(rows_zero)
    assert public_rows[0]["page"] == 0
    assert public_rows[0]["text"] == rows_zero[0]["text"]
    assert public_rows[0]["start"] == rows_zero[0]["start"]
    assert public_rows[0]["end"] == rows_zero[0]["end"]
    assert _row_key(public_rows[0]) == _row_key(rows_zero[0])

    excerpts = _passage_excerpts(None, [rows_zero[0]["unit_id"]], rows_zero[0]["text"], rows_zero)
    assert excerpts[0]["page"] == 0
    assert excerpts[0]["chunk_id"] == rows_zero[0]["unit_id"]
    assert excerpts[0]["start"] == rows_zero[0]["start"]
    assert excerpts[0]["end"] == rows_zero[0]["end"]
    assert f"{excerpts[0]['chunk_id']}@{excerpts[0]['start']}:{excerpts[0]['end']}" == _row_key(rows_zero[0])
    assert rows_zero[0]["text"] == CONTENT[rows_zero[0]["start"]:rows_zero[0]["end"]]

    public_packed = _public_spans(packed_zero)
    assert public_packed[0]["page"] == 0
    assert public_packed[0]["text"] == packed_zero[0]["text"]
    assert public_packed[0]["start"] == packed_zero[0]["start"]
    assert public_packed[0]["end"] == packed_zero[0]["end"]
    assert _row_key(public_packed[0]) == _row_key(packed_zero[0])

    packed_excerpts = _passage_excerpts(
        None,
        [packed_zero[0]["unit_id"]],
        packed_zero[0]["text"],
        packed_zero,
    )
    assert packed_excerpts[0]["page"] == 0
    assert packed_excerpts[0]["start"] == packed_zero[0]["start"]
    assert packed_excerpts[0]["end"] == packed_zero[0]["end"]
    assert packed_zero[0]["text"] == CONTENT[packed_zero[0]["start"]:packed_zero[0]["end"]]

    chosen = _candidate(packed_zero[0])
    assert chosen["page"] == 0
    assert chosen["text"] == packed_zero[0]["text"]
    assert chosen["start"] == packed_zero[0]["start"]
    assert chosen["end"] == packed_zero[0]["end"]
    assert _row_key(chosen) == _row_key(packed_zero[0])

    bare = dict(packed_zero[0])
    bare.pop("page")
    bare_none = dict(packed_zero[0])
    bare_none["page"] = None
    for source in (bare, bare_none):
        rendered = _candidate(source)
        assert rendered["page"] == 1
        assert rendered["text"] == packed_zero[0]["text"]
        assert rendered["start"] == packed_zero[0]["start"]
        assert rendered["end"] == packed_zero[0]["end"]
        assert _row_key(rendered) == _row_key(packed_zero[0])
        public = _public_spans([source])
        assert public[0]["page"] == 1
        assert public[0]["text"] == packed_zero[0]["text"]
        assert public[0]["start"] == packed_zero[0]["start"]
        assert public[0]["end"] == packed_zero[0]["end"]
        blank_excerpts = _passage_excerpts(None, [source["unit_id"]], source["text"], [source])
        assert blank_excerpts[0]["page"] == 1
        assert blank_excerpts[0]["start"] == packed_zero[0]["start"]
        assert blank_excerpts[0]["end"] == packed_zero[0]["end"]
        assert f"{blank_excerpts[0]['chunk_id']}@{blank_excerpts[0]['start']}:{blank_excerpts[0]['end']}" == _row_key(
            packed_zero[0]
        )

    span_zero = {"page": 0, "unit_id": CHUNK_ID, "text": CONTENT, "start": 0, "end": len(CONTENT)}
    view = _source_view(_chunk(7), CHUNK_ID, span_zero)
    assert view["page_number"] == 0
    assert view["content"] == CONTENT
    assert view["chunk_id"] == CHUNK_ID

    view = _source_view(zero, CHUNK_ID, {})
    assert view["page_number"] == 0
    assert view["content"] == CONTENT

    view = _source_view(zero, CHUNK_ID, {"page": None})
    assert view["page_number"] == 0
    assert view["content"] == CONTENT

    view = _source_view(absent, CHUNK_ID, {})
    assert view["page_number"] == 1
    assert view["content"] == CONTENT

    view = _source_view(explicit_none, CHUNK_ID, {"page": None})
    assert view["page_number"] == 1
    assert view["content"] == CONTENT

    from_row = _candidate_from_source(
        {"chunk_id": CHUNK_ID, "content": CONTENT, "page_number": 0, "doc_name": "Native note"},
        {"chunks": {CHUNK_ID: {"content": CONTENT, "page_number": 7, "doc_id": "html-native"}}},
    )
    assert from_row["page"] == 0
    assert from_row["text"] == CONTENT
    assert from_row["start"] == 0
    assert from_row["end"] == len(CONTENT)
    assert _row_key(from_row) == f"{CHUNK_ID}@0:{len(CONTENT)}"

    from_chunk = _candidate_from_source(
        {"chunk_id": CHUNK_ID, "content": CONTENT, "doc_name": "Native note"},
        {"chunks": {CHUNK_ID: {"content": CONTENT, "page_number": 0, "doc_id": "html-native"}}},
    )
    assert from_chunk["page"] == 0
    assert from_chunk["text"] == CONTENT
    assert from_chunk["start"] == 0
    assert from_chunk["end"] == len(CONTENT)
    assert _row_key(from_chunk) == _row_key(from_row)

    from_none = _candidate_from_source(
        {"chunk_id": CHUNK_ID, "content": CONTENT, "page_number": None},
        {"chunks": {CHUNK_ID: {"content": CONTENT, "page_number": None}}},
    )
    assert from_none["page"] == 1
    assert from_none["text"] == CONTENT
    assert from_none["start"] == 0
    assert from_none["end"] == len(CONTENT)
    assert _row_key(from_none) == _row_key(from_row)

    from_absent = _candidate_from_source(
        {"chunk_id": CHUNK_ID, "content": CONTENT},
        {"chunks": {CHUNK_ID: {"content": CONTENT}}},
    )
    assert from_absent["page"] == 1
    assert from_absent["text"] == CONTENT
    assert from_absent["start"] == from_row["start"]
    assert from_absent["end"] == from_row["end"]
    assert _row_key(from_absent) == _row_key(from_row)
