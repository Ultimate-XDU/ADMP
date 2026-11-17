"""Directed reference traversal and one-batch target span recovery."""

import pytest

import knowledge.graph_rag.hybrid_index as hybrid_index
import knowledge.graph_rag.runtime as runtime
import knowledge.graph_rag.service as service_module
from knowledge.graph_rag.config import MAX_PATH_EDGES
from knowledge.graph_rag.graph_ops import enumerate_paths, expand_hops, filter_graph
from knowledge.graph_rag.hierarchy import build_structure, select_structure
from knowledge.graph_rag.hybrid_index import retrieval_limits
from knowledge.graph_rag.service import GraphService, _answer_from_evidence, _tools

QUERY = "ZXQ-REF-17"
LATE = "LATEWINDOW"
PREFIX = "pad " * 400


def _corpus():
    documents = {
        "d1": {"id": "d1", "title": "手册甲", "name": "手册甲"},
        "d2": {"id": "d2", "title": "手册乙", "name": "手册乙"},
    }
    chunks = {
        "c1": {
            "chunk_id": "c1",
            "doc_id": "d1",
            "doc_name": "手册甲",
            "content": "Pointer ZXQ-REF-17 only.",
            "original_format": "html",
            "page_number": 0,
            "section_path": ["总则", "引用"],
            "heading": "引用",
            "references": ["手册乙"],
            "locator": {"label": "引用", "anchor": "cite"},
        },
        "sib": {
            "chunk_id": "sib",
            "doc_id": "d1",
            "doc_name": "手册甲",
            "content": "Sibling paragraph about wind.",
            "original_format": "html",
            "page_number": 0,
            "section_path": ["其他"],
            "heading": "其他",
        },
        "c2": {
            "chunk_id": "c2",
            "doc_id": "d2",
            "doc_name": "手册乙",
            "content": PREFIX + "ZXQ-REF-17 LATEWINDOW target clause.",
            "original_format": "html",
            "page_number": 0,
            "section_path": ["正文"],
            "heading": "正文",
        },
        "c3": {
            "chunk_id": "c3",
            "doc_id": "d2",
            "doc_name": "手册乙",
            "content": "Filler paragraph about weather.",
            "original_format": "html",
            "page_number": 0,
            "section_path": ["附则"],
            "heading": "附则",
        },
    }
    return documents, chunks


def _structure():
    documents, chunks = _corpus()
    return build_structure(documents, chunks, []), chunks


def test_reference_expansion_is_directed_and_keeps_sibling_closed():
    structure, _chunks = _structure()
    refs = [edge for edge in structure["edges"] if edge.get("edgeKind") == "references"]
    assert len(refs) == 1
    original = refs[0]
    assert original["id"]
    assert original["directed"] is True
    assert original["relationType"] in {"REFERS_TO", "AMENDS", "SUPERSEDES"}
    assert original["sourceIds"] == ["c1"]
    assert original["source"] == "chk:c1"
    assert original["target"] == "doc:d2"
    assert LATE not in str(original.get("provenance"))

    semantic, _truncated = enumerate_paths(
        structure["nodes"], structure["edges"], ["chk:c1", "doc:d2"], scope="semantic"
    )
    assert semantic == []
    semantic_expand = expand_hops(structure, ["chk:c1"], MAX_PATH_EDGES, scope="semantic")
    assert "doc:d2" not in {node["id"] for node in semantic_expand["nodes"]}

    paths, _truncated = enumerate_paths(structure["nodes"], structure["edges"], ["chk:c1"], scope="all")
    joined = [path for path in paths if {"chk:c1", "doc:d2", "chk:c2"} <= set(path["nodeIds"])]
    assert joined
    edge_by_id = {edge["id"]: edge for edge in structure["edges"]}
    for path in joined:
        assert path["nodeIds"].index("chk:c1") < path["nodeIds"].index("doc:d2") < path["nodeIds"].index("chk:c2")
        assert path["semanticHops"] == 0
        assert path["hops"] == len(path["edgeIds"])
        contains = sum(1 for edge_id in path["edgeIds"] if edge_by_id[edge_id]["edgeKind"] == "contains")
        assert path["hierarchyDepth"] == contains
        assert any(edge_by_id[edge_id]["edgeKind"] == "references" for edge_id in path["edgeIds"])
        assert "chk:sib" not in path["nodeIds"]

    ref_ids = {edge["id"] for edge in refs}
    doc_paths, _truncated = enumerate_paths(
        structure["nodes"], structure["edges"], ["doc:d1", "doc:d2"], scope="all"
    )
    chunk_ids = {"chk:c1", "chk:c2", "chk:c3", "chk:sib"}
    assert doc_paths
    for path in doc_paths:
        assert path["semanticHops"] == 0
        if not ref_ids.intersection(path["edgeIds"]):
            assert len(chunk_ids.intersection(path["nodeIds"])) <= 1

    expanded = expand_hops(structure, ["chk:c1"], MAX_PATH_EDGES, scope="all")
    expanded_ids = {node["id"] for node in expanded["nodes"]}
    assert {"chk:c1", "doc:d2", "chk:c2"} <= expanded_ids
    assert "chk:sib" not in expanded_ids
    kept = [edge for edge in expanded["edges"] if edge["id"] == original["id"]]
    assert kept
    assert kept[0]["source"] == original["source"]
    assert kept[0]["target"] == original["target"]
    assert kept[0]["directed"] is True
    assert kept[0]["relationType"] == original["relationType"]
    assert kept[0]["sourceIds"] == original["sourceIds"]
    assert kept[0]["provenance"] == original["provenance"]
    assert all(path["semanticHops"] == 0 for path in expanded["paths"])

    reverse = expand_hops(structure, ["chk:c2"], MAX_PATH_EDGES, scope="hierarchy")
    reverse_ids = {node["id"] for node in reverse["nodes"]}
    assert "chk:c1" not in reverse_ids
    assert "chk:sib" not in reverse_ids
    assert "chk:c3" not in reverse_ids


def test_select_structure_seeds_retrieved_leaf_and_filter_blocks_target():
    structure, chunks = _structure()
    browsed = select_structure(structure, "引用", 8)
    labels = {node["label"] for node in browsed["nodes"]}
    ids = {node["id"] for node in browsed["nodes"]}
    assert {"手册甲", "总则", "引用", "手册乙"} <= labels
    assert "其他" not in labels and "附则" not in labels
    assert "chk:sib" not in ids and "chk:c3" not in ids
    assert all(path["semanticHops"] == 0 for path in browsed["paths"])

    assert select_structure(structure, "missing-token", 8)["nodes"] == []
    seeded = select_structure(structure, "missing-token", 8, ["not-a-chunk", "c1"])
    seeded_ids = {node["id"] for node in seeded["nodes"]}
    assert "chk:c1" in seeded_ids
    assert "doc:d2" in seeded_ids
    assert "chk:sib" not in seeded_ids
    assert "chk:c2" not in seeded_ids
    assert "chk:c3" not in seeded_ids
    assert any(edge.get("edgeKind") == "references" and edge.get("target") == "doc:d2" for edge in seeded["edges"])

    filtered = filter_graph(structure, ["d1"], chunks)
    filtered_ids = {node["id"] for node in filtered["nodes"]}
    assert "chk:c1" in filtered_ids
    assert "doc:d2" not in filtered_ids
    assert "chk:c2" not in filtered_ids
    assert all(edge.get("target") != "doc:d2" for edge in filtered["edges"])


def _state(query, doc_ids, pending):
    return {
        "question": query,
        "doc_ids": doc_ids,
        "pending": pending,
        "observations": [],
        "evidence": [],
        "trace": [],
        "graph_parts": [],
        "tool_calls": 0,
    }


def _block(content, chunk_id, document_key, title, heading, references=None):
    block = {
        "content": content,
        "chunk_id": chunk_id,
        "document_key": document_key,
        "document_title": title,
        "original_format": "html",
        "page_number": 0,
        "section_path": [heading],
        "section_title": heading,
        "heading": heading,
        "locator": {"label": heading, "anchor": chunk_id},
        "source_url": "https://example.test/" + document_key,
    }
    if references is not None:
        block["references"] = references
    return block


def _find(service, marker):
    return next(chunk for chunk in service.catalog.chunks.values() if marker in chunk["content"])


def _leaf_id(snapshot, chunk_id):
    for node in snapshot["structure"]["nodes"]:
        owned = [str(item) for item in node.get("sourceIds") or []]
        if node.get("type") in {"chunk", "table", "figure"} and chunk_id in owned:
            assert node["id"] == "chk:" + chunk_id
            return node["id"]
    raise AssertionError(chunk_id)


def _doc_id(snapshot, doc_id):
    for node in snapshot["structure"]["nodes"]:
        if node.get("type") == "document" and str(node.get("docId") or "") == doc_id:
            assert node["id"] == "doc:" + doc_id
            return node["id"]
    raise AssertionError(doc_id)


def _span_chunk(span):
    return str(span.get("chunkId") or span.get("unit_id") or span.get("chunk_id") or "")


def _late_spans(spans, chunk_id):
    return [span for span in spans if _span_chunk(span) == chunk_id and LATE in str(span.get("text") or "")]


def _answer_parts(answered):
    if isinstance(answered, dict):
        return (
            answered.get("sources") or [],
            answered.get("context_spans") or answered.get("contextSpans") or [],
        )
    return answered[1], answered[2]


@pytest.fixture()
def reference_tools(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPH_DISABLE_VECTORS", "1")
    monkeypatch.setattr(service_module, "auto_build_enabled", lambda: False)

    def _rerank(*_args, **_kwargs):
        return None

    monkeypatch.setattr(runtime, "rerank_scores", _rerank)
    if hasattr(hybrid_index, "rerank_scores"):
        monkeypatch.setattr(hybrid_index, "rerank_scores", _rerank)

    service = GraphService(tmp_path / "graph")
    service.store_managed_blocks(
        "manual-a.html",
        [
            _block("Pointer ZXQ-REF-17 to the second manual.", "cite-b", "manual-a", "手册甲", "引乙", ["手册乙"]),
            _block("Repeat ZXQ-REF-17 toward the second manual.", "cite-b2", "manual-a", "手册甲", "再引", ["手册乙"]),
            _block("Bridge ZXQ-REF-17 into the third manual.", "cite-c", "manual-a", "手册甲", "外引", ["手册丙"]),
            _block("Sibling paragraph about wind.", "sib", "manual-a", "手册甲", "其他"),
        ],
    )
    service.store_managed_blocks(
        "manual-b.html",
        [
            _block(PREFIX + "ZXQ-REF-17 LATEWINDOW target clause.", "late", "manual-b", "手册乙", "正文"),
            _block("Filler paragraph about weather.", "filler", "manual-b", "手册乙", "附则"),
        ],
    )
    service.store_managed_blocks(
        "manual-c.html",
        [_block("Third manual ZXQ-REF-17 supplement.", "extra", "manual-c", "手册丙", "补充")],
    )
    snapshot = service._snapshot()
    for node in snapshot.get("structure", {}).get("nodes") or []:
        if node.get("type") in {"chunk", "table", "figure"}:
            node["label"] = "irrelevant-heading"
            locator = node.get("locator")
            if isinstance(locator, dict):
                locator["label"] = "irrelevant-heading"
    calls = []
    real = service_module.retrieve_spans

    def wrapped(snapshot, query, doc_ids=None, **kwargs):
        calls.append(
            {
                "query": query,
                "doc_ids": None if doc_ids is None else [str(item) for item in doc_ids],
                "limit": kwargs.get("limit"),
                "rerank_pool": kwargs.get("rerank_pool"),
            }
        )
        return real(snapshot, query, doc_ids, **kwargs)

    monkeypatch.setattr(service_module, "retrieve_spans", wrapped)
    chunks = {
        "cite_b": _find(service, "Pointer ZXQ-REF-17"),
        "cite_b2": _find(service, "Repeat ZXQ-REF-17"),
        "cite_c": _find(service, "Bridge ZXQ-REF-17"),
        "sib": _find(service, "Sibling paragraph"),
        "late": _find(service, LATE),
        "filler": _find(service, "Filler paragraph"),
        "extra": _find(service, "Third manual"),
    }
    return {
        "service": service,
        "snapshot": snapshot,
        "tools": _tools(snapshot),
        "calls": calls,
        "chunks": chunks,
    }


def test_tools_search_expand_one_target_batch_and_answer_window(reference_tools, monkeypatch):
    bundle = reference_tools
    service = bundle["service"]
    snapshot = bundle["snapshot"]
    tools = bundle["tools"]
    calls = bundle["calls"]
    chunks = bundle["chunks"]
    cite_b = chunks["cite_b"]
    cite_b2 = chunks["cite_b2"]
    cite_c = chunks["cite_c"]
    sibling = chunks["sib"]
    late = chunks["late"]
    filler = chunks["filler"]
    extra = chunks["extra"]
    assert service.search(QUERY, scope="semantic")["nodes"] == []
    assert late["content"].index(LATE) > 0
    assert {cite_b["doc_id"], late["doc_id"], extra["doc_id"]} == {
        cite_b["doc_id"],
        late["doc_id"],
        extra["doc_id"],
    }
    assert len({cite_b["doc_id"], late["doc_id"], extra["doc_id"]}) == 3

    leaf_ids = [
        _leaf_id(snapshot, cite_b["chunk_id"]),
        _leaf_id(snapshot, cite_b2["chunk_id"]),
        _leaf_id(snapshot, cite_c["chunk_id"]),
    ]
    doc_b = _doc_id(snapshot, late["doc_id"])
    doc_c = _doc_id(snapshot, extra["doc_id"])
    original_edges = {
        edge["id"]: edge
        for edge in snapshot["structure"]["edges"]
        if edge.get("edgeKind") == "references"
    }
    assert original_edges

    calls.clear()
    state = _state(QUERY, None, {"query": QUERY})
    search_update = tools["graph_search"](state)
    assert isinstance(search_update["tool_calls"], int)
    assert search_update["tool_calls"] == 1
    assert len(calls) == 2
    assert all(call["doc_ids"] is None for call in calls)
    search_obs = search_update["observations"][0]
    assert "个节点" in search_obs["summary"]
    node_ids = search_obs["nodeIds"]
    for leaf_id in leaf_ids:
        assert leaf_id in node_ids[:8]
    assert doc_b in node_ids[:8]
    assert doc_c in node_ids[:8]
    assert _leaf_id(snapshot, sibling["chunk_id"]) not in node_ids
    assert _leaf_id(snapshot, filler["chunk_id"]) not in node_ids
    kinds = {node["id"]: node.get("kind") for node in search_update["graph_parts"][0]["nodes"]}
    first_semantic = next((index for index, node_id in enumerate(node_ids) if kinds.get(node_id) != "structure"), None)
    if first_semantic is not None:
        assert node_ids.index(leaf_ids[0]) < first_semantic
        assert node_ids.index(doc_b) < first_semantic
    assert filler["chunk_id"] not in search_obs["sourceIds"]
    assert sibling["chunk_id"] not in search_obs["sourceIds"]
    assert cite_b["chunk_id"] in search_obs["sourceIds"]
    assert all(not str(source_id).startswith("chk:") for source_id in search_obs["sourceIds"])

    calls.clear()
    state = {**state, **search_update, "pending": {"query": QUERY, "nodeIds": leaf_ids, "hops": 1}}
    expand_update = tools["graph_expand"](state)
    assert isinstance(expand_update["tool_calls"], int)
    assert expand_update["tool_calls"] == 2
    assert len(calls) == 1
    limits = retrieval_limits(snapshot, state)
    assert limits["candidates"] <= 8
    assert limits["rerank_pool"] == 16
    assert calls[0]["query"] == QUERY
    assert calls[0]["limit"] == limits["candidates"]
    assert calls[0]["rerank_pool"] == limits["rerank_pool"]
    assert isinstance(calls[0]["doc_ids"], list)
    assert len(calls[0]["doc_ids"]) == len(set(calls[0]["doc_ids"])) == 2
    assert set(calls[0]["doc_ids"]) == {late["doc_id"], extra["doc_id"]}
    expand_obs = expand_update["observations"][-1]
    assert "个节点" in expand_obs["summary"]
    assert late["chunk_id"] in expand_obs["sourceIds"]
    assert extra["chunk_id"] in expand_obs["sourceIds"]
    assert filler["chunk_id"] not in expand_obs["sourceIds"]
    assert sibling["chunk_id"] not in expand_obs["sourceIds"]
    assert all(not str(source_id).startswith("chk:") for source_id in expand_obs["sourceIds"])
    late_rows = [row for row in expand_update["evidence"] if row.get("chunk_id") == late["chunk_id"]]
    assert late_rows
    assert late_rows[0]["page_number"] == 0

    expand_graph = expand_update["graph_parts"][-1]
    seen_ref = False
    for edge in expand_graph["edges"]:
        if edge.get("edgeKind") != "references":
            continue
        seen_ref = True
        original = original_edges[edge["id"]]
        assert edge["directed"] is True
        assert edge["relationType"] == original["relationType"]
        assert edge["relationType"] in {"REFERS_TO", "AMENDS", "SUPERSEDES"}
        assert edge["source"] == original["source"]
        assert edge["target"] == original["target"]
        assert edge["sourceIds"] == original["sourceIds"]
        assert edge["provenance"] == original["provenance"]
        assert late["chunk_id"] not in [str(item) for item in edge.get("sourceIds") or []]
        assert LATE not in str(edge.get("provenance"))
    assert seen_ref

    recorded = _late_spans(expand_obs["spans"], late["chunk_id"])
    assert recorded
    span = recorded[0]
    assert late["content"][span["start"] : span["end"]] == span["text"]
    assert span["start"] > 0
    assert late["content"].find(LATE) >= span["start"]
    assert late["content"].find(LATE) < span["end"]
    assert str(span.get("docId") or span.get("doc_id") or "") == late["doc_id"]

    seen = {}

    def _chat(prompt, system_prompt="", **_kwargs):
        seen["prompt"] = prompt
        return "LATEWINDOW acknowledged.[1]"

    monkeypatch.setattr(runtime, "ollama_chat", _chat)
    if hasattr(service_module, "ollama_chat"):
        monkeypatch.setattr(service_module, "ollama_chat", _chat)
    state = {**state, **expand_update}
    sources, context_spans = _answer_parts(_answer_from_evidence(state))
    answered_spans = _late_spans(context_spans, late["chunk_id"])
    assert answered_spans
    answered = answered_spans[0]
    assert late["content"][answered["start"] : answered["end"]] == answered["text"]
    assert answered["start"] > 0
    assert answered["text"] in seen["prompt"]
    assert LATE in seen["prompt"]
    late_sources = [
        row
        for row in sources
        if str(row.get("chunk_id") or row.get("chunkId") or "") == late["chunk_id"]
    ]
    assert late_sources
    assert late_sources[0].get("page_number", late_sources[0].get("pageNumber")) == 0
    assert len(calls) == 1


def test_graph_expand_target_batch_skips_structure_supplement_and_keeps_history(monkeypatch):
    documents = {
        "d1": {"id": "d1", "title": "Guidance A", "name": "Guidance A"},
        "d2": {"id": "d2", "title": "Guidance B", "name": "Guidance B"},
    }
    chunks = {
        "cite": {
            "chunk_id": "cite",
            "doc_id": "d1",
            "doc_name": "Guidance A",
            "content": "An explicit reference to Guidance B.",
            "references": ["Guidance B"],
            "heading": "Pointer",
            "page_number": 0,
            "original_format": "html",
        }
    }
    for index in range(16):
        source_id = f"target-{index:02d}"
        if index < 8:
            tail = "paddingword " * 100
        else:
            tail = f"useful fact {index}"
        chunks[source_id] = {
            "chunk_id": source_id,
            "doc_id": "d2",
            "doc_name": "Guidance B",
            "content": "zqtarget " + tail,
            "heading": "Section",
            "page_number": 0,
            "original_format": "html",
        }
    snapshot = {
        "documents": documents,
        "chunks": chunks,
        "graph": {"nodes": [], "edges": []},
        "structure": build_structure(documents, chunks, []),
    }
    state = _state("zqtarget", ["d1", "d2"], {"nodeIds": ["chk:cite"], "hops": 1, "query": "zqtarget"})
    state["tool_calls"] = 1
    state["observations"] = [{"action": "graph_search", "query": "zqtarget", "sourceIds": ["cite", "target-00"], "nodeIds": ["chk:cite"]}]
    historical = service_module._evidence(chunks["target-00"], 0.0, "zqtarget")
    state["evidence"] = [historical]
    target_batches = 0
    real_retrieve = service_module.retrieve_spans

    def _counting_retrieve(*args, **kwargs):
        nonlocal target_batches
        target_batches += 1
        return real_retrieve(*args, **kwargs)

    monkeypatch.setattr(runtime, "rerank_scores", lambda *args, **kwargs: None)
    monkeypatch.setattr(service_module, "retrieve_spans", _counting_retrieve)
    result = _tools(snapshot)["graph_expand"](state)
    observation = result["observations"][-1]
    span_ids = {str(span.get("unit_id") or "") for span in observation.get("spans") or []}
    span_ids.discard("")
    target_source_ids = {source_id for source_id in observation.get("sourceIds") or [] if str(source_id).startswith("target-")}
    assert target_source_ids == span_ids
    assert len(target_source_ids) <= retrieval_limits(snapshot, state)["candidates"]
    assert "cite" in (observation.get("sourceIds") or [])
    assert target_batches == 1
    assert any(row is historical or row == historical for row in result.get("evidence") or [])


def test_tools_expand_skips_without_in_scope_targets(reference_tools):
    bundle = reference_tools
    snapshot = bundle["snapshot"]
    tools = bundle["tools"]
    calls = bundle["calls"]
    chunks = bundle["chunks"]
    cite_b = chunks["cite_b"]
    sibling = chunks["sib"]
    late = chunks["late"]
    filler = chunks["filler"]
    extra = chunks["extra"]

    calls.clear()
    blocked = tools["graph_expand"](
        _state(
            QUERY,
            [cite_b["doc_id"]],
            {"query": QUERY, "nodeIds": [_leaf_id(snapshot, cite_b["chunk_id"])], "hops": MAX_PATH_EDGES},
        )
    )
    assert calls == []
    blocked_obs = blocked["observations"][0]
    assert isinstance(blocked["tool_calls"], int)
    for chunk in (late, filler, extra):
        assert chunk["chunk_id"] not in blocked_obs["sourceIds"]
        assert _doc_id(snapshot, chunk["doc_id"]) not in blocked_obs["nodeIds"]
        assert _leaf_id(snapshot, chunk["chunk_id"]) not in blocked_obs["nodeIds"]

    calls.clear()
    quiet = tools["graph_expand"](
        _state(
            QUERY,
            None,
            {"query": QUERY, "nodeIds": [_leaf_id(snapshot, sibling["chunk_id"])], "hops": MAX_PATH_EDGES},
        )
    )
    assert calls == []
    quiet_obs = quiet["observations"][0]
    for chunk in (late, filler, extra):
        assert chunk["chunk_id"] not in quiet_obs["sourceIds"]
    assert _leaf_id(snapshot, cite_b["chunk_id"]) not in quiet_obs["nodeIds"]
    assert _doc_id(snapshot, late["doc_id"]) not in quiet_obs["nodeIds"]
