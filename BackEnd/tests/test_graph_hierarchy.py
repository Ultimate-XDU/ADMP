"""Hierarchy scope, cross-script retrieval, and single-mode diagnostics."""

import pytest

from knowledge.graph_rag.config import MAX_PATH_EDGES
from knowledge.graph_rag.graph_ops import enumerate_paths
from knowledge.graph_rag.hierarchy import build_structure, chunk_locator, select_structure
from knowledge.graph_rag.hybrid_index import (
    DEFAULT_CANDIDATES,
    bridge_keep,
    budget_for,
    compatible_bridge_labels,
    retrieval_limits,
    retrieve_spans,
)
from knowledge.graph_rag.router import (
    AskBody,
    EvidenceModel,
    ExpandBody,
    GraphAnswerModel,
    GraphDataModel,
    SearchBody,
    graph_ask,
    graph_evidence,
    graph_expand,
    graph_search,
)
from knowledge.graph_rag.service import GraphService, _tools


def _validate(model, payload):
    model.model_validate(payload)


def test_structure_ancestors_locator_and_no_semantic_hop():
    documents = {
        "d1": {"id": "d1", "title": "手册甲", "name": "手册甲"},
        "d2": {"id": "d2", "title": "手册乙", "name": "手册乙"},
    }
    chunks = {
        "c1": {
            "chunk_id": "c1",
            "doc_id": "d1",
            "doc_name": "手册甲",
            "content": "本段只是一条指针。",
            "original_format": "html",
            "page_number": 0,
            "section_path": ["总则", "引用"],
            "heading": "引用",
            "references": ["手册乙"],
            "locator": {"label": "引用", "anchor": "cite"},
        },
        "c2": {
            "chunk_id": "c2",
            "doc_id": "d2",
            "doc_name": "手册乙",
            "content": "目标文档第一段。",
            "original_format": "html",
            "page_number": 0,
            "heading": "正文",
        },
        "c3": {
            "chunk_id": "c3",
            "doc_id": "d2",
            "doc_name": "手册乙",
            "content": "目标文档第二段。",
            "original_format": "html",
            "page_number": 0,
            "heading": "附则",
        },
    }
    locator = chunk_locator(chunks["c1"], documents["d1"])
    assert locator["anchor"] == "cite"
    assert "page" not in locator
    first = build_structure(documents, chunks, [])
    second = build_structure(documents, chunks, [])
    assert [node["id"] for node in first["nodes"]] == [node["id"] for node in second["nodes"]]
    leaf = next(node for node in first["nodes"] if node["id"] == "chk:c1")
    assert leaf["locator"]["anchor"] == "cite"
    assert "page" not in leaf["locator"]
    refs = [edge for edge in first["edges"] if edge["edgeKind"] == "references"]
    assert refs and refs[0]["sourceIds"] == ["c1"]
    selected = select_structure(first, "引用", 8)
    labels = {node["label"] for node in selected["nodes"]}
    assert {"手册甲", "总则", "引用", "手册乙"} <= labels
    assert "正文" not in labels and "附则" not in labels
    assert all(path["semanticHops"] == 0 for path in selected["paths"])
    semantic, _truncated = enumerate_paths(first["nodes"], first["edges"], ["doc:d1", "chk:c1", "doc:d2"], scope="semantic")
    assert semantic == []
    paths, _truncated = enumerate_paths(
        first["nodes"],
        first["edges"],
        [node["id"] for node in first["nodes"] if node["type"] == "document"],
        scope="all",
    )
    chunk_ids = {node["id"] for node in first["nodes"] if node["type"] == "chunk"}
    assert paths
    ref_edge_ids = {edge["id"] for edge in first["edges"] if edge.get("edgeKind") == "references"}
    for path in paths:
        assert path["semanticHops"] == 0
        if not ref_edge_ids.intersection(path["edgeIds"]):
            assert len(chunk_ids.intersection(path["nodeIds"])) <= 1


def test_cross_script_retrieval_keeps_explicit_doc_filter():
    chunk = {
        "chunk_id": "c1",
        "doc_id": "d1",
        "doc_name": "Guidance",
        "content": "Maintain five hundred foot vertical separation.",
        "page_number": 0,
        "original_format": "html",
    }
    snapshot = {"chunks": {"c1": chunk}}
    query = "垂直分离标准是什么"

    def encode(texts, _query):
        return [0.8 for _text in texts]

    def rerank(_query, texts):
        return [0.7 for _text in texts]

    spans = retrieve_spans(snapshot, query, encode=encode, rerank=rerank)
    assert len(spans) == 1
    assert spans[0]["unit_id"] == "c1"
    assert spans[0]["doc_id"] == "d1"
    assert chunk["content"][spans[0]["start"] : spans[0]["end"]] == spans[0]["text"]
    assert retrieve_spans(snapshot, query, ["other"], encode=encode, rerank=rerank) == []
    assert bridge_keep(query, chunk, user_filter=False) is True
    assert compatible_bridge_labels(query, ["vertical separation"]) == ["vertical separation"]


@pytest.fixture()
def graph_service(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPH_DISABLE_VECTORS", "1")
    monkeypatch.setattr("knowledge.graph_rag.service.auto_build_enabled", lambda: False)
    service = GraphService(tmp_path / "graph")
    service.store_managed_blocks(
        "manual-a.html",
        [
            {
                "content": "本段只是一条指针。",
                "chunk_id": "cite",
                "document_key": "manual-a",
                "document_title": "手册甲",
                "original_format": "html",
                "page_number": 0,
                "section_path": ["总则", "引用"],
                "section_title": "引用",
                "heading": "引用",
                "locator": {"label": "引用", "anchor": "cite"},
                "references": ["手册乙"],
                "source_url": "https://example.test/manual-a",
            }
        ],
    )
    service.store_managed_blocks(
        "manual-b.html",
        [
            {
                "content": "目标文档第一段。",
                "chunk_id": "body",
                "document_key": "manual-b",
                "document_title": "手册乙",
                "original_format": "html",
                "page_number": 0,
                "heading": "正文",
                "section_title": "正文",
                "source_url": "https://example.test/manual-b",
            },
            {
                "content": "目标文档第二段。",
                "chunk_id": "extra",
                "document_key": "manual-b",
                "document_title": "手册乙",
                "original_format": "html",
                "page_number": 0,
                "heading": "附则",
                "section_title": "附则",
                "source_url": "https://example.test/manual-b",
            },
        ],
    )
    return service


def test_service_search_expand_router(graph_service, monkeypatch):
    service = graph_service
    stored = list(service.catalog.chunks.values())
    assert len({chunk["doc_id"] for chunk in stored}) == 2
    citing = next(chunk for chunk in stored if chunk["content"] == "本段只是一条指针。")
    targets = [chunk for chunk in stored if chunk["content"].startswith("目标文档")]
    assert service.search("引用", scope="semantic")["nodes"] == []

    monkeypatch.delenv("GRAPH_DISABLE_VECTORS", raising=False)

    def boom(*_args, **_kwargs):
        raise AssertionError("hierarchy browse loaded a model")

    monkeypatch.setattr("knowledge.graph_rag.runtime.encode_texts", boom)
    monkeypatch.setattr("knowledge.graph_rag.runtime.rerank_scores", boom)
    monkeypatch.setattr("knowledge.graph_rag.router.get_service", lambda: service)

    browsed = service.search("", scope="hierarchy")
    again = service.search("", scope="hierarchy")
    assert [node["id"] for node in browsed["nodes"]] == [node["id"] for node in again["nodes"]]
    assert browsed["nodes"] and all(node["kind"] == "structure" for node in browsed["nodes"])
    combined = service.search("", scope="all")
    assert any(node["kind"] == "structure" for node in combined["nodes"])
    assert all(path["semanticHops"] == 0 for path in combined["paths"])

    focused = graph_search(SearchBody(query="引用", scope="hierarchy", limit=8))
    _validate(GraphDataModel, focused)
    labels = {node["label"] for node in focused["nodes"]}
    assert {"手册甲", "总则", "引用", "手册乙"} <= labels
    assert "正文" not in labels and "附则" not in labels
    assert any(edge["edgeKind"] == "contains" for edge in focused["edges"])
    assert any(edge["edgeKind"] == "references" for edge in focused["edges"])
    assert focused["paths"] and all(path["semanticHops"] == 0 for path in focused["paths"])
    leaves = [node for node in focused["nodes"] if node["type"] == "chunk" and node["id"] == "chk:" + citing["chunk_id"]]
    assert leaves and leaves[0]["label"] == "本段只是一条指针。" and leaves[0]["locator"]["anchor"] == "cite"
    assert "page" not in leaves[0]["locator"]
    full = build_structure(service.catalog.documents, service.catalog.chunks, [])
    assert {node["id"] for node in focused["nodes"]} <= {node["id"] for node in full["nodes"]}

    assert service.search("引用", ["missing-doc"], scope="hierarchy")["nodes"] == []
    doc_node = next(node["id"] for node in focused["nodes"] if node["type"] == "document" and node["label"] == "手册甲")
    expanded = graph_expand(ExpandBody(nodeIds=[doc_node], hops=MAX_PATH_EDGES, scope="hierarchy"))
    _validate(GraphDataModel, expanded)
    assert any(node["label"] in {"总则", "引用"} for node in expanded["nodes"])
    assert any(edge["edgeKind"] == "contains" for edge in expanded["edges"])
    assert all(edge["edgeKind"] in {"contains", "references"} for edge in expanded["edges"])
    assert all(path["semanticHops"] == 0 for path in expanded["paths"])
    semantic = service.expand([doc_node], 2, scope="semantic")
    assert semantic["nodes"] == [] and semantic["paths"] == []

    evidence = graph_evidence(chunkId=citing["chunk_id"])
    _validate(EvidenceModel, evidence)
    assert evidence["page_number"] == 0
    assert evidence["page_count"] == 0
    assert evidence["locator"]["anchor"] == "cite"
    assert evidence["doc_id"] == citing["doc_id"]
    assert evidence["document_key"] == "manual-a"

    update = _tools(service._snapshot())["graph_search"](
        {
            "question": "引用",
            "doc_ids": None,
            "pending": {"query": "引用"},
            "observations": [],
            "evidence": [],
            "trace": [],
            "graph_parts": [],
            "tool_calls": 0,
        }
    )
    source_ids = update["observations"][0]["sourceIds"]
    assert citing["chunk_id"] in source_ids
    assert all(chunk["chunk_id"] not in source_ids for chunk in targets)
    assert len(source_ids) <= retrieval_limits({})["candidates"]


def test_single_ask_reports_real_context(graph_service, monkeypatch):
    service = graph_service
    question = "本段只是一条指针吗"
    monkeypatch.setattr("knowledge.graph_rag.router.get_service", lambda: service)
    seen = {}

    def chat(prompt, system_prompt="", **_kwargs):
        seen["prompt"] = prompt
        return "本段只是一条指针。[1]"

    monkeypatch.setattr("knowledge.graph_rag.runtime.ollama_chat", chat)
    graph_result = graph_ask(AskBody(question=question, mode="graph"))
    _validate(GraphAnswerModel, graph_result)
    assert graph_result["stopReason"] != "graph_unavailable"
    assert graph_result["answer"]

    result = graph_ask(AskBody(question=question, mode="single", diagnostics=True))
    _validate(GraphAnswerModel, result)
    assert [step["action"] for step in result["trace"]] == ["document_retrieve", "finish"]
    limits = retrieval_limits({})
    budgets = result["diagnostics"]["budgets"]
    assert budgets["candidates"] == limits["candidates"] == DEFAULT_CANDIDATES
    assert budgets["rerankPool"] == limits["rerank_pool"]
    assert budgets["contextChars"] == limits["context_chars"]
    assert budgets["passageChars"] == limits["passage_chars"]
    assert budgets["toolCalls"] == budget_for({"question": question})
    by_id = {chunk["chunk_id"]: chunk for chunk in service.catalog.chunks.values()}

    def assert_slice(spans):
        assert spans
        for span in spans:
            chunk = by_id[span["chunkId"]]
            assert chunk["content"][span["start"] : span["end"]] == span["text"]
            assert span["text"] in seen["prompt"]
            assert span["docId"] == chunk["doc_id"]

    assert_slice(result["diagnostics"]["candidateSpans"])
    assert_slice(result["diagnostics"]["contextSpans"])
    assert result["sources"][0]["page_number"] == 0
    assert result["sources"][0]["locator"]["anchor"] == "cite"
    quiet = graph_ask(AskBody(question=question, mode="single", diagnostics=False))
    assert "diagnostics" not in quiet

    def chat_fail(prompt, system_prompt="", **_kwargs):
        seen["prompt"] = prompt
        raise RuntimeError("model down")

    monkeypatch.setattr("knowledge.graph_rag.runtime.ollama_chat", chat_fail)
    failed = graph_ask(AskBody(question=question, mode="single", diagnostics=True))
    assert failed["stopReason"] == "error"
    assert_slice(failed["diagnostics"]["contextSpans"])
    assert_slice(failed["diagnostics"]["candidateSpans"])
