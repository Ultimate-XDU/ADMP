import asyncio
import json
from unittest.mock import patch

import pytest

from knowledge.graph_rag.service import GraphService

DOC_ID = "product-doc-17"
PASSAGE = "\n".join(
    f"第{index}款 经营者应当建立进货查验记录并保留票据、合同和检验报告。"
    for index in range(12)
)
IDENTITY = (
    "id",
    "name",
    "sourceUrl",
    "kind",
    "storagePath",
    "title",
    "publisher",
    "publishedAt",
    "originalFormat",
    "documentKey",
)


def _service(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    (corpus / "units").mkdir(parents=True)
    (corpus / "manifest.json").write_text("[]\n", encoding="utf-8")
    monkeypatch.setenv("GRAPH_CORPUS_DIR", str(corpus))
    monkeypatch.setenv("GRAPH_MANAGEMENT_STATE", str(tmp_path / "missing-management.json"))
    service = GraphService(tmp_path / "graph")
    service.projection = {"nodes": [], "edges": []}
    service.rebuild_vectors = lambda: None
    service.refresh_projection = lambda: None
    monkeypatch.setattr("knowledge.graph_rag.service.progress_path", lambda: tmp_path / "graph-progress.json")
    return service


def _seed(service):
    chunks = []
    for index in range(4):
        chunks.append({
            "chunk_id": f"{DOC_ID}:block:{index}",
            "doc_id": DOC_ID,
            "content": f"片段{index}\n{PASSAGE}",
            "publisher": "市监局",
            "published_at": "2024-05-01",
            "original_format": "docx",
            "document_title": "食品安全监督管理办法",
            "document_key": DOC_ID,
            "source_url": "https://example.test/product-doc-17",
        })
    service.catalog.replace_chunks(
        DOC_ID,
        "食品安全监督管理办法",
        "https://example.test/product-doc-17",
        chunks,
        "managed",
        "uploads/product-doc-17.docx",
    )
    return [
        chunk_id
        for chunk_id, chunk in service.catalog.chunks.items()
        if chunk.get("doc_id") == DOC_ID
    ]


def _forbid_corpus(*_args, **_kwargs):
    raise AssertionError("stored product catalog must be preferred over the corpus adapter")


async def _no_insert(_chunk_id, _texts):
    return None


@pytest.mark.parametrize("scope", ["all", "smoke"])
def test_build_preserves_product_catalog(tmp_path, monkeypatch, scope):
    service = _service(tmp_path, monkeypatch)
    before_ids = _seed(service)
    before_doc = dict(service.catalog.documents[DOC_ID])
    before_chunks = {chunk_id: dict(chunk) for chunk_id, chunk in service.catalog.chunks.items()}
    sentinel = before_ids[-1]
    service.catalog.add_source_map({"existing-window-sentinel": sentinel})

    with patch("knowledge.graph_rag.service.is_corpus", return_value=True), \
         patch("knowledge.graph_rag.service.chunks_for", side_effect=_forbid_corpus), \
         patch("knowledge.graph_rag.service.selection_report", side_effect=_forbid_corpus), \
         patch("knowledge.graph_rag.service.manifest_by_id", side_effect=_forbid_corpus), \
         patch("knowledge.graph_rag.runtime.insert_windows", new=_no_insert):
        meta = service._doc_meta(DOC_ID)
        selected, report = service._resolve_chunks(DOC_ID, scope)
        asyncio.run(service._build([DOC_ID], scope))

    assert meta["name"] == before_doc["name"]
    assert meta["sourceUrl"] == before_doc["sourceUrl"]
    assert meta["kind"] == before_doc["kind"]
    assert meta["storagePath"] == before_doc["storagePath"]
    selected_ids = [row["chunk_id"] for row in selected]
    assert report["available"] == len(before_ids)
    assert set(selected_ids) <= set(before_ids)
    assert all(row["doc_id"] == DOC_ID for row in selected)
    if scope == "smoke":
        assert selected_ids == before_ids[:2]
        assert report["sampled"] is True
    else:
        assert selected_ids == before_ids
        assert report["fullText"] is True

    catalog = service.catalog
    assert (catalog.build.get("error") or "") == ""
    after = catalog.documents[DOC_ID]
    assert after["id"] == DOC_ID
    for field in IDENTITY:
        assert after.get(field) == before_doc.get(field)
    assert after["availableChunks"] == before_doc["availableChunks"] == len(before_ids)
    assert set(catalog.chunks) == set(before_chunks)
    for chunk_id, chunk in before_chunks.items():
        assert catalog.chunks[chunk_id] == chunk
        assert catalog.chunks[chunk_id]["doc_id"] == DOC_ID
    assert catalog.source_map["existing-window-sentinel"] == sentinel
    produced = {
        key: value
        for key, value in catalog.source_map.items()
        if key != "existing-window-sentinel"
    }
    assert produced
    assert set(produced.values()) <= set(selected_ids)
    for target in produced.values():
        assert catalog.chunks[target]["chunk_id"] == target
        assert catalog.chunks[target]["doc_id"] == DOC_ID


def test_ask_uses_stored_chunks_when_semantic_nodes_absent(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)
    _seed(service)
    service.projection = {"nodes": [], "edges": []}
    service._ensure_vectors = lambda: None
    status = service.status_payload()
    assert status["nodes"] == 0
    assert status["chunks"] > 0
    answer = {
        "evidence": [],
        "answer": "",
        "stop_reason": "insufficient_evidence",
        "tool_calls": 0,
        "graph_parts": [],
        "trace": [],
    }
    with patch("knowledge.graph_rag.service.run_agent", return_value=answer) as agent:
        for mode in ("graph", "agent", "single"):
            result = service.ask("文件规定了哪些义务？", mode, [DOC_ID])
            assert result["stopReason"] != "graph_unavailable"
    assert [call.args[2] for call in agent.call_args_list] == ["graph", "agent", "single"]


def test_ask_unavailable_without_stored_evidence(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)
    service.projection = {"nodes": [], "edges": []}
    status = service.status_payload()
    assert status["nodes"] == 0
    assert status["chunks"] == 0
    with patch("knowledge.graph_rag.service.run_agent") as agent:
        for mode in ("graph", "agent", "single"):
            result = service.ask("文件规定了哪些义务？", mode)
            assert result["stopReason"] == "graph_unavailable"
    assert agent.call_count == 0


def test_build_rebuilds_vectors_once_per_batch(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)
    calls = {"rebuild": 0}

    def rebuild():
        calls["rebuild"] += 1

    service.rebuild_vectors = rebuild
    for index in (1, 2):
        doc_id = f"batch-doc-{index}"
        service.catalog.replace_chunks(
            doc_id,
            f"文档{index}",
            f"https://example.test/{doc_id}",
            [{
                "chunk_id": f"{doc_id}:block:0",
                "doc_id": doc_id,
                "content": f"片段{index}\n{PASSAGE}",
                "publisher": "市监局",
                "published_at": "2024-05-01",
                "original_format": "docx",
                "document_title": f"文档{index}",
                "document_key": doc_id,
                "source_url": f"https://example.test/{doc_id}",
            }],
            "managed",
            f"uploads/{doc_id}.docx",
        )

    def _no_windows(_chunk_id, _content):
        return []

    with patch("knowledge.graph_rag.runtime.prepared_windows", new=_no_windows), \
         patch("knowledge.graph_rag.runtime.insert_windows", new=_no_insert):
        asyncio.run(service._build(["batch-doc-1", "batch-doc-2"], "all"))

    assert calls["rebuild"] == 1
    progress = json.loads((tmp_path / "graph-progress.json").read_text(encoding="utf-8"))
    assert progress["build"]["completed_documents"] == 2
