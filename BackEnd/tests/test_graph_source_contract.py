from knowledge.graph_rag.hierarchy import build_structure
from knowledge.graph_rag.router import EvidenceModel
from knowledge.graph_rag.service import GraphService, _block_chunk, _evidence, _public_evidence


def test_block_chunk_keeps_single_character_scalar():
    locator = {"label": "/rows/0", "page": 0}
    first = _block_chunk(
        "noaa-1",
        {
            "chunk_id": "noaa-1:block:0",
            "content": "0",
            "doc_name": "obs.json",
            "document_title": "NOAA观测",
            "page_number": 0,
            "source_url": "https://example.test/obs.json",
            "locator": locator,
            "publisher": "NOAA",
            "original_format": "json",
        },
        "fallback.json",
    )
    second = _block_chunk(
        "noaa-1",
        {
            "chunk_id": "noaa-1:block:1",
            "content": "0",
            "doc_name": "obs.json",
            "page_number": 0,
            "locator": {"label": "/rows/1", "page": 0},
        },
        "fallback.json",
    )
    assert first is not None
    assert second is not None
    assert first["content"] == "0"
    assert second["content"] == "0"
    assert first["chunk_id"] == "noaa-1:block:0"
    assert second["chunk_id"] == "noaa-1:block:1"
    assert first["locator"] == locator
    assert first["publisher"] == "NOAA"
    assert first["page_number"] == 0
    assert first["source_url"] == "https://example.test/obs.json"
    assert first["doc_name"] == "obs.json"
    assert first["document_title"] == "NOAA观测"
    assert _block_chunk("noaa-1", {"chunk_id": "noaa-1:block:2", "content": "  "}, "obs.json") is None


def test_public_evidence_keeps_filename_title_locator_and_page_zero():
    chunk = _block_chunk(
        "pdf-9",
        {
            "chunk_id": "pdf-9:block:0",
            "content": "正文内容",
            "doc_name": "real-file.pdf",
            "document_title": "人类可读标题",
            "page_number": 0,
            "original_format": "pdf",
            "source_url": "https://cdn.example.test/real-file.pdf",
            "locator": {"label": "第1页", "page": 0},
        },
        "人类可读标题",
    )
    assert chunk["doc_name"] == "real-file.pdf"
    assert chunk["document_title"] == "人类可读标题"
    assert chunk["page_number"] == 0
    assert chunk["page_count"] == 0
    assert chunk["source_url"] == "https://cdn.example.test/real-file.pdf"
    assert chunk["locator"] == {"label": "第1页", "page": 0}
    fallback = _block_chunk(
        "pdf-9",
        {
            "chunk_id": "pdf-9:block:1",
            "content": "只有标题的正文",
            "document_title": "只有标题",
            "page_number": 0,
            "source_url": "https://cdn.example.test/real-file.pdf",
        },
        "real-file.pdf",
    )
    assert fallback["doc_name"] == "real-file.pdf"
    assert fallback["document_title"] == "只有标题"
    staged = _evidence(chunk, 0.25, "正文内容")
    assert staged["chunk_id"] == "pdf-9:block:0"
    assert staged["doc_name"] == "real-file.pdf"
    assert staged["document_title"] == "人类可读标题"
    assert staged["page_number"] == 0
    assert staged["locator"]["label"] == "第1页"
    evidence = _public_evidence(staged)
    assert evidence["doc_name"] == "real-file.pdf"
    assert evidence["document_title"] == "人类可读标题"
    assert evidence["page_number"] == 0
    assert evidence["source_url"] == "https://cdn.example.test/real-file.pdf"
    assert evidence["locator"]["label"] == "第1页"
    model = EvidenceModel(**evidence)
    assert model.doc_name == "real-file.pdf"
    assert model.document_title == "人类可读标题"
    assert model.page_number == 0
    assert model.source_url == "https://cdn.example.test/real-file.pdf"
    assert model.locator is not None
    assert model.locator.label == "第1页"


def test_hierarchy_leaf_label_previews_source_content():
    doc_id = "doc-leaf"
    long_text = "测" * 100
    chunks = {
        "doc-leaf:block:0": {
            "chunk_id": "doc-leaf:block:0",
            "doc_id": doc_id,
            "content": "0",
            "page_number": 0,
            "locator": {"label": "第1页", "page": 0},
            "source_type": "text",
        },
        "doc-leaf:block:1": {
            "chunk_id": "doc-leaf:block:1",
            "doc_id": doc_id,
            "content": long_text,
            "page_number": 0,
            "locator": {"label": "第1页", "page": 0},
            "source_type": "table",
        },
        "doc-leaf:block:2": {
            "chunk_id": "doc-leaf:block:2",
            "doc_id": doc_id,
            "content": "  锚点\n说明  ",
            "locator": {"label": "policy.html#article-3"},
            "source_type": "figure",
        },
    }
    documents = {doc_id: {"title": "人类可读标题", "name": "人类可读标题"}}
    built = build_structure(
        documents,
        chunks,
        [{"id": "ent:duty", "label": "进货查验义务", "sourceIds": ["doc-leaf:block:0"]}],
    )
    nodes = {node["id"]: node for node in built["nodes"]}
    assert nodes[f"doc:{doc_id}"]["label"] == "人类可读标题"
    first = nodes["chk:doc-leaf:block:0"]
    second = nodes["chk:doc-leaf:block:1"]
    third = nodes["chk:doc-leaf:block:2"]
    assert first["label"] == "0"
    assert second["label"] == "测" * 80
    assert len(second["label"]) == 80
    assert third["label"] == "锚点 说明"
    assert first["label"] != second["label"]
    assert first["locator"]["label"] == "第1页"
    assert second["locator"]["label"] == "第1页"
    assert third["locator"]["label"] == "policy.html#article-3"
    assert first["sourceIds"] == ["doc-leaf:block:0"]
    assert {first["type"], second["type"], third["type"]} <= {"chunk", "table", "figure"}
    assert "ent:duty" not in nodes or nodes["ent:duty"]["label"] == "进货查验义务"


def test_catalog_reload_restores_storage_filename(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    (corpus / "units").mkdir(parents=True)
    (corpus / "manifest.json").write_text("[]\n", encoding="utf-8")
    monkeypatch.setenv("GRAPH_CORPUS_DIR", str(corpus))
    monkeypatch.setenv("GRAPH_MANAGEMENT_STATE", str(tmp_path / "missing-management.json"))
    root = tmp_path / "graph"
    service = GraphService(root)
    doc_id = "pdf-doc-9"
    storage = "uploads/reports/real-file.pdf"
    first_id = f"{doc_id}:block:0"
    second_id = f"{doc_id}:block:1"
    service.catalog.replace_chunks(
        doc_id,
        "年度报告",
        "https://example.test/real-file.pdf",
        [
            {
                "chunk_id": first_id,
                "doc_id": doc_id,
                "content": "正文零",
                "doc_name": "年度报告",
                "page_number": 0,
                "source_url": "https://example.test/real-file.pdf",
                "locator": {"label": "第1页", "page": 0},
            },
            {
                "chunk_id": second_id,
                "doc_id": doc_id,
                "content": "正文一",
                "doc_name": "年度报告",
                "document_title": "已有标题",
                "page_number": 0,
                "source_url": "https://example.test/real-file.pdf",
                "locator": {"label": "第1页", "page": 0},
            },
        ],
        "managed",
        storage,
    )
    assert service.catalog.documents[doc_id]["storagePath"] == storage
    service.catalog.chunks[first_id]["doc_name"] = "年度报告"
    service.catalog.chunks[first_id].pop("document_title", None)
    service.catalog.chunks[second_id]["doc_name"] = "年度报告"
    service.catalog.chunks[second_id]["document_title"] = "已有标题"
    service.catalog.add_source_map({"win-0": first_id, "win-1": second_id})
    service.catalog.save()
    before_ids = set(service.catalog.chunks)
    before_map = dict(service.catalog.source_map)
    before_inner = {key: chunk["chunk_id"] for key, chunk in service.catalog.chunks.items()}

    reloaded = GraphService(root)
    assert set(reloaded.catalog.chunks) == before_ids
    assert reloaded.catalog.source_map == before_map
    assert reloaded.catalog.chunks[first_id]["chunk_id"] == before_inner[first_id]
    assert reloaded.catalog.chunks[second_id]["chunk_id"] == before_inner[second_id]
    assert reloaded.catalog.chunks[first_id]["doc_name"] == "real-file.pdf"
    assert reloaded.catalog.chunks[first_id]["document_title"] == "年度报告"
    assert reloaded.catalog.chunks[second_id]["doc_name"] == "real-file.pdf"
    assert reloaded.catalog.chunks[second_id]["document_title"] == "已有标题"
    assert reloaded.catalog.chunks[first_id]["page_number"] == 0
    assert reloaded.catalog.chunks[first_id]["locator"]["label"] == "第1页"
    assert reloaded.catalog.chunks[first_id]["source_url"] == "https://example.test/real-file.pdf"
    assert reloaded.catalog.documents[doc_id]["storagePath"] == storage
