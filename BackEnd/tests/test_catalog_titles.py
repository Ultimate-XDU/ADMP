"""ensure_corpus keeps a managed title when the frozen manifest title is empty."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from knowledge.graph_rag.catalog import Catalog


def _write_catalog(path: Path, documents: dict, chunks: dict, source_map: dict) -> None:
    path.write_text(
        json.dumps({"documents": documents, "chunks": chunks, "source_map": source_map}),
        encoding="utf-8",
    )


def test_ensure_corpus_empty_manifest_title_preserves_managed_name(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.json"
    chunk_id = "nasa-polycarp-readme:0"
    content = "PolyCARP detects conflicts among ownship and traffic."
    chunks = {
        chunk_id: {
            "chunk_id": chunk_id,
            "doc_id": "nasa-polycarp-readme",
            "content": content,
        }
    }
    source_map = {"poly:0": chunk_id}
    _write_catalog(
        catalog_path,
        documents={
            "nasa-polycarp-readme": {
                "id": "nasa-polycarp-readme",
                "name": "nasa-polycarp-readme",
                "title": "PolyCARP",
                "sourceUrl": "https://example.test/polycarp",
                "kind": "corpus",
                "indexed": True,
                "availableChunks": 1,
                "ingestedChunks": 1,
                "extractedChunkIds": [chunk_id],
            },
            "nasa-icarous-overview": {
                "id": "nasa-icarous-overview",
                "name": "ICAROUS briefing",
                "sourceUrl": "",
                "kind": "corpus",
                "indexed": False,
                "availableChunks": 0,
                "ingestedChunks": 0,
                "extractedChunkIds": [],
            },
        },
        chunks=chunks,
        source_map=source_map,
    )
    manifests = {
        "nasa-polycarp-readme": {"title": "", "source_url": "https://example.test/polycarp"},
        "nasa-icarous-overview": {"title": "   ", "source_url": "https://example.test/icarous"},
    }

    with (
        patch("knowledge.graph_rag.catalog.corpus_ids", return_value=list(manifests)),
        patch("knowledge.graph_rag.catalog.manifest_by_id", return_value=manifests),
        patch("knowledge.graph_rag.catalog.available_count", return_value=1),
    ):
        catalog = Catalog(catalog_path)
        before_chunks = json.loads(json.dumps(catalog.chunks))
        before_map = json.loads(json.dumps(catalog.source_map))
        catalog.ensure_corpus()
        catalog.ensure_corpus()

    poly = catalog.documents["nasa-polycarp-readme"]
    icarous = catalog.documents["nasa-icarous-overview"]
    assert poly["name"] == "PolyCARP"
    assert poly["title"] == "PolyCARP"
    assert icarous["name"] == "ICAROUS briefing"
    assert poly["extractedChunkIds"] == [chunk_id]
    assert poly["indexed"] is True
    assert poly["ingestedChunks"] == 1
    assert catalog.chunks == before_chunks
    assert catalog.source_map == before_map
    assert catalog.chunks[chunk_id]["content"] == content
    assert catalog.chunks[chunk_id]["chunk_id"] == chunk_id

    saved = json.loads(catalog_path.read_text(encoding="utf-8"))
    assert saved["documents"]["nasa-polycarp-readme"]["name"] == "PolyCARP"
    assert saved["chunks"] == chunks
    assert saved["source_map"] == source_map
