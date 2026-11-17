"""Catalog load folds same-source identities onto manifest doc ids once."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from knowledge.graph_rag.catalog import Catalog


class CatalogIdentityMigrationTest(unittest.TestCase):
    def test_load_merges_same_source_and_reload_is_idempotent(self) -> None:
        urn = "urn:ieee:alpha"
        canonical = "doc-alpha"
        source = "https://publisher.example/alpha"
        other_source = "https://publisher.example/other"
        manifest = {
            canonical: {
                "doc_id": canonical,
                "title": "Alpha",
                "source_url": source,
                "file": "alpha.pdf",
            }
        }
        payload = {
            "documents": {
                urn: {
                    "id": urn,
                    "name": "Alpha upload",
                    "sourceUrl": source,
                    "documentKey": urn,
                    "kind": "managed",
                    "storagePath": "work/alpha.pdf",
                    "indexed": True,
                    "availableChunks": 2,
                    "ingestedChunks": 2,
                    "extractedChunkIds": ["c1", "c2"],
                    "publisher": "IEEE",
                    "title": "Alpha Paper",
                },
                canonical: {
                    "id": canonical,
                    "name": "Alpha",
                    "sourceUrl": source,
                    "kind": "corpus",
                    "storagePath": "",
                    "indexed": False,
                    "availableChunks": 2,
                    "ingestedChunks": 0,
                    "extractedChunkIds": [],
                },
                "other-doc": {
                    "id": "other-doc",
                    "name": "Other",
                    "sourceUrl": other_source,
                    "documentKey": "urn:other",
                    "kind": "managed",
                    "storagePath": "work/other.pdf",
                    "indexed": True,
                    "availableChunks": 1,
                    "ingestedChunks": 1,
                    "extractedChunkIds": ["c3"],
                },
            },
            "chunks": {
                "c1": {
                    "chunk_id": "c1",
                    "doc_id": urn,
                    "content": "one",
                    "source_url": source,
                    "document_key": urn,
                },
                "c2": {"chunk_id": "c2", "doc_id": urn, "content": "two", "source_url": source},
                "c3": {"chunk_id": "c3", "doc_id": "other-doc", "content": "three", "source_url": other_source},
            },
            "source_map": {"unit-1": "c1", "unit-3": "c3"},
            "build": {"building": False, "status": "ready"},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "catalog.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with patch("knowledge.graph_rag.catalog.manifest_by_id", return_value=manifest), patch(
                "knowledge.graph_rag.catalog.is_corpus",
                side_effect=lambda doc_id: doc_id in manifest,
            ):
                catalog = Catalog(path)
                self.assertEqual(set(catalog.documents), {canonical, "other-doc"})
                merged = catalog.documents[canonical]
                self.assertEqual(merged["id"], canonical)
                self.assertEqual(merged["documentKey"], urn)
                self.assertEqual(merged["sourceUrl"], source)
                self.assertTrue(merged["indexed"])
                self.assertEqual(merged["extractedChunkIds"], ["c1", "c2"])
                self.assertEqual(merged["ingestedChunks"], 2)
                self.assertEqual(merged["publisher"], "IEEE")
                self.assertEqual(merged["title"], "Alpha Paper")
                self.assertEqual(merged["storagePath"], "work/alpha.pdf")
                self.assertEqual(catalog.chunks["c1"]["doc_id"], canonical)
                self.assertEqual(catalog.chunks["c1"]["chunk_id"], "c1")
                self.assertEqual(catalog.chunks["c1"]["content"], "one")
                self.assertEqual(catalog.chunks["c1"]["document_key"], urn)
                self.assertEqual(catalog.chunks["c2"]["doc_id"], canonical)
                self.assertEqual(catalog.chunks["c2"]["chunk_id"], "c2")
                self.assertEqual(catalog.chunks["c3"]["doc_id"], "other-doc")
                self.assertEqual(catalog.chunks["c3"]["content"], "three")
                self.assertEqual(list(catalog.chunks), ["c1", "c2", "c3"])
                self.assertEqual(catalog.source_map, {"unit-1": "c1", "unit-3": "c3"})
                other = catalog.documents["other-doc"]
                self.assertEqual(other["name"], "Other")
                self.assertEqual(other["documentKey"], "urn:other")
                self.assertEqual(other["sourceUrl"], other_source)
                self.assertEqual(other["extractedChunkIds"], ["c3"])
                stamp = path.stat().st_mtime_ns
                again = Catalog(path)
                self.assertEqual(path.stat().st_mtime_ns, stamp)
                self.assertEqual(set(again.documents), {canonical, "other-doc"})
                self.assertEqual(again.documents[canonical]["documentKey"], urn)
                self.assertEqual(again.chunks["c1"]["doc_id"], canonical)
                self.assertEqual(again.chunks["c2"]["doc_id"], canonical)
                self.assertEqual(again.chunks["c3"]["doc_id"], "other-doc")
                self.assertEqual(list(again.chunks), ["c1", "c2", "c3"])
                self.assertEqual(again.source_map, {"unit-1": "c1", "unit-3": "c3"})
