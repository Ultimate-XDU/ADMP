from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

import numpy as np

from knowledge.graph_rag.hybrid_index import (
    RRF_K,
    _alias_order,
    _dense_order,
    bm25_order,
    catalog_brief,
    retrieve_spans,
    row_key,
    rrf_order,
    window_rows,
)
from knowledge.graph_rag.service import _planner_model, _planner_prompt


def _window_chunk(doc_id: str, chunk_id: str, content: str, section: str = "part", title: str = "manual") -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "content": content,
        "doc_name": title,
        "page_number": 1,
        "section_title": section,
        "heading": section,
        "publisher": "",
        "language": "en",
        "source_url": "",
    }


class GraphRankingPlannerTest(unittest.TestCase):
    def test_filtered_dense_matrix_uses_row_keys(self) -> None:
        chunks = {
            "drop-me": _window_chunk("drop", "drop-me", "outside row"),
            "keep-hi": _window_chunk("keep", "keep-hi", "kept high"),
            "keep-lo": _window_chunk("keep", "keep-lo", "kept low"),
        }
        rows = window_rows(chunks)
        self.assertEqual({row["unit_id"] for row in rows}, {"drop-me", "keep-hi", "keep-lo"})
        ordered = (
            [row for row in rows if row["unit_id"] == "keep-lo"]
            + [row for row in rows if row["unit_id"] == "drop-me"]
            + [row for row in rows if row["unit_id"] == "keep-hi"]
        )
        stored_ids = [row_key(row) for row in ordered]
        keep_rows = [row for row in rows if row["doc_id"] == "keep"]
        self.assertNotEqual(stored_ids, [row_key(row) for row in keep_rows])
        self.assertNotEqual(stored_ids, [row_key(row) for row in rows])
        matrix = np.zeros((len(ordered), 2), dtype=np.float32)
        weight = {"keep-hi": 1.0, "drop-me": 0.75, "keep-lo": 0.25}
        for index, row in enumerate(ordered):
            matrix[index, 0] = weight[row["unit_id"]]
        snapshot = {"chunks": chunks, "window_matrix": matrix, "window_ids": stored_ids}
        recorded = []

        def fake_encode(texts):
            recorded.append([str(item) for item in texts])
            return np.asarray([[1.0, 0.0]], dtype=np.float32)

        def encode_documents(texts, query):
            raise AssertionError(texts)

        with patch("knowledge.graph_rag.runtime.encode_texts", fake_encode):
            order = _dense_order("channel", keep_rows, snapshot, encode=encode_documents)
            hits = retrieve_spans(
                snapshot,
                "channel",
                ["keep"],
                encode=encode_documents,
                rerank=lambda query, texts: None,
            )
        unit_of = {row_key(row): row["unit_id"] for row in rows}
        units = [unit_of[key] for key in order]
        self.assertEqual(units[0], "keep-hi")
        self.assertIn("keep-lo", units)
        self.assertNotIn("drop-me", units)
        self.assertEqual(recorded, [["channel"], ["channel"]])
        self.assertTrue(hits)
        self.assertEqual(hits[0]["unit_id"], "keep-hi")
        self.assertTrue(all(hit["doc_id"] == "keep" for hit in hits))
        self.assertNotIn("drop-me", [hit["unit_id"] for hit in hits])

    def test_round_robin_pool_and_rerank_order(self) -> None:
        chunks = {}
        for index in range(20):
            chunks[f"w{index:02d}"] = _window_chunk(
                "doc",
                f"w{index:02d}",
                f"alphatoken {index:02d}",
                section="alphatoken",
                title="binder",
            )
        chunks["body"] = _window_chunk("doc", "body", "quartzwindowtext", section="notes", title="binder")
        query = "alphatoken"

        def encode(texts, _query):
            return [10.0 if "quartzwindowtext" in text else 0.05 for text in texts]

        def score_of(index, text):
            if "quartzwindowtext" in text:
                return 50.0
            return 1.0 / (1 + index)

        rows = window_rows(chunks)
        channels = [
            bm25_order(query, rows),
            _dense_order(query, rows, {"chunks": chunks}, encode),
            _alias_order(query, rows),
        ]
        fused = rrf_order([ranked for ranked in channels if ranked])
        fused_keys = [key for key, _score, _ranks in fused]
        body_key = next(row_key(row) for row in rows if row["unit_id"] == "body")
        self.assertGreaterEqual(fused_keys.index(body_key), 16)
        unit_by_key = {row_key(row): row["unit_id"] for row in rows}
        expected_off = [unit_by_key[key] for key in fused_keys[:8]]
        self.assertNotIn("body", expected_off)

        off = retrieve_spans(
            {"chunks": chunks},
            query,
            encode=encode,
            rerank=lambda _query, _texts: None,
            limit=8,
            rerank_pool=16,
        )
        self.assertEqual([hit["unit_id"] for hit in off], expected_off)
        self.assertAlmostEqual(off[0]["score"], fused[0][1])

        def broken(_query, _texts):
            raise RuntimeError("rerank unavailable")

        raised = retrieve_spans(
            {"chunks": chunks},
            query,
            encode=encode,
            rerank=broken,
            limit=8,
            rerank_pool=16,
        )
        self.assertEqual([hit["unit_id"] for hit in raised], expected_off)

        seen = []

        def rerank(_query, texts):
            seen.append(list(texts))
            return [score_of(index, text) for index, text in enumerate(texts)]

        hits = retrieve_spans(
            {"chunks": chunks},
            query,
            encode=encode,
            rerank=rerank,
            limit=8,
            rerank_pool=16,
        )
        self.assertEqual(len(seen), 1)
        pool = seen[0]
        self.assertLessEqual(len(pool), 16)
        self.assertTrue(any("quartzwindowtext" in text for text in pool))
        text_to_unit = {}
        for row in rows:
            text_to_unit.setdefault(row["text"], row["unit_id"])
        order = sorted(range(len(pool)), key=lambda index: (-score_of(index, pool[index]), index))
        expected_hits = [text_to_unit[pool[index]] for index in order[:8]]
        self.assertEqual([hit["unit_id"] for hit in hits], expected_hits)
        self.assertEqual(hits[0]["unit_id"], "body")
        self.assertEqual(hits[0]["ranks"]["dense"], 1)
        self.assertEqual(hits[0]["ranks"]["rerank"], 1)
        self.assertNotIn("lexical", hits[0]["ranks"])
        for rank, hit in enumerate(hits, start=1):
            self.assertAlmostEqual(hit["score"], 1.0 / (RRF_K + rank))
            self.assertNotEqual(hit["score"], 50.0)

    def test_catalog_brief_prefers_human_title(self) -> None:
        snapshot = {
            "chunks": {
                "c1": {
                    "chunk_id": "c1",
                    "doc_id": "d1",
                    "doc_name": "physical-file.pdf",
                    "document_title": "Human Title",
                    "publisher": "Office",
                    "language": "zh",
                    "section_title": "Intro",
                },
                "c2": {
                    "chunk_id": "c2",
                    "doc_id": "d2",
                    "doc_name": "plain-name.pdf",
                    "publisher": "",
                    "language": "en",
                    "section_title": "",
                },
            }
        }
        brief = {row["id"]: row for row in catalog_brief(snapshot, None)}
        self.assertEqual(brief["d1"]["title"], "Human Title")
        self.assertEqual(brief["d2"]["title"], "plain-name.pdf")
        self.assertEqual(snapshot["chunks"]["c1"]["doc_name"], "physical-file.pdf")
        self.assertEqual(brief["d1"]["language"], "zh")
        self.assertIn("Intro", brief["d1"]["sections"])

    def test_planner_prompt_is_bounded_to_relevant_state(self) -> None:
        marker = "repeated-source-uri-marker"
        long_id = marker + "/" + ("u" * 400)
        docs = []
        for index in range(103):
            docs.append(
                {
                    "id": f"doc-{index:03d}",
                    "title": f"Unrelated Volume {index:03d}",
                    "publisher": "Pub",
                    "language": "en",
                    "sections": [f"Sec{index:03d}-{part}-" + ("x" * 40) for part in range(12)],
                }
            )
        docs[4] = {
            "id": "doc-004",
            "title": "filename-004.pdf",
            "publisher": "Pub",
            "language": "en",
            "sections": ["Intro"],
        }
        docs[9] = {
            "id": "doc-009",
            "title": "Named Manual",
            "publisher": "Pub",
            "language": "en",
            "sections": ["Body"],
        }
        question = "QUESTION-KEEP-PHRASE Named Manual and the remaining part"
        actions = [
            "graph_search",
            "graph_expand",
            "document_retrieve",
            "graph_search",
            "document_retrieve",
            "inspect_evidence",
        ]
        observations = []
        for step, action in enumerate(actions):
            excerpt = f"MID{step}-" + ("m" * 500)
            if step == 0:
                excerpt = "EARLY_ONLY_MARK" + ("e" * 900)
            if step == 5:
                excerpt = "LATEST_EVIDENCE_MARK"
            node_ids = ["node-a"] if step == 0 else [f"node-{step}"]
            if step == 1:
                node_ids = ["bridge-node-77"]
            if step == 5:
                node_ids = ["latest-node-3"]
            source_ids = [long_id, f"dup-{step}"]
            if step == 5:
                source_ids = ["latest-source-9", long_id]
            observations.append(
                {
                    "action": action,
                    "query": f"query-step-{step}",
                    "summary": ("summary pad " * 50) + f" step {step}",
                    "nodeIds": node_ids,
                    "sourceIds": source_ids,
                    "labels": {"bridge-node-77": "Bridge Label"} if step == 1 else {},
                    "excerpts": [{"chunk_id": f"chunk-{step}", "excerpt": excerpt}],
                    "spans": [
                        {
                            "unit_id": f"chunk-{step}",
                            "doc_id": "doc-004",
                            "text": "span",
                            "start": 0,
                            "end": 4,
                            "page": 1,
                        }
                    ],
                    "documents": docs,
                }
            )
        state = {
            "question": question,
            "doc_ids": [f"doc-{index:03d}" for index in range(103)],
            "tool_calls": 6,
            "observations": observations,
            "evidence": [
                {
                    "chunk_id": "chunk-5",
                    "doc_id": "doc-004",
                    "content": "passage body",
                    "doc_name": "filename-004.pdf",
                    "document_title": "Touched Guide",
                    "source_url": long_id,
                    "page_number": 1,
                    "section_title": "Intro",
                }
            ],
            "trace": [{"step": 1, "action": "graph_search"}],
            "graph_parts": [],
        }
        before = copy.deepcopy(state)
        captured = []

        def fake_chat(prompt, **kwargs):
            captured.append(prompt)
            return "{}"

        with patch("knowledge.graph_rag.runtime.ollama_chat", fake_chat):
            self.assertEqual(_planner_model(state), "{}")
        self.assertEqual(state, before)
        self.assertEqual(len(state["doc_ids"]), 103)
        self.assertIn("doc-057", state["doc_ids"])
        self.assertEqual(len(captured), 1)
        prompt = captured[0]
        self.assertIn(question, prompt)
        self.assertIn("every part of the question", prompt)
        self.assertIn("Query a missing part before finish", prompt)
        self.assertIn("LATEST_EVIDENCE_MARK", prompt)
        self.assertNotIn("EARLY_ONLY_MARK", prompt)
        self.assertIn("latest-source-9", prompt)
        self.assertIn("latest-node-3", prompt)
        self.assertIn("bridge-node-77", prompt)
        self.assertIn("Bridge Label", prompt)
        self.assertIn("Touched Guide", prompt)
        self.assertIn("doc-004", prompt)
        self.assertIn("doc-009", prompt)
        self.assertIn("Intro", prompt)
        self.assertNotIn("filename-004.pdf", prompt)
        self.assertNotIn("Unrelated Volume", prompt)
        self.assertNotIn("doc-057", prompt)
        self.assertNotIn("Sec057", prompt)
        self.assertEqual(prompt.count(marker), 1)
        self.assertLess(len(prompt), 8000)
        for step, action in enumerate(actions):
            self.assertIn(f"{action}: query-step-{step}", prompt)


def test_planner_prompt_retains_graph_search_seed_after_empty_retrieve():
    seed = "seed-node-alpha"
    state = {
        "question": "which limit applies",
        "tool_calls": 2,
        "observations": [
            {
                "action": "graph_search",
                "query": "limit",
                "summary": "found one seed",
                "nodeIds": [seed],
                "sourceIds": [],
                "labels": {},
                "excerpts": [],
                "spans": [],
                "documents": [],
            },
            {
                "action": "document_retrieve",
                "query": "limit",
                "summary": "retrieved text",
                "nodeIds": [],
                "sourceIds": [],
                "labels": {},
                "excerpts": [],
                "spans": [],
                "documents": [],
            },
        ],
        "evidence": [],
    }
    snapshot = copy.deepcopy(state)
    prompt = _planner_prompt(state)
    assert seed in prompt
    assert state == snapshot


if __name__ == "__main__":
    unittest.main()
