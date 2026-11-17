"""Controller, filter, and lifecycle tests. No extraction model is called."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ["GRAPH_DISABLE_VECTORS"] = "1"

from knowledge.graph_rag.agent import policy_action, run_agent
from knowledge.graph_rag.config import ollama_host
from knowledge.graph_rag.corpus import _bibliography_unit
from knowledge.graph_rag.graph_ops import expand_hops, project_graph
from knowledge.graph_rag.router import SearchBody, graph_search
from knowledge.graph_rag.service import (
    _select_answer_sources,
    _tools,
    get_service,
    normalize_doc_ids,
    reset_service,
)


def _chunk(doc_id: str, chunk_id: str, content: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "content": content,
        "doc_name": doc_id,
        "page_number": 1,
        "page_count": 1,
        "section_title": "body",
        "heading": "body",
        "parent_heading": doc_id,
        "source_url": "",
        "source_type": "text",
        "highlight_text": "",
    }


def _snapshot() -> dict:
    return {
        "chunks": {
            "c1": _chunk("d1", "c1", "icing on the wing changes lift"),
            "c2": _chunk("d1", "c2", "stall speed increases after ice"),
        },
        "graph": {
            "nodes": [
                {"id": "n1", "label": "icing", "type": "condition", "sourceIds": ["c1"], "description": "icing"},
                {"id": "n2", "label": "wing", "type": "part", "sourceIds": ["c1"], "description": "wing"},
                {"id": "n3", "label": "stall", "type": "condition", "sourceIds": ["c2"], "description": "stall"},
            ],
            "edges": [
                {"id": "e1", "source": "n1", "target": "n2", "label": "forms_on", "sourceIds": ["c1"]},
                {"id": "e2", "source": "n2", "target": "n3", "label": "associated_with", "sourceIds": ["c2"]},
            ],
        },
        "passage_ids": [],
        "passage_matrix": None,
        "entity_ids": [],
        "entity_matrix": None,
    }


class GraphControllerTests(unittest.TestCase):
    def test_policy_searches_bridge_then_reserves_inspect(self) -> None:
        state = {
            "question": "icing wing",
            "tool_calls": 2,
            "observations": [
                {
                    "action": "graph_search",
                    "query": "icing wing",
                    "nodeIds": ["n1", "n2"],
                    "sourceIds": ["c1"],
                    "labels": {"n1": "icing", "n2": "wing"},
                },
                {
                    "action": "graph_expand",
                    "query": "icing wing",
                    "nodeIds": ["n1", "n2", "n3"],
                    "sourceIds": ["c1", "c2"],
                    "labels": {"n1": "icing", "n2": "wing", "n3": "stall"},
                },
            ],
        }
        bridge = policy_action(state)
        self.assertEqual(bridge["action"], "graph_search")
        self.assertIn("stall", bridge["query"])
        state["tool_calls"] = 3
        state["observations"].append(
            {
                "action": "graph_search",
                "query": "stall",
                "nodeIds": ["n3"],
                "sourceIds": ["c2"],
                "labels": {"n3": "stall"},
            }
        )
        self.assertEqual(policy_action(state)["action"], "inspect_evidence")

    def test_compiled_graph_preserves_state_and_bridge_passages(self) -> None:
        seen = {}

        def answer_fn(state: dict) -> dict:
            sources = _select_answer_sources(state)
            seen["question"] = state.get("question")
            seen["sources"] = [row["chunk_id"] for row in sources]
            seen["evidence"] = [row["chunk_id"] for row in state.get("evidence") or []]
            return {"answer": "见 [1] 与 [2]。", "stop_reason": "answered", "sources": sources}

        result = run_agent("icing wing", None, "graph", _tools(_snapshot()), answer_fn, None)
        actions = [step["action"] for step in result["trace"]]
        self.assertEqual(actions, ["graph_search", "graph_expand", "graph_search", "inspect_evidence", "finish"])
        self.assertEqual(result["question"], "icing wing")
        self.assertEqual(result["tool_calls"], 4)
        self.assertEqual(seen["question"], "icing wing")
        self.assertIn("c1", seen["evidence"])
        self.assertIn("c2", seen["evidence"])
        self.assertEqual(seen["sources"][:2], ["c1", "c2"])
        self.assertEqual([row["chunk_id"] for row in result["evidence"]], seen["sources"])
        self.assertTrue(all(path["hops"] <= 3 for part in result["graph_parts"] for path in part.get("paths") or []))

    def test_model_cannot_spend_the_last_call_on_another_search(self) -> None:
        queries = iter(["icing wing", "stall", "wing icing", "extra query"])

        def action_fn(_state: dict) -> str:
            query = next(queries)
            return (
                '{"action":"graph_search","query":"%s","nodeIds":[],"chunkIds":[],"hops":1}' % query
            )

        result = run_agent("icing wing", None, "agent", _tools(_snapshot()), _answer, action_fn)
        actions = [step["action"] for step in result["trace"]]
        self.assertEqual(actions, ["graph_search", "graph_search", "graph_search", "inspect_evidence", "finish"])
        self.assertEqual(result["question"], "icing wing")
        self.assertTrue(result["evidence"])

    def test_inspect_then_finish(self) -> None:
        def action_fn(state: dict) -> str:
            if int(state.get("tool_calls") or 0) == 0:
                return '{"action":"graph_search","query":"icing wing","nodeIds":[],"chunkIds":[],"hops":1}'
            if int(state.get("tool_calls") or 0) == 1:
                return '{"action":"inspect_evidence","query":"icing wing","nodeIds":[],"chunkIds":["c1"],"hops":1}'
            return '{"action":"finish","query":"","nodeIds":[],"chunkIds":[],"hops":1}'

        result = run_agent("icing wing", None, "agent", _tools(_snapshot()), _answer, action_fn)
        actions = [step["action"] for step in result["trace"]]
        self.assertEqual(actions, ["graph_search", "inspect_evidence", "finish"])
        self.assertIn("c1", [row["chunk_id"] for row in result["evidence"]])
        self.assertEqual(result["stop_reason"], "answered")

    def test_edge_passages_are_not_dropped_for_later_hits(self) -> None:
        filler = []
        for index in range(8):
            filler.append(
                {
                    "chunk_id": f"f{index}",
                    "content": "x" * 900,
                    "score": 10 - index,
                    "doc_name": "d",
                    "page_number": 1,
                    "section_title": "",
                    "heading": "",
                }
            )
        bridge = {
            "chunk_id": "bridge",
            "content": "stall after icing " * 20,
            "score": 0.01,
            "doc_name": "d",
            "page_number": 2,
            "section_title": "handling",
            "heading": "handling",
        }
        state = {
            "question": "stall",
            "evidence": filler + [bridge],
            "observations": [{"nodeIds": ["n3"]}],
            "graph_parts": [
                {
                    "nodes": [{"id": "n3", "label": "stall", "type": "condition", "sourceIds": ["bridge"]}],
                    "edges": [
                        {"id": "e9", "source": "n2", "target": "n3", "label": "associated_with", "sourceIds": ["bridge"]}
                    ],
                }
            ],
        }
        selected = _select_answer_sources(state)
        self.assertEqual(selected[0]["chunk_id"], "bridge")
        self.assertIn("bridge", [row["chunk_id"] for row in selected])

    def test_distinct_document_query_is_accepted_and_duplicate_is_rejected(self) -> None:
        from knowledge.graph_rag.agent import validate_action

        state = {
            "tool_calls": 1,
            "question": "icing and wind shear",
            "observations": [
                {"action": "document_retrieve", "query": "icing wing", "sourceIds": ["c1"], "nodeIds": []},
            ],
        }
        accepted = validate_action(
            {"action": "document_retrieve", "query": "wind shear", "nodeIds": [], "chunkIds": [], "hops": 1},
            state,
        )
        self.assertIsNotNone(accepted)
        self.assertEqual(accepted["action"], "document_retrieve")
        self.assertEqual(accepted["query"], "wind shear")
        self.assertIsNone(
            validate_action(
                {"action": "document_retrieve", "query": "  Icing   Wing ", "nodeIds": [], "chunkIds": [], "hops": 1},
                state,
            )
        )

        def action_fn(current: dict) -> str:
            used = int(current.get("tool_calls") or 0)
            if used == 0:
                return '{"action":"document_retrieve","query":"icing wing","nodeIds":[],"chunkIds":[],"hops":1}'
            if used == 1:
                return '{"action":"document_retrieve","query":"  icing   wing  ","nodeIds":[],"chunkIds":[],"hops":1}'
            if used == 2:
                return '{"action":"document_retrieve","query":"wind shear","nodeIds":[],"chunkIds":[],"hops":1}'
            return '{"action":"finish","query":"","nodeIds":[],"chunkIds":[],"hops":1}'

        result = run_agent("icing and wind shear", None, "agent", _tools(_snapshot()), _answer, action_fn)
        retrieved = [step["query"] for step in result["trace"] if step["action"] == "document_retrieve"]
        self.assertEqual(retrieved, ["icing wing", "wind shear"])

    def test_current_observation_leads_with_the_new_ranked_source(self) -> None:
        from knowledge.graph_rag.service import _planner_model

        snapshot = _snapshot()
        snapshot["chunks"]["old-ice"] = _chunk("d1", "old-ice", "icing forms on the wing in the wind")
        snapshot["chunks"]["new-shear"] = _chunk("d1", "new-shear", "low level wind shear near the runway")
        tools = _tools(snapshot)
        state = {
            "question": "icing and wind shear",
            "doc_ids": None,
            "tool_calls": 0,
            "observations": [],
            "trace": [],
            "graph_parts": [],
            "evidence": [],
            "pending": {"action": "document_retrieve", "query": "icing wing"},
        }
        state.update(tools["document_retrieve"](state))
        state["pending"] = {"action": "document_retrieve", "query": "wind shear runway"}
        state.update(tools["document_retrieve"](state))
        excerpts = state["observations"][-1]["excerpts"]
        self.assertEqual(state["observations"][-1]["sourceIds"][0], "new-shear")
        self.assertEqual(excerpts[0]["chunk_id"], "new-shear")
        self.assertLess(
            [row["chunk_id"] for row in excerpts].index("new-shear"),
            [row["chunk_id"] for row in excerpts].index("old-ice"),
        )
        self.assertIn("wind shear", excerpts[0]["excerpt"])

        captured = []

        def fake_chat(prompt: str, **kwargs) -> str:
            captured.append(prompt)
            return "{}"

        with patch("knowledge.graph_rag.runtime.ollama_chat", fake_chat):
            _planner_model(state)
        self.assertIn("new-shear", captured[0])
        self.assertIn("wind shear", captured[0])
        self.assertIn("every part of the question", captured[0])
        self.assertIn("Query a missing part before finish", captured[0])

    def test_later_targeted_and_bridge_sources_survive_packing(self) -> None:
        question = "结冰以及风切变"
        early = []
        for index in range(8):
            early.append(
                {
                    "chunk_id": f"e{index}",
                    "content": ("结冰以及风切变都会影响飞行。" * 80)[:960],
                    "score": 5,
                    "doc_name": "early",
                    "page_number": 1,
                    "section_title": "body",
                    "heading": "body",
                }
            )
        targeted = {
            "chunk_id": "shear",
            "content": ("进近阶段需要单独核对第二项条件。" * 80)[:960],
            "score": 0.1,
            "doc_name": "later",
            "page_number": 4,
            "section_title": "body",
            "heading": "body",
        }
        targeted_state = {
            "question": question,
            "evidence": early + [targeted],
            "observations": [
                {
                    "action": "graph_search",
                    "query": "结冰",
                    "sourceIds": [row["chunk_id"] for row in early],
                    "nodeIds": [f"n{index}" for index in range(8)],
                },
                {"action": "document_retrieve", "query": "第二项条件", "sourceIds": ["shear"], "nodeIds": []},
            ],
            "graph_parts": [
                {
                    "nodes": [{"id": f"n{index}", "label": f"n{index}", "sourceIds": [f"e{index}"]} for index in range(8)],
                    "edges": [
                        {
                            "id": f"edge{index}",
                            "source": f"n{index}",
                            "target": f"n{index + 1}",
                            "label": "related_to",
                            "sourceIds": [f"e{index}"],
                        }
                        for index in range(7)
                    ],
                }
            ],
        }
        selected = _select_answer_sources(targeted_state)
        ids = [row["chunk_id"] for row in selected]
        self.assertIn("shear", ids)
        self.assertEqual(next(row for row in selected if row["chunk_id"] == "shear")["content"], targeted["content"])

        bridge = {
            "chunk_id": "bridge-step",
            "content": ("新的桥接段落只在扩展后出现。" * 80)[:960],
            "score": 0.1,
            "doc_name": "later",
            "page_number": 5,
            "section_title": "body",
            "heading": "body",
        }
        bridge_state = {
            "question": question,
            "evidence": early + [bridge],
            "observations": [
                {
                    "action": "graph_search",
                    "query": "结冰",
                    "sourceIds": [row["chunk_id"] for row in early],
                    "nodeIds": [f"n{index}" for index in range(8)],
                },
                {
                    "action": "graph_expand",
                    "query": "结冰",
                    "sourceIds": [row["chunk_id"] for row in early] + ["bridge-step"],
                    "nodeIds": [f"n{index}" for index in range(8)] + ["n-bridge"],
                },
            ],
            "graph_parts": targeted_state["graph_parts"]
            + [
                {
                    "nodes": [{"id": "n-bridge", "label": "bridge", "sourceIds": ["bridge-step"]}],
                    "edges": [
                        {
                            "id": "edge-bridge",
                            "source": "n0",
                            "target": "n-bridge",
                            "label": "related_to",
                            "sourceIds": ["bridge-step"],
                        }
                    ],
                }
            ],
        }
        packed = _select_answer_sources(bridge_state)
        packed_ids = [row["chunk_id"] for row in packed]
        self.assertIn("bridge-step", packed_ids)
        self.assertEqual(next(row for row in packed if row["chunk_id"] == "bridge-step")["content"], bridge["content"])

    def test_partial_cited_answer_is_not_marked_fully_answered(self) -> None:
        from knowledge.graph_rag.config import ABSTAIN_TEXT
        from knowledge.graph_rag.qa_bridge import answer_with_graph
        from knowledge.graph_rag.service import _answer_from_evidence
        from qa.response import attach_retrieval_fields

        state = {
            "question": "结冰以及风切变",
            "evidence": [
                {
                    "chunk_id": "c1",
                    "content": "机翼结冰会提高失速速度。",
                    "doc_name": "d",
                    "page_number": 1,
                    "section_title": "结冰",
                    "heading": "结冰",
                    "score": 0.8,
                }
            ],
            "observations": [],
            "graph_parts": [],
        }
        mixed = f"结冰会提高失速速度 [1]。另一部分：{ABSTAIN_TEXT}"
        with patch("knowledge.graph_rag.runtime.ollama_chat", return_value=mixed):
            result = _answer_from_evidence(state)
        self.assertEqual(result["stop_reason"], "partial_evidence")
        self.assertEqual(result["answer"], mixed)
        self.assertEqual([row["chunk_id"] for row in result["sources"]], ["c1"])
        self.assertEqual(result["sources"][0]["content"], "机翼结冰会提高失速速度。")

        pure = f"{ABSTAIN_TEXT} [1]"
        with patch("knowledge.graph_rag.runtime.ollama_chat", return_value=pure):
            abstained = _answer_from_evidence(state)
        self.assertEqual(abstained["stop_reason"], "insufficient_evidence")
        self.assertNotIn("提高失速速度", abstained["answer"])

        prefix = "无关内容。" * 200
        suffix = "低空风切变出现在跑道附近。"
        late = {
            "chunk_id": "late",
            "content": prefix + suffix,
            "doc_name": "d",
            "page_number": 2,
            "section_title": "body",
            "heading": "body",
            "score": 0.4,
        }
        self.assertGreater(len(prefix), 900)
        self.assertNotIn("风切变", prefix)
        seen = []

        def fake_chat(prompt: str, **kwargs) -> str:
            seen.append(prompt)
            return "见 [1]。"

        with patch("knowledge.graph_rag.runtime.ollama_chat", fake_chat):
            windowed = _answer_from_evidence(
                {"question": "风切变", "evidence": [late], "observations": [], "graph_parts": []}
            )
        self.assertIn(suffix, seen[0])
        self.assertEqual(windowed["sources"][0]["chunk_id"], "late")
        self.assertEqual(windowed["sources"][0]["content"], late["content"])
        self.assertEqual(windowed["stop_reason"], "answered")

        class _Service:
            def ask(self, question: str, mode: str) -> dict:
                return {
                    "answer": mixed,
                    "sources": [{"chunk_id": "c1", "score": 0.9, "content": "机翼结冰会提高失速速度。"}],
                    "stopReason": "partial_evidence",
                    "trace": [],
                    "graph": {},
                }

        with patch("knowledge.graph_rag.qa_bridge.get_service", return_value=_Service()):
            payload = answer_with_graph("结冰以及风切变", mode="agent")
        self.assertEqual(payload["stopReason"], "partial_evidence")
        self.assertEqual(payload["abstainReason"], "partial_evidence")
        self.assertEqual(payload["confidence"], 0.0)
        attached = attach_retrieval_fields(payload)
        self.assertEqual(attached["citations"], [{"id": 1, "sourceIndex": 0, "chunkId": "c1"}])


def _answer(state: dict) -> dict:
    sources = list(state.get("evidence") or [])
    return {"answer": "见 [1]。", "stop_reason": "answered", "sources": sources}


class GraphApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._old = {
            key: os.environ.get(key)
            for key in (
                "GRAPH_DATA_DIR",
                "GRAPH_AUTO_BUILD",
                "GRAPH_OLLAMA_HOST",
                "LLM_BASE_URL",
                "GRAPH_MANAGEMENT_STATE",
            )
        }
        os.environ["GRAPH_DATA_DIR"] = self._tmp.name
        os.environ["GRAPH_DISABLE_VECTORS"] = "1"
        os.environ.pop("GRAPH_AUTO_BUILD", None)
        reset_service()

    def tearDown(self) -> None:
        reset_service()
        for key, value in self._old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self._tmp.cleanup()

    def _load_two_docs(self) -> None:
        service = get_service()
        service.catalog.replace_chunks("d1", "one", "", [_chunk("d1", "c1", "icing on the wing")], "managed")
        service.catalog.replace_chunks("d2", "two", "", [_chunk("d2", "c2", "unrelated hydraulic pump")], "managed")
        service.projection = {
            "nodes": [
                {"id": "n1", "label": "icing", "type": "condition", "sourceIds": ["c1"], "description": "icing"},
                {"id": "n2", "label": "pump", "type": "part", "sourceIds": ["c2"], "description": "pump"},
            ],
            "edges": [],
        }

    def test_empty_doc_ids_mean_all_documents_and_limit_is_clamped(self) -> None:
        self.assertIsNone(normalize_doc_ids([]))
        self._load_two_docs()
        both = graph_search(SearchBody(query="", docIds=[], limit=100))
        self.assertEqual({node["id"] for node in both["nodes"]}, {"n1", "n2"})
        only = graph_search(SearchBody(query="", docIds=["d1"], limit=100))
        self.assertEqual([node["id"] for node in only["nodes"]], ["n1"])
        service = get_service()
        nodes = []
        chunks = []
        for index in range(90):
            chunk_id = f"p{index}"
            chunks.append(_chunk("d1", chunk_id, f"passage {index}"))
            nodes.append({"id": f"n{index}", "label": f"n{index}", "type": "entity", "sourceIds": [chunk_id]})
        service.catalog.replace_chunks("d1", "one", "", chunks, "managed")
        service.projection = {"nodes": nodes, "edges": []}
        bounded = service.search("", None, 100)
        self.assertEqual(len(bounded["nodes"]), 80)
        self.assertTrue(bounded["truncated"])
        self.assertTrue(all(path["hops"] <= 3 for path in bounded["paths"]))

    def test_blank_browse_does_not_embed(self) -> None:
        self._load_two_docs()
        service = get_service()
        class _Matrix:
            def copy(self):
                return self

        service.passage_matrix = _Matrix()
        service.passage_ids = ["c1"]
        called = []

        def _encode(texts):
            called.append(list(texts))
            raise RuntimeError("embed")

        with patch("knowledge.graph_rag.runtime.encode_texts", _encode):
            graph = service.search("", [], 30)
            self.assertEqual(len(graph["nodes"]), 2)
            self.assertEqual(called, [])
            service.search("icing wing", [], 30)
            self.assertTrue(called)

    def test_edges_without_original_passages_are_dropped(self) -> None:
        projected = project_graph(
            [
                {"id": "a", "entity_name": "A", "source_id": "chunk-1"},
                {"id": "b", "entity_name": "B", "source_id": "missing"},
            ],
            [
                {"source": "a", "target": "b", "keywords": "rel", "source_id": "chunk-1"},
                {"source": "a", "target": "a", "keywords": "loop", "source_id": "chunk-1"},
            ],
            {"chunk-1": "orig"},
            {"orig": {"chunk_id": "orig", "doc_id": "d1"}},
        )
        self.assertEqual([node["id"] for node in projected["nodes"]], ["a"])
        self.assertEqual(projected["edges"], [])

    def test_requested_hops_above_three_are_truncated(self) -> None:
        nodes = [{"id": str(index), "label": str(index), "type": "t", "sourceIds": ["c"]} for index in range(6)]
        edges = [
            {"id": f"e{index}", "source": str(index), "target": str(index + 1), "label": "rel", "sourceIds": ["c"]}
            for index in range(5)
        ]
        expanded = expand_hops({"nodes": nodes, "edges": edges}, ["0"], 4)
        self.assertTrue(expanded["truncated"])
        self.assertTrue(expanded["paths"])
        self.assertTrue(all(path["hops"] <= 3 for path in expanded["paths"]))

    def test_store_does_not_start_extraction(self) -> None:
        from knowledge.graph_rag import schedule_document_ingest

        schedule_document_ingest("note.txt", [{"chunk_id": "c1", "content": "一段真实原文"}])
        self.assertFalse((Path(self._tmp.name) / "lightrag").exists())
        status = get_service().status_payload()
        self.assertEqual(status["chunks"], 1)
        self.assertEqual(status["nodes"], 0)
        self.assertEqual(status["status"], "stored")
        self.assertFalse(status["building"])

    def test_delete_opens_persisted_lightrag_after_restart(self) -> None:
        from knowledge.graph_rag import runtime

        service = get_service()
        service.catalog.replace_chunks(
            "managed:fixture.txt",
            "fixture.txt",
            "",
            [_chunk("managed:fixture.txt", "managed:fixture.txt:c1", "disposable original paragraph")],
            "managed",
        )
        (Path(self._tmp.name) / "lightrag").mkdir()
        deleted = []

        async def fake_instance():
            class _Rag:
                async def adelete_by_doc_id(self, doc_id: str, delete_llm_cache: bool = False) -> None:
                    deleted.append(doc_id)

            return _Rag()

        runtime._rag = None
        with patch.object(runtime, "_rag_instance", fake_instance):
            service.delete_document("fixture.txt")
        self.assertIn("managed:fixture.txt:c1", deleted)
        self.assertEqual(service.status_payload()["chunks"], 0)

    def test_bibliography_pages_are_not_pilot_passages(self) -> None:
        self.assertTrue(_bibliography_unit({"section": "参考文献", "text": "Author. Title. 1998."}))
        self.assertFalse(_bibliography_unit({"section": "结冰", "text": "机翼表面结冰会改变升力。"}))

    def test_ollama_host_follows_the_normal_endpoint_unless_isolated(self) -> None:
        os.environ.pop("GRAPH_OLLAMA_HOST", None)
        os.environ["LLM_BASE_URL"] = "http://127.0.0.1:11434/v1"
        self.assertEqual(ollama_host(), "http://127.0.0.1:11434")
        os.environ["GRAPH_OLLAMA_HOST"] = "http://127.0.0.1:11435"
        self.assertEqual(ollama_host(), "http://127.0.0.1:11435")
        source = Path(__file__).resolve().parents[1].joinpath("knowledge/graph_rag/runtime.py").read_text(encoding="utf-8")
        self.assertNotIn("promote_structured_thinking", source)

    def test_multi_mode_does_not_call_the_old_retriever(self) -> None:
        import qa.entry as entry

        with patch("qa.entry.retrieve", side_effect=AssertionError("old retriever")):
            payload = entry._compose_structured_answer("结冰问题", engine=object(), retrieval_mode="multi")
        self.assertEqual(payload["stopReason"], "graph_unavailable")
        self.assertEqual(payload["abstainReason"], "graph_unavailable")
        self.assertEqual(payload["sources"], [])

    def test_auto_dispatch_uses_the_graph_agent_only_when_nodes_exist(self) -> None:
        import qa.entry as entry
        from knowledge.graph_rag.qa_bridge import prepare_graph_dispatch

        history = [
            {"role": "user", "content": "《结冰条件下的飞行》对失速速度有什么规定？"},
            {
                "role": "assistant",
                "content": "结冰会提高失速速度。",
                "payload": {
                    "sources": [
                        {
                            "doc_name": "caac-icing-sc25-040.pdf",
                            "section_title": "失速速度",
                            "chunk_id": "ice-1",
                        }
                    ]
                },
            },
        ]
        service = get_service()
        service.projection = {"nodes": [{"id": "n1", "label": "icing", "type": "condition", "sourceIds": ["c1"]}], "edges": []}
        seen = []

        def fake_graph(question: str, mode: str = "agent") -> dict:
            seen.append((question, mode))
            return {
                "answer": "图谱回答",
                "sources": [],
                "confidence": 0.0,
                "abstainReason": "",
                "evidenceIds": [],
                "queryType": "graph",
                "retrievalTrace": {"steps": [], "mode": mode, "planner": "langgraph", "reason": "answered"},
                "visualContext": {"used": False, "pages": [], "reason": ""},
                "stopReason": "answered",
                "graph": {},
                "graphTrace": [],
            }

        with patch("knowledge.graph_rag.qa_bridge.answer_with_graph", fake_graph), patch(
            "qa.entry.retrieve", side_effect=AssertionError("old retriever")
        ), patch("qa.entry.HybridSearch", side_effect=AssertionError("legacy engine")):
            entry._compose_structured_answer(
                "上述规定以及风切变有什么区别？",
                engine=None,
                history=history,
                retrieval_mode="auto",
            )
            for requested, expected in (("multi", "agent"), ("agent", "agent"), ("graph", "graph")):
                entry._compose_structured_answer("结冰以及风切变", engine=None, retrieval_mode=requested)
        self.assertEqual(seen[0][1], "agent")
        self.assertIn("caac-icing-sc25-040", seen[0][0])
        self.assertIn("失速速度", seen[0][0])
        self.assertIn("上述规定以及风切变有什么区别？", seen[0][0])
        self.assertEqual([mode for _question, mode in seen[1:]], ["agent", "agent", "graph"])

        contextual, mode, reason = prepare_graph_dispatch("上述规定以及风切变有什么区别？", history, "auto")
        self.assertEqual(mode, "agent")
        self.assertEqual(reason, "")
        self.assertIn("失速速度", contextual)
        plain, mode, reason = prepare_graph_dispatch("雷暴形成需要哪三个大气条件？", None, "auto")
        self.assertEqual((plain, mode, reason), ("雷暴形成需要哪三个大气条件？", None, ""))
        _query, mode, reason = prepare_graph_dispatch("比较雷暴以及风切变的差别", None, "single")
        self.assertEqual((mode, reason), (None, ""))
        self.assertEqual(prepare_graph_dispatch("比较雷暴以及风切变的差别", None, "graph")[1], "graph")
        self.assertEqual(prepare_graph_dispatch("比较雷暴以及风切变的差别", None, "multi")[1], "agent")

        service.projection = {"nodes": [], "edges": []}
        _query, mode, reason = prepare_graph_dispatch("比较雷暴以及风切变的差别", None, "auto")
        self.assertEqual((mode, reason), (None, "graph_not_built"))
        self.assertEqual(prepare_graph_dispatch("比较雷暴以及风切变的差别", None, "agent")[1], "agent")
        with patch("knowledge.graph_rag.qa_bridge.get_service", side_effect=RuntimeError("graph down")):
            _query, mode, reason = prepare_graph_dispatch("比较雷暴以及风切变的差别", None, "auto")
        self.assertEqual((mode, reason), (None, "graph_not_built"))

        endpoint = Path(__file__).resolve().parents[1].joinpath("api/aerospace_api.py").read_text(encoding="utf-8")
        self.assertIn("engine=None if graph_mode else get_engine()", endpoint)

    def test_auto_complex_without_graph_is_one_hybrid_search(self) -> None:
        import qa.entry as entry

        engine = _RecordingEngine()
        llm_systems = []

        def fake_llm(system: str, user: str) -> str:
            llm_systems.append(system)
            return "风切变和雷暴需要分别查看证据。"

        with patch("qa.entry.call_qwen", fake_llm), patch("qa.prompt.call_qwen", fake_llm), patch(
            "qa.entry.HybridSearch", side_effect=AssertionError("legacy engine")
        ), patch("knowledge.graph_rag.qa_bridge.answer_with_graph", side_effect=AssertionError("graph")):
            payload = entry.ask_structured("比较雷暴以及风切变的差别", engine=engine, retrieval_mode="auto")
        self.assertEqual(engine.queries, ["比较雷暴以及风切变的差别"])
        self.assertEqual(payload["retrievalTrace"]["mode"], "single")
        self.assertEqual(payload["retrievalTrace"]["planner"], "none")
        self.assertEqual(payload["retrievalTrace"]["reason"], "graph_not_built")
        self.assertEqual(len(payload["retrievalTrace"]["steps"]), 1)
        self.assertFalse(any("检索规划器" in system for system in llm_systems))

        constructed = []

        class _FakeHybrid:
            def __init__(self, *args, **kwargs):
                constructed.append(True)

            def __enter__(self):
                return engine

            def __exit__(self, exc_type, exc, tb):
                return False

        engine.queries.clear()
        with patch("qa.entry.call_qwen", fake_llm), patch("qa.prompt.call_qwen", fake_llm), patch(
            "qa.entry.HybridSearch", _FakeHybrid
        ), patch("knowledge.graph_rag.qa_bridge.answer_with_graph", side_effect=AssertionError("graph")):
            payload = entry.ask_structured("比较雷暴以及风切变的差别", engine=None, retrieval_mode="auto")
        self.assertEqual(constructed, [True])
        self.assertEqual(engine.queries, ["比较雷暴以及风切变的差别"])
        self.assertEqual(payload["retrievalTrace"]["reason"], "graph_not_built")
        self.assertEqual(payload["retrievalTrace"]["mode"], "single")

        with patch("qa.entry.retrieve", side_effect=RuntimeError("search down")):
            failed = entry._compose_structured_answer("比较雷暴以及风切变的差别", engine=object(), retrieval_mode="auto")
        self.assertEqual(failed["abstainReason"], "search_error")
        self.assertEqual(failed["retrievalTrace"]["reason"], "graph_not_built")
        self.assertEqual(failed["retrievalTrace"]["mode"], "single")
        self.assertEqual(failed["retrievalTrace"]["planner"], "none")

    def test_simple_auto_stays_on_single_retrieval_when_graph_exists(self) -> None:
        import qa.entry as entry

        service = get_service()
        service.projection = {"nodes": [{"id": "n1", "label": "storm", "type": "condition", "sourceIds": ["c1"]}], "edges": []}
        engine = _RecordingEngine()
        history = [
            {"role": "user", "content": "《结冰条件下的飞行》对失速速度有什么规定？"},
            {
                "role": "assistant",
                "content": "结冰会提高失速速度。",
                "payload": {"sources": [{"doc_name": "caac-icing-sc25-040.pdf", "section_title": "失速速度", "chunk_id": "ice-1"}]},
            },
        ]

        def fake_llm(system: str, user: str) -> str:
            return "三个条件见证据。"

        with patch("qa.entry.call_qwen", fake_llm), patch("qa.prompt.call_qwen", fake_llm), patch(
            "knowledge.graph_rag.qa_bridge.answer_with_graph", side_effect=AssertionError("graph")
        ):
            plain = entry.ask_structured("雷暴形成需要哪三个大气条件？", engine=engine, retrieval_mode="auto")
            follow = entry.ask_structured(
                "上述规定对起飞有什么要求？",
                engine=engine,
                history=history,
                retrieval_mode="auto",
            )
        self.assertEqual(engine.queries[0], "雷暴形成需要哪三个大气条件？")
        self.assertEqual(plain["retrievalTrace"]["mode"], "single")
        self.assertEqual(plain["retrievalTrace"]["planner"], "none")
        self.assertEqual(plain["retrievalTrace"]["reason"], "single")
        self.assertEqual(len(engine.queries), 2)
        self.assertIn("caac-icing-sc25-040", engine.queries[1])
        self.assertIn("失速速度", engine.queries[1])
        self.assertIn("上述规定对起飞有什么要求？", engine.queries[1])
        self.assertEqual(follow["retrievalTrace"]["reason"], "single")
        self.assertEqual(follow["retrievalTrace"]["mode"], "single")

    def test_persisted_managed_documents_register_without_reading_files(self) -> None:
        import builtins

        from knowledge.graph_rag import register_loaded_documents
        from knowledge.graph_rag.config import BACKEND_ROOT
        from knowledge.graph_rag.corpus import corpus_ids

        secret = "SECRET_GRAPH_BYTES_DO_NOT_READ"
        state_path = Path(self._tmp.name) / "library-state.json"
        state_payload = {
            "documents": [
                {
                    "fileName": "大型飞机运行规则",
                    "storageFileName": "graph-catalog-probe-not-a-real-upload.pdf",
                    "summary": secret,
                },
                {"fileName": "没有存储路径", "summary": secret},
                "not-a-record",
            ]
        }
        state_path.write_text(json.dumps(state_payload, ensure_ascii=False), encoding="utf-8")
        os.environ["GRAPH_MANAGEMENT_STATE"] = str(state_path)
        guarded_names = {"graph-catalog-probe-not-a-real-upload.pdf", "direct.pdf"}
        real_open = builtins.open

        def guarded_open(file, *args, **kwargs):
            target = file if isinstance(file, (str, Path)) else ""
            if target and Path(target).name in guarded_names:
                raise AssertionError(f"opened managed storage {target}")
            return real_open(file, *args, **kwargs)

        service = get_service()
        with patch("builtins.open", guarded_open), patch(
            "utils.state_store.load_state", side_effect=AssertionError("state store lock")
        ):
            added = register_loaded_documents(
                [
                    {"fileName": "直接登记", "storageFileName": "direct.pdf", "summary": secret},
                    {"fileName": "缺路径"},
                ]
            )
            rows = service.documents_payload()
        self.assertEqual(added, 1)
        self.assertFalse((Path(self._tmp.name) / "lightrag").exists())
        self.assertEqual(Path(service.catalog.path).parent, Path(self._tmp.name))
        catalog_text = service.catalog.path.read_text(encoding="utf-8")
        self.assertNotIn(secret, catalog_text)
        ids = {row["id"] for row in rows}
        self.assertIn("managed:graph-catalog-probe-not-a-real-upload.pdf", ids)
        self.assertIn("managed:direct.pdf", ids)
        self.assertTrue(set(corpus_ids()).issubset(ids))
        self.assertNotIn("managed:没有存储路径", ids)
        managed = service.catalog.documents["managed:graph-catalog-probe-not-a-real-upload.pdf"]
        self.assertEqual(managed["name"], "大型飞机运行规则")
        self.assertEqual(
            managed["storagePath"],
            str(BACKEND_ROOT / "aerospace_docs" / "graph-catalog-probe-not-a-real-upload.pdf"),
        )
        self.assertFalse(managed["indexed"])
        self.assertEqual(managed["availableChunks"], 0)
        self.assertEqual(managed["ingestedChunks"], 0)
        self.assertEqual(managed["extractedChunkIds"], [])
        self.assertFalse(any(chunk.get("doc_id") == managed["id"] for chunk in service.catalog.chunks.values()))
        listed = next(row for row in rows if row["id"] == managed["id"])
        self.assertEqual(listed["chunks"], 0)
        self.assertFalse(listed["indexed"])

        managed["indexed"] = True
        managed["ingestedChunks"] = 1
        managed["extractedChunkIds"] = ["c-keep"]
        service.catalog.chunks["c-keep"] = {
            "chunk_id": "c-keep",
            "doc_id": managed["id"],
            "content": "kept-passage",
        }
        state_payload["documents"][0]["fileName"] = "新名称"
        state_path.write_text(json.dumps(state_payload, ensure_ascii=False), encoding="utf-8")
        self.assertEqual(service.register_persisted_managed_documents(), 1)
        updated = service.catalog.documents[managed["id"]]
        self.assertEqual(updated["name"], "新名称")
        self.assertTrue(updated["indexed"])
        self.assertEqual(updated["extractedChunkIds"], ["c-keep"])
        self.assertEqual(service.catalog.chunks["c-keep"]["content"], "kept-passage")
        self.assertNotIn(secret, service.catalog.path.read_text(encoding="utf-8"))
        with patch.object(service.catalog, "save", wraps=service.catalog.save) as saved:
            self.assertEqual(service.register_persisted_managed_documents(), 0)
            again = service.documents_payload()
        self.assertEqual(saved.call_count, 0)
        shown = next(row for row in again if row["id"] == managed["id"])
        self.assertTrue(shown["indexed"])
        self.assertEqual(shown["chunks"], 1)
        self.assertEqual(state_path.read_text(encoding="utf-8"), json.dumps(state_payload, ensure_ascii=False))


class _RecordingEngine:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search_for_qa(self, query: str) -> list[dict]:
        self.queries.append(query)
        return [
            {
                "chunk_id": "c1",
                "content": "低空风切变出现在离地五百米以下。",
                "doc_name": "weather.pdf",
                "page_number": 3,
                "section_title": "风切变",
                "score": 0.8,
            }
        ]


if __name__ == "__main__":
    unittest.main()
