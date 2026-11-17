"""Window retrieval, source routing, and the retrieval-only evaluation interface.

Fixtures are synthetic. Gold questions are not read. Model calls are patched.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from knowledge.graph_rag.agent import budget_for
from knowledge.graph_rag.config import MAX_TOOL_CALLS, REPO_ROOT
from knowledge.graph_rag.corpus import choose_units, describe_coverage, unit_to_chunk
from knowledge.graph_rag.eval_retrieval import _merge_graph_candidates, load_queries, retrieval_budget, run_query, run_query_file
from knowledge.graph_rag.graph_ops import citations_from_answer
from knowledge.graph_rag.hybrid_index import question_facets, retrieve_spans, rrf_order, split_spans
from knowledge.graph_rag.router import EvidenceModel
from knowledge.graph_rag.service import _answer_from_evidence, _planner_model, _public_evidence, _tools, get_service, reset_service


def _chunk(doc_id: str, chunk_id: str, content: str, title: str = "", section: str = "body", language: str = "en") -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "content": content,
        "doc_name": title or doc_id,
        "page_number": 1,
        "page_count": 1,
        "section_title": section,
        "heading": section,
        "parent_heading": title or doc_id,
        "source_url": "",
        "source_type": "text",
        "highlight_text": "",
        "publisher": "",
        "language": language,
        "file_stem": doc_id,
    }


def _no_model(query, texts):
    return None


def _no_vectors(texts, query):
    return None


class WindowRetrievalTests(unittest.TestCase):
    def test_late_page_window_is_retrieved_exactly(self) -> None:
        prefix = "无关内容。" * 500
        content = prefix + "\n\nzqlatefact is the late clause."
        self.assertGreater(len(prefix), 2000)
        spans = split_spans(content)
        late = [item for item in spans if "zqlatefact" in content[item[0] : item[1]]]
        self.assertTrue(late)
        self.assertGreater(late[0][0], 2000)
        self.assertEqual(content[late[0][0] : late[0][1]], content[late[0][0] : late[0][1]])
        seen = []

        def encode(texts, query):
            seen.extend(texts)
            return None

        snapshot = {"chunks": {"p1": _chunk("d1", "p1", content, section="x")}}
        hits = retrieve_spans(snapshot, "zqlatefact", encode=encode, rerank=_no_model)
        self.assertTrue(any("zqlatefact" in text for text in seen))
        self.assertGreater(len(seen), 1)
        self.assertTrue(all(len(text) < 2000 for text in seen))
        self.assertNotIn(content[:2000], seen)
        self.assertEqual(hits[0]["unit_id"], "p1")
        self.assertGreater(hits[0]["start"], 2000)
        self.assertEqual(hits[0]["text"], content[hits[0]["start"] : hits[0]["end"]])
        self.assertAlmostEqual(hits[0]["score"], 1.0 / 61.0, places=6)

    def test_identifier_and_language_route_without_a_special_case(self) -> None:
        english = _chunk("en-doc", "en1", "general discussion of a load case", "Wing Load Notes", "intro", "en")
        chinese = _chunk("zh-doc", "zh1", "这里记录结冰试验条件。", "结冰试验手册", "概述", "zh")
        sec9 = _chunk("sec-doc", "s9", "limit value 0.014 applies in this section", "Limits", "SEC-9")
        sec1 = _chunk("sec-doc", "s1", "limit value 0.014 applies in the other section", "Limits", "SEC-1")
        snapshot = {"chunks": {"en1": english, "zh1": chinese, "s9": sec9, "s1": sec1}}
        english_hits = retrieve_spans(
            snapshot,
            "What does Wing Load Notes report for the load case",
            encode=_no_vectors,
            rerank=_no_model,
        )
        self.assertTrue(english_hits)
        self.assertTrue(all(hit["doc_id"] == "en-doc" for hit in english_hits))
        chinese_hits = retrieve_spans(snapshot, "结冰试验手册里的试验条件", encode=_no_vectors, rerank=_no_model)
        self.assertTrue(chinese_hits)
        self.assertTrue(all(hit["doc_id"] == "zh-doc" for hit in chinese_hits))
        filtered = retrieve_spans(
            snapshot,
            "结冰试验手册里的试验条件",
            ["en-doc"],
            encode=_no_vectors,
            rerank=_no_model,
        )
        self.assertTrue(all(hit["doc_id"] == "en-doc" for hit in filtered))
        self.assertFalse(any(hit["unit_id"] == "zh1" for hit in filtered))
        section_hits = retrieve_spans(snapshot, "SEC-9 limit value", encode=_no_vectors, rerank=_no_model)
        self.assertEqual(section_hits[0]["unit_id"], "s9")
        self.assertEqual(section_hits[0]["text"], sec9["content"])

    def test_table_and_formula_text_is_not_rewritten(self) -> None:
        content = "Heading row\n| flap | 12.50 | 0.014 |\nCL=1.25*alpha+0.7\n"
        snapshot = {"chunks": {"t1": _chunk("d1", "t1", content, section="x")}}
        hits = retrieve_spans(snapshot, "flap 12.50", encode=_no_vectors, rerank=_no_model)
        self.assertEqual(hits[0]["text"], content[hits[0]["start"] : hits[0]["end"]])
        self.assertIn("| flap | 12.50 | 0.014 |", hits[0]["text"])
        self.assertIn("CL=1.25*alpha+0.7", hits[0]["text"])
        self.assertNotIn("12.5|", hits[0]["text"])

    def test_rerank_order_determines_hits(self) -> None:
        fused = rrf_order([["A", "C", "B"], ["B", "C", "A"], ["A", "B", "C"]])
        self.assertEqual(fused[0][0], "A")
        snapshot = {
            "chunks": {
                "a": _chunk("d", "a", "alpha signal", section="x"),
                "b": _chunk("d", "b", "beta-only signal", section="x"),
                "c": _chunk("d", "c", "gamma signal", section="x"),
            }
        }
        calls = []

        def encode(texts, query):
            order = {"alpha signal": 0.1, "beta-only signal": 10.0, "gamma signal": 0.2}
            return [order[text] for text in texts]

        def rerank(query, texts):
            calls.append(list(texts))
            return [1000.0 if "beta-only" in text else 0.0 for text in texts]

        hits = retrieve_spans(snapshot, "alpha", encode=encode, rerank=rerank, rerank_pool=8)
        self.assertEqual(len(calls), 1)
        self.assertGreaterEqual(len(calls[0]), 2)
        self.assertTrue(all(len(text) < 500 for text in calls[0]))
        self.assertEqual(hits[0]["unit_id"], "b")
        self.assertEqual(hits[0]["ranks"].get("rerank"), 1)
        self.assertNotEqual(hits[0]["score"], 1000.0)
        self.assertLess(hits[0]["score"], 1.0)
        self.assertGreater(hits[0]["score"], hits[1]["score"])
        self.assertTrue(all(hit["score"] != 1000.0 for hit in hits))

    def test_facets_keep_both_windows_and_partial_status(self) -> None:
        gap = "pad token.\n" * 200
        content = "ZQALPHA is the first condition.\n" + gap + "ZQBETA is the second condition."
        self.assertGreater(len(content), 900)
        source = _chunk("d", "page", content, section="x")
        source["score"] = 0.4
        question = "ZQALPHA and ZQBETA"
        self.assertEqual(question_facets(question), ["ZQALPHA", "ZQBETA"])
        seen = []

        def fake_chat(prompt: str, **kwargs) -> str:
            seen.append(prompt)
            return "第一项见 [1]。"

        with patch("knowledge.graph_rag.runtime.ollama_chat", fake_chat):
            answered = _answer_from_evidence(
                {"question": question, "evidence": [source], "observations": [], "graph_parts": []}
            )
        self.assertIn("ZQALPHA", seen[0])
        self.assertIn("ZQBETA", seen[0])
        self.assertEqual(answered["sources"][0]["content"], content)
        self.assertEqual(answered["stop_reason"], "answered")

        only_alpha = dict(source)
        only_alpha["content"] = "ZQALPHA is the only stored condition."
        mixed = "ZQALPHA holds [1]. ZQBETA also holds."
        with patch("knowledge.graph_rag.runtime.ollama_chat", return_value=mixed):
            cited = _answer_from_evidence(
                {"question": question, "evidence": [only_alpha], "observations": [], "graph_parts": []}
            )
        # Contract change: a missing facet is not partial_evidence. Facet
        # overlap is a retrieval hint. Cited text on nonempty span text is
        # answered unless the answer itself abstains.
        self.assertEqual(cited["stop_reason"], "answered")
        self.assertEqual(cited["answer"], mixed)
        self.assertIn("[1]", cited["answer"])
        self.assertEqual(cited["sources"][0]["content"], only_alpha["content"])


class GraphSeedTests(unittest.TestCase):
    def _snapshot(self) -> dict:
        return {
            "chunks": {
                "seed": _chunk("d1", "seed", "zqseedfact is recorded in the english note", "English Notes", "body"),
                "inc": _chunk("d2", "inc", "结冰天气只在另一份资料里出现。", "另一份资料", "概述", "zh"),
            },
            "graph": {
                "nodes": [
                    {"id": "n-inc", "label": "结冰", "type": "condition", "sourceIds": ["inc"], "description": "结冰天气"},
                ],
                "edges": [],
            },
        }

    def test_hybrid_seed_survives_a_graph_miss(self) -> None:
        tools = _tools(self._snapshot())
        state = {
            "question": "zqseedfact",
            "doc_ids": None,
            "tool_calls": 0,
            "observations": [],
            "trace": [],
            "graph_parts": [],
            "evidence": [],
            "pending": {"action": "graph_search", "query": "zqseedfact", "nodeIds": [], "chunkIds": [], "hops": 1},
        }
        update = tools["graph_search"](state)
        observation = update["observations"][-1]
        self.assertEqual(observation["sourceIds"][0], "seed")
        self.assertNotIn("inc", observation["sourceIds"])
        self.assertNotIn("n-inc", observation["nodeIds"])

    def test_doc_filter_applies_to_every_tool(self) -> None:
        snapshot = {
            "chunks": {
                "c1": _chunk("d1", "c1", "alpha fact in the kept document", "Kept Notes Manual"),
                "c2": _chunk("d2", "c2", "beta fact in the other document", "Other Notes Manual"),
            },
            "graph": {
                "nodes": [
                    {"id": "n1", "label": "alpha", "type": "entity", "sourceIds": ["c1"], "description": "alpha"},
                    {"id": "n2", "label": "beta", "type": "entity", "sourceIds": ["c2"], "description": "beta"},
                ],
                "edges": [
                    {"id": "e1", "source": "n1", "target": "n2", "label": "related_to", "sourceIds": ["c1"]},
                ],
            },
        }
        tools = _tools(snapshot)
        base = {
            "question": "Other Notes Manual beta",
            "doc_ids": ["d1"],
            "tool_calls": 0,
            "observations": [],
            "trace": [],
            "graph_parts": [],
            "evidence": [],
        }
        search = tools["graph_search"](
            {**base, "pending": {"action": "graph_search", "query": "Other Notes Manual beta", "nodeIds": [], "chunkIds": [], "hops": 1}}
        )
        self._assert_only_d1(snapshot, search)
        expand_state = {**base, "observations": search["observations"], "evidence": search.get("evidence") or []}
        expand = tools["graph_expand"](
            {**expand_state, "pending": {"action": "graph_expand", "query": "Other Notes Manual beta", "nodeIds": ["n1"], "chunkIds": [], "hops": 2}}
        )
        self._assert_only_d1(snapshot, expand)
        retrieve = tools["document_retrieve"](
            {**base, "pending": {"action": "document_retrieve", "query": "Other Notes Manual beta", "nodeIds": [], "chunkIds": [], "hops": 1}}
        )
        self._assert_only_d1(snapshot, retrieve)
        inspect = tools["inspect_evidence"](
            {**base, "pending": {"action": "inspect_evidence", "query": "beta", "nodeIds": [], "chunkIds": ["c1", "c2"], "hops": 1}}
        )
        self._assert_only_d1(snapshot, inspect)
        self.assertEqual(inspect["observations"][-1]["sourceIds"], ["c1"])

    def _assert_only_d1(self, snapshot: dict, update: dict) -> None:
        observation = update["observations"][-1]
        for source_id in observation["sourceIds"]:
            self.assertEqual(snapshot["chunks"][source_id]["doc_id"], "d1")
        for row in update.get("evidence") or []:
            self.assertNotIn("beta fact", row.get("content") or "")

    def test_bridge_replaces_only_a_redundant_seed(self) -> None:
        seeds = [
            {"unit_id": "a1", "start": 0, "end": 8, "text": "ZQALPHA", "score": 0.2},
            {"unit_id": "a2", "start": 0, "end": 8, "text": "ZQALPHA", "score": 0.1},
        ]
        extras = [
            {"unit_id": "noise", "start": 0, "end": 4, "text": "pump", "score": 9.0},
            {"unit_id": "b1", "start": 0, "end": 6, "text": "ZQBETA", "score": 0.01},
        ]
        merged = _merge_graph_candidates(seeds, extras, "ZQALPHA and ZQBETA", 2)
        self.assertEqual([row["unit_id"] for row in merged], ["a1", "b1"])
        kept = _merge_graph_candidates(seeds[:1], extras, "ZQALPHA and ZQBETA", 1)
        self.assertEqual(kept[0]["unit_id"], "a1")

    def test_planner_sees_metadata_and_the_missing_aspect(self) -> None:
        snapshot = {
            "chunks": {"a1": _chunk("d1", "a1", "ZQALPHA is stored here", "Alpha Manual", language="en")},
            "graph": {"nodes": [], "edges": []},
        }
        state = {
            "question": "ZQALPHA and ZQBETA",
            "doc_ids": None,
            "tool_calls": 0,
            "observations": [],
            "trace": [],
            "graph_parts": [],
            "evidence": [],
            "pending": {"action": "document_retrieve", "query": "ZQALPHA", "nodeIds": [], "chunkIds": [], "hops": 1},
        }
        state.update(_tools(snapshot)["document_retrieve"](state))
        documents = state["observations"][-1]["documents"]
        self.assertEqual(documents[0]["title"], "Alpha Manual")
        self.assertEqual(documents[0]["language"], "en")
        self.assertIn("ZQBETA", state["observations"][-1]["missingAspects"])
        captured = []

        def fake_chat(prompt: str, **kwargs) -> str:
            captured.append(prompt)
            return "{}"

        with patch("knowledge.graph_rag.runtime.ollama_chat", fake_chat):
            _planner_model(state)
        prompt = captured[0]
        self.assertIn("Alpha Manual", prompt)
        self.assertIn("ZQBETA", prompt)
        self.assertIn("Missing aspects", prompt)
        self.assertIn("every part of the question", prompt)
        self.assertIn("Query a missing part before finish", prompt)
        self.assertEqual(budget_for({"question": "icing wing"}), MAX_TOOL_CALLS)
        self.assertEqual(budget_for({"question": "icing and wind shear"}), 5)


class EvaluationInterfaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._vectors = os.environ.get("GRAPH_DISABLE_VECTORS")
        os.environ["GRAPH_DISABLE_VECTORS"] = "1"

    def tearDown(self) -> None:
        if self._vectors is None:
            os.environ.pop("GRAPH_DISABLE_VECTORS", None)
        else:
            os.environ["GRAPH_DISABLE_VECTORS"] = self._vectors

    def _snapshot(self) -> dict:
        return {
            "chunks": {
                "seed": _chunk("d1", "seed", "zqseedfact is recorded in the english note", "English Notes"),
                "inc": _chunk("d2", "inc", "结冰天气只在另一份资料里出现。", "另一份资料", language="zh"),
            },
            "graph": {
                "nodes": [
                    {"id": "n-inc", "label": "结冰", "type": "condition", "sourceIds": ["inc"], "description": "结冰天气"},
                ],
                "edges": [],
            },
            "passage_ids": [],
            "passage_matrix": None,
            "entity_ids": [],
            "entity_matrix": None,
        }

    def test_retrieval_modes_return_real_spans_without_an_answer_call(self) -> None:
        def boom(*args, **kwargs):
            raise AssertionError("answer model was called")

        query = {"query_id": "q-seed", "query": "zqseedfact", "gold": ["hidden"], "evidence": ["nope"]}
        with patch("knowledge.graph_rag.runtime.ollama_chat", boom):
            hybrid = run_query(query, "hybrid", self._snapshot(), candidates=4, context_chars=1200, passage_chars=300, rerank_pool=4)
            graph = run_query(query, "hybrid_graph", self._snapshot(), candidates=4, context_chars=1200, passage_chars=300, rerank_pool=4)
            baseline = run_query(query, "baseline", self._snapshot(), candidates=4, context_chars=1200, passage_chars=300, rerank_pool=4)
        for result in (hybrid, graph, baseline):
            self.assertFalse(result["generated"])
            self.assertIsNone(result["answer"])
            self.assertGreaterEqual(result["timings"]["seconds"], 0.0)
            self.assertNotIn("error", result)
            blob = json.dumps(result)
            for banned in ("recall", "precision", "ndcg", "mrr"):
                self.assertNotIn(banned, blob.lower())
            self.assertTrue(result["candidates"])
            content = self._snapshot()["chunks"][result["candidates"][0]["unit_id"]]["content"]
            top = result["candidates"][0]
            self.assertEqual(top["text"], content[top["start"] : top["end"]])
            self.assertIn("zqseedfact", top["text"])
        self.assertEqual(hybrid["budgets"], graph["budgets"])
        self.assertEqual(baseline["budgets"]["candidates"], hybrid["budgets"]["candidates"])
        self.assertEqual(baseline["budgets"]["context_chars"], hybrid["budgets"]["context_chars"])
        self.assertEqual(hybrid["candidates"][0]["unit_id"], "seed")
        self.assertEqual(graph["candidates"][0]["unit_id"], "seed")
        hybrid_top = hybrid["candidates"][0]
        self.assertEqual(hybrid_top["ranks"], {"lexical": 1, "alias": 1})
        self.assertAlmostEqual(hybrid_top["score"], 2.0 / 61.0, places=6)
        self.assertGreater(baseline["candidates"][0]["score"], 0.5)
        frozen = (REPO_ROOT / "work" / "eval20" / "baseline-code" / "BackEnd" / "knowledge" / "graph_rag" / "service.py").read_text(encoding="utf-8")
        self.assertIn("[:2000]", frozen)
        self.assertIn("* 0.45 + score * 0.55", frozen)

    def test_load_queries_drops_gold_and_file_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "queries.jsonl"
            path.write_text(
                json.dumps(
                    {"query_id": "q1", "query": "zqseedfact", "doc_ids": ["d1"], "gold": ["secret"], "evidence": ["x"]}
                )
                + "\n",
                encoding="utf-8",
            )
            loaded = load_queries(path)
            self.assertEqual(loaded, [{"query_id": "q1", "query": "zqseedfact", "doc_ids": ["d1"]}])
            output = Path(tmp) / "hits.jsonl"
            rows = run_query_file(path, "hybrid", output, snapshot=self._snapshot(), candidates=4, context_chars=800, passage_chars=200, rerank_pool=4)
            self.assertEqual(len(rows), 1)
            saved = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(saved["candidates"][0]["unit_id"], "seed")
            self.assertIsNone(saved["answer"])
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.jsonl"
            bad.write_text(json.dumps({"query_id": "q", "gold": ["x"]}) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_queries(bad)
        multi = retrieval_budget("ZQALPHA and ZQBETA")
        self.assertEqual(multi["tool_calls"], 5)
        self.assertEqual(multi["tool_call_base"], 4)
        self.assertIn("5", multi["note"])

    def test_agent_mode_is_the_path_that_calls_the_model(self) -> None:
        calls = []

        def fake(prompt: str, **kwargs) -> str:
            calls.append(prompt)
            if prompt.startswith("Choose the next"):
                if len(calls) == 1:
                    return '{"action":"graph_search","query":"zqseedfact","nodeIds":[],"chunkIds":[],"hops":1}'
                return '{"action":"finish","query":"","nodeIds":[],"chunkIds":[],"hops":1}'
            return "见 [1]。"

        with patch("knowledge.graph_rag.runtime.ollama_chat", fake):
            result = run_query({"query_id": "q", "query": "zqseedfact"}, "agent", self._snapshot())
        self.assertTrue(calls)
        self.assertTrue(result["generated"])
        self.assertTrue(result["answer"])


class ContractAndCoverageTests(unittest.TestCase):
    def test_public_evidence_and_citations_keep_the_old_shape(self) -> None:
        row = _public_evidence(
            {
                "chunk_id": "c1",
                "content": "| flap | 12.50 |",
                "doc_name": "notes",
                "page_number": 3,
                "section_title": "SEC-9",
                "heading": "SEC-9",
                "parent_heading": "notes",
                "score": 0.2,
                "page_count": 4,
                "highlight_text": "12.50",
                "source_type": "table",
                "source_url": "https://example.test/notes",
            }
        )
        parsed = EvidenceModel(**row)
        self.assertEqual(parsed.chunk_id, "c1")
        self.assertEqual(parsed.content, "| flap | 12.50 |")
        self.assertEqual(parsed.source_type, "table")
        self.assertEqual(
            citations_from_answer("见 [1]。", [{"chunk_id": "c1"}]),
            [{"id": 1, "sourceIndex": 0, "chunkId": "c1"}],
        )

    def test_scope_all_keeps_every_unit_and_pilot_samples_only_long_documents(self) -> None:
        rows = [{"unit_id": str(index), "text": "x" * 130, "page": index, "section": "body"} for index in range(20)]
        self.assertEqual(len(choose_units(rows, "all")), 20)
        self.assertEqual(len(choose_units(rows, "pilot")), 12)
        self.assertEqual(len(choose_units(rows[:6], "pilot")), 6)
        bibliography = {"unit_id": "bib", "text": "参考文献\nAuthor. Title.", "page": 0, "section": "body"}
        self.assertEqual(len(choose_units([bibliography] + rows, "all")), 21)
        self.assertNotIn("bib", [unit["unit_id"] for unit in choose_units([bibliography] + rows, "pilot")])
        full = describe_coverage("all", 20, 20, 20)
        self.assertTrue(full["fullGraph"])
        self.assertEqual(full["label"], "full")
        partial = describe_coverage("pilot", 20, 12, 12)
        self.assertFalse(partial["fullGraph"])
        self.assertIn("not a full knowledge graph", partial["note"])
        chunk = unit_to_chunk(
            {
                "unit_id": "u-table",
                "doc_id": "not-a-catalog-document",
                "page": 4,
                "section": "SEC-9",
                "citation_url": "https://example.test/table",
                "text": "| flap | 12.50 | 0.014 |\nCL=1.25*alpha+0.7",
                "source_type": "table",
            }
        )
        self.assertEqual(chunk["source_type"], "table")
        self.assertEqual(chunk["content"], "| flap | 12.50 | 0.014 |\nCL=1.25*alpha+0.7")
        self.assertEqual(chunk["chunk_id"], "u-table")
        self.assertEqual(chunk["section_title"], "SEC-9")

    def test_window_index_does_not_rewrite_the_graph_and_delete_clears_it(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        previous = os.environ.get("GRAPH_DATA_DIR")
        vectors = os.environ.get("GRAPH_DISABLE_VECTORS")
        os.environ["GRAPH_DATA_DIR"] = tmp.name
        os.environ["GRAPH_DISABLE_VECTORS"] = "1"
        reset_service()
        try:
            service = get_service()
            content = ("pad.\n" * 800) + "\n\nzqlatefact tail"
            service.catalog.replace_chunks("d1", "notes", "", [_chunk("d1", "c1", content)], "managed")
            graph_dir = Path(tmp.name) / "lightrag"
            graph_dir.mkdir()
            graphml = graph_dir / "graph_chunk_entity_relation.graphml"
            graphml.write_text("<graphml/>", encoding="utf-8")
            os.environ["GRAPH_DISABLE_VECTORS"] = "0"
            captured = []

            def encode(texts):
                captured.append(list(texts))
                return [[0.1, 0.2] for _text in texts]

            with patch("knowledge.graph_rag.runtime.encode_texts", encode):
                service._build_window_index()
            self.assertEqual(graphml.read_text(encoding="utf-8"), "<graphml/>")
            self.assertFalse((Path(tmp.name) / "vectors.npz").exists())
            self.assertTrue((Path(tmp.name) / "window_vectors.npz").exists())
            self.assertTrue(any("zqlatefact" in text for text in captured[0]))
            self.assertGreater(len(captured[0]), 1)
            self.assertTrue(all(len(text) < 2000 for text in captured[0]))
            service.window_ids = ["stale"]
            service.window_matrix = [[1.0]]
            service.window_fingerprint = "stale"
            os.environ["GRAPH_DISABLE_VECTORS"] = "1"
            service.delete_document("notes")
            self.assertEqual(service.window_ids, [])
            self.assertIsNone(service.window_matrix)
            self.assertEqual(service.status_payload()["chunks"], 0)
        finally:
            reset_service()
            if previous is None:
                os.environ.pop("GRAPH_DATA_DIR", None)
            else:
                os.environ["GRAPH_DATA_DIR"] = previous
            if vectors is None:
                os.environ.pop("GRAPH_DISABLE_VECTORS", None)
            else:
                os.environ["GRAPH_DISABLE_VECTORS"] = vectors
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
