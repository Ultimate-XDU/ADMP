"""Focused tests for retrieval mode, history resolution, and citation links."""

import unittest

from qa.entry import _compose_structured_answer
from qa.prompt import _build_contextual_search_query, _is_followup_query
from qa.response import attach_retrieval_fields, citations_from_answer
from qa.vision import visual_context_for_sources
from search.hybrid_search import HybridSearch
from search.query_analysis import _is_valid_search_query
from search.retrieval_coordinator import evidence_follow_query, parse_planner_output, retrieve, resolve_retrieval_mode
from utils.query_utils import classify_query_intent
from utils.retrieval_contract import build_retrieval_contract


def _write_two_page_pdf(directory):
    import fitz
    from pathlib import Path

    path = Path(directory) / "two-pages.pdf"
    document = fitz.open()
    try:
        document.new_page().insert_text((72, 72), "PAGE-ONE-MARKER")
        document.new_page().insert_text((72, 72), "PAGE-TWO-MARKER")
        document.save(path)
    finally:
        document.close()
    return path


def _visual_page_sources():
    return [
        {"doc_name": "two-pages.pdf", "page_number": 1, "chunk_id": "a", "content": "chunk-page1-a"},
        {"doc_name": "two-pages.pdf", "page_number": 1, "chunk_id": "b", "content": "chunk-page1-b"},
        {"doc_name": "notes.docx", "page_number": 1, "chunk_id": "d", "content": "chunk-docx"},
        {"doc_name": "two-pages.pdf", "page_number": 2, "chunk_id": "c", "content": "chunk-page2"},
        {"doc_name": "two-pages.pdf", "page_number": 3, "chunk_id": "e", "content": "chunk-page3-not-attached"},
    ]


class RecordingEngine:
    def __init__(self, batches):
        self.batches = list(batches)
        self.queries = []

    def search_for_qa(self, query, structural_depth=1, semantic_extra=0):
        self.queries.append(query)
        if not self.batches:
            return []
        return self.batches.pop(0)


class RetrievalApiTests(unittest.TestCase):
    def test_citations_map_only_markers_that_exist(self):
        sources = [
            {"chunk_id": "chunk-a", "doc_name": "a.pdf"},
            {"chunk_id": "chunk-b", "doc_name": "b.pdf"},
        ]
        citations = citations_from_answer("结论见 [2]，重复 [2]，越界 [9]。", sources)
        self.assertEqual(
            citations,
            [{"id": 2, "sourceIndex": 1, "chunkId": "chunk-b"}],
        )
        self.assertEqual(citations_from_answer("没有编号。", sources), [])

    def test_saved_history_keeps_empty_trace_and_follows_source_order(self):
        from api.aerospace_api import QARequest, QAResponse, _normalize_cached_payload_sources

        request = QARequest(question="雷暴形成需要什么条件？")
        self.assertEqual(request.retrievalMode, "auto")
        response = QAResponse(
            answer="三个条件 [1]。",
            sources=[{"content": "水汽", "doc_name": "a.pdf", "heading": "", "parent_heading": "", "score": 0.2, "chunk_id": "a"}],
            citations=[{"id": 1, "sourceIndex": 0, "chunkId": "a"}],
            retrievalTrace={"steps": [{"hop": 1, "query": "雷暴", "resultCount": 1}], "mode": "single"},
        )
        self.assertEqual(response.citations[0].sourceIndex, 0)

        old = _normalize_cached_payload_sources("旧回答", {"answer": "旧回答", "sources": [{"chunk_id": "old"}]})
        self.assertEqual(old["retrievalTrace"], {"steps": [], "mode": ""})
        self.assertEqual(old["citations"], [])
        self.assertNotIn("visualContext", old)

        reordered = _normalize_cached_payload_sources(
            "见 cited.pdf",
            {
                "sources": [
                    {"doc_name": "other.pdf", "chunk_id": "o"},
                    {"doc_name": "cited.pdf", "chunk_id": "c"},
                ],
                "citations": [{"id": 2, "sourceIndex": 1, "chunkId": "c"}],
                "retrievalTrace": {"steps": [{"hop": 1, "query": "q", "resultCount": 2}], "mode": "single"},
            },
        )
        self.assertEqual(reordered["sources"][0]["chunk_id"], "c")
        self.assertEqual(reordered["citations"], [{"id": 2, "sourceIndex": 0, "chunkId": "c"}])
        self.assertEqual(reordered["retrievalTrace"]["steps"][0]["hop"], 1)
        self.assertNotIn("visualContext", reordered)

        with_pages = _normalize_cached_payload_sources(
            "见 cited.pdf",
            {
                "sources": [
                    {"doc_name": "other.pdf", "chunk_id": "o"},
                    {"doc_name": "cited.pdf", "chunk_id": "c"},
                ],
                "citations": [{"id": 2, "sourceIndex": 1, "chunkId": "c"}],
                "visualContext": {
                    "used": True,
                    "pages": [
                        {"sourceIndex": 1, "page": 4},
                        {"sourceIndex": 0, "page": 2},
                    ],
                    "reason": "",
                },
                "retrievalTrace": {"steps": [{"hop": 1, "query": "q", "resultCount": 2}], "mode": "single"},
            },
        )
        self.assertEqual(with_pages["sources"][0]["chunk_id"], "c")
        self.assertEqual(with_pages["citations"], [{"id": 2, "sourceIndex": 0, "chunkId": "c"}])
        self.assertEqual(
            with_pages["visualContext"]["pages"],
            [{"sourceIndex": 0, "page": 4}, {"sourceIndex": 1, "page": 2}],
        )
        self.assertEqual(with_pages["visualContext"]["used"], True)
        self.assertEqual(with_pages["visualContext"]["reason"], "")

    def test_old_payload_does_not_invent_trace_or_citations(self):
        payload = attach_retrieval_fields(
            {
                "answer": "旧回答没有引用标记。",
                "sources": [{"chunk_id": "old", "doc_name": "old.pdf"}],
                "abstainReason": "",
            }
        )
        self.assertEqual(payload["retrievalTrace"], {"steps": [], "mode": ""})
        self.assertEqual(payload["citations"], [])

    def test_abstain_drops_citations(self):
        payload = attach_retrieval_fields(
            {
                "answer": "当前证据不足 [1]",
                "sources": [{"chunk_id": "c1"}],
                "abstainReason": "low_confidence",
                "retrievalTrace": {"steps": [{"hop": 1, "query": "q", "resultCount": 1}], "mode": "single"},
            }
        )
        self.assertEqual(payload["citations"], [])
        self.assertEqual(payload["retrievalTrace"]["mode"], "single")

    def test_followup_uses_prior_assistant_evidence(self):
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
        resolved = _build_contextual_search_query("上述规定对起飞有什么要求？", history)
        self.assertIn("caac-icing-sc25-040", resolved)
        self.assertIn("失速速度", resolved)
        self.assertIn("上述规定对起飞有什么要求？", resolved)
        self.assertLess(resolved.index("失速速度"), resolved.index("上述规定"))

        engine = RecordingEngine([[]])
        _compose_structured_answer(
            "上述规定对起飞有什么要求？",
            engine=engine,
            history=history,
            retrieval_mode="single",
        )
        self.assertTrue(engine.queries)
        self.assertIn("失速速度", engine.queries[0])
        self.assertEqual(len(engine.queries), 1)

    def test_plain_question_is_unchanged(self):
        question = "雷暴形成需要哪三个大气条件？"
        self.assertEqual(_build_contextual_search_query(question, None), question)
        self.assertEqual(resolve_retrieval_mode(question, "auto"), "single")
        self.assertEqual(resolve_retrieval_mode(question, "multi"), "multi")
        self.assertEqual(resolve_retrieval_mode("比较雷暴以及风切变的差别", "auto"), "multi")

    def test_second_hop_is_conditioned_on_first_hop_evidence(self):
        hop1 = [
            {
                "chunk_id": "w1",
                "doc_name": "caac-weather-guide.pdf",
                "section_title": "低空风切变",
                "content": "低空风切变出现在离地 500 米以下。",
                "page_number": 12,
            }
        ]
        hop2 = [
            {
                "chunk_id": "w2",
                "doc_name": "caac-weather-guide.pdf",
                "section_title": "起飞",
                "content": "起飞阶段遭遇风切变应中止起飞。",
                "page_number": 40,
            }
        ]
        question = "它对起飞有什么影响？"
        self.assertIn("低空风切变", evidence_follow_query(question, hop1))

        def planner(original, rows):
            self.assertEqual(original, question)
            self.assertEqual(rows[0]["chunk_id"], "w1")
            return "低空风切变导致的起飞中断条件", ["w1"]

        engine = RecordingEngine([hop1, hop2])
        results, trace = retrieve(engine, question, mode="multi", planner=planner)
        self.assertEqual(len(engine.queries), 2)
        self.assertEqual(engine.queries[0], question)
        self.assertEqual(engine.queries[1], "低空风切变导致的起飞中断条件")
        self.assertEqual(trace["planner"], "llm")
        self.assertEqual(trace["steps"][1]["basisChunkIds"], ["w1"])
        self.assertEqual(trace["steps"][1]["resultCount"], 1)
        self.assertIn("w2", {row["chunk_id"] for row in results})

        single = RecordingEngine([hop1, hop2])
        _, single_trace = retrieve(single, question, mode="single")
        self.assertEqual(len(single.queries), 1)
        self.assertEqual(single_trace["planner"], "none")

        def timed_out(original, rows):
            raise TimeoutError("timed out")

        fallback = RecordingEngine([hop1, hop2])
        _, fallback_trace = retrieve(fallback, question, mode="multi", planner=timed_out)
        self.assertEqual(fallback_trace["planner"], "heuristic_fallback")
        self.assertEqual(fallback_trace["reason"], "timeout")
        self.assertIn("低空风切变", fallback.queries[1])
        self.assertEqual(fallback_trace["steps"][1]["basisChunkIds"], ["w1"])

    def test_no_second_hop_without_new_evidence_terms(self):
        question = "低空风切变对起飞有什么影响？"
        rows = [{"chunk_id": "w1", "section_title": "低空风切变", "content": "低空风切变", "page_number": 1}]
        self.assertEqual(evidence_follow_query(question, rows), "")
        engine = RecordingEngine([rows, [{"chunk_id": "other", "section_title": "别的", "content": "x"}]])
        _, trace = retrieve(engine, question, mode="multi", planner=lambda original, evidence: None)
        self.assertEqual(len(engine.queries), 1)
        self.assertEqual(len(trace["steps"]), 1)
        self.assertEqual(trace["planner"], "heuristic_fallback")
        self.assertEqual(trace["reason"], "no_useful_subquery")

        called = []

        def unused(original, evidence):
            called.append(original)
            return "should-not-run", ["w1"]

        empty = RecordingEngine([[]])
        _, empty_trace = retrieve(empty, "比较雷暴以及风切变", mode="multi", planner=unused)
        self.assertEqual(called, [])
        self.assertEqual(empty_trace["planner"], "none")
        self.assertEqual(empty_trace["reason"], "empty_evidence")

    def test_rerank_order_ignores_logit_magnitude(self):
        content = "结冰条件下起飞需要额外速度余度，机组应保持既定程序并监控失速裕度。"
        metadata = {
            "content": content,
            "retrieval_text": content,
            "page_number": 12,
            "page_count": 30,
            "source_type": "text",
            "doc_name": "icing.pdf",
            "parse_confidence": 0.8,
            "title_body_consistency": 0.6,
        }
        profile = {
            "kw_weight": 1.0,
            "vec_weight": 1.0,
            "structure_weight": 1.0,
            "article_weight": 1.0,
            "heading_weight": 1.0,
            "table_weight": 1.0,
            "rrf_k": 60,
            "query_type": "general",
            "query_signature": {},
            "target_doc_name": "",
            "target_roles": [],
        }
        better_rank = {
            "rrf_score": 0.02,
            "rerank_score": 0.1,
            "rerank_rank": 0,
            "rerank_window": 2,
            "channels": [],
            "content": content,
        }
        worse_rank = {
            "rrf_score": 0.02,
            "rerank_score": 12.0,
            "rerank_rank": 1,
            "rerank_window": 2,
            "channels": [],
            "content": content,
        }
        engine = object.__new__(HybridSearch)
        better = HybridSearch._score_candidate(engine, "结冰起飞", better_rank, metadata, profile)
        worse = HybridSearch._score_candidate(engine, "结冰起飞", worse_rank, metadata, profile)
        self.assertGreater(better, worse)

    def test_english_and_chinese_questions_pass_the_entry_gate(self):
        english = "What three atmospheric ingredients does the CAAC pilot weather guide require for a thunderstorm to form?"
        self.assertTrue(_is_valid_search_query(english))
        self.assertTrue(_is_valid_search_query("雷暴形成需要哪三个条件"))
        self.assertFalse(_is_valid_search_query("??"))
        self.assertFalse(_is_valid_search_query("12"))
        self.assertFalse(_is_valid_search_query(" "))

    def test_current_article_wins_over_history_and_pronoun_resolves(self):
        history = [
            {"role": "user", "content": "CCAR-25第25.101条规定了什么？"},
            {
                "role": "assistant",
                "content": "上一轮回答。",
                "payload": {"sources": [{"doc_name": "CCAR-25.pdf", "section_title": "第25.101条", "chunk_id": "old"}]},
            },
        ]
        resolved = _build_contextual_search_query("继续解释第25.103条。", history)
        self.assertIn("CCAR-25", resolved)
        self.assertIn("25.103", resolved)
        self.assertNotIn("25.101", resolved)
        self.assertNotIn("规定了什么", resolved)
        contract = build_retrieval_contract(resolved)
        self.assertEqual(contract.article_id, "25.103")

        self.assertTrue(_is_followup_query("它是什么？"))
        icing = [
            {"role": "user", "content": "什么是结冰？"},
            {
                "role": "assistant",
                "content": "结冰是一种现象。",
                "payload": {"sources": [{"doc_name": "caac-icing.pdf", "section_title": "结冰", "chunk_id": "ice"}]},
            },
        ]
        pronoun = _build_contextual_search_query("它是什么？", icing)
        self.assertIn("结冰", pronoun)
        self.assertIn("caac-icing", pronoun)
        self.assertNotIn("什么是结冰", pronoun)

    def test_instrument_word_is_not_a_table_request(self):
        self.assertNotEqual(classify_query_intent("压力仪表受结冰堵塞影响时有哪些现象？").intent, "table")
        self.assertEqual(classify_query_intent("请查看 October 2014 告警统计表").intent, "table")

    def test_hybrid_orders_by_reranker_and_keeps_cross_lingual_hit(self):
        class Vec:
            def tolist(self):
                return [0.0, 0.2]

        class Embedder:
            def encode(self, texts, normalize_embeddings=True):
                return [Vec()]

        class Reranker:
            def predict(self, pairs, batch_size=16):
                self.pairs = pairs
                return [0.2, 4.5]

        engine = object.__new__(HybridSearch)
        engine.embedder = Embedder()
        engine.reranker = Reranker()
        engine._keyword_search = lambda query, profile, top_k=80: [
            {"chunk_id": "keyword-hit", "content": "中文整句", "score": 3.0}
        ]
        engine._vector_search = lambda vec, top_k=80: [
            {"chunk_id": "nasa-en", "content": "flap extension increases tailplane stall risk with ice", "score": 0.4}
        ]
        engine._get_metadata_batch = lambda ids: {
            chunk_id: {"content": chunk_id, "doc_name": "nasa.pdf", "page_number": 2, "page_count": 8}
            for chunk_id in ids
        }

        question = "为什么放襟翼会增加积冰尾翼失速风险？"
        rows = HybridSearch.search_hybrid(engine, question)
        self.assertEqual(rows[0]["chunk_id"], "nasa-en")
        self.assertGreater(rows[0]["rerank_score"], rows[1]["rerank_score"])
        self.assertIn("nasa-en", {row["chunk_id"] for row in rows})

        engine.reranker = None
        engine._keyword_search = lambda query, profile, top_k=80: []
        kept = HybridSearch.search_hybrid(engine, question)
        self.assertEqual(kept[0]["chunk_id"], "nasa-en")
        self.assertEqual(kept[0]["content"], "nasa-en")

    def test_planner_output_is_validated_once(self):
        parsed = parse_planner_output(
            '说明\n{"query":"低空风切变对起飞的影响","basisChunkIds":["w1","missing"]}',
            ["w1"],
        )
        self.assertEqual(parsed, ("低空风切变对起飞的影响", ["w1"]))
        self.assertIsNone(parse_planner_output("不是 JSON", ["w1"]))
        self.assertEqual(parse_planner_output('{"query":"","basisChunkIds":[]}', ["w1"]), ("", []))

    def test_visual_failure_reports_the_error(self):
        import qa.vision as vision

        failed = visual_context_for_sources("图里是什么？", [{"doc_name": "missing-page.pdf", "page_number": 1, "chunk_id": "p"}])
        self.assertFalse(failed["used"])
        self.assertIn("FileNotFoundError", failed["reason"])

        original = vision.render_pdf_page
        vision.render_pdf_page = lambda doc_name, page: b"png"
        try:
            def complete(prompt, images):
                self.assertIn("[1]", prompt)
                self.assertEqual(images, [b"png"])
                return "页面显示三个条件 [1]。"

            seen = visual_context_for_sources(
                "图里是什么？",
                [{"doc_name": "guide.pdf", "page_number": 4, "chunk_id": "p"}],
                complete=complete,
            )
            self.assertTrue(seen["used"])
            self.assertEqual(seen["pages"], [{"sourceIndex": 0, "page": 4}])
            self.assertIn("[1]", seen["answer"])

            broken = visual_context_for_sources(
                "图里是什么？",
                [{"doc_name": "guide.pdf", "page_number": 4, "chunk_id": "p"}],
                complete=lambda prompt, images: (_ for _ in ()).throw(TimeoutError("timed out")),
            )
            self.assertFalse(broken["used"])
            self.assertIn("TimeoutError", broken["reason"])
            self.assertIn("timed out", broken["reason"])
        finally:
            vision.render_pdf_page = original

    def test_distinct_pdf_pages_keep_first_source_index(self):
        import tempfile
        from pathlib import Path

        import qa.vision as vision

        with tempfile.TemporaryDirectory() as tmp:
            _write_two_page_pdf(tmp)
            original = vision.DOC_DIR
            vision.DOC_DIR = Path(tmp)
            try:
                captured = {}

                def complete(prompt, images):
                    captured["prompt"] = prompt
                    captured["images"] = list(images)
                    return "看到两页 [1] [4]"

                text = "文字证据 chunk-page1-a chunk-page1-b chunk-docx chunk-page2 chunk-page3-not-attached"
                seen = visual_context_for_sources(
                    "页面问题",
                    _visual_page_sources(),
                    complete=complete,
                    user_prompt=text,
                )
            finally:
                vision.DOC_DIR = original

        self.assertTrue(seen["used"])
        self.assertEqual(seen["pages"], [{"sourceIndex": 0, "page": 1}, {"sourceIndex": 3, "page": 2}])
        images = captured["images"]
        self.assertEqual(len(images), 2)
        self.assertTrue(images[0].startswith(b"\x89PNG"))
        self.assertTrue(images[1].startswith(b"\x89PNG"))
        self.assertNotEqual(images[0], images[1])
        self.assertIn(text, captured["prompt"])
        self.assertIn("[1] 第 1 页", captured["prompt"])
        self.assertIn("[4] 第 2 页", captured["prompt"])
        self.assertNotIn("[2] 第 1 页", captured["prompt"])
        self.assertNotIn("[5] 第 3 页", captured["prompt"])

    def test_visual_generation_includes_all_text_evidence_once(self):
        import tempfile
        from pathlib import Path

        import qa.entry as entry
        import qa.vision as vision
        from qa.prompt import BASE_SYSTEM_PROMPT

        question = "第二跳文字证据还在吗？"
        order = []
        captured = {}
        with tempfile.TemporaryDirectory() as tmp:
            _write_two_page_pdf(tmp)
            original_dir = vision.DOC_DIR
            original_text = entry.call_qwen
            original_vision = vision.call_qwen_vision
            vision.DOC_DIR = Path(tmp)

            def fake_text(system, prompt):
                order.append("text")
                return "被丢弃的文字答案"

            def fake_vision(system, prompt, images):
                order.append("vision")
                captured["system"] = system
                captured["prompt"] = prompt
                captured["images"] = list(images)
                return "图像和文字一起回答 [1] [5]"

            entry.call_qwen = fake_text
            vision.call_qwen_vision = fake_vision
            try:
                result = _compose_structured_answer(
                    question,
                    engine=RecordingEngine([_visual_page_sources()]),
                    retrieval_mode="single",
                    include_visual=True,
                )
            finally:
                entry.call_qwen = original_text
                vision.call_qwen_vision = original_vision
                vision.DOC_DIR = original_dir

        self.assertEqual(order, ["vision"])
        self.assertEqual(captured["system"], BASE_SYSTEM_PROMPT)
        self.assertIn(question, captured["prompt"])
        for marker in (
            "chunk-page1-a",
            "chunk-page1-b",
            "chunk-docx",
            "chunk-page2",
            "chunk-page3-not-attached",
        ):
            self.assertIn(marker, captured["prompt"])
        self.assertIn("[4] 第 2 页", captured["prompt"])
        self.assertNotIn("[5] 第 3 页", captured["prompt"])
        self.assertEqual(len(captured["images"]), 2)
        self.assertTrue(captured["images"][0].startswith(b"\x89PNG"))
        self.assertNotEqual(captured["images"][0], captured["images"][1])
        self.assertEqual(result["answer"], "图像和文字一起回答 [1] [5]")
        self.assertEqual(result["evidenceIds"], ["a", "b", "d", "c", "e"])
        self.assertEqual(result["visualContext"]["used"], True)
        self.assertEqual(result["visualContext"]["reason"], "")
        self.assertEqual(
            result["visualContext"]["pages"],
            [{"sourceIndex": 0, "page": 1}, {"sourceIndex": 3, "page": 2}],
        )
        self.assertEqual([row["chunk_id"] for row in result["sources"]], ["a", "b", "d", "c", "e"])

    def test_visual_model_failure_uses_one_text_fallback(self):
        import tempfile
        from pathlib import Path

        import qa.entry as entry
        import qa.vision as vision
        from qa.prompt import BASE_SYSTEM_PROMPT

        question = "模型失败时保留文字回答吗？"
        order = []
        captured = {}
        with tempfile.TemporaryDirectory() as tmp:
            _write_two_page_pdf(tmp)
            original_dir = vision.DOC_DIR
            original_text = entry.call_qwen
            original_vision = vision.call_qwen_vision
            vision.DOC_DIR = Path(tmp)

            def fake_vision(system, prompt, images):
                order.append("vision")
                captured["vision_prompt"] = prompt
                captured["images"] = list(images)
                raise TimeoutError("timed out")

            def fake_text(system, prompt):
                order.append("text")
                captured["text_system"] = system
                captured["text_prompt"] = prompt
                return "文字回退 [5]"

            entry.call_qwen = fake_text
            vision.call_qwen_vision = fake_vision
            try:
                result = _compose_structured_answer(
                    question,
                    engine=RecordingEngine([_visual_page_sources()]),
                    retrieval_mode="single",
                    include_visual=True,
                )
            finally:
                entry.call_qwen = original_text
                vision.call_qwen_vision = original_vision
                vision.DOC_DIR = original_dir

        self.assertEqual(order, ["vision", "text"])
        self.assertIn("chunk-page3-not-attached", captured["vision_prompt"])
        self.assertEqual(len(captured["images"]), 2)
        self.assertEqual(captured["text_system"], BASE_SYSTEM_PROMPT)
        self.assertIn(question, captured["text_prompt"])
        self.assertIn("chunk-page3-not-attached", captured["text_prompt"])
        self.assertIn("chunk-page1-b", captured["text_prompt"])
        self.assertEqual(result["answer"], "文字回退 [5]")
        self.assertFalse(result["visualContext"]["used"])
        self.assertIn("TimeoutError", result["visualContext"]["reason"])
        self.assertIn("timed out", result["visualContext"]["reason"])
        self.assertEqual(
            result["visualContext"]["pages"],
            [{"sourceIndex": 0, "page": 1}, {"sourceIndex": 3, "page": 2}],
        )
        self.assertEqual(result["evidenceIds"], ["a", "b", "d", "c", "e"])

    def test_unavailable_pdf_page_uses_one_text_generation(self):
        import qa.entry as entry
        import qa.vision as vision

        order = []
        prompts = []
        original_text = entry.call_qwen
        original_vision = vision.call_qwen_vision

        def fake_vision(system, prompt, images):
            order.append("vision")
            return "不应该调用视觉模型"

        def fake_text(system, prompt):
            order.append("text")
            prompts.append(prompt)
            return "只有文字 [1]"

        entry.call_qwen = fake_text
        vision.call_qwen_vision = fake_vision
        try:
            missing = _compose_structured_answer(
                "缺页时怎么回答？",
                engine=RecordingEngine([[
                    {"doc_name": "missing-page.pdf", "page_number": 1, "chunk_id": "p", "content": "still-in-text"},
                ]]),
                retrieval_mode="single",
                include_visual=True,
            )
            missing_order = list(order)
            order.clear()
            docx = _compose_structured_answer(
                "文档证据在吗？",
                engine=RecordingEngine([[
                    {"doc_name": "notes.docx", "page_number": 1, "chunk_id": "d", "content": "still-in-text"},
                ]]),
                retrieval_mode="single",
                include_visual=True,
            )
        finally:
            entry.call_qwen = original_text
            vision.call_qwen_vision = original_vision

        self.assertEqual(missing_order, ["text"])
        self.assertIn("still-in-text", prompts[0])
        self.assertIn("still-in-text", prompts[1])
        self.assertEqual(missing["answer"], "只有文字 [1]")
        self.assertFalse(missing["visualContext"]["used"])
        self.assertIn("FileNotFoundError", missing["visualContext"]["reason"])
        self.assertEqual(missing["visualContext"]["pages"], [])
        self.assertEqual(missing["evidenceIds"], ["p"])
        self.assertEqual(order, ["text"])
        self.assertEqual(docx["answer"], "只有文字 [1]")
        self.assertFalse(docx["visualContext"]["used"])
        self.assertEqual(docx["visualContext"]["reason"], "no_pdf_page")
        self.assertEqual(docx["evidenceIds"], ["d"])


if __name__ == "__main__":
    unittest.main()
