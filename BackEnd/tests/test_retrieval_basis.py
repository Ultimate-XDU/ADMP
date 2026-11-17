"""Planner basis chunks stay available with the second-hop evidence."""

import unittest

from search.common import QA_TOP_K
from search.retrieval_coordinator import merge_hop_results, retrieve


class RecordingEngine:
    def __init__(self, batches):
        self.batches = list(batches)
        self.queries = []

    def search_for_qa(self, query, structural_depth=1, semantic_extra=0):
        self.queries.append(query)
        if not self.batches:
            return []
        return self.batches.pop(0)


def _row(chunk_id, content, section_title=None, page_number=1):
    return {
        "chunk_id": chunk_id,
        "doc_name": "guide.pdf",
        "section_title": section_title or chunk_id,
        "content": content,
        "page_number": page_number,
    }


class RetrievalBasisTests(unittest.TestCase):
    def test_third_source_basis_is_kept_with_answer_evidence(self):
        hop1 = [
            _row("first-1", "第一段背景", page_number=1),
            _row("first-2", "第二段背景", page_number=2),
            _row("bridge-3", "桥接事实：离地 500 米", section_title="离地高度", page_number=3),
        ]
        hop2 = [
            _row("answer-1", "起飞中止", page_number=10),
            _row("answer-2", "复飞", page_number=11),
            _row("answer-1", "重复的起飞中止", page_number=12),
            _row("bridge-3", "另一份桥接副本", page_number=13),
        ]
        question = "比较风切变以及起飞中止"
        follow = "离地 500 米时的起飞中止条件"

        def planner(original, rows):
            self.assertEqual(original, question)
            self.assertEqual([row["chunk_id"] for row in rows], ["first-1", "first-2", "bridge-3"])
            return follow, ["bridge-3"]

        engine = RecordingEngine([hop1, hop2])
        results, trace = retrieve(engine, question, mode="multi", planner=planner)

        self.assertEqual(engine.queries, [question, follow])
        self.assertEqual(trace["planner"], "llm")
        self.assertEqual(trace["reason"], "follow_up")
        self.assertEqual(trace["steps"][1]["basisChunkIds"], ["bridge-3"])

        ids = [row["chunk_id"] for row in results]
        self.assertEqual(ids, ["bridge-3", "answer-1", "answer-2", "first-1"])
        self.assertLessEqual(len(results), QA_TOP_K)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertIn("bridge-3", ids)
        self.assertIn("answer-1", ids)
        self.assertIn("answer-2", ids)
        originals = hop1 + hop2
        for row in results:
            self.assertTrue(any(row is item for item in originals))
        self.assertIs(results[0], hop1[2])
        self.assertEqual(results[0]["chunk_id"], "bridge-3")
        self.assertEqual(results[0]["content"], "桥接事实：离地 500 米")
        self.assertIs(results[1], hop2[0])
        self.assertEqual(results[1]["content"], "起飞中止")

        merged = merge_hop_results(hop1, hop2, basis_ids=["bridge-3"])
        self.assertEqual([row["chunk_id"] for row in merged], ids)
        self.assertIs(merged[0], hop1[2])
        self.assertIs(merged[1], hop2[0])
        self.assertLessEqual(len(merged), QA_TOP_K)

        untouched = merge_hop_results(hop1, hop2[:2])
        self.assertEqual(
            [row["chunk_id"] for row in untouched],
            ["first-1", "first-2", "answer-1", "answer-2"],
        )

    def test_heuristic_fallback_keeps_cited_third_source(self):
        hop1 = [
            _row("first-1", "风切变", section_title="风切变", page_number=1),
            _row("first-2", "复飞", section_title="复飞", page_number=2),
            _row("bridge-3", "离地 500 米出现风切变。", section_title="离地高度", page_number=3),
        ]
        hop2 = [
            _row("answer-1", "中止起飞", page_number=8),
            _row("answer-2", "继续进近", page_number=9),
        ]
        question = "比较风切变以及复飞"

        def timed_out(original, rows):
            raise TimeoutError("timed out")

        engine = RecordingEngine([hop1, hop2])
        results, trace = retrieve(engine, question, mode="multi", planner=timed_out)

        self.assertEqual(trace["planner"], "heuristic_fallback")
        self.assertEqual(trace["reason"], "timeout")
        self.assertEqual(trace["steps"][1]["basisChunkIds"], ["first-1", "first-2", "bridge-3"])
        self.assertEqual(len(engine.queries), 2)
        ids = [row["chunk_id"] for row in results]
        self.assertEqual(ids, ["first-1", "first-2", "bridge-3", "answer-1"])
        self.assertLessEqual(len(results), QA_TOP_K)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertIn("bridge-3", ids)
        self.assertIn("answer-1", ids)
        self.assertIs(results[2], hop1[2])
        self.assertEqual(results[2]["chunk_id"], "bridge-3")
        self.assertIs(results[3], hop2[0])


if __name__ == "__main__":
    unittest.main()
