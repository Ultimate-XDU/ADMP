"""Cited answers on nonempty spans stay in place.

Normal use retrieved an English retirement span and a Chinese answer with a
valid citation. Facet overlap was empty, so the cited answer was replaced.
aspect_support remains a retrieval hint. These tests call coverage_status
and _coverage_stop.
"""

import unittest

from knowledge.graph_rag.config import ABSTAIN_TEXT
from knowledge.graph_rag.hybrid_index import aspect_support, coverage_status
from knowledge.graph_rag.service import _coverage_stop

ENGLISH_SPAN = (
    "The average age of retirement of aircraft is25-30years for passenger aircraft and30-40for freighters"
)
QUESTION = "客运航空器退役机龄以及货机退役机龄"
SPANS = [
    {
        "unit_id": "c1",
        "doc_id": "easa",
        "start": 0,
        "end": len(ENGLISH_SPAN),
        "text": ENGLISH_SPAN,
    }
]
SOURCES = [
    {
        "chunk_id": "c1",
        "content": ENGLISH_SPAN,
        "doc_name": "EASA",
        "page_number": 1,
    }
]
# The live citation index was [8]. coverage_status is given the validated count.
CITED = "客运航空器平均退役机龄为25-30年，货机为30-40年。[8]"
# _coverage_stop reads citation numbers from the answer against the packed sources.
CITED_FOR_STOP = "客运航空器平均退役机龄为25-30年，货机为30-40年。[1]"


class GraphAnswerSupportTests(unittest.TestCase):
    def test_cross_language_cited_answer_remains(self) -> None:
        supported, missing = aspect_support(QUESTION, SPANS)
        self.assertEqual(supported, [])
        self.assertTrue(missing)
        self.assertEqual(
            coverage_status(CITED, QUESTION, SPANS, 1, ABSTAIN_TEXT),
            "answered",
        )
        self.assertEqual(
            _coverage_stop(CITED_FOR_STOP, SOURCES, QUESTION, SPANS),
            "answered",
        )

    def test_empty_context_is_insufficient(self) -> None:
        self.assertEqual(
            coverage_status(CITED, QUESTION, [], 1, ABSTAIN_TEXT),
            "insufficient_evidence",
        )
        self.assertEqual(
            coverage_status(CITED, QUESTION, [{"text": "  "}], 1, ABSTAIN_TEXT),
            "insufficient_evidence",
        )
        self.assertEqual(
            _coverage_stop(CITED_FOR_STOP, SOURCES, QUESTION, []),
            "insufficient_evidence",
        )
        self.assertEqual(
            _coverage_stop(CITED_FOR_STOP, SOURCES, QUESTION, [{"text": ""}]),
            "insufficient_evidence",
        )

    def test_explicit_partial_answer_remains_partial(self) -> None:
        counted = f"客运航空器平均退役机龄为25-30年。[8]{ABSTAIN_TEXT}"
        partial = f"客运航空器平均退役机龄为25-30年。[1]{ABSTAIN_TEXT}"
        supported, missing = aspect_support(QUESTION, SPANS)
        self.assertEqual(supported, [])
        self.assertTrue(missing)
        self.assertEqual(
            coverage_status(counted, QUESTION, SPANS, 1, ABSTAIN_TEXT),
            "partial_evidence",
        )
        self.assertEqual(
            _coverage_stop(partial, SOURCES, QUESTION, SPANS),
            "partial_evidence",
        )

    def test_uncited_answer_is_rejected(self) -> None:
        bare = "客运航空器平均退役机龄为25-30年，货机为30-40年。"
        self.assertEqual(
            coverage_status(bare, QUESTION, SPANS, 0, ABSTAIN_TEXT),
            "uncited_answer",
        )
        self.assertEqual(
            _coverage_stop(bare, SOURCES, QUESTION, SPANS),
            "uncited_answer",
        )

    def test_all_abstention_is_insufficient(self) -> None:
        self.assertEqual(
            coverage_status(f"{ABSTAIN_TEXT} [8]", QUESTION, SPANS, 1, ABSTAIN_TEXT),
            "insufficient_evidence",
        )
        self.assertEqual(
            _coverage_stop(f"{ABSTAIN_TEXT} [1]", SOURCES, QUESTION, SPANS),
            "insufficient_evidence",
        )
        self.assertEqual(
            coverage_status(ABSTAIN_TEXT, QUESTION, SPANS, 0, ABSTAIN_TEXT),
            "insufficient_evidence",
        )
        self.assertEqual(
            _coverage_stop(ABSTAIN_TEXT, SOURCES, QUESTION, SPANS),
            "insufficient_evidence",
        )
