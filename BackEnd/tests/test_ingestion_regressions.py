"""Regression checks for ingestion defects on frozen NASA text and the ASRS PDF.

These tests call the production metadata, splitter, and chunker. They do not
reimplement the parser and they do not touch indexes.
"""

import unittest
from pathlib import Path

from search.common import _build_structural_metadata, clean_block_text
from search.document_pipeline import extract_document_units, semantic_split
from search.index_manager import build_document_chunks

NASA_DOCX = Path("/home/yuexing/src/ADMP/work/eval/baseline/documents/nasa-icing-performance.docx")
ASRS_PDF = Path(
    "/mnt/c/Users/MSIK/Documents/Codex/2026-11-10/ubuntu-github-admp-rag-pandawiki-https/work/aviation-corpus/nasa-asrs-419.pdf"
)
FAKE_ARTICLE_IDS = {"0.014", "0.3", "6.0", "0.7"}


class _Embedder:
    def encode(self, *args, **kwargs):
        raise AssertionError("boundary packing must not call the embedder")


def _recover(original: str, chunks: list) -> None:
    position = 0
    for index, chunk in enumerate(chunks):
        if index == 0:
            if not original.startswith(chunk):
                raise AssertionError(chunk[:80])
            position = len(chunk)
            continue
        found = None
        for start in range(max(0, position - len(chunk)), position + 1):
            if original[start : start + len(chunk)] == chunk and start <= position <= start + len(chunk):
                found = start
                break
        if found is None:
            raise AssertionError(chunk[:80])
        if found > 0:
            previous = original[found - 1]
            if not (previous.isspace() or chunk[:1].isspace() or previous in "。！？；?!."):
                raise AssertionError(f"cut inside a token at {found}: {original[found - 8:found + 8]!r}")
        position = found + len(chunk)
    if position != len(original):
        raise AssertionError(f"recovered {position} of {len(original)}")


class IngestionRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not NASA_DOCX.is_file():
            raise unittest.SkipTest(f"missing frozen docx: {NASA_DOCX}")
        cls.units, _health = extract_document_units(str(NASA_DOCX))
        cls.chunks = build_document_chunks(cls.units, _Embedder())
        cls.performance = next(unit["content"] for unit in cls.units if "0.014 inches" in unit.get("content", ""))

    def test_measurements_stay_verbatim(self):
        self.assertIn("0.014 inches", self.performance)
        for value in ("0.3", "6.0", "0.7"):
            self.assertTrue(any(value in chunk["content"] for chunk in self.chunks), value)
        for chunk in self.chunks:
            self.assertNotIn(chunk.get("article_id") or "", FAKE_ARTICLE_IDS)
            for fake in ("第0.014条", "第0.3条", "第6.0条", "第0.7条"):
                self.assertNotIn(fake, chunk["content"])
        metadata = _build_structural_metadata(self.performance)
        self.assertEqual(metadata["article_id"], "")
        self.assertEqual(metadata["article_label"], "")
        self.assertIn("0.014 inches", metadata and self.performance)

    def test_english_split_preserves_nasa_paragraph(self):
        original = clean_block_text(self.performance)
        chunks = semantic_split(original, _Embedder(), max_len=240, overlap_chars=40)
        self.assertGreater(len(chunks), 1)
        _recover(original, chunks)
        self.assertTrue(any("0.014" in chunk for chunk in chunks))
        self.assertNotIn("0.01\n4", "\n".join(chunks))
        joined = "\n".join(chunk["content"] for chunk in self.chunks)
        self.assertIn("0.014 inches", joined)
        for chunk in self.chunks:
            self.assertFalse(chunk["content"].lstrip().startswith("ff,"))
            self.assertFalse(chunk["content"].rstrip().endswith("Thirty minu"))

    def test_legal_clause_labels(self):
        chinese = _build_structural_metadata("第25.103条 起飞速度应当满足该条所述的失速准则。")
        self.assertEqual(chinese["article_id"], "25.103")
        self.assertEqual(chinese["article_label"], "第25.103条")
        signed = _build_structural_metadata(
            "See §25.103 for the stall speed criterion that applies to this configuration and the stated weight."
        )
        self.assertEqual(signed["article_id"], "25.103")
        self.assertEqual(signed["article_label"], "§25.103")
        measurement = _build_structural_metadata("AOA=0.3 deg and AOA=6.0 deg and AOA=0.7 deg with 0.014 inches of ice.")
        self.assertEqual(measurement["article_id"], "")
        self.assertNotIn("第", measurement["article_label"])

    def test_asrs_column_order_without_claiming_tables(self):
        if not ASRS_PDF.is_file():
            self.skipTest(f"missing frozen pdf: {ASRS_PDF}")
        units, health = extract_document_units(str(ASRS_PDF))
        page2 = [unit for unit in units if unit.get("page_number") == 2]
        text = "\n".join(unit["content"] for unit in page2)
        depart = text.find("We decided to depart")
        edge = text.find("on the leading edge of the wing")
        alerts = text.find("ASRS Alerts Issued")
        intake = text.find("October 2014 Report Intake")
        carrier = text.find("Air Carrier/Air Taxi Pilots")
        mailbox = text.find("P.O. Box")
        self.assertGreaterEqual(depart, 0)
        self.assertLess(depart, edge)
        self.assertFalse(depart < alerts < edge)
        self.assertFalse(intake < mailbox < carrier)
        self.assertTrue(all(unit.get("page_number") == 2 and unit.get("bbox") for unit in page2))
        self.assertIn("4,757", text)
        self.assertIn("7,596", text)
        # PyMuPDF find_tables does not return these newsletter grids.
        self.assertEqual(health.get("table_count"), 0)
        chunks = build_document_chunks(units, _Embedder())
        page2_chunks = [chunk["content"] for chunk in chunks if chunk.get("page_number") == 2]
        self.assertTrue(any("Thirty minutes" in content for content in page2_chunks))
        for content in page2_chunks:
            self.assertFalse(content.rstrip().endswith("Thirty minu"))
            self.assertFalse(content.lstrip().startswith("ponsibility"))


if __name__ == "__main__":
    unittest.main()
