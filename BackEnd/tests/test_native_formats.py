"""Native parsers keep exact tokens and locators. No model and no database."""

import tempfile
import unittest
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from qa.response import _normalize_response_source
from search.document_pipeline import extract_document_units
from search.index_manager import build_document_chunks
from search.native_parser import document_type_for_name, normalize_document_key


class _Embedder:
    def encode(self, *args, **kwargs):
        raise AssertionError("parser tests must not embed")


def _chunks(units):
    return build_document_chunks(units, _Embedder())


def _write(directory: Path, name: str, text: str) -> Path:
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


class NativeFormatTests(unittest.TestCase):
    def test_html_nested_headings_table_and_figure(self):
        html = """<!DOCTYPE html><html><head><title>File Title</title></head><body>
        <h1 id="ops">Operations</h1>
        <p>Intro thickness 0.014 inches.</p>
        <h2 id="stall">Stall speed</h2>
        <p>Limit 6.0 degrees for TAIL-1001.</p>
        <h3 id="detail">Detail</h3>
        <p>Angle 0.7 deg.</p>
        <table id="limits"><tr><th>tail</th><th>thickness</th></tr><tr><td>TAIL-1001</td><td>0.014</td></tr></table>
        <figure id="fig1"><img alt="ice photo 0.014"><figcaption>Figure caption TAIL-1001</figcaption></figure>
        </body></html>"""
        with tempfile.TemporaryDirectory() as directory:
            path = _write(Path(directory), "sample.htm", html)
            units, _health = extract_document_units(str(path), {"source_url": "https://example.test/ops", "publisher": "Desk", "document_title": "Given Title", "published_at": "2024-05-06", "original_format": "HTML", "document_key": "key-ops"})
        detail = next(unit for unit in units if "0.7" in unit["content"])
        self.assertEqual(detail["section_path"], ["Operations", "Stall speed", "Detail"])
        self.assertEqual(detail["locator"]["anchor"], "detail")
        self.assertIn("#detail", detail["locator"]["label"])
        self.assertEqual(detail["page_number"], 0)
        self.assertEqual(detail["original_format"], "HTML")
        self.assertEqual(detail["source_url"], "https://example.test/ops")
        self.assertEqual(detail["document_title"], "Given Title")
        self.assertEqual(detail["document_key"], "key-ops")
        table = next(unit for unit in units if unit.get("role") == "table")
        self.assertIn("TAIL-1001", table["content"])
        self.assertIn("0.014", table["content"])
        figure = next(unit for unit in units if unit.get("role") == "figure")
        self.assertIn("ice photo 0.014", figure["content"])
        self.assertIn("Figure caption TAIL-1001", figure["content"])
        self.assertEqual(figure["locator"]["anchor"], "fig1")
        chunks = _chunks(units)
        self.assertTrue(any("0.7" in chunk["content"] and chunk["locator"]["anchor"] == "detail" for chunk in chunks))
        self.assertTrue(all(chunk["page_number"] == 0 for chunk in chunks))

    def test_markdown_sections_and_code(self):
        text = "# Operations\n\nIntro thickness 0.014 inches.\n\n## Stall speed\n\nBefore the block.\n\n```\nTAIL-1001\nvalue 6.0\n```\n"
        with tempfile.TemporaryDirectory() as directory:
            path = _write(Path(directory), "sample.md", text)
            units, _health = extract_document_units(str(path))
        code = next(unit for unit in units if unit.get("role") == "code")
        self.assertEqual(code["content"], "TAIL-1001\nvalue 6.0")
        self.assertIn("6.0", code["content"])
        self.assertGreaterEqual(code["locator"]["rowStart"], 1)
        self.assertGreaterEqual(code["locator"]["rowEnd"], code["locator"]["rowStart"])
        self.assertEqual(code["section_path"], ["Operations", "Stall speed"])
        self.assertEqual(code["page_number"], 0)
        intro = next(unit for unit in units if "0.014" in unit["content"])
        self.assertEqual(intro["section_path"], ["Operations"])
        chunks = _chunks(units)
        self.assertTrue(any(chunk.get("role") == "code" and "TAIL-1001" in chunk["content"] and "6.0" in chunk["content"] for chunk in chunks))

    def test_unicode_txt_line_range(self):
        text = "第一行结冰说明\n厚度 0.014 英寸，温度 −40℃。\n\n第二段 TAIL-1001\n保持 6.0 度。\n"
        with tempfile.TemporaryDirectory() as directory:
            path = _write(Path(directory), "sample.txt", text)
            units, _health = extract_document_units(str(path))
        first = next(unit for unit in units if "0.014" in unit["content"])
        self.assertIn("结冰", first["content"])
        self.assertIn("−40℃", first["content"])
        self.assertEqual(first["locator"]["rowStart"], 1)
        self.assertEqual(first["locator"]["rowEnd"], 2)
        self.assertEqual(first["page_number"], 0)
        second = next(unit for unit in units if "TAIL-1001" in unit["content"])
        self.assertIn("6.0", second["content"])
        self.assertEqual(second["locator"]["rowStart"], 4)
        self.assertEqual(second["locator"]["rowEnd"], 5)
        chunks = _chunks(units)
        covered = next(chunk for chunk in chunks if "−40℃" in chunk["content"])
        self.assertLessEqual(covered["locator"]["rowStart"], 1)
        self.assertGreaterEqual(covered["locator"]["rowEnd"], 2)

    def test_quoted_multiline_csv(self):
        text = 'tail,note,thickness\n"TAIL-1001","leading edge\nstill quoted, with comma",0.014\n"TAIL-1002","quote ""inner""",6.0\n'
        with tempfile.TemporaryDirectory() as directory:
            path = _write(Path(directory), "sample.csv", text)
            units, _health = extract_document_units(str(path))
        multiline = next(unit for unit in units if "TAIL-1001" in unit["content"])
        self.assertIn("still quoted, with comma", multiline["content"])
        self.assertIn("0.014", multiline["content"])
        self.assertEqual(multiline["locator"]["rowStart"], 2)
        self.assertEqual(multiline["locator"]["header"], ["tail", "note", "thickness"])
        self.assertEqual(multiline["locator"]["lineStart"], 2)
        self.assertEqual(multiline["locator"]["lineEnd"], 3)
        self.assertEqual(multiline["page_number"], 0)
        quoted = next(unit for unit in units if "TAIL-1002" in unit["content"])
        self.assertIn('quote "inner"', quoted["content"])
        self.assertIn("6.0", quoted["content"])
        chunks = _chunks(units)
        self.assertTrue(any("0.014" in chunk["content"] and chunk["locator"]["rowStart"] == 2 for chunk in chunks))

    def test_nested_json_paths(self):
        text = '{"flights":[{"tail":"TAIL-1001","ice":{"thickness":0.014,"note":"leading"}},{"tail":"TAIL-1002","ice":{"thickness":6.0,"note":"stall"}}],"code":"B-42"}'
        with tempfile.TemporaryDirectory() as directory:
            path = _write(Path(directory), "sample.json", text)
            units, _health = extract_document_units(str(path))
        first = next(unit for unit in units if unit["locator"].get("jsonPointer") == "/flights/0/ice/thickness")
        second = next(unit for unit in units if unit["locator"].get("jsonPointer") == "/flights/1/ice/thickness")
        self.assertEqual(first["content"], "0.014")
        self.assertEqual(first["locator"]["path"], "$.flights[0].ice.thickness")
        self.assertEqual(second["content"], "6.0")
        self.assertEqual(second["locator"]["path"], "$.flights[1].ice.thickness")
        self.assertEqual(first["page_number"], 0)
        self.assertTrue(any(unit["content"] == "B-42" and unit["locator"]["jsonPointer"] == "/code" for unit in units))
        chunks = _chunks(units)
        kept = next(chunk for chunk in chunks if chunk["content"] == "0.014")
        self.assertEqual(kept["locator"]["jsonPointer"], "/flights/0/ice/thickness")
        self.assertEqual(kept["locator"]["path"], "$.flights[0].ice.thickness")

    def test_repeated_xml_elements(self):
        text = "<reports><report id=\"R-1\"><value>0.014</value><tail>TAIL-1001</tail></report><report id=\"R-2\"><value>0.014</value><tail>TAIL-1002</tail></report></reports>"
        with tempfile.TemporaryDirectory() as directory:
            path = _write(Path(directory), "sample.xml", text)
            units, _health = extract_document_units(str(path))
        values = [unit for unit in units if unit["locator"]["path"].endswith("/value[1]")]
        self.assertEqual(sorted(unit["locator"]["path"] for unit in values), ["/reports/report[1]/value[1]", "/reports/report[2]/value[1]"])
        self.assertTrue(all(unit["content"] == "0.014" for unit in values))
        self.assertTrue(any("id=R-1" in unit["content"] for unit in units))
        self.assertTrue(any(unit["content"] == "TAIL-1002" for unit in units))
        self.assertTrue(all(unit["page_number"] == 0 for unit in units))
        chunks = _chunks(units)
        paths = {chunk["locator"]["path"] for chunk in chunks if chunk["content"] == "0.014"}
        self.assertEqual(paths, {"/reports/report[1]/value[1]", "/reports/report[2]/value[1]"})

    def test_metadata_and_chunk_ids_stay_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _write(Path(directory), "sample.txt", "厚度 0.014\n")
            metadata = {"source_url": "https://example.test/a", "publisher": "Desk", "document_title": "Notes", "published_at": "2024-01-02", "original_format": "TXT", "document_key": "same-key"}
            first, _health = extract_document_units(str(path), metadata)
            second, _health = extract_document_units(str(path), metadata)
        self.assertEqual([unit["content"] for unit in first], [unit["content"] for unit in second])
        self.assertTrue(all(unit["document_key"] == "same-key" and unit["publisher"] == "Desk" for unit in first))
        self.assertEqual([chunk["chunk_id"] for chunk in _chunks(first)], [chunk["chunk_id"] for chunk in _chunks(second)])

    def test_document_key_and_suffixes(self):
        self.assertEqual(document_type_for_name("page.htm"), "HTML")
        self.assertEqual(document_type_for_name("notes.markdown"), "MD")
        digest = "abc123"
        mirrored = normalize_document_key("", "https://example.test/a", "Desk", "2024-01-02", digest)
        again = normalize_document_key("", "https://example.test/a", "Desk", "2024-01-02", digest)
        self.assertEqual(mirrored, again)
        self.assertNotEqual(mirrored, normalize_document_key("", "https://example.test/a", "Desk", "2024-01-02", "other"))
        self.assertEqual(normalize_document_key("explicit", "https://example.test/a", "Desk", "2024-01-02", digest), "explicit")
        self.assertEqual(normalize_document_key("", "", "", "", digest), normalize_document_key("", "", "", "", digest))

    def test_source_fields_survive_response_normalization(self):
        normalized = _normalize_response_source({
            "content": "0.014",
            "doc_name": "sample.json",
            "page_number": 0,
            "score": 0.2,
            "source_url": "https://example.test/a",
            "document_key": "key-ops",
            "locator": {"label": "$.flights[0].ice.thickness", "kind": "json", "path": "$.flights[0].ice.thickness"},
            "section_path": ["flights", "0", "ice", "thickness"],
        })
        self.assertEqual(normalized["page_number"], 0)
        self.assertEqual(normalized["source_url"], "https://example.test/a")
        self.assertEqual(normalized["document_key"], "key-ops")
        self.assertEqual(normalized["locator"]["path"], "$.flights[0].ice.thickness")

    def test_pptx_slide_and_xlsx_cell(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pptx = root / "deck.pptx"
            slide = """<?xml version="1.0" encoding="UTF-8"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld><p:spTree>
    <p:sp><p:nvSpPr><p:nvPr><p:ph type="title"/></p:nvPr></p:nvSpPr><p:txBody><a:p><a:r><a:t>Operations</a:t></a:r></a:p></p:txBody></p:sp>
    <p:sp><p:txBody><a:p><a:r><a:t>thickness 0.014</a:t></a:r></a:p></p:txBody></p:sp>
  </p:spTree></p:cSld>
</p:sld>"""
            with zipfile.ZipFile(pptx, "w") as archive:
                archive.writestr("ppt/slides/slide1.xml", slide)
            units, _health = extract_document_units(str(pptx))
            self.assertTrue(any(unit["page_number"] == 1 and "0.014" in unit["content"] and unit["locator"]["page"] == 1 for unit in units))
            sheet = """<?xml version="1.0" encoding="UTF-8"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>
    <row r="1"><c r="A1" t="inlineStr"><is><t>tail</t></is></c><c r="B1" t="inlineStr"><is><t>thickness</t></is></c></row>
    <row r="2"><c r="A2" t="inlineStr"><is><t>TAIL-1001</t></is></c><c r="B2"><v>0.014</v></c></row>
  </sheetData>
</worksheet>"""
            workbook = """<?xml version="1.0" encoding="UTF-8"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets><sheet name="Limits" sheetId="1" r:id="rId1"/></sheets>
</workbook>"""
            rels = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>"""
            xlsx = root / "book.xlsx"
            with zipfile.ZipFile(xlsx, "w") as archive:
                archive.writestr("xl/workbook.xml", workbook)
                archive.writestr("xl/_rels/workbook.xml.rels", rels)
                archive.writestr("xl/worksheets/sheet1.xml", sheet)
            units, _health = extract_document_units(str(xlsx))
            row = next(unit for unit in units if "TAIL-1001" in unit["content"])
            self.assertEqual(row["page_number"], 0)
            self.assertEqual(row["locator"]["rowStart"], 2)
            self.assertEqual(row["locator"]["path"], "Limits!A2:B2")
            self.assertIn("0.014", row["content"])
            self.assertIn("thickness=0.014", row["content"])
            self.assertEqual(row["locator"]["label"], "Limits!A2:B2")
