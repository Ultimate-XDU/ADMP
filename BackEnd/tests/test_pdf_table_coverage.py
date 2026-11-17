"""False table boxes must keep prose the extractor omitted, without cloning real cells."""

import os
import re
import tempfile
from pathlib import Path
from unittest.mock import patch

import fitz

from search.document_pipeline import _extract_pdf_units


ASRS_534_PAGE1_LINES = [
    "experienced an updraft, downdraft, severe turbulence, and",
    "severe precipitation, and needed to get out of the situation.",
    "He told me to fly heading 180 and gave me a block altitude",
    "from FL270 to FL280 upon my request. I climbed back up to",
    "FL280 on a 180 heading and quickly exited the IMC. Once",
    "stable again in VMC, the Controller asked if we had any",
    "aircraft damage or injuries.\u2026 I responded, \u201cNegative\u201d to",
    "both. He then asked our intentions, and I said I would like to",
    "fly [routing]\u2026north to Knoxville. He granted this permission",
    "and did a great job assisting us. The rest of the flight was",
    "uneventful.\u2026 The next time, I will be much more cautious",
    "flying in the vicinity of thunderstorms and definitively find a",
    "better course than in the vicinity of thunderstorms, despite",
    "other aircraft ahead of me making it through.",
    "Part 121 \u2013 Could It Get Any Worse?",
    "An A321 Flight Attendant (FA) chronicles significant",
    "effects and concerns of high temperatures combined",
    "with inadequate air conditioning (A/C) during a stressful",
    "n  Prior to aircraft arrival, the Captain informed the crew",
    "that there was an issue with the air conditioning on the",
    "plane. It was 105 degrees\u2026this particular day. We began",
    "boarding with very limited airflow in the cabin, and the",
    "temperature began to increase to a very uncomfortable",
    "level. We closed the boarding door, and before the cockpit",
    "door was closed, the Captain had informed the [Flight",
    "Attendant]\u2026that we might have an issue with our tires.",
    "While waiting for maintenance to come out, the temperature",
    "in the cabin began getting even worse. Passengers were",
    "vocal about their discomfort. Maintenance took about 40",
    "minutes to tell us that we would need to deplane. During that",
    "time, the Captain made one announcement to passengers",
    "letting them know what was happening. The Captain",
    "claimed that he informed Tower we would need to deplane",
    "and needed an agent to bring the jet-bridge back to the",
    "plane. That took an additional 30 minutes to get an agent to",
    "open the door. During this time, passengers began getting",
    "impatient (rightfully so), and a few passengers threatened",
    "to open emergency exits. Others started telling other",
    "passengers to call 911 for help. Once the door was finally",
    "opened, the agent refused to let any passengers off until",
    "she took the temperature in the cabin. Drenched in sweat,",
]

ASRS_544_PAGE1_LINES = [
    "Aviation Maintenance Technicians (AMTs) bear the solemn",
    "responsibility of keeping our aircraft safe and airworthy, and",
    "on occasion, they assume additional duties during the course",
    "of a flight. Operating in what is commonly considered the",
    "most hazardous work environment, AMTs guard their own",
    "safety and that of co-workers, crewmembers, and passengers.",
    "Inquisitive and innovative, AMTs often quell threats and",
    "hazards before they become incidents and accidents.",
    "In May, we often remember Charles E. \u201cCharlie\u201d Taylor,",
    "the Wright Brothers\u2019 legendary bicycle mechanic, who",
    "turned engine mechanic and became the country\u2019s first real",
    "AMT. Charlie was born on May 24, 1868 in Cerro Gordo,",
    "Illinois. His life is an iconic American story. In his tradition,",
    "ASRS honors and thanks all AMTs for their expertise and",
    "professionalism in maintaining America\u2019s aircraft.",
    "This month, CALLBACK offers selected incidents that",
    "demonstrate significant impact, positive or negative, that",
    "AMTs can have on all flights. AMTs work tirelessly to",
    "mitigate or eliminate real and potential threats. Explore the",
    "narratives as you identify the issues and the actions taken by",
    "Part 121 \u2013 Inflation Protection",
    "Servicing a B767, this air carrier AMT calls for better safety",
    "and protections through new or updated functional inflation",
    "tools for all Technicians servicing aircraft tires with nitrogen.",
    "n The tire inflation tools available for use when servicing",
    "tires with nitrogen do not allow for personnel to stand",
    "forward or aft of the tire that is being serviced. The fitting",
    "at the end of the hose has to be held onto the inflation valve",
    "by hand, thus meaning you have to stand in front of the tire",
    "during servicing. The AMM (Aircraft Maintenance Manual)",
    "reference states in a warning that \u201cmake sure that the",
    "servicing cart and all personnel are forward or aft of the tire",
    "before you do the servicing procedure. If the wheel comes",
    "apart, the pieces can cause damage to equipment and kill or",
    "cause injuries to personnel.\u201d The tooling we have doesn\u2019t",
    "allow personnel to stand forward or aft of the wheel. I have",
    "always felt uncomfortable using this tooling during this task,",
    "and so do several of my fellow colleagues. \u2026the tooling we",
    "have works well when new but with time and use it wears.",
    "Suggestion: Buy a different inflation tool that allows staff to",
]


def _corpus_root():
    configured = os.environ.get("ADMP_AVIATION_CORPUS")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[2] / "work" / "corpus-100"


def _norm(text):
    return re.sub(r"\s+", " ", text or "").strip()


def _parse_pdf(path, doc_name):
    # Data loss is in the PyMuPDF fallback. Do not let MinerU replace that path.
    with patch("search.document_parser_mineru.is_mineru_available", return_value=False):
        return _extract_pdf_units(str(path), doc_name)


def _page_blob(units, page_number):
    return _norm(
        "\n".join(
            unit.get("content") or ""
            for unit in units
            if unit.get("page_number") == page_number
        )
    )


def _column_lines(pdf_path, page_index, min_chars=20):
    """Visual lines from the PDF text layer, split at the page midpoint."""
    document = fitz.open(pdf_path)
    try:
        page = document[page_index]
        midpoint = float(page.rect.width) / 2.0
        columns = {"left": [], "right": []}
        for block in page.get_text("dict").get("blocks", []):
            if block.get("type") != 0:
                continue
            x0, _y0, x1, _y1 = block.get("bbox", (0, 0, 0, 0))
            side = "left" if (float(x0) + float(x1)) / 2.0 < midpoint else "right"
            for line in block.get("lines", []):
                text = _norm("".join(span.get("text", "") for span in line.get("spans", [])))
                if len(text) >= min_chars:
                    columns[side].append(text)
        return columns["left"], columns["right"]
    finally:
        document.close()


def _assert_on_page(units, page_number, line, label):
    needle = _norm(line)
    assert needle in _page_blob(units, page_number), (
        f"{label} missing on page {page_number}: {line!r}"
    )
    matched_pages = {
        unit["page_number"]
        for unit in units
        if needle in _norm(unit.get("content") or "")
    }
    assert page_number in matched_pages, (
        f"{label} units tagged {sorted(matched_pages)} for {line!r}"
    )


def _assert_nasa_page(doc_id, filename, page_number, evidence_lines):
    pdf_path = _corpus_root() / "raw" / filename
    assert pdf_path.is_file(), f"required corpus PDF is missing: {pdf_path}"
    units, health = _parse_pdf(pdf_path, doc_id)
    assert health["parser_used"] == "pymupdf"
    blob = _page_blob(units, page_number)
    missing = [line for line in evidence_lines if _norm(line) not in blob]
    assert not missing, (
        f"{doc_id} page {page_number} missing {len(missing)} canonical lines, first={missing[0]!r}"
    )
    left_lines, right_lines = _column_lines(pdf_path, page_number - 1)
    assert len(left_lines) >= 10, f"{doc_id} left column kept {len(left_lines)} long lines"
    assert len(right_lines) >= 10, f"{doc_id} right column kept {len(right_lines)} long lines"
    for side, side_lines in (("left", left_lines), ("right", right_lines)):
        absent = [line for line in side_lines if line not in blob]
        assert not absent, (
            f"{doc_id} page {page_number} {side} column missing {len(absent)} lines, first={absent[0]!r}"
        )
        anchor = max(side_lines, key=len)
        _assert_on_page(units, page_number, anchor, f"{doc_id} {side} column")
    _assert_on_page(units, page_number, evidence_lines[0], f"{doc_id} evidence start")
    _assert_on_page(units, page_number, evidence_lines[-1], f"{doc_id} evidence end")


def test_nasa_asrs_534_page1_keeps_both_columns():
    _assert_nasa_page("nasa-asrs-534", "nasa-asrs-534.pdf", 1, ASRS_534_PAGE1_LINES)


def test_nasa_asrs_544_page1_keeps_both_columns():
    _assert_nasa_page("nasa-asrs-544", "nasa-asrs-544.pdf", 1, ASRS_544_PAGE1_LINES)


def _write_ruled_table(path, above, below, rows):
    document = fitz.open()
    try:
        page = document.new_page(width=612, height=792)
        page.insert_text(fitz.Point(72, 64), above, fontsize=11)
        left, top = 72.0, 180.0
        col_widths = (220.0, 180.0)
        row_height = 36.0
        for row_index, row in enumerate(rows):
            x = left
            y = top + row_index * row_height
            for col_index, value in enumerate(row):
                rect = fitz.Rect(x, y, x + col_widths[col_index], y + row_height)
                page.draw_rect(rect, color=(0, 0, 0), width=1)
                page.insert_textbox(
                    fitz.Rect(rect.x0 + 6, rect.y0 + 8, rect.x1 - 6, rect.y1 - 4),
                    value,
                    fontsize=10,
                )
                x += col_widths[col_index]
        bottom = top + len(rows) * row_height
        page.insert_text(fitz.Point(72, bottom + 40), below, fontsize=11)
        document.save(str(path))
    finally:
        document.close()


def test_ruled_pdf_keeps_both_table_columns_once():
    above = "Dispatch note: crews shall review the status board before departure."
    below = "Release note: no outstanding maintenance items remain."
    rows = [
        ["System", "Status"],
        ["Hydraulic pump", "Nominal"],
        ["Cabin pressure", "Stable"],
    ]
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "ruled_status_board.pdf"
        _write_ruled_table(pdf_path, above, below, rows)
        units, health = _parse_pdf(pdf_path, "ruled-status-board")
    assert health["parser_used"] == "pymupdf"
    assert health["table_count"] >= 1
    table_units = [
        unit for unit in units if unit.get("source_type") == "table" or unit.get("is_table")
    ]
    plain_units = [
        unit for unit in units if unit.get("source_type") != "table" and not unit.get("is_table")
    ]
    assert table_units
    table_blob = _norm("\n".join(unit.get("content") or "" for unit in table_units))
    plain_blob = _norm("\n".join(unit.get("content") or "" for unit in plain_units))
    for row_index, row in enumerate(rows):
        for col_index, value in enumerate(row):
            side = "left" if col_index == 0 else "right"
            assert value in table_blob, f"ruled table {side} column missing {value!r}"
            assert value not in plain_blob, f"ruled table duplicated {value!r} outside the table unit"
            _assert_on_page(table_units, 1, value, f"ruled table {side} row {row_index}")
    for sentence in (above, below):
        assert sentence in plain_blob
        _assert_on_page(plain_units, 1, sentence, "ruled prose")
