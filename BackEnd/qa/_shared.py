import re
from typing import Dict


def _compact_evidence_text(value: str) -> str:
    return re.sub(r"[^\u4e00-\u9fffA-Za-z0-9.]+", "", str(value or "").lower())


def _is_structural_evidence(row: Dict) -> bool:
    source_type = str(row.get("source_type") or "")
    layout_type = str(row.get("layout_type") or "")
    return source_type.startswith("structural") or layout_type == "structural" or bool(row.get("structural_node_id"))
