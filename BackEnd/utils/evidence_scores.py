"""Shared evidence score normalization helpers.

Retrieval uses ranking scores for ordering, while API responses and prompts
need bounded display scores.  This module keeps that contract explicit so
callers do not accidentally reuse one score scale for another purpose.
"""

from __future__ import annotations

import math
from typing import Any, Mapping


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _has_value(row: Mapping[str, Any], key: str) -> bool:
    value = row.get(key)
    return value is not None and value != ""


def _score_components(row: Mapping[str, Any]) -> Mapping[str, Any]:
    components = row.get("score_components")
    return components if isinstance(components, Mapping) else {}


def normalize_structural_score(raw_score: float) -> float:
    """Compress an unbounded structural heuristic score to [0, 1)."""
    bounded = max(_safe_float(raw_score), 0.0)
    return bounded / (1.0 + bounded)


def calibrate_ranking_score(raw_score: float, *, scale: float = 3.5) -> float:
    """Compress an internal ranking score to a stable public evidence score."""
    bounded = max(_safe_float(raw_score), 0.0)
    safe_scale = max(_safe_float(scale, 3.5), 0.01)
    return 1.0 - math.exp(-bounded / safe_scale)


def evidence_ranking_score(row: Mapping[str, Any]) -> float:
    """Return the score intended for internal ordering and evidence selection."""
    if _has_value(row, "ranking_score"):
        return max(_safe_float(row.get("ranking_score")), 0.0)

    components = _score_components(row)
    for key in ("ranking", "structural", "rrf", "rerank"):
        if _has_value(components, key):
            return max(_safe_float(components.get(key)), 0.0)

    for key in ("raw_structural_score", "final_score", "rrf_score", "rerank_score"):
        if _has_value(row, key):
            return max(_safe_float(row.get(key)), 0.0)

    return max(_safe_float(row.get("score", 0.0)), 0.0)


def evidence_display_score(row: Mapping[str, Any]) -> float:
    """Return the bounded score intended for prompts and API display."""
    if _has_value(row, "display_score"):
        return max(min(_safe_float(row.get("display_score")), 1.0), 0.0)

    components = _score_components(row)
    if _has_value(components, "display"):
        return max(min(_safe_float(components.get("display")), 1.0), 0.0)

    raw_score = _safe_float(row.get("score", 0.0))
    if 0.0 <= raw_score <= 1.0:
        return raw_score
    return calibrate_ranking_score(evidence_ranking_score(row))


def ensure_evidence_score_fields(source: Mapping[str, Any]) -> dict:
    """Return a copy of *source* with the standard score fields populated."""
    normalized = dict(source)
    ranking_score = evidence_ranking_score(normalized)
    display_score = evidence_display_score(normalized)
    normalized["ranking_score"] = ranking_score
    normalized["display_score"] = display_score
    # Backward-compatible API alias: "score" is intentionally display-scale.
    normalized["score"] = display_score
    components = dict(_score_components(normalized))
    components["ranking"] = ranking_score
    components["display"] = display_score
    normalized["score_components"] = components
    return normalized
