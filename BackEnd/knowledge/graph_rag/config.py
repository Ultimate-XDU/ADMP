"""Paths and runtime limits for the graph backend."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "BackEnd"

MAX_TOOL_CALLS = 4
# Multi-fact questions may use a few more calls. GRAPH_MAX_TOOL_CALLS overrides this.
MAX_TOOL_CALLS_CAP = 6
MAX_PATH_EDGES = 3
MAX_PATHS = 12
DEFAULT_SEARCH_LIMIT = 30
MAX_SEARCH_LIMIT = 80
MAX_EXPAND_NODES = 60
WINDOW_CHARS = 1400
WINDOW_OVERLAP = 120
ANSWER_PASSAGES = 6
ANSWER_PASSAGE_CHARS = 900
# Passage text placed in the answer prompt. Edge-backed passages are kept first.
ANSWER_CHAR_BUDGET = 5600

ABSTAIN_TEXT = "当前证据不足，无法给出可靠结论。"
CAUSATION_NOTE = (
    "抽取到的关系路径只表示原文里的联系，不能把路径本身当成因果关系的证明。"
    "接口里的 paths 是这些边上的结构连通，不是代理本次实际调用的工具轨迹。"
)


def data_dir() -> Path:
    configured = os.environ.get("GRAPH_DATA_DIR", "").strip()
    path = Path(configured) if configured else REPO_ROOT / "work" / "graph-data"
    return path


def corpus_dir() -> Path:
    configured = os.environ.get("GRAPH_CORPUS_DIR", "").strip()
    return Path(configured) if configured else REPO_ROOT / "work" / "corpus"


def progress_path() -> Path:
    return REPO_ROOT / "work" / "graph-progress.json"


def ollama_host() -> str:
    """Isolated runs set GRAPH_OLLAMA_HOST. Otherwise follow the normal LLM endpoint."""
    explicit = os.environ.get("GRAPH_OLLAMA_HOST", "").strip()
    if explicit:
        return explicit.rstrip("/")
    base = os.environ.get("LLM_BASE_URL", "").strip().rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return base or "http://127.0.0.1:11434"


def llm_model() -> str:
    return (
        os.environ.get("GRAPH_LLM_MODEL", "").strip()
        or os.environ.get("LLM_MODEL", "").strip()
        or "qwen3.5:9b"
    )


def extract_scope() -> str:
    scope = os.environ.get("GRAPH_EXTRACT_SCOPE", "pilot").strip().lower()
    if scope not in {"pilot", "all", "smoke"}:
        return "pilot"
    return scope


def auto_build_enabled() -> bool:
    return os.environ.get("GRAPH_AUTO_BUILD", "").strip().lower() in {"1", "true", "yes"}


def tool_call_budget(facet_count: int = 1) -> int:
    """Tool-call cap. The base stays 4. A multi-fact question may add at most two calls.

    ``GRAPH_MAX_TOOL_CALLS`` replaces that rule with a fixed cap from 1 to 8.
    Callers report the returned number; it is not an unbounded loop.
    """
    raw = os.environ.get("GRAPH_MAX_TOOL_CALLS", "").strip()
    if raw:
        try:
            value = int(raw)
        except ValueError:
            value = MAX_TOOL_CALLS
        return max(1, min(value, 8))
    extra = min(max(int(facet_count) - 1, 0), 2) if int(facet_count) >= 2 else 0
    return min(MAX_TOOL_CALLS + extra, MAX_TOOL_CALLS_CAP)


def _first_existing(candidates: list[str]) -> str:
    for item in candidates:
        if item and Path(item).exists():
            return item
    return candidates[-1] if candidates else ""


def embedding_model_path() -> str:
    return _first_existing(
        [
            os.environ.get("AEROSPACE_EMBEDDING_MODEL", ""),
            "/mnt/d/ADMP/models/BAAI--bge-m3",
            str(Path.home() / ".cache/modelscope/hub/models/BAAI/bge-m3"),
        ]
    )


def rerank_model_path() -> str:
    return _first_existing(
        [
            os.environ.get("AEROSPACE_RERANK_MODEL", ""),
            "/mnt/d/ADMP/models/BAAI--bge-reranker-v2-m3",
        ]
    )
