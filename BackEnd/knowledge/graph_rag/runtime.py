"""Ollama generation, CPU embeddings, and the LightRAG extraction runtime."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from knowledge.graph_rag.config import (
    WINDOW_CHARS,
    WINDOW_OVERLAP,
    data_dir,
    embedding_model_path,
    llm_model,
    ollama_host,
    rerank_model_path,
)
from knowledge.graph_rag.graph_ops import split_windows

logger = logging.getLogger(__name__)

_embedder = None
_reranker = None
_rerank_failed = False
_model_lock = threading.Lock()
_loop: Optional[asyncio.AbstractEventLoop] = None
_loop_thread: Optional[threading.Thread] = None
_loop_lock = threading.Lock()
_rag = None
_rag_lock = threading.Lock()
_THINK = re.compile(r"<think>.*?</think>", re.S)


def event_loop() -> asyncio.AbstractEventLoop:
    global _loop, _loop_thread
    with _loop_lock:
        if _loop is None:
            _loop = asyncio.new_event_loop()

            def _run() -> None:
                asyncio.set_event_loop(_loop)
                _loop.run_forever()

            _loop_thread = threading.Thread(target=_run, name="graph-lightrag", daemon=True)
            _loop_thread.start()
        return _loop


def run_async(coro, timeout: Optional[float] = None):
    future = asyncio.run_coroutine_threadsafe(coro, event_loop())
    return future.result(timeout)


def _log_raw(record: dict) -> None:
    if os.environ.get("GRAPH_LOG_RAW_LLM", "1").strip() == "0":
        return
    path = data_dir() / "raw-llm.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(record)
    record["ts"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def ollama_chat(
    prompt: str,
    system_prompt: Optional[str] = None,
    history: Optional[list] = None,
    json_mode: bool = False,
    max_tokens: int = 1500,
    timeout: int = 180,
) -> str:
    """Call the local Ollama server. think and num_ctx are set on this request only."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for item in history or []:
        if isinstance(item, dict) and item.get("role") and item.get("content") is not None:
            messages.append({"role": str(item["role"]), "content": str(item["content"])})
    messages.append({"role": "user", "content": str(prompt or "")})
    payload = {
        "model": llm_model(),
        "messages": messages,
        "stream": False,
        "think": False,
        "keep_alive": "30m",
        "options": {"num_ctx": 8192, "temperature": 0, "num_predict": max_tokens},
    }
    if json_mode:
        payload["format"] = "json"
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{ollama_host()}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    content = ""
    thinking = ""
    error = ""
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = json.loads(response.read().decode("utf-8"))
        message = raw.get("message") or {}
        content = str(message.get("content") or "")
        thinking = str(message.get("thinking") or "")
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        error = f"{type(exc).__name__}: {exc}"
        logger.warning("Ollama 请求失败: %s", error)
    content = _THINK.sub("", content).strip()
    _log_raw(
        {
            "model": llm_model(),
            "host": ollama_host(),
            "think": False,
            "num_ctx": 8192,
            "json_mode": json_mode,
            "system": system_prompt or "",
            "prompt": prompt,
            "content": content,
            "thinking": thinking,
            "error": error,
        }
    )
    if error and not content:
        raise RuntimeError(error)
    return content


def encode_texts(texts: list[str]):
    import numpy as np

    model = _embedding_model()
    cleaned = [str(text or "")[:4000] for text in texts] or [""]
    with _model_lock:
        vectors = model.encode(cleaned, normalize_embeddings=True, batch_size=8, show_progress_bar=False)
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    return array


def _embedding_model():
    global _embedder
    if _embedder is None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        from sentence_transformers import SentenceTransformer

        path = embedding_model_path()
        logger.info("加载 BGE-M3（CPU）: %s", path)
        _embedder = SentenceTransformer(path, device="cpu")
    return _embedder


def rerank_scores(query: str, texts: list[str]) -> Optional[list[float]]:
    global _reranker, _rerank_failed
    if _rerank_failed or not texts:
        return None
    try:
        if _reranker is None:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            from sentence_transformers import CrossEncoder

            path = rerank_model_path()
            logger.info("加载 BGE reranker（CPU）: %s", path)
            _reranker = CrossEncoder(path, device="cpu")
        pairs = [(query, str(text or "")[:2000]) for text in texts]
        with _model_lock:
            scores = _reranker.predict(pairs, show_progress_bar=False)
        return [float(score) for score in scores]
    except Exception as exc:
        _rerank_failed = True
        logger.warning("重排序不可用，改用向量和词法分数: %s", exc)
        return None


async def _embed(texts: list[str], **_kwargs):
    return await asyncio.to_thread(encode_texts, list(texts))


async def _llm(prompt, system_prompt=None, history_messages=None, **kwargs):
    response_format = kwargs.get("response_format")
    json_mode = bool(kwargs.get("keyword_extraction") or kwargs.get("entity_extraction"))
    if isinstance(response_format, dict) and str(response_format.get("type") or "").startswith("json"):
        json_mode = True
    return await asyncio.to_thread(
        ollama_chat,
        str(prompt or ""),
        system_prompt,
        list(history_messages or []),
        json_mode,
        1500,
        180,
    )


def prepared_windows(doc_key: str, text: str) -> list[tuple[str, str]]:
    """Map each extraction window to the LightRAG chunk id LightRAG will store."""
    from lightrag.utils import sanitize_text_for_encoding
    from lightrag.utils_pipeline import make_custom_chunk_id

    rows = []
    seen = set()
    for part in split_windows(text, WINDOW_CHARS, WINDOW_OVERLAP):
        cleaned = sanitize_text_for_encoding(part)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        rows.append((make_custom_chunk_id(doc_key, cleaned), cleaned))
    return rows


async def _rag_instance():
    global _rag
    with _rag_lock:
        if _rag is not None:
            return _rag
        from lightrag import LightRAG
        from lightrag.utils import EmbeddingFunc

        working = data_dir() / "lightrag"
        working.mkdir(parents=True, exist_ok=True)
        rag = LightRAG(
            working_dir=str(working),
            llm_model_func=_llm,
            embedding_func=EmbeddingFunc(embedding_dim=1024, max_token_size=8192, func=_embed),
            graph_storage="NetworkXStorage",
            vector_storage="NanoVectorDBStorage",
            llm_model_name=llm_model(),
            llm_model_max_async=1,
            embedding_func_max_async=1,
            entity_extract_max_gleaning=0,
            entity_extraction_use_json=True,
            default_llm_timeout=180,
            enable_llm_cache=True,
            enable_llm_cache_for_entity_extract=True,
            chunk_token_size=WINDOW_CHARS,
        )
        _rag = rag
    await rag.initialize_storages()
    return rag


async def insert_windows(doc_key: str, windows: list[str]) -> None:
    rag = await _rag_instance()
    cleaned = [window for window in windows if str(window or "").strip()]
    if not cleaned:
        return
    try:
        await rag.ainsert_custom_chunks("\n".join(cleaned), cleaned, doc_key)
    except Exception as exc:
        message = str(exc).lower()
        if "already" in message and ("committed" in message or "storage" in message):
            logger.info("LightRAG 文档已存在，按成功处理: %s", doc_key)
            return
        raise


async def delete_lightrag_docs(doc_keys: list[str]) -> None:
    keys = [str(key) for key in doc_keys if str(key).strip()]
    if not keys:
        return
    storage = data_dir() / "lightrag"
    # A restarted API has not constructed LightRAG yet. Persisted storage still
    # has to be opened so the deleted document's graph rows are removed.
    if _rag is None and not storage.exists():
        return
    rag = await _rag_instance()
    for doc_key in dict.fromkeys(keys):
        try:
            await rag.adelete_by_doc_id(doc_key)
        except Exception as exc:
            logger.warning("删除 LightRAG 文档失败 %s: %s", doc_key, exc)


def graphml_path(root: Optional[Path] = None) -> Optional[Path]:
    base = Path(root) if root is not None else data_dir()
    folder = base / "lightrag"
    if not folder.exists():
        return None
    matches = sorted(folder.glob("**/graph_chunk_entity_relation.graphml"))
    if not matches:
        matches = sorted(path for path in folder.glob("**/graph_*.graphml") if path.is_file())
    return matches[0] if matches else None


def read_raw_graph(root: Optional[Path] = None) -> tuple[list[dict], list[dict]]:
    path = graphml_path(root)
    if path is None or not path.exists() or path.stat().st_size == 0:
        return [], []
    import networkx as nx

    graph = nx.read_graphml(path)
    nodes = []
    for node_id, data in graph.nodes(data=True):
        row = {str(key): value for key, value in dict(data).items()}
        row["id"] = str(node_id)
        nodes.append(row)
    edges = []
    if graph.is_multigraph():
        triples = ((source, target, data) for source, target, _key, data in graph.edges(keys=True, data=True))
    else:
        triples = graph.edges(data=True)
    for source, target, data in triples:
        row = {str(key): value for key, value in dict(data).items()}
        row["source"] = str(source)
        row["target"] = str(target)
        edges.append(row)
    return nodes, edges
