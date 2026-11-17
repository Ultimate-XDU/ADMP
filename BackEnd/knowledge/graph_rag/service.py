"""Graph lifecycle, retrieval, and the agent entry points."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Optional

from knowledge.graph_rag.agent import bridge_labels, normalize_query, run_agent
from knowledge.graph_rag.catalog import Catalog, _atomic_write, _now, canonical_document_key, managed_id
from knowledge.graph_rag.config import (
    ABSTAIN_TEXT,
    ANSWER_CHAR_BUDGET,
    ANSWER_PASSAGE_CHARS,
    CAUSATION_NOTE,
    DEFAULT_SEARCH_LIMIT,
    MAX_PATH_EDGES,
    MAX_SEARCH_LIMIT,
    BACKEND_ROOT,
    auto_build_enabled,
    data_dir,
    extract_scope,
    progress_path,
)
from knowledge.graph_rag.corpus import chunks_for, describe_coverage, is_corpus, manifest_by_id, selection_report
from knowledge.graph_rag.hierarchy import build_structure, chunk_locator, select_structure, unpaginated
from knowledge.graph_rag.hybrid_index import (
    budget_for,
    budget_note,
    catalog_brief,
    coverage_status,
    missing_aspects,
    named_doc_ids,
    pack_context,
    question_facets,
    retrieval_limits,
    retrieve_spans,
    unit_ranking,
)
from knowledge.graph_rag.graph_ops import (
    citations_from_answer,
    edge_kind,
    empty_graph,
    expand_hops,
    filter_graph,
    graph_with_paths,
    highlight,
    induced_subgraph,
    lexical_score,
    merge_graphs,
    split_windows,
    project_graph,
    public_graph,
)

logger = logging.getLogger(__name__)


def _persisted_document_records() -> list:
    """Read library documents from JSON. Do not use the state store: it locks and may rewrite the file."""
    configured = os.environ.get("GRAPH_MANAGEMENT_STATE", "").strip()
    if configured:
        path = Path(configured)
    else:
        from utils.state_store import DEFAULT_STATE_FILE

        path = DEFAULT_STATE_FILE
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    documents = payload.get("documents") if isinstance(payload, dict) else None
    return documents if isinstance(documents, list) else []


def _block_page(block: dict, fmt: str) -> int:
    raw = block.get("page_number")
    if raw is None or raw == "":
        raw = block.get("page")
    if raw is None or raw == "":
        return 0 if unpaginated(fmt) else 1
    try:
        page = int(raw)
    except (TypeError, ValueError):
        return 0
    if page > 0 and unpaginated(fmt):
        return 0
    return page


def _block_chunk(doc_id: str, block: dict, fallback_name: str) -> Optional[dict]:
    content = str(block.get("content") or block.get("text") or "").strip()
    if not content:
        return None
    raw_id = str(block.get("chunk_id") or block.get("unit_id") or "").strip()
    if not raw_id:
        raw_id = hashlib.sha1(content.encode("utf-8")).hexdigest()[:16]
    # Keep a parser-supplied id. Prefix only a bare local id so two documents cannot collide.
    if ":" in raw_id or raw_id.startswith(doc_id):
        chunk_id = raw_id
    else:
        chunk_id = f"{doc_id}:{raw_id}"
    fmt = str(block.get("original_format") or block.get("format") or "").strip().lower()
    page_number = _block_page(block, fmt)
    try:
        page_count = int(block.get("page_count") if block.get("page_count") not in (None, "") else (0 if page_number == 0 else 1))
    except (TypeError, ValueError):
        page_count = 0 if page_number == 0 else 1
    heading = str(block.get("heading_text") or block.get("heading") or block.get("section_title") or "")
    filename = str(block.get("doc_name") or "").strip() or str(fallback_name or "").strip() or doc_id
    chunk = {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "content": content,
        "doc_name": filename,
        "page_number": page_number,
        "page_count": page_count if page_count > 0 else (0 if page_number == 0 else 1),
        "section_title": str(block.get("section_title") or ""),
        "heading": heading,
        "parent_heading": str(block.get("parent_heading") or ""),
        "source_url": str(block.get("source_url") or block.get("citation_url") or ""),
        "source_type": str(block.get("source_type") or "text"),
        "highlight_text": str(block.get("highlight_text") or ""),
    }
    document_key = str(block.get("document_key") or block.get("documentKey") or "").strip()
    if document_key:
        chunk["document_key"] = document_key
    if fmt:
        chunk["original_format"] = fmt
    for source, target in (
        ("publisher", "publisher"),
        ("published_at", "published_at"),
        ("document_title", "document_title"),
    ):
        value = block.get(source)
        if value not in (None, ""):
            chunk[target] = value
    if isinstance(block.get("section_path"), list):
        chunk["section_path"] = [str(item) for item in block["section_path"] if str(item).strip()]
    elif isinstance(block.get("section_path"), str) and block.get("section_path").strip():
        chunk["section_path"] = block.get("section_path")
    if isinstance(block.get("locator"), dict):
        chunk["locator"] = dict(block["locator"])
    for field in ("references", "amends", "supersedes", "article", "paragraph", "record_id"):
        if block.get(field) not in (None, "", []):
            chunk[field] = block.get(field)
    return chunk


class GraphService:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.catalog = Catalog(self.directory / "catalog.json")
        self.projection_path = self.directory / "projection.json"
        self.projection = {"nodes": [], "edges": []}
        self.passage_ids: list[str] = []
        self.passage_matrix = None
        self.entity_ids: list[str] = []
        self.entity_matrix = None
        self.window_ids: list[str] = []
        self.window_matrix = None
        self.window_fingerprint = ""
        self._load_projection()
        self._load_vectors()
        self._load_window_index()

    def _load_projection(self) -> None:
        from knowledge.graph_rag.runtime import graphml_path, read_raw_graph

        if graphml_path(self.directory) is not None:
            try:
                raw_nodes, raw_edges = read_raw_graph(self.directory)
                projected = project_graph(raw_nodes, raw_edges, dict(self.catalog.source_map), dict(self.catalog.chunks))
                if projected["nodes"] or projected["edges"] or not self.projection_path.exists():
                    self.projection = projected
                    self._store_projection()
                    return
            except Exception as exc:
                logger.warning("从 LightRAG 图文件恢复失败: %s", exc)
        if self.projection_path.exists():
            try:
                payload = json.loads(self.projection_path.read_text(encoding="utf-8"))
                self.projection = {"nodes": payload.get("nodes") or [], "edges": payload.get("edges") or []}
                return
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("读取投影失败: %s", exc)
        self.projection = {"nodes": [], "edges": []}

    def _store_projection(self) -> None:
        _atomic_write(
            self.projection_path,
            {"nodes": self.projection.get("nodes") or [], "edges": self.projection.get("edges") or []},
        )

    def _load_vectors(self) -> None:
        matrix_path = self.directory / "vectors.npz"
        id_path = self.directory / "vector_ids.json"
        if not matrix_path.exists() or not id_path.exists():
            return
        try:
            import numpy as np

            payload = np.load(matrix_path)
            ids = json.loads(id_path.read_text(encoding="utf-8"))
            self.passage_matrix = payload["passages"] if "passages" in payload else None
            self.entity_matrix = payload["entities"] if "entities" in payload else None
            self.passage_ids = list(ids.get("passages") or [])
            self.entity_ids = list(ids.get("entities") or [])
        except Exception as exc:
            logger.warning("读取向量索引失败: %s", exc)

    def _load_window_index(self) -> None:
        """Load the retrieval window matrix. It is not part of the LightRAG graph."""
        meta_path = self.directory / "window_meta.json"
        matrix_path = self.directory / "window_vectors.npz"
        if not meta_path.exists() or not matrix_path.exists():
            return
        try:
            import numpy as np

            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            payload = np.load(matrix_path)
            self.window_ids = [str(item) for item in meta.get("ids") or []]
            self.window_fingerprint = str(meta.get("fingerprint") or "")
            self.window_matrix = payload["windows"] if "windows" in payload.files else None
        except Exception as exc:
            logger.warning("读取窗口索引失败: %s", exc)
            self.window_ids = []
            self.window_matrix = None
            self.window_fingerprint = ""

    def _window_fingerprint(self, chunks: dict) -> str:
        parts = []
        for chunk_id in sorted(chunks):
            content = str(chunks[chunk_id].get("content") or "")
            parts.append(chunk_id + ":" + hashlib.sha1(content.encode("utf-8")).hexdigest())
        return hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()

    def _build_window_index(self) -> None:
        """Embed every window of stored text. Does not rewrite LightRAG files or vectors.npz."""
        if os.environ.get("GRAPH_DISABLE_VECTORS") == "1":
            return
        with self.catalog.lock:
            chunks = {key: dict(value) for key, value in self.catalog.chunks.items()}
        from knowledge.graph_rag.hybrid_index import row_key, window_rows

        rows = window_rows(chunks)
        fingerprint = self._window_fingerprint(chunks)
        if not rows:
            self.window_ids = []
            self.window_matrix = None
            self.window_fingerprint = fingerprint
            return
        try:
            from knowledge.graph_rag.runtime import encode_texts
            import numpy as np

            matrix = encode_texts([row["text"] for row in rows])
        except Exception as exc:
            logger.warning("窗口索引未更新: %s", exc)
            self.window_matrix = None
            return
        ids = [row_key(row) for row in rows]
        matrix_path = self.directory / "window_vectors.npz"
        meta_path = self.directory / "window_meta.json"
        np.savez(matrix_path, windows=np.asarray(matrix, dtype=np.float32))
        _atomic_write(meta_path, {"ids": ids, "fingerprint": fingerprint})
        self.window_ids = ids
        self.window_matrix = matrix
        self.window_fingerprint = fingerprint

    def _ensure_window_index(self) -> None:
        if os.environ.get("GRAPH_DISABLE_VECTORS") == "1":
            return
        with self.catalog.lock:
            chunks = {key: dict(value) for key, value in self.catalog.chunks.items()}
        fingerprint = self._window_fingerprint(chunks)
        if self.window_matrix is not None and self.window_fingerprint == fingerprint and self.window_ids:
            return
        self._build_window_index()

    def refresh_projection(self) -> None:
        from knowledge.graph_rag.runtime import read_raw_graph

        try:
            raw_nodes, raw_edges = read_raw_graph(self.directory)
        except Exception as exc:
            logger.warning("读取 LightRAG 图失败，保留上一份投影: %s", exc)
            return
        with self.catalog.lock:
            self.projection = project_graph(raw_nodes, raw_edges, dict(self.catalog.source_map), dict(self.catalog.chunks))
            self._store_projection()

    def rebuild_vectors(self) -> None:
        if os.environ.get("GRAPH_DISABLE_VECTORS") == "1":
            return
        with self.catalog.lock:
            chunks = [dict(chunk) for chunk in self.catalog.chunks.values()]
            nodes = [dict(node) for node in self.projection.get("nodes") or []]
        if not chunks and not nodes:
            return
        try:
            from knowledge.graph_rag.runtime import encode_texts
            import numpy as np

            passage_ids = [chunk["chunk_id"] for chunk in chunks]
            # Legacy chunk prefixes stay in vectors.npz so a frozen baseline can reuse this directory.
            # Improved retrieval embeds full windows in window_vectors.npz instead of this prefix.
            passage_matrix = encode_texts([chunk["content"][:2000] for chunk in chunks]) if chunks else None
            entity_ids = [node["id"] for node in nodes]
            entity_text = [
                f"{node.get('label') or ''}\n{node.get('type') or ''}\n{str(node.get('description') or '')[:400]}"
                for node in nodes
            ]
            entity_matrix = encode_texts(entity_text) if entity_text else None
        except Exception as exc:
            logger.warning("向量索引未更新: %s", exc)
            return
        with self.catalog.lock:
            self.passage_ids = passage_ids
            self.passage_matrix = passage_matrix
            self.entity_ids = entity_ids
            self.entity_matrix = entity_matrix
            matrix_path = self.directory / "vectors.npz"
            id_path = self.directory / "vector_ids.json"
            save = {}
            if passage_matrix is not None:
                save["passages"] = np.asarray(passage_matrix, dtype=np.float32)
            if entity_matrix is not None:
                save["entities"] = np.asarray(entity_matrix, dtype=np.float32)
            if save:
                np.savez(matrix_path, **save)
            _atomic_write(id_path, {"passages": passage_ids, "entities": entity_ids})
        self._build_window_index()

    def _ensure_vectors(self) -> None:
        if os.environ.get("GRAPH_DISABLE_VECTORS") == "1":
            return
        with self.catalog.lock:
            dirty = len(self.passage_ids) != len(self.catalog.chunks) or self.passage_matrix is None
            has_chunks = bool(self.catalog.chunks)
        if has_chunks and dirty:
            self.rebuild_vectors()
            return
        self._ensure_window_index()

    def status_payload(self) -> dict:
        with self.catalog.lock:
            documents = self.catalog.public_documents()
            build = dict(self.catalog.build)
            chunk_count = len(self.catalog.chunks)
            nodes = list(self.projection.get("nodes") or [])
            edges = list(self.projection.get("edges") or [])
        building = bool(build.get("building"))
        error = str(build.get("error") or "").strip()
        if building:
            status = "building"
        elif nodes:
            status = "ready"
        elif error:
            status = "error"
        elif chunk_count:
            status = "stored"
        else:
            status = "empty"
        structure = build_structure(self.catalog.documents, self.catalog.chunks, nodes)
        structured_docs = {node.get("docId") for node in structure["nodes"] if node.get("type") == "document" and node.get("sourceIds")}
        semantic_docs = set()
        for node in nodes:
            for source_id in node.get("sourceIds") or []:
                chunk = self.catalog.chunks.get(source_id)
                if not chunk:
                    continue
                doc_id = str(chunk.get("doc_id") or "")
                if doc_id:
                    semantic_docs.add(doc_id)
        payload = {
            "status": status,
            "documents": len(documents),
            "nodes": len(nodes),
            "edges": len(edges),
            "chunks": chunk_count,
            "building": building,
            "completedDocuments": int(build.get("completedDocuments") or 0),
            "totalDocuments": int(build.get("totalDocuments") or 0),
            "structureNodes": len(structure["nodes"]),
            "semanticNodes": len(nodes),
            "structuredDocuments": len(structured_docs),
            "semanticDocuments": len(semantic_docs),
        }
        if error and not building:
            payload["error"] = error
        coverage = self._coverage_payload()
        if coverage is not None:
            payload["graphCoverage"] = coverage
        return payload

    def documents_payload(self) -> list[dict]:
        try:
            self.register_persisted_managed_documents()
        except Exception as exc:
            logger.warning("读取已保存文档目录失败: %s", exc)
        with self.catalog.lock:
            return self.catalog.public_documents()

    def register_persisted_managed_documents(self, records: Optional[list] = None) -> int:
        """Register managed library metadata already stored by the application.

        ``records is None`` reads the management JSON directly. This does not
        take the state-store lock and does not merge or rewrite that file.
        File bytes are not opened; extraction stays on an explicit build.
        """
        if records is None:
            records = _persisted_document_records()
        upload_dir = BACKEND_ROOT / "aerospace_docs"
        changed = 0
        with self.catalog.lock:
            for record in records or []:
                if not isinstance(record, dict):
                    continue
                storage_name = str(record.get("storageFileName") or "").strip()
                if not storage_name:
                    continue
                doc_id = managed_id(storage_name)
                if not doc_id:
                    continue
                storage_path = str(upload_dir / Path(storage_name).name)
                name = str(record.get("title") or record.get("fileName") or Path(storage_name).name)
                metadata = {
                    "sourceUrl": str(record.get("sourceUrl") or record.get("source_url") or ""),
                    "title": str(record.get("title") or ""),
                    "publisher": str(record.get("publisher") or ""),
                    "publishedAt": str(record.get("publishedAt") or ""),
                    "originalFormat": str(record.get("originalFormat") or ""),
                    "documentKey": str(record.get("documentKey") or record.get("document_key") or ""),
                }
                doc_id = canonical_document_key(
                    document_key=metadata["documentKey"],
                    source_url=metadata["sourceUrl"],
                    file_name=storage_name,
                ) or doc_id
                self.catalog.fold_into(managed_id(storage_name), doc_id)
                if self.catalog.register_metadata(doc_id, name, storage_path, metadata):
                    changed += 1
            if changed:
                self.catalog.save()
        return changed

    def register_managed(self, record: dict, storage_path: str = "") -> dict:
        file_name = str(record.get("storageFileName") or record.get("fileName") or "")
        metadata = {
            "title": str(record.get("title") or ""),
            "publisher": str(record.get("publisher") or ""),
            "publishedAt": str(record.get("publishedAt") or ""),
            "originalFormat": str(record.get("originalFormat") or ""),
            "documentKey": str(record.get("documentKey") or record.get("document_key") or ""),
            "sourceUrl": str(record.get("sourceUrl") or record.get("source_url") or ""),
        }
        doc_id = canonical_document_key(
            document_key=metadata["documentKey"],
            source_url=metadata["sourceUrl"],
            file_name=file_name,
        )
        if not doc_id:
            return {}
        with self.catalog.lock:
            alias = managed_id(file_name)
            self.catalog.fold_into(alias, doc_id)
            self.catalog.register(doc_id, str(record.get("title") or record.get("fileName") or file_name), metadata["sourceUrl"], storage_path, metadata)
        return {"id": doc_id}

    def store_managed_blocks(self, file_name: str, blocks: list) -> dict:
        sample = next((block for block in blocks or [] if isinstance(block, dict)), {})
        doc_id = canonical_document_key(
            document_key=str(sample.get("document_key") or sample.get("documentKey") or ""),
            source_url=str(sample.get("source_url") or sample.get("sourceUrl") or ""),
            file_name=file_name,
        )
        if not doc_id:
            return {"id": "", "chunks": 0, "building": False}
        name = str(sample.get("document_title") or Path(file_name).name)
        chunks = [chunk for block in blocks or [] if (chunk := _block_chunk(doc_id, block, name))]
        source_url = next((str(chunk.get("source_url") or "") for chunk in chunks if chunk.get("source_url")), "")
        with self.catalog.lock:
            self.catalog.fold_into(managed_id(file_name), doc_id)
            self.catalog.replace_chunks(doc_id, name, source_url, chunks, "managed")
        if auto_build_enabled() and chunks:
            self.start_build([doc_id], scope="all")
        return {"id": doc_id, "chunks": len(chunks), "building": bool(self.catalog.build.get("building"))}

    def delete_document(self, doc_name: str) -> dict:
        with self.catalog.lock:
            targets = []
            primary = self.catalog.canonical_id(doc_name)
            targets.append(primary)
            alternate = managed_id(doc_name)
            if alternate and alternate not in targets:
                targets.append(alternate)
            requested = str(doc_name or "")
            if requested:
                for stored_id, document in self.catalog.documents.items():
                    if stored_id in targets or not isinstance(document, dict):
                        continue
                    legacy_name = str(document.get("name") or "")
                    basename = Path(str(document.get("storagePath") or "")).name
                    if requested == legacy_name or (basename and requested == basename):
                        targets.append(stored_id)
            removed = []
            seen_targets = set()
            for doc_id in targets:
                if doc_id in seen_targets or doc_id not in self.catalog.documents:
                    continue
                seen_targets.add(doc_id)
                removed.extend(self.catalog.delete(doc_id))
            doc_id = primary
        self.window_ids = []
        self.window_matrix = None
        self.window_fingerprint = ""
        self.refresh_projection()
        from knowledge.graph_rag import runtime

        if removed and Path(runtime.data_dir()).resolve() == self.directory.resolve():
            try:
                runtime.run_async(runtime.delete_lightrag_docs(list(dict.fromkeys(removed))))
                self.refresh_projection()
            except Exception as exc:
                logger.warning("LightRAG 删除未完成: %s", exc)
        return {"id": doc_id, "removed": len(set(removed))}

    def evidence_payload(self, chunk_id: str, query: str = "", doc_ids: Optional[list[str]] = None) -> Optional[dict]:
        doc_ids = normalize_doc_ids(doc_ids)
        with self.catalog.lock:
            chunk = self.catalog.chunk(chunk_id)
        if not chunk:
            return None
        if doc_ids is not None and chunk.get("doc_id") not in set(doc_ids):
            return None
        return _evidence(chunk, lexical_score(query, chunk.get("content") or "") if query else 0.0, query)

    def search(self, query: str, doc_ids: Optional[list[str]] = None, limit: Optional[int] = None, scope: Optional[str] = None) -> dict:
        doc_ids = normalize_doc_ids(doc_ids)
        query = str(query or "")
        chosen = _scope_name(scope)
        # Hierarchy is a catalog walk. An empty browse, and any hierarchy query, stays off the embedding model.
        if chosen != "hierarchy" and query.strip():
            self._ensure_vectors()
        snapshot = self._snapshot()
        return public_graph(_scoped_search(snapshot, query, doc_ids, _limit(limit), chosen))

    def expand(self, node_ids: list[str], hops: int, doc_ids: Optional[list[str]] = None, scope: Optional[str] = None) -> dict:
        doc_ids = normalize_doc_ids(doc_ids)
        chosen = _scope_name(scope)
        snapshot = self._snapshot()
        if chosen == "hierarchy":
            graph = _structure_graph(snapshot, doc_ids)
        elif chosen == "all":
            semantic = _visible_graph(snapshot, doc_ids)
            structure = _structure_graph(snapshot, doc_ids)
            graph = {
                "nodes": list(semantic.get("nodes") or []) + list(structure.get("nodes") or []),
                "edges": list(semantic.get("edges") or []) + list(structure.get("edges") or []),
            }
        else:
            graph = _visible_graph(snapshot, doc_ids)
        expanded = expand_hops(graph, node_ids, hops, chosen)
        if int(hops) > MAX_PATH_EDGES:
            expanded["truncated"] = True
        return public_graph(expanded)

    def ask(self, question: str, mode: str, doc_ids: Optional[list[str]] = None, diagnostics: bool = False) -> dict:
        doc_ids = normalize_doc_ids(doc_ids)
        status = self.status_payload()
        # Stored passages are evidence for every mode. LightRAG entities are optional.
        if status["chunks"] <= 0 and status["nodes"] <= 0:
            return _unavailable(status)
        self._ensure_vectors()
        snapshot = self._snapshot()
        tools = _tools(snapshot)
        action_fn = _planner_model if mode == "agent" else None

        def answer_fn(state: dict) -> dict:
            return _answer_from_evidence(state)

        result = run_agent(question, doc_ids, mode, tools, answer_fn, action_fn)
        sources = list(result.get("evidence") or [])
        answer = str(result.get("answer") or "")
        stop_reason = str(result.get("stop_reason") or "insufficient_evidence")
        if stop_reason == "answered" and int(result.get("tool_calls") or 0) >= budget_for(result):
            stop_reason = "tool_budget"
        graph = public_graph(merge_graphs(list(result.get("graph_parts") or []), _seed_nodes(result)))
        trace = [_public_step(step, index) for index, step in enumerate(result.get("trace") or [], start=1)]
        citations = citations_from_answer(answer, sources) if stop_reason in {"answered", "tool_budget", "partial_evidence"} else []
        payload = {
            "answer": answer,
            "sources": [_public_evidence(row) for row in sources],
            "citations": citations,
            "graph": graph,
            "trace": trace,
            "stopReason": stop_reason,
        }
        if diagnostics:
            payload["diagnostics"] = _diagnostics_payload(snapshot, result)
        return payload

    def start_build(self, doc_ids: list[str], scope: Optional[str] = None) -> dict:
        cleaned = [str(doc_id).strip() for doc_id in doc_ids if str(doc_id).strip()]
        if not cleaned:
            raise ValueError("docIds 不能为空")
        with self.catalog.lock:
            self.catalog.ensure_corpus()
            if self.catalog.build.get("building"):
                return self.status_payload()
            missing = [doc_id for doc_id in cleaned if not self.catalog.knows(doc_id)]
            if missing:
                raise ValueError("未知文档: " + ", ".join(missing))
            canonical = [self.catalog.canonical_id(doc_id) for doc_id in cleaned]
            self.catalog.set_build(
                building=True,
                status="building",
                completedDocuments=0,
                totalDocuments=len(canonical),
                error="",
                currentDoc="",
            )
        chosen_scope = scope or extract_scope()
        from knowledge.graph_rag.runtime import event_loop

        future = asyncio.run_coroutine_threadsafe(self._build(canonical, chosen_scope), event_loop())

        def _finished(done) -> None:
            try:
                done.result()
            except Exception as exc:
                logger.exception("图谱构建任务失败")
                self.catalog.set_build(building=False, status="error", error=str(exc)[:500])
                self._write_progress("pilot_extraction", "failed")

        future.add_done_callback(_finished)
        return self.status_payload()

    def build_blocking(self, doc_ids: list[str], scope: Optional[str] = None, timeout: float = 86400) -> dict:
        import time

        self.start_build(doc_ids, scope=scope)
        deadline = time.time() + timeout
        while self.status_payload().get("building"):
            if time.time() > deadline:
                raise TimeoutError("图谱构建超时")
            time.sleep(1)
        return self.status_payload()

    async def _build(self, doc_ids: list[str], scope: str) -> None:
        from knowledge.graph_rag.runtime import insert_windows, prepared_windows

        errors: list[str] = []
        try:
            for index, doc_id in enumerate(doc_ids):
                self.catalog.set_build(currentDoc=doc_id, completedDocuments=index)
                self._write_progress("pilot_extraction", "running")
                try:
                    chunks, report = self._resolve_chunks(doc_id, scope)
                except Exception as exc:
                    logger.exception("解析文档失败 %s", doc_id)
                    errors.append(f"{doc_id}: {exc}")
                    self._mark_doc_finished(index + 1)
                    continue
                if not chunks:
                    errors.append(f"{doc_id}: 没有可抽取的原文")
                    self._mark_doc_finished(index + 1)
                    continue
                _, stored_chunks = self._stored_product(doc_id)
                with self.catalog.lock:
                    # Scope only chooses extraction input. Keep an existing product catalog.
                    if not stored_chunks:
                        meta = self._doc_meta(doc_id)
                        self.catalog.replace_chunks(doc_id, meta["name"], meta["sourceUrl"], chunks, meta["kind"], meta["storagePath"])
                    extracted = set(self.catalog.documents.get(doc_id, {}).get("extractedChunkIds") or [])
                for chunk in chunks:
                    chunk_id = chunk["chunk_id"]
                    if chunk_id in extracted:
                        continue
                    try:
                        windows = prepared_windows(chunk_id, chunk["content"])
                        if not windows:
                            continue
                        with self.catalog.lock:
                            self.catalog.add_source_map({window_id: chunk_id for window_id, _text in windows})
                        await insert_windows(chunk_id, [text for _window_id, text in windows])
                        with self.catalog.lock:
                            self.catalog.mark_extracted(doc_id, chunk_id)
                        self.refresh_projection()
                    except Exception as exc:
                        logger.exception("抽取失败 %s", chunk_id)
                        errors.append(f"{chunk_id}: {exc}")
                with self.catalog.lock:
                    self.catalog.mark_document_done(doc_id)
                    self._store_coverage(doc_id, report)
                self._mark_doc_finished(index + 1)
            self.refresh_projection()
            await asyncio.to_thread(self.rebuild_vectors)
        finally:
            error = "; ".join(errors)[:800]
            nodes = len(self.projection.get("nodes") or [])
            status = "ready" if nodes else ("error" if error else "empty")
            self.catalog.set_build(building=False, status=status, error=error, currentDoc="")
            self._write_progress("pilot_extraction", "done" if nodes else "failed")

    def _mark_doc_finished(self, completed: int) -> None:
        self.catalog.set_build(completedDocuments=completed)
        self._write_progress("pilot_extraction", "running")

    def _stored_product(self, doc_id: str) -> tuple[dict, list[dict]]:
        with self.catalog.lock:
            current = dict(self.catalog.documents.get(doc_id) or {})
            chunks = [dict(chunk) for chunk in self.catalog.chunks.values() if chunk.get("doc_id") == doc_id]
        return current, chunks


    def _doc_meta(self, doc_id: str) -> dict:
        current, chunks = self._stored_product(doc_id)
        if chunks or not is_corpus(doc_id):
            return {
                "name": current.get("name") or doc_id,
                "sourceUrl": current.get("sourceUrl") or "",
                "kind": current.get("kind") or "managed",
                "storagePath": current.get("storagePath") or "",
            }
        meta = manifest_by_id().get(doc_id) or {}
        return {
            "name": meta.get("title") or doc_id,
            "sourceUrl": meta.get("source_url") or "",
            "kind": "corpus",
            "storagePath": "",
        }


    def _resolve_chunks(self, doc_id: str, scope: str) -> tuple[list[dict], dict]:
        current, stored = self._stored_product(doc_id)
        if not stored and is_corpus(doc_id):
            report = selection_report(doc_id, scope)
            return chunks_for(doc_id, scope), report
        rows = stored or self._parse_storage(current)
        available = len(rows)
        if scope == "smoke":
            rows = rows[:2]
        report = {
            "scope": scope,
            "available": available,
            "selected": len(rows),
            "sampled": len(rows) < available,
            "fullText": scope == "all" and available > 0 and len(rows) == available,
        }
        return rows, report

    def _store_coverage(self, doc_id: str, report: dict) -> None:
        document = self.catalog.documents.get(doc_id)
        if not document or not report:
            return
        extracted = len(document.get("extractedChunkIds") or [])
        document["extractScope"] = report.get("scope") or ""
        document["sourceUnits"] = int(report.get("available") or 0)
        document["selectedUnits"] = int(report.get("selected") or 0)
        document["graphCoverage"] = describe_coverage(
            str(report.get("scope") or ""),
            int(report.get("available") or 0),
            int(report.get("selected") or 0),
            extracted,
        )
        self.catalog.save()

    def _coverage_payload(self) -> Optional[dict]:
        with self.catalog.lock:
            rows = []
            for document in self.catalog.documents.values():
                coverage = document.get("graphCoverage")
                if isinstance(coverage, dict):
                    rows.append({"id": str(document.get("id") or ""), **coverage})
        if not rows:
            return None
        full = all(bool(row.get("fullGraph")) for row in rows)
        return {
            "full": full,
            "documents": rows,
            "note": (
                "full means every built document extracted all parsed units"
                if full
                else "partial graph coverage; not a full knowledge graph"
            ),
        }

    def _parse_storage(self, document: dict) -> list[dict]:
        path = str(document.get("storagePath") or "")
        if not path or not Path(path).exists():
            return []
        from search.document_pipeline import extract_document_units

        units, _health = extract_document_units(path)
        doc_id = str(document.get("id") or "")
        chunks = [_block_chunk(doc_id, unit, document.get("name") or doc_id) for unit in units]
        return [chunk for chunk in chunks if chunk]

    def _snapshot(self) -> dict:
        with self.catalog.lock:
            documents = {key: dict(value) for key, value in self.catalog.documents.items()}
            chunks = {key: dict(value) for key, value in self.catalog.chunks.items()}
            nodes = [dict(node) for node in self.projection.get("nodes") or []]
            edges = [dict(edge) for edge in self.projection.get("edges") or []]
            passage_ids = list(self.passage_ids)
            passage_matrix = None if self.passage_matrix is None else self.passage_matrix.copy()
            entity_ids = list(self.entity_ids)
            entity_matrix = None if self.entity_matrix is None else self.entity_matrix.copy()
            window_ids = list(self.window_ids)
            window_matrix = None if self.window_matrix is None else self.window_matrix.copy()
        return {
            "chunks": chunks,
            "structure": build_structure(documents, chunks, nodes),
            "graph": {"nodes": nodes, "edges": edges},
            "passage_ids": passage_ids,
            "passage_matrix": passage_matrix,
            "entity_ids": entity_ids,
            "entity_matrix": entity_matrix,
            "window_ids": window_ids,
            "window_matrix": window_matrix,
        }

    def _write_progress(self, rung: str, state: str) -> None:
        path = progress_path()
        try:
            payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except (OSError, json.JSONDecodeError):
            payload = {}
        build = self.catalog.build
        payload["updated_at"] = _now()
        rungs = payload.setdefault("rungs", {})
        rungs["spine"] = "done"
        rungs["api"] = rungs.get("api") or "not_started"
        rungs[rung] = state
        payload["build"] = {
            "phase": rung,
            "state": state,
            "current_doc": build.get("currentDoc") or "",
            "completed_documents": int(build.get("completedDocuments") or 0),
            "total_documents": int(build.get("totalDocuments") or 0),
            "error": build.get("error") or "",
            "nodes": len(self.projection.get("nodes") or []),
            "edges": len(self.projection.get("edges") or []),
            "chunks": len(self.catalog.chunks),
        }
        coverage = self._coverage_payload()
        if coverage is not None:
            payload["graphCoverage"] = coverage
        try:
            _atomic_write(path, payload)
        except OSError as exc:
            logger.warning("写入 graph-progress 失败: %s", exc)


def normalize_doc_ids(doc_ids: Optional[list[str]]) -> Optional[list[str]]:
    """Empty selections mean every document. The graph UI sends docIds: []."""
    if not doc_ids:
        return None
    cleaned = []
    for item in doc_ids:
        text = str(item or "").strip()
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned or None


def _limit(limit: Optional[int]) -> int:
    if limit is None:
        return DEFAULT_SEARCH_LIMIT
    try:
        value = int(limit)
    except (TypeError, ValueError):
        return DEFAULT_SEARCH_LIMIT
    return max(1, min(value, MAX_SEARCH_LIMIT))


def _visible_graph(snapshot: dict, doc_ids: Optional[list[str]]) -> dict:
    return filter_graph(snapshot["graph"], doc_ids, snapshot["chunks"])


def _scope_name(scope: Optional[str]) -> str:
    value = str(scope or "semantic")
    return value if value in {"hierarchy", "semantic", "all"} else "semantic"


def _structure_graph(snapshot: dict, doc_ids: Optional[list[str]]) -> dict:
    return filter_graph(snapshot.get("structure") or {"nodes": [], "edges": []}, doc_ids, snapshot.get("chunks") or {})


def _scoped_search(snapshot: dict, query: str, doc_ids: Optional[list[str]], limit: int, scope: str) -> dict:
    if scope == "hierarchy":
        return select_structure(_structure_graph(snapshot, doc_ids), query, limit)
    graph, _sources, _labels = _search_graph(
        snapshot,
        query,
        doc_ids,
        limit,
        "all" if scope == "all" else "semantic",
    )
    return graph


def _structure_source_ids(snapshot: dict, query: str, doc_ids: Optional[list[str]], limit: int) -> list[str]:
    """Passages for a matched parent/section, plus the citing side of an explicit reference.

    A document hub stores every child id for rollup. Those ids are not copied into evidence.
    REFERS_TO contributes the passage that states the reference, not the target document.
    """
    limit = max(int(limit), 0)
    if limit == 0:
        return []
    view = select_structure(_structure_graph(snapshot, doc_ids), query, limit)
    chunks = snapshot.get("chunks") or {}
    matched: set[str] = set()
    for node in view.get("nodes") or []:
        locator = node.get("locator") if isinstance(node.get("locator"), dict) else {}
        label_score = lexical_score(query, str(node.get("label") or ""))
        locator_score = lexical_score(query, str(locator.get("label") or ""))
        if label_score > 0 or locator_score > 0:
            matched.add(node["id"])
    found: list[str] = []

    def add(source_id: str) -> None:
        if source_id and source_id in chunks and source_id not in found and len(found) < limit:
            found.append(source_id)

    leaves = {"chunk", "table", "figure"}
    for node in view.get("nodes") or []:
        if node.get("id") not in matched or node.get("type") in {"document", "source"}:
            continue
        source_ids = [str(item) for item in node.get("sourceIds") or []]
        if node.get("type") not in leaves:
            source_ids = sorted(
                (
                    source_id
                    for source_id in source_ids
                    if lexical_score(query, str((chunks.get(source_id) or {}).get("content") or "")) > 0
                ),
                key=lambda source_id: lexical_score(query, str(chunks[source_id].get("content") or "")),
                reverse=True,
            )
        for source_id in source_ids:
            add(source_id)
    for edge in view.get("edges") or []:
        if edge.get("edgeKind") != "references":
            continue
        for source_id in edge.get("sourceIds") or []:
            add(str(source_id))
        for item in edge.get("provenance") or []:
            if isinstance(item, dict):
                add(str(item.get("sourceId") or ""))
    return found


def _vector_scores(matrix, ids: list[str], query: str, allowed: set[str]) -> dict[str, float]:
    if matrix is None or not ids:
        return {}
    try:
        from knowledge.graph_rag.runtime import encode_texts

        vector = encode_texts([query])[0]
        scores = matrix @ vector
    except Exception as exc:
        logger.warning("查询向量失败: %s", exc)
        return {}
    found = {}
    for index, chunk_id in enumerate(ids):
        if allowed is not None and chunk_id not in allowed:
            continue
        score = float(scores[index])
        if score > 0:
            found[chunk_id] = score
    return found


def _passage_ranking(snapshot: dict, query: str, doc_ids: Optional[list[str]]) -> list[tuple[str, float]]:
    """Best window per original unit, ordered by reciprocal rank fusion."""
    limits = retrieval_limits(snapshot)
    spans = retrieve_spans(
        snapshot,
        query,
        doc_ids,
        limit=limits["candidates"],
        rerank_pool=limits["rerank_pool"],
    )
    return unit_ranking(spans)


def _graph_source_ids(graph: dict) -> list[str]:
    found: list[str] = []
    for item in list(graph.get("nodes") or []) + list(graph.get("edges") or []):
        for source_id in item.get("sourceIds") or []:
            if source_id not in found:
                found.append(source_id)
    return found


def _bounded_projection(graph: dict, limit: int) -> tuple[dict, list[str], dict]:
    """Return a stored neighborhood for an empty browse query. No embedding."""
    nodes = list(graph.get("nodes") or [])
    truncated = len(nodes) > limit
    chosen_ids = [node["id"] for node in nodes[:limit]]
    if not chosen_ids:
        return empty_graph(), [], {}
    subgraph = induced_subgraph(graph, chosen_ids, chosen_ids[:6])
    if truncated:
        subgraph["truncated"] = True
    labels = {node["id"]: node.get("label") or node["id"] for node in subgraph.get("nodes") or []}
    return subgraph, _graph_source_ids(subgraph), labels


def _search_graph(snapshot: dict, query: str, doc_ids: Optional[list[str]], limit: int, scope: str = "semantic"):
    scope = "all" if scope == "all" else "semantic"
    graph = _visible_graph(snapshot, doc_ids)
    seed_ids: list[str] = []
    if not str(query or "").strip():
        subgraph, ordered, labels = _bounded_projection(graph, limit)
    else:
        limits = retrieval_limits(snapshot)
        spans = retrieve_spans(
            snapshot,
            query,
            doc_ids,
            limit=limits["candidates"],
            rerank_pool=limits["rerank_pool"],
        )
        seed_ids = [unit_id for unit_id, _score in unit_ranking(spans)]
        seed_set = set(seed_ids)
        user_filter = doc_ids is not None
        named = None if user_filter else named_doc_ids(query, snapshot.get("chunks") or {})
        named_set = set(named or [])
        allowed_nodes = {node["id"] for node in graph["nodes"]}
        entity_scores = _vector_scores(snapshot.get("entity_matrix"), snapshot.get("entity_ids") or [], query, allowed_nodes)
        ranked = []
        for node in graph["nodes"]:
            sources = [str(item) for item in node.get("sourceIds") or []]
            source_docs = {
                str((snapshot.get("chunks") or {}).get(source_id, {}).get("doc_id") or "")
                for source_id in sources
            }
            if named_set and not (source_docs & named_set) and not (set(sources) & seed_set):
                continue
            blob = f"{node.get('label') or ''} {node.get('description') or ''}"
            owns_seed = bool(set(sources) & seed_set)
            lexical = lexical_score(query, blob)
            score = max(lexical, entity_scores.get(node["id"], 0.0))
            if owns_seed:
                score += 0.45
            if score > 0 or owns_seed:
                ranked.append((score, node))
        ranked.sort(key=lambda item: -item[0])
        chosen = [node for _score, node in ranked[:limit]]
        chosen_ids = [node["id"] for node in chosen]
        subgraph = induced_subgraph(graph, chosen_ids, chosen_ids[:6]) if chosen_ids else empty_graph()
        if len(ranked) > limit:
            subgraph["truncated"] = True
        labels = {node["id"]: node.get("label") or node["id"] for node in subgraph.get("nodes") or []}
        ordered = list(seed_ids)
        for source_id in _graph_source_ids(subgraph):
            if source_id in ordered:
                continue
            chunk = (snapshot.get("chunks") or {}).get(source_id)
            if not chunk:
                continue
            ordered.append(source_id)
    if scope != "all":
        return subgraph, ordered, labels
    structure = select_structure(_structure_graph(snapshot, doc_ids), query, limit, seed_ids or None)
    merged = {
        "nodes": list(subgraph.get("nodes") or []) + list(structure.get("nodes") or []),
        "edges": list(subgraph.get("edges") or []) + list(structure.get("edges") or []),
    }
    seeds = [node["id"] for node in list(structure.get("nodes") or [])[:6]]
    seen = set(seeds)
    for node in list(subgraph.get("nodes") or [])[:6]:
        if node["id"] not in seen:
            seeds.append(node["id"])
            seen.add(node["id"])
    result = graph_with_paths(merged, seeds or [node["id"] for node in merged["nodes"][:6]], scope="all")
    result["truncated"] = bool(subgraph.get("truncated") or structure.get("truncated") or result.get("truncated"))
    labels = {node["id"]: node.get("label") or node["id"] for node in result.get("nodes") or []}
    return result, ordered, labels


def _evidence(chunk: dict, score: float, query: str) -> dict:
    raw_page = chunk.get("page_number")
    page_number = int(raw_page) if raw_page not in (None, "") else 0
    raw_count = chunk.get("page_count")
    page_count = int(raw_count) if raw_count not in (None, "") else (0 if page_number == 0 else 1)
    payload = {
        "chunk_id": chunk["chunk_id"],
        "content": chunk.get("content") or "",
        "doc_name": chunk.get("doc_name") or "",
        "page_number": page_number,
        "section_title": chunk.get("section_title") or "",
        "source_url": chunk.get("source_url") or "",
        "heading": chunk.get("heading") or "",
        "parent_heading": chunk.get("parent_heading") or "",
        "score": float(score),
        "page_count": page_count,
        "highlight_text": chunk.get("highlight_text") or highlight(chunk.get("content") or "", query),
        "source_type": chunk.get("source_type") or "text",
    }
    document_title = str(chunk.get("document_title") or "").strip()
    if document_title:
        payload["document_title"] = document_title
    doc_id = str(chunk.get("doc_id") or "").strip()
    if doc_id:
        payload["doc_id"] = doc_id
    document_key = str(chunk.get("document_key") or "").strip()
    if document_key:
        payload["document_key"] = document_key
    locator = chunk.get("locator") if isinstance(chunk.get("locator"), dict) else None
    if locator and str(locator.get("label") or "").strip():
        payload["locator"] = dict(locator)
    return payload


def _public_evidence(row: dict) -> dict:
    raw_page = row.get("page_number")
    page_number = int(raw_page) if raw_page not in (None, "") else 0
    raw_count = row.get("page_count")
    page_count = int(raw_count) if raw_count not in (None, "") else (0 if page_number == 0 else 1)
    payload = {
        "chunk_id": str(row.get("chunk_id") or ""),
        "content": str(row.get("content") or ""),
        "doc_name": str(row.get("doc_name") or ""),
        "page_number": page_number,
        "section_title": str(row.get("section_title") or ""),
        "heading": str(row.get("heading") or ""),
        "parent_heading": str(row.get("parent_heading") or ""),
        "score": float(row.get("score") or 0.0),
        "page_count": page_count,
        "highlight_text": str(row.get("highlight_text") or ""),
        "source_type": str(row.get("source_type") or "text"),
    }
    source_url = str(row.get("source_url") or "")
    if source_url:
        payload["source_url"] = source_url
    document_title = str(row.get("document_title") or "").strip()
    if document_title:
        payload["document_title"] = document_title
    doc_id = str(row.get("doc_id") or "")
    if doc_id:
        payload["doc_id"] = doc_id
    document_key = str(row.get("document_key") or "")
    if document_key:
        payload["document_key"] = document_key
    locator = row.get("locator") if isinstance(row.get("locator"), dict) else None
    if locator and str(locator.get("label") or "").strip():
        payload["locator"] = dict(locator)
    return payload


def _public_step(step: dict, number: int) -> dict:
    return {
        "step": number,
        "action": str(step.get("action") or ""),
        "query": str(step.get("query") or ""),
        "summary": str(step.get("summary") or ""),
        "nodeIds": [str(item) for item in step.get("nodeIds") or []],
        "sourceIds": [str(item) for item in step.get("sourceIds") or []],
    }


def _diagnostic_span(span: dict) -> dict:
    return {
        "chunkId": str(span.get("unit_id") or span.get("chunk_id") or span.get("chunkId") or ""),
        "docId": str(span.get("doc_id") or span.get("docId") or ""),
        "start": int(span.get("start") or 0),
        "end": int(span.get("end") or 0),
        "text": str(span.get("text") or ""),
    }


def _diagnostics_payload(snapshot: dict, result: dict) -> dict:
    """Budgets and spans already produced by retrieval and answer packing."""
    limits = retrieval_limits(snapshot, result)
    packed = result.get("context_limits") or limits
    candidates = []
    for observation in result.get("observations") or []:
        for span in observation.get("spans") or []:
            candidates.append(_diagnostic_span(span))
    return {
        "budgets": {
            "candidates": int(limits["candidates"]),
            "rerankPool": int(limits["rerank_pool"]),
            "contextChars": int(packed["context_chars"]),
            "passageChars": int(packed["passage_chars"]),
            "toolCalls": budget_for(result),
        },
        "candidateSpans": candidates,
        "contextSpans": [_diagnostic_span(span) for span in result.get("context_spans") or []],
    }


def _seed_nodes(result: dict) -> list[str]:
    for observation in result.get("observations") or []:
        if observation.get("nodeIds"):
            return list(observation["nodeIds"])[:6]
    return []


def _passage_excerpts(evidence: Optional[list], source_ids: list[str], query: str = "", spans: Optional[list] = None) -> list[dict]:
    """Show this tool's sources in the order it ranked them, not older evidence order."""
    span_by_unit = {}
    for span in spans or []:
        unit_id = str(span.get("unit_id") or "")
        if unit_id and unit_id not in span_by_unit:
            span_by_unit[unit_id] = span
    by_id = {}
    for row in evidence or []:
        chunk_id = str(row.get("chunk_id") or "")
        if chunk_id and chunk_id not in by_id:
            by_id[chunk_id] = row
    rows = []
    ordered = [str(item) for item in source_ids if str(item).strip()] or list(span_by_unit)
    for chunk_id in ordered:
        span = span_by_unit.get(chunk_id)
        row = by_id.get(chunk_id)
        if span is not None:
            text = highlight(str(span.get("text") or ""), query, 180)
            excerpt = {
                "chunk_id": chunk_id,
                "excerpt": text,
                "start": int(span.get("start") or 0),
                "end": int(span.get("end") or 0),
                "doc_name": str(span.get("doc_name") or ""),
                "section": str(span.get("section") or ""),
                "page": int(span["page"]) if span.get("page") is not None else 1,
            }
        elif row is not None:
            text = highlight(str(row.get("content") or ""), query, 180)
            excerpt = {"chunk_id": chunk_id, "excerpt": text}
        else:
            continue
        if not excerpt.get("excerpt"):
            continue
        rows.append(excerpt)
        if len(rows) >= 4:
            break
    return rows


def _bind_budget(snapshot: dict, state: dict) -> None:
    budget = dict(snapshot.get("retrieval_budget") or {})
    budget.update(state.get("retrieval_budget") or {})
    if budget:
        snapshot["retrieval_budget"] = budget


def _public_spans(spans: list[dict]) -> list[dict]:
    rows = []
    for span in spans or []:
        rows.append(
            {
                "unit_id": str(span.get("unit_id") or ""),
                "start": int(span.get("start") or 0),
                "end": int(span.get("end") or 0),
                "text": str(span.get("text") or ""),
                "doc_id": str(span.get("doc_id") or ""),
                "doc_name": str(span.get("doc_name") or ""),
                "section": str(span.get("section") or ""),
                "page": int(span["page"]) if span.get("page") is not None else 1,
                "score": float(span.get("score") or 0.0),
                "ranks": dict(span.get("ranks") or {}),
            }
        )
    return rows


def _attach_evidence(snapshot: dict, state: dict, source_ids: list[str], query: str, scores: Optional[dict] = None) -> list:
    """Keep every original passage a tool already touched, including bridge edges."""
    evidence = []
    seen = set()
    for row in state.get("evidence") or []:
        chunk_id = str(row.get("chunk_id") or "")
        if chunk_id and chunk_id not in seen:
            evidence.append(row)
            seen.add(chunk_id)
    allowed = None if state.get("doc_ids") is None else set(state.get("doc_ids") or [])
    score_map = dict(scores or {})
    for observation in state.get("observations") or []:
        score_map.update(observation.get("scores") or {})
    for chunk_id in source_ids:
        if chunk_id in seen:
            continue
        chunk = snapshot["chunks"].get(chunk_id)
        if not chunk:
            continue
        if allowed is not None and chunk.get("doc_id") not in allowed:
            continue
        score = score_map.get(chunk_id)
        if score is None:
            score = lexical_score(query, chunk.get("content") or "")
        evidence.append(_evidence(chunk, float(score), query))
        seen.add(chunk_id)
    return evidence


def _record(state: dict, action: str, query: str, summary: str, node_ids: list[str], source_ids: list[str], labels: dict, graph: Optional[dict], evidence: Optional[list] = None, scores: Optional[dict] = None, spans: Optional[list] = None, documents: Optional[list] = None) -> dict:
    observations = list(state.get("observations") or [])
    trace = list(state.get("trace") or [])
    public_spans = _public_spans(spans or [])
    observation = {
        "action": action,
        "query": query,
        "summary": summary,
        "nodeIds": node_ids,
        "sourceIds": source_ids,
        "labels": labels,
        "scores": scores or {},
        "excerpts": _passage_excerpts(evidence, source_ids, query, public_spans),
        "spans": public_spans,
        "documents": list(documents or []),
    }
    observations.append(observation)
    missing = missing_aspects(
        {
            "question": state.get("question") or "",
            "observations": observations,
            "evidence": evidence if evidence is not None else state.get("evidence") or [],
        }
    )
    observation["missingAspects"] = missing
    if missing:
        summary = summary + " 仍缺：" + "、".join(missing[:4]) + "。"
        observation["summary"] = summary
    trace.append(
        {
            "step": len(trace) + 1,
            "action": action,
            "query": query,
            "summary": summary,
            "nodeIds": node_ids,
            "sourceIds": source_ids,
        }
    )
    parts = list(state.get("graph_parts") or [])
    if graph is not None:
        parts.append(graph)
    update = {
        "tool_calls": int(state.get("tool_calls") or 0) + 1,
        "observations": observations,
        "trace": trace,
        "graph_parts": parts,
    }
    if evidence is not None:
        update["evidence"] = evidence
    return update


def _tools(snapshot: dict) -> dict:
    def graph_search(state: dict) -> dict:
        _bind_budget(snapshot, state)
        pending = state.get("pending") or {}
        query = str(pending.get("query") or state.get("question") or "")
        limits = retrieval_limits(snapshot, state)
        spans = retrieve_spans(
            snapshot,
            query,
            state.get("doc_ids"),
            limit=limits["candidates"],
            rerank_pool=limits["rerank_pool"],
        )
        graph, source_ids, labels = _search_graph(snapshot, query, state.get("doc_ids"), 24, "all")
        scores = {}
        for span in spans:
            scores.setdefault(str(span.get("unit_id") or ""), float(span.get("score") or 0.0))
        for source_id in _structure_source_ids(snapshot, query, state.get("doc_ids"), limits["candidates"]):
            if source_id not in source_ids:
                source_ids.append(source_id)
        evidence = _attach_evidence(snapshot, state, source_ids, query, scores)
        retrieved = []
        seen_units: set[str] = set()
        for span in spans:
            unit_id = str(span.get("unit_id") or "")
            if unit_id and unit_id not in seen_units:
                retrieved.append(unit_id)
                seen_units.add(unit_id)
        leaf_for_source: dict[str, str] = {}
        for node in graph.get("nodes") or []:
            if node.get("kind") != "structure" or node.get("type") not in {"chunk", "table", "figure"}:
                continue
            for source_id in node.get("sourceIds") or []:
                leaf_for_source.setdefault(str(source_id), node["id"])
        frontier: list[str] = []
        seen_frontier: set[str] = set()
        for unit_id in retrieved:
            leaf_id = leaf_for_source.get(unit_id)
            if leaf_id and leaf_id not in seen_frontier:
                frontier.append(leaf_id)
                seen_frontier.add(leaf_id)
        for edge in graph.get("edges") or []:
            if edge_kind(edge) != "references":
                continue
            cites = {str(item) for item in edge.get("sourceIds") or []}
            if edge.get("source") not in seen_frontier and not (cites & seen_units):
                continue
            target_id = str(edge.get("target") or "")
            if target_id and target_id not in seen_frontier:
                frontier.append(target_id)
                seen_frontier.add(target_id)
        node_ids = frontier + [node["id"] for node in graph.get("nodes") or [] if node["id"] not in seen_frontier]
        order = {node_id: index for index, node_id in enumerate(node_ids)}
        graph["nodes"] = sorted(
            list(graph.get("nodes") or []),
            key=lambda node: (order.get(node["id"], len(order)), str(node["id"])),
        )
        summary = (
            f"图谱检索命中 {len(graph['nodes'])} 个节点、{len(graph['edges'])} 条关系，"
            f"对应 {len(source_ids)} 段原文。混合种子 {len(spans)} 条仍保留。{CAUSATION_NOTE}"
        )
        return _record(
            state,
            "graph_search",
            query,
            summary,
            node_ids,
            source_ids,
            labels,
            graph,
            evidence=evidence,
            scores=scores,
            spans=spans,
            documents=catalog_brief(snapshot, state.get("doc_ids")),
        )

    def graph_expand(state: dict) -> dict:
        _bind_budget(snapshot, state)
        pending = state.get("pending") or {}
        node_ids = list(pending.get("nodeIds") or [])
        hops = int(pending.get("hops") or 2)
        query = str(pending.get("query") or state.get("question") or "")
        doc_ids = state.get("doc_ids")
        visible = _visible_graph(snapshot, doc_ids)
        structure = _structure_graph(snapshot, doc_ids)
        mixed = {
            "nodes": list(visible.get("nodes") or []) + list(structure.get("nodes") or []),
            "edges": list(visible.get("edges") or []) + list(structure.get("edges") or []),
        }
        before = {node_id for item in state.get("observations") or [] for node_id in item.get("nodeIds") or []}
        previous_sources = {source_id for item in state.get("observations") or [] for source_id in item.get("sourceIds") or []}
        graph = expand_hops(mixed, node_ids, hops, scope="all")
        labels = {node["id"]: node.get("label") or node["id"] for node in graph["nodes"]}
        chunks = snapshot.get("chunks") or {}
        kept_ids: list[str] = []
        fresh_ids: list[str] = []
        seen_sources: set[str] = set()

        def consider(source_id: str) -> None:
            source_id = str(source_id or "")
            if not source_id or source_id in seen_sources or source_id not in chunks:
                return
            seen_sources.add(source_id)
            kept_ids.append(source_id)
            if source_id not in previous_sources:
                fresh_ids.append(source_id)

        for node in graph.get("nodes") or []:
            if node.get("kind") == "structure":
                continue
            for source_id in node.get("sourceIds") or []:
                consider(source_id)
        for edge in graph.get("edges") or []:
            kind = edge_kind(edge)
            if kind == "semantic":
                for source_id in edge.get("sourceIds") or []:
                    consider(source_id)
            elif kind == "references":
                for source_id in edge.get("sourceIds") or []:
                    consider(source_id)
                for item in edge.get("provenance") or []:
                    if isinstance(item, dict):
                        consider(str(item.get("sourceId") or ""))
        fresh_ids.sort(
            key=lambda source_id: lexical_score(query, chunks[source_id].get("content") or ""),
            reverse=True,
        )
        fresh_set = set(fresh_ids)
        source_ids = fresh_ids + [source_id for source_id in kept_ids if source_id not in fresh_set]
        by_id = {node["id"]: node for node in graph.get("nodes") or []}
        allowed = None if doc_ids is None else {str(item) for item in doc_ids}
        target_doc_ids: list[str] = []
        seen_docs: set[str] = set()
        for edge in graph.get("edges") or []:
            if edge_kind(edge) != "references":
                continue
            target = by_id.get(edge.get("target"))
            if not target:
                continue
            doc_id = str(target.get("docId") or "")
            if not doc_id or doc_id in seen_docs:
                continue
            if allowed is not None and doc_id not in allowed:
                continue
            seen_docs.add(doc_id)
            target_doc_ids.append(doc_id)
        limits = retrieval_limits(snapshot, state)
        for source_id in _structure_source_ids(snapshot, query, doc_ids, limits["candidates"]):
            chunk = chunks.get(source_id) or {}
            if str(chunk.get("doc_id") or "") in seen_docs:
                continue
            if source_id not in source_ids:
                source_ids.append(source_id)
        # retrieve_spans([]) is unfiltered; an empty target set is a skip, not a call.
        spans = []
        if target_doc_ids:
            spans = retrieve_spans(
                snapshot,
                query,
                target_doc_ids,
                limit=limits["candidates"],
                rerank_pool=limits["rerank_pool"],
            )
        scores = {}
        for span in spans:
            unit_id = str(span.get("unit_id") or "")
            if not unit_id:
                continue
            scores.setdefault(unit_id, float(span.get("score") or 0.0))
            if unit_id in chunks and unit_id not in source_ids:
                source_ids.append(unit_id)
        evidence = _attach_evidence(snapshot, state, source_ids, query, scores)
        fresh = [labels[node_id] for node_id in labels if node_id not in before and labels[node_id]]
        bridge = "、".join(fresh[:6]) if fresh else "无"
        summary = (
            f"关系扩展 {min(max(hops, 1), MAX_PATH_EDGES)} 跳，得到 {len(graph['nodes'])} 个节点、"
            f"{len(graph['edges'])} 条关系。新的桥接节点：{bridge}。{CAUSATION_NOTE}"
        )
        return _record(
            state,
            "graph_expand",
            query,
            summary,
            [node["id"] for node in graph["nodes"]],
            source_ids,
            labels,
            graph,
            evidence=evidence,
            scores=scores,
            spans=spans,
            documents=catalog_brief(snapshot, state.get("doc_ids")),
        )

    def document_retrieve(state: dict) -> dict:
        _bind_budget(snapshot, state)
        pending = state.get("pending") or {}
        query = str(pending.get("query") or state.get("question") or "")
        limits = retrieval_limits(snapshot, state)
        spans = retrieve_spans(
            snapshot,
            query,
            state.get("doc_ids"),
            limit=limits["candidates"],
            rerank_pool=limits["rerank_pool"],
        )
        ranked = unit_ranking(spans)
        source_ids = [chunk_id for chunk_id, _score in ranked]
        scores = {chunk_id: score for chunk_id, score in ranked}
        evidence = _attach_evidence(snapshot, state, source_ids, query, scores)
        summary = f"原文检索返回 {len(source_ids)} 段。这一步只定位原文，不新增图谱关系。"
        return _record(
            state,
            "document_retrieve",
            query,
            summary,
            [],
            source_ids,
            {},
            None,
            evidence=evidence,
            scores=scores,
            spans=spans,
            documents=catalog_brief(snapshot, state.get("doc_ids")),
        )

    def inspect_evidence(state: dict) -> dict:
        pending = state.get("pending") or {}
        query = str(pending.get("query") or state.get("question") or "")
        requested = [str(chunk_id) for chunk_id in pending.get("chunkIds") or []]
        evidence = _attach_evidence(snapshot, state, requested, query)
        accepted = [chunk_id for chunk_id in requested if any(row.get("chunk_id") == chunk_id for row in evidence)]
        summary = f"核对了 {len(accepted)} 段原文。"
        return _record(state, "inspect_evidence", query, summary, [], accepted, {}, None, evidence=evidence)

    return {
        "graph_search": graph_search,
        "graph_expand": graph_expand,
        "document_retrieve": document_retrieve,
        "inspect_evidence": inspect_evidence,
    }


def _planner_prompt(state: dict) -> str:
    """Read-only planner view. Callers keep evidence, observations, and doc scope."""

    def clean_ids(values, limit=None):
        rows = []
        seen = set()
        for value in values or []:
            item = str(value).strip()
            if not item or item in seen:
                continue
            seen.add(item)
            rows.append(item)
            if limit is not None and len(rows) >= limit:
                break
        return rows

    observations = list(state.get("observations") or [])
    question = str(state.get("question") or "")
    queries = [question]
    for item in observations:
        query = str(item.get("query") or "").strip()
        if query:
            queries.append(query)
    evidence_title = {}
    touched = set()
    for row in state.get("evidence") or []:
        doc_id = str(row.get("doc_id") or "").strip()
        if not doc_id:
            continue
        touched.add(doc_id)
        human = str(row.get("document_title") or "").strip()
        if human and doc_id not in evidence_title:
            evidence_title[doc_id] = human
    for item in observations:
        for span in item.get("spans") or []:
            doc_id = str(span.get("doc_id") or "").strip()
            if doc_id:
                touched.add(doc_id)
        for source in item.get("sourceIds") or []:
            source_id = str(source).strip()
            if source_id:
                touched.add(source_id)
    latest_ids = set()
    if observations:
        last = observations[-1]
        chunk_ids = {str(item).strip() for item in (last.get("sourceIds") or [])}
        for excerpt in last.get("excerpts") or []:
            chunk_ids.add(str(excerpt.get("chunk_id") or "").strip())
        for span in last.get("spans") or []:
            doc_id = str(span.get("doc_id") or "").strip()
            unit_id = str(span.get("unit_id") or "").strip()
            if doc_id and (unit_id in chunk_ids or doc_id in chunk_ids or not chunk_ids):
                latest_ids.add(doc_id)
        for row in state.get("evidence") or []:
            if str(row.get("chunk_id") or "").strip() in chunk_ids:
                doc_id = str(row.get("doc_id") or "").strip()
                if doc_id:
                    latest_ids.add(doc_id)
    listed = []
    seen_docs = set()
    for item in observations:
        for document in item.get("documents") or []:
            doc_id = str(document.get("id") or "").strip()
            if not doc_id or doc_id in seen_docs:
                continue
            seen_docs.add(doc_id)
            listed.append((doc_id, document))
    for row in state.get("evidence") or []:
        doc_id = str(row.get("doc_id") or "").strip()
        if not doc_id or doc_id in seen_docs:
            continue
        seen_docs.add(doc_id)
        section = str(row.get("section_title") or "").strip()
        listed.append(
            (
                doc_id,
                {
                    "id": doc_id,
                    "title": row.get("document_title") or row.get("doc_name") or doc_id,
                    "publisher": "",
                    "language": "",
                    "sections": [section] if section else [],
                },
            )
        )
    labels = bridge_labels(observations)
    bridge_names = {str(label).strip().casefold() for label in labels if str(label).strip()}

    def display_title(doc_id, document):
        return str(
            document.get("document_title") or evidence_title.get(doc_id) or document.get("title") or doc_id
        ).strip()

    def title_named(title):
        probe = title.strip()
        if len(probe) < 3:
            return False
        folded = probe.casefold()
        return any(folded in query.casefold() for query in queries if query)

    picked = []
    for index, (doc_id, document) in enumerate(listed):
        title = display_title(doc_id, document)
        raw = str(document.get("title") or "").strip()
        named = title_named(title) or (raw and raw != title and title_named(raw))
        bridged = title.casefold() in bridge_names or doc_id.casefold() in bridge_names
        if doc_id in latest_ids or named:
            rank = 0
        elif doc_id in touched or bridged:
            rank = 1
        else:
            continue
        picked.append((rank, index, doc_id, document))
    picked.sort()
    catalog_lines = []
    for _rank, _index, doc_id, document in picked[:12]:
        title = display_title(doc_id, document)[:160]
        sections = []
        for section in list(document.get("sections") or [])[:4]:
            text = str(section).strip()
            if text:
                sections.append(text[:100])
        catalog_lines.append(
            f"{doc_id} | {title} | {document.get('publisher') or ''} | {document.get('language') or ''} | sections={sections}"
        )
    excerpt_budget = 1200
    excerpt_text = [""] * len(observations)
    for index in range(len(observations) - 1, -1, -1):
        if excerpt_budget <= 0:
            break
        parts = []
        for excerpt in observations[index].get("excerpts") or []:
            piece = f" | {excerpt.get('chunk_id')}: {excerpt.get('excerpt')}"
            if len(piece) > excerpt_budget:
                if excerpt_budget > 24:
                    parts.append(piece[:excerpt_budget])
                excerpt_budget = 0
                break
            parts.append(piece)
            excerpt_budget -= len(piece)
        excerpt_text[index] = "".join(parts)
    seen_nodes = set()
    fresh_nodes = []
    for item in observations:
        current = clean_ids(item.get("nodeIds"))
        fresh = []
        if item.get("action") in {"graph_search", "graph_expand"}:
            fresh = [node_id for node_id in current if node_id not in seen_nodes][:8]
        fresh_nodes.append(fresh)
        seen_nodes.update(current)
    lines = []
    last = len(observations) - 1
    for index, item in enumerate(observations):
        action = str(item.get("action") or "")
        query = str(item.get("query") or "")
        summary = " ".join(str(item.get("summary") or "").split())
        summary = summary[:360] if index == last else summary[:100]
        if index == last:
            nodes = fresh_nodes[index] + [
                node_id for node_id in clean_ids(item.get("nodeIds"), 8) if node_id not in fresh_nodes[index]
            ]
            nodes = nodes[:8]
            sources = clean_ids(item.get("sourceIds"), 8)
        elif fresh_nodes[index]:
            nodes = fresh_nodes[index]
            sources = []
        else:
            nodes = []
            sources = []
        line = f"{action}: {query} | {summary}"
        if nodes:
            line += f" nodeIds={nodes}"
        if sources:
            line += f" sourceIds={sources}"
        line += excerpt_text[index]
        lines.append(line)
    return (
        "Choose the next retrieval action. Reply with one JSON object only, no reasoning.\n"
        'Keys: action, query, nodeIds, chunkIds, hops.\n'
        "action is one of graph_search, graph_expand, document_retrieve, inspect_evidence, finish.\n"
        "graph_search finds entities and relations for one query.\n"
        "graph_expand follows known nodeIds along relations to a bridge entity.\n"
        "document_retrieve reads original passages for one part of the question. "
        "A later document_retrieve may use a different query for a part the earlier passages do not cover.\n"
        "inspect_evidence reads passages already found.\n"
        "finish only when the passages address every part of the question. "
        "Citations, or a previous retrieve, do not mean the whole question is covered. "
        "Query a missing part before finish.\n"
        f"Tool calls used: {int(state.get('tool_calls') or 0)}. {budget_note(state)}.\n"
        f"Question: {question}\n"
        f"Missing aspects: {missing_aspects(state) or 'none'}\n"
        f"Document metadata: {catalog_lines or '(none)'}\n"
        f"Bridge labels: {labels}\n"
        "Observations:\n"
        + ("\n".join(lines) or "(none)")
    )


def _planner_model(state: dict) -> str:
    from knowledge.graph_rag.runtime import ollama_chat

    return ollama_chat(
        _planner_prompt(state),
        system_prompt="You only emit a JSON action.",
        json_mode=True,
        max_tokens=300,
        timeout=90,
    )


def _retrieval_groups(state: dict) -> list[list[str]]:
    """One source list per distinct retrieval query and per expand that adds a passage."""
    groups: list[list[str]] = []
    seen_queries: set[str] = set()
    prior_sources: set[str] = set()
    for observation in state.get("observations") or []:
        action = str(observation.get("action") or "")
        source_ids = [str(item) for item in observation.get("sourceIds") or [] if str(item).strip()]
        if action in {"graph_search", "document_retrieve"}:
            key = normalize_query(observation.get("query") or "")
            if key not in seen_queries and source_ids:
                seen_queries.add(key)
                groups.append(source_ids)
        elif action == "graph_expand":
            fresh = [source_id for source_id in source_ids if source_id not in prior_sources]
            if fresh:
                groups.append(fresh)
        prior_sources.update(source_ids)
    return groups


def _question_relevance(question: str, row: dict) -> tuple[float, float]:
    content = str(row.get("content") or "")
    return (lexical_score(question, content), float(row.get("score") or 0.0))


def _select_answer_sources(state: dict) -> list[dict]:
    """Keep one passage from each retrieval or bridge step, then fill by the question.

    Earlier graph neighborhoods do not consume the whole budget before a later
    targeted or bridge passage. The returned rows are exactly the rows numbered
    in the prompt, and each row still carries its original passage.
    """
    evidence = []
    seen = set()
    for row in state.get("evidence") or []:
        chunk_id = str(row.get("chunk_id") or "")
        if not chunk_id or chunk_id in seen:
            continue
        seen.add(chunk_id)
        evidence.append(row)
    by_id = {row["chunk_id"]: row for row in evidence}
    reserved: list[str] = []
    reserved_set: set[str] = set()
    for group in _retrieval_groups(state):
        for source_id in group:
            if source_id in by_id and source_id not in reserved_set:
                reserved.append(source_id)
                reserved_set.add(source_id)
                break
    question = str(state.get("question") or "")
    facets = question_facets(question)
    for facet in facets:
        best_id = ""
        best_score = 0.0
        for row in evidence:
            chunk_id = row["chunk_id"]
            if chunk_id in reserved_set:
                continue
            score = lexical_score(facet, str(row.get("content") or ""))
            if score > best_score:
                best_id = chunk_id
                best_score = score
        if best_id and best_score > 0:
            reserved.append(best_id)
            reserved_set.add(best_id)
    rest = sorted(
        (row for row in evidence if row["chunk_id"] not in reserved_set),
        key=lambda row: _question_relevance(question, row),
        reverse=True,
    )
    ordered = [by_id[source_id] for source_id in reserved] + rest
    limits = retrieval_limits(state=state)
    selected = []
    used = 0
    for row in ordered:
        cost = min(len(str(row.get("content") or "")), limits["passage_chars"])
        if selected and used + cost > limits["context_chars"]:
            continue
        selected.append(row)
        used += cost
    return selected


def _relevant_window(content: str, query: str, limit: int) -> str:
    """Pick one existing window of the original passage. The stored text is unchanged."""
    body = str(content or "")
    if len(body) <= limit:
        return body
    windows = split_windows(body, limit, 120)
    if not windows:
        return body[:limit]
    if os.environ.get("GRAPH_DISABLE_VECTORS") != "1" and len(windows) > 1:
        try:
            from knowledge.graph_rag.runtime import rerank_scores

            scores = rerank_scores(query, windows)
        except Exception:
            scores = None
        if scores and len(scores) == len(windows):
            best = max(range(len(windows)), key=lambda index: float(scores[index]))
            return windows[best]
    best = windows[0]
    best_score = lexical_score(query, best)
    for piece in windows[1:]:
        score = lexical_score(query, piece)
        if score > best_score:
            best = piece
            best_score = score
    return best


def _coverage_stop(answer: str, sources: list[dict], question: str = "", spans: Optional[list] = None) -> str:
    """Status from citations plus the spans actually packed for this question."""
    citations = citations_from_answer(answer, sources)
    if spans is None:
        # Callers that only have the answer text still need the old citation check.
        if not citations:
            return "insufficient_evidence" if ABSTAIN_TEXT in (answer or "") else "uncited_answer"
        if ABSTAIN_TEXT not in (answer or ""):
            return "answered"
        remainder = re.sub(r"\[\d+\]", "", (answer or "").replace(ABSTAIN_TEXT, ""))
        remainder = re.sub(r"[\s\W_]+", "", remainder)
        return "partial_evidence" if remainder else "insufficient_evidence"
    return coverage_status(answer, question, spans, len(citations), ABSTAIN_TEXT)


def _answer_from_evidence(state: dict) -> dict:
    from knowledge.graph_rag.runtime import ollama_chat

    sources = _select_answer_sources(state)
    if not sources:
        limits = retrieval_limits(state=state)
        return {
            "answer": ABSTAIN_TEXT,
            "stop_reason": "insufficient_evidence",
            "sources": [],
            "context_spans": [],
            "context_limits": limits,
        }
    question = str(state.get("question") or "")
    limits = retrieval_limits(state=state)
    facets = question_facets(question)
    selected_ids = {str(source.get("chunk_id") or "") for source in sources}
    matched_spans: list = []
    matched_units: set[str] = set()
    for observation in state.get("observations") or []:
        for span in observation.get("spans") or []:
            unit_id = span["unit_id"]
            if unit_id not in selected_ids:
                continue
            matched_spans.append(span)
            matched_units.add(unit_id)
    packed = pack_context(
        question,
        sources,
        context_chars=limits["context_chars"],
        passage_chars=limits["passage_chars"],
        facets=facets,
        matched_spans=matched_spans,
    )
    packed_by_unit: dict[str, list] = {}
    for span in packed:
        packed_by_unit.setdefault(span["unit_id"], []).append(span)
    status_spans = list(packed)
    numbered = []
    index_by_chunk = {}
    used = sum(len(str(span.get("text") or "")) for span in packed)
    for index, source in enumerate(sources, start=1):
        chunk_id = source["chunk_id"]
        spans = packed_by_unit.get(chunk_id) or []
        if spans:
            body = "\n".join(f"[{span['start']}:{span['end']}] {span['text']}" for span in spans)
        elif chunk_id in matched_units:
            continue
        else:
            content = str(source.get("content") or "")
            body = _relevant_window(content, question, limits["passage_chars"])
            if used and used + len(body) > limits["context_chars"]:
                continue
            start = content.find(body) if body else -1
            if start >= 0 and body and content[start : start + len(body)] == body:
                status_spans.append(
                    {
                        "unit_id": chunk_id,
                        "doc_id": str(source.get("doc_id") or ""),
                        "start": start,
                        "end": start + len(body),
                        "text": body,
                    }
                )
            used += len(body)
        index_by_chunk[chunk_id] = index
        numbered.append(
            f"[{index}] 文档：{source.get('doc_name') or ''} 页：{source.get('page_number') or ''} 章节：{source.get('section_title') or source.get('heading') or ''}\n{body}"
        )
    relations = []
    merged = merge_graphs(list(state.get("graph_parts") or []), _seed_nodes(state))
    labels = {node["id"]: node.get("label") or node["id"] for node in merged["nodes"]}
    for edge in merged["edges"]:
        refs = [index_by_chunk[source_id] for source_id in edge.get("sourceIds") or [] if source_id in index_by_chunk]
        if not refs:
            continue
        ref_text = "".join(f"[{ref}]" for ref in refs[:4])
        relations.append(f"{labels.get(edge['source'], edge['source'])} --{edge.get('label') or 'related_to'}--> {labels.get(edge['target'], edge['target'])} {ref_text}")
        if len(relations) >= 12:
            break
    prompt = (
        f"问题：{state.get('question') or ''}\n\n"
        "原文：\n"
        + "\n\n".join(numbered)
        + "\n\n抽取关系（只帮助定位原文，不是因果证明）：\n"
        + ("\n".join(relations) or "（无）")
        + f"\n\n{CAUSATION_NOTE}\n回答时用原文编号引用。"
    )
    system = (
        "你是航空资料问答助手。只根据给出的编号原文回答，不要补充资料里没有的事实。"
        "问题有多个部分时，每个部分使用对应的片段。不要只用整题里的一个窗口丢掉另一个部分的条件。"
        "如果给出的片段已经覆盖问题的每个方面，直接回答并引用，不要再写证据不足。"
        "只覆盖一部分时，被覆盖的部分要引用。"
        f"原文没有支持的部分必须写明：{ABSTAIN_TEXT}"
        "不要为了让回答看起来完整而编造未支持的事实。"
        f"如果整个问题都无法支持，只输出：{ABSTAIN_TEXT}"
        "引用格式是 [编号]。不要输出思考过程。"
        f"{CAUSATION_NOTE}"
    )
    try:
        answer = ollama_chat(prompt, system_prompt=system, max_tokens=800, timeout=180)
    except Exception as exc:
        logger.warning("生成回答失败: %s", exc)
        return {
            "answer": ABSTAIN_TEXT,
            "stop_reason": "error",
            "sources": sources,
            "context_spans": status_spans,
            "context_limits": limits,
        }
    stop = _coverage_stop(answer, sources, question, status_spans)
    if stop in {"uncited_answer", "insufficient_evidence"}:
        return {
            "answer": ABSTAIN_TEXT,
            "stop_reason": stop,
            "sources": sources,
            "context_spans": status_spans,
            "context_limits": limits,
        }
    return {
        "answer": answer,
        "stop_reason": stop,
        "sources": sources,
        "context_spans": status_spans,
        "context_limits": limits,
    }


def _unavailable(status: dict) -> dict:
    if status.get("building"):
        answer = "知识图谱正在构建，当前还没有可检索的实体和关系。"
    elif status.get("status") == "error":
        answer = "知识图谱不可用。" + (f" {status.get('error')}" if status.get("error") else "")
    elif status.get("status") == "stored":
        answer = "原文已登记，知识图谱尚未构建。请先执行图谱构建。"
    else:
        answer = "知识图谱当前不可用，无法执行图谱检索。"
    return {
        "answer": answer,
        "sources": [],
        "citations": [],
        "graph": empty_graph(),
        "trace": [],
        "stopReason": "graph_unavailable",
    }


_service: Optional[GraphService] = None
_service_dir: Optional[str] = None
_service_lock = threading.Lock()


def get_service() -> GraphService:
    global _service, _service_dir
    directory = str(data_dir())
    with _service_lock:
        if _service is None or _service_dir != directory:
            _service = GraphService(Path(directory))
            _service_dir = directory
        return _service


def reset_service() -> None:
    global _service, _service_dir
    with _service_lock:
        _service = None
        _service_dir = None
