"""Persistent passage catalog. Original passages stay the provenance record."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from knowledge.graph_rag.corpus import available_count, corpus_ids, is_corpus, manifest_by_id


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def managed_id(file_name: str) -> str:
    name = Path(str(file_name or "")).name
    if not name:
        return ""
    if name.startswith("managed:"):
        return name
    return f"managed:{name}"


def _norm_url(value: str) -> str:
    return str(value or "").strip().rstrip("/").casefold()


def corpus_id_for_source(source_url: str = "", file_name: str = "") -> str:
    """Manifest doc_id when a managed source is the seeded corpus row."""
    token = str(file_name or "").strip()
    if token.startswith("managed:"):
        token = token[len("managed:") :]
    if token and is_corpus(token):
        return token
    url = _norm_url(source_url)
    base = Path(token).name.casefold() if token else ""
    if not url and not base:
        return ""
    for row in manifest_by_id().values():
        if url and _norm_url(row.get("source_url") or "") == url:
            return str(row.get("doc_id") or "")
        row_file = Path(str(row.get("file") or "")).name.casefold()
        if base and row_file and base == row_file:
            return str(row.get("doc_id") or "")
    return ""


def canonical_document_key(document_key: str = "", source_url: str = "", file_name: str = "", existing_id: str = "") -> str:
    """One id for a work, from the seeded manifest or the native document key.

    A managed upload whose source matches the seeded corpus reuses that doc_id.
    """
    matched = corpus_id_for_source(source_url, file_name) or corpus_id_for_source("", existing_id)
    if matched:
        return matched
    explicit = str(document_key or "").strip()
    if explicit:
        return corpus_id_for_source("", explicit) or explicit
    current = str(existing_id or "").strip()
    if current:
        if is_corpus(current) or current.startswith("managed:"):
            return corpus_id_for_source("", current) or current
        if "." not in Path(current).name:
            return current
        return managed_id(current)
    raw = str(file_name or "").strip()
    if not raw:
        return ""
    if is_corpus(raw) or "." not in Path(raw).name:
        return raw
    return managed_id(raw)


def chunk_document_key(chunk: dict) -> str:
    return canonical_document_key(
        document_key=str(chunk.get("document_key") or chunk.get("documentKey") or ""),
        source_url=str(chunk.get("source_url") or chunk.get("sourceUrl") or ""),
        existing_id=str(chunk.get("doc_id") or ""),
    )


def _manifest_doc_id(doc_id: str, document: dict) -> str:
    """Manifest id for a stored document. Empty when the source is not in the corpus."""
    if is_corpus(doc_id):
        return doc_id
    source_url = str(document.get("sourceUrl") or "")
    if source_url:
        matched = corpus_id_for_source(source_url, "")
        if matched:
            return matched
    storage_name = Path(str(document.get("storagePath") or "")).name
    for token in (document.get("documentKey"), document.get("name"), storage_name, doc_id):
        token = str(token or "").strip()
        if not token:
            continue
        matched = corpus_id_for_source("", token)
        if matched:
            return matched
    return ""


def _combine_document(canonical: str, items: list[tuple[str, dict]]) -> dict:
    """Fold same-source rows into one document. Provenance documentKey stays as stored."""

    def rank(item: tuple[str, dict]) -> tuple:
        doc = item[1]
        return (
            1 if doc.get("extractedChunkIds") else 0,
            1 if doc.get("indexed") else 0,
            1 if str(doc.get("storagePath") or "") else 0,
            int(doc.get("ingestedChunks") or 0),
        )

    merged = dict(sorted(items, key=rank, reverse=True)[0][1])
    held = {"id", "documentKey", "extractedChunkIds", "indexed", "ingestedChunks", "availableChunks", "partial"}
    for _, doc in items:
        for field, value in doc.items():
            if field in held or merged.get(field) not in (None, "", [], {}):
                continue
            merged[field] = value
    extracted: list = []
    seen = set()
    for _, doc in items:
        for chunk_id in doc.get("extractedChunkIds") or []:
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            extracted.append(chunk_id)
    merged["id"] = canonical
    merged["extractedChunkIds"] = extracted
    merged["indexed"] = any(bool(doc.get("indexed")) for _, doc in items)
    merged["ingestedChunks"] = max(len(extracted), max(int(doc.get("ingestedChunks") or 0) for _, doc in items))
    merged["availableChunks"] = max(int(doc.get("availableChunks") or 0) for _, doc in items)
    if any(doc.get("partial") for _, doc in items):
        merged["partial"] = True
    provenance = ""
    for _, doc in items:
        key = str(doc.get("documentKey") or "").strip()
        if key and key != canonical:
            provenance = key
            break
    if not provenance:
        for doc_id, _doc in items:
            if doc_id != canonical:
                provenance = doc_id
                break
    if provenance:
        merged["documentKey"] = provenance
    if is_corpus(canonical) or any(doc.get("kind") == "corpus" for _, doc in items):
        merged["kind"] = "corpus"
    return merged


def _empty() -> dict:
    return {
        "documents": {},
        "chunks": {},
        "source_map": {},
        "build": {
            "building": False,
            "status": "empty",
            "completedDocuments": 0,
            "totalDocuments": 0,
            "error": "",
            "currentDoc": "",
            "updatedAt": _now(),
        },
    }


class Catalog:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.RLock()
        self.data = self._load()
        # Startup boundary: fold identities once before seed or public reads.
        if self._migrate_identities():
            self.save()

    def _load(self) -> dict:
        if not self.path.exists():
            return _empty()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return _empty()
        base = _empty()
        if isinstance(payload, dict):
            base["documents"] = payload.get("documents") or {}
            base["chunks"] = payload.get("chunks") or {}
            base["source_map"] = payload.get("source_map") or {}
            build = payload.get("build") or {}
            if isinstance(build, dict):
                base["build"].update(build)
        # A crashed build must not stay locked after process restart.
        if base["build"].get("building"):
            base["build"]["building"] = False
            if not base["build"].get("error"):
                base["build"]["error"] = "构建进程中断，已保留已完成的文档。"
            base["build"]["status"] = "error" if not base["chunks"] else "ready"
        return base

    def save(self) -> None:
        _atomic_write(self.path, self.data)

    @property
    def documents(self) -> dict:
        return self.data["documents"]

    @property
    def chunks(self) -> dict:
        return self.data["chunks"]

    @property
    def source_map(self) -> dict:
        return self.data["source_map"]

    @property
    def build(self) -> dict:
        return self.data["build"]

    def _migrate_identities(self) -> bool:
        """Rekey stored documents onto manifest ids and restore physical filenames. Caller saves once when this changes data."""
        documents = self.documents
        remap: dict[str, str] = {}
        for doc_id, document in list(documents.items()):
            canonical = _manifest_doc_id(doc_id, document)
            if canonical and canonical != doc_id:
                remap[doc_id] = canonical
        changed = False
        if remap:
            groups: dict[str, list[str]] = {}
            for alias, canonical in remap.items():
                groups.setdefault(canonical, []).append(alias)
            for canonical, aliases in groups.items():
                items = [(alias, documents[alias]) for alias in aliases]
                current = documents.get(canonical)
                if isinstance(current, dict):
                    items.append((canonical, current))
                merged = _combine_document(canonical, items)
                for alias in aliases:
                    documents.pop(alias, None)
                documents[canonical] = merged
            for chunk in self.chunks.values():
                target = remap.get(str(chunk.get("doc_id") or ""))
                if target:
                    chunk["doc_id"] = target
            changed = True
        for chunk in self.chunks.values():
            document = documents.get(str(chunk.get("doc_id") or ""))
            if not isinstance(document, dict):
                continue
            storage = str(document.get("storagePath") or "").strip()
            if not storage:
                continue
            basename = Path(storage).name
            if not basename or basename in {".", ".."}:
                continue
            current_name = str(chunk.get("doc_name") or "")
            if current_name == basename:
                continue
            if current_name.strip() and not str(chunk.get("document_title") or "").strip():
                chunk["document_title"] = current_name
            chunk["doc_name"] = basename
            changed = True
        return changed

    def ensure_corpus(self) -> None:
        changed = False
        for doc_id in corpus_ids():
            meta = manifest_by_id()[doc_id]
            current = self.documents.get(doc_id)
            available = available_count(doc_id)
            if current is None:
                self.documents[doc_id] = {
                    "id": doc_id,
                    "name": meta.get("title") or doc_id,
                    "sourceUrl": meta.get("source_url") or "",
                    "kind": "corpus",
                    "storagePath": "",
                    "indexed": False,
                    "availableChunks": available,
                    "ingestedChunks": 0,
                    "extractedChunkIds": [],
                }
                changed = True
                continue
            manifest_title = str(meta.get("title") or "").strip()
            stored_title = str(current.get("title") or "").strip()
            stored_name = str(current.get("name") or "").strip()
            # Managed title wins. An empty frozen-manifest title must not
            # replace a human name, and a name already overwritten with the
            # doc_id alias is restored from the stored title.
            if stored_title:
                desired_name = stored_title
            elif manifest_title:
                desired_name = manifest_title
            elif stored_name:
                desired_name = stored_name
            else:
                desired_name = doc_id
            if current.get("name") != desired_name or current.get("availableChunks") != available:
                current["name"] = desired_name
                current["sourceUrl"] = meta.get("source_url") or ""
                current["availableChunks"] = available
                current["kind"] = "corpus"
                changed = True
        if changed:
            self.save()

    def knows(self, doc_id: str) -> bool:
        if doc_id in self.documents or is_corpus(doc_id):
            return True
        key = canonical_document_key(file_name=doc_id, existing_id=doc_id)
        return bool(key and key in self.documents)

    def canonical_id(self, doc_id: str) -> str:
        key = canonical_document_key(file_name=doc_id, existing_id=doc_id)
        if key and key in self.documents:
            return key
        if doc_id in self.documents or is_corpus(doc_id):
            return doc_id
        managed = managed_id(doc_id)
        if managed and managed in self.documents:
            return managed
        return key or doc_id

    def public_documents(self) -> list[dict]:
        self.ensure_corpus()
        order: list[str] = []
        grouped: dict[str, dict] = {}
        for doc_id in list(corpus_ids()) + list(self.documents):
            document = self.documents.get(doc_id)
            if not isinstance(document, dict):
                continue
            key = canonical_document_key(
                document_key=str(document.get("documentKey") or ""),
                source_url=str(document.get("sourceUrl") or ""),
                file_name=str(document.get("name") or ""),
                existing_id=doc_id,
            )
            if not key:
                continue
            current = grouped.get(key)
            if current is None:
                order.append(key)
                grouped[key] = dict(document)
                grouped[key]["id"] = key
                continue
            for field in ("name", "sourceUrl", "publisher", "originalFormat", "title", "documentKey"):
                if not current.get(field) and document.get(field):
                    current[field] = document[field]
            if document.get("indexed"):
                current["indexed"] = True
                current["ingestedChunks"] = max(int(current.get("ingestedChunks") or 0), int(document.get("ingestedChunks") or 0))
            current["availableChunks"] = max(int(current.get("availableChunks") or 0), int(document.get("availableChunks") or 0))
        stored: dict[str, int] = {}
        for chunk in self.chunks.values():
            key = chunk_document_key(chunk)
            if key:
                stored[key] = stored.get(key, 0) + 1
        rows = []
        for key in order:
            document = grouped[key]
            indexed = bool(document.get("indexed"))
            if indexed:
                chunks = int(document.get("ingestedChunks") or stored.get(key) or 0)
            else:
                chunks = stored.get(key) or int(document.get("availableChunks") or 0)
            row = {
                "id": key,
                "name": str(document.get("title") or document.get("name") or key),
                "sourceUrl": str(document.get("sourceUrl") or ""),
                "chunks": chunks,
                "indexed": indexed,
            }
            publisher = str(document.get("publisher") or "").strip()
            fmt = str(document.get("originalFormat") or document.get("format") or "").strip()
            if publisher:
                row["publisher"] = publisher
            if fmt:
                row["format"] = fmt
            rows.append(row)
        return rows

    def fold_into(self, alias_id: str, canonical_id: str) -> None:
        """Move one alias onto the canonical document. Chunk ids stay as stored."""
        if not alias_id or not canonical_id or alias_id == canonical_id:
            return
        alias = self.documents.get(alias_id)
        if alias is None:
            for chunk in self.chunks.values():
                if chunk.get("doc_id") == alias_id:
                    chunk["doc_id"] = canonical_id
            return
        current = self.documents.get(canonical_id)
        if current is None:
            moved = dict(alias)
            moved["id"] = canonical_id
            moved["documentKey"] = canonical_id
            self.documents[canonical_id] = moved
        else:
            for field in ("name", "sourceUrl", "publisher", "publishedAt", "originalFormat", "title", "storagePath"):
                if not current.get(field) and alias.get(field):
                    current[field] = alias[field]
            if not current.get("documentKey"):
                current["documentKey"] = canonical_id
        self.documents.pop(alias_id, None)
        for chunk in self.chunks.values():
            if chunk.get("doc_id") == alias_id:
                chunk["doc_id"] = canonical_id
        self.save()

    def replace_chunks(self, doc_id: str, name: str, source_url: str, chunks: list[dict], kind: str, storage_path: str = "", metadata: Optional[dict] = None) -> None:
        previous = self.documents.get(doc_id) or {}
        old_ids = {chunk_id for chunk_id, chunk in self.chunks.items() if chunk.get("doc_id") == doc_id}
        new_ids = []
        for chunk in chunks:
            chunk_id = str(chunk.get("chunk_id") or "")
            if not chunk_id or not str(chunk.get("content") or "").strip():
                continue
            chunk = dict(chunk)
            chunk["doc_id"] = doc_id
            chunk["chunk_id"] = chunk_id
            self.chunks[chunk_id] = chunk
            new_ids.append(chunk_id)
        for chunk_id in old_ids - set(new_ids):
            self.chunks.pop(chunk_id, None)
            stale = [key for key, value in self.source_map.items() if value == chunk_id]
            for key in stale:
                self.source_map.pop(key, None)
        extracted = [chunk_id for chunk_id in previous.get("extractedChunkIds") or [] if chunk_id in new_ids]
        indexed = bool(extracted) and set(extracted) == set(new_ids)
        record = {
            "id": doc_id,
            "name": name or previous.get("name") or doc_id,
            "sourceUrl": source_url or previous.get("sourceUrl") or "",
            "kind": previous.get("kind") if previous.get("kind") == "corpus" else kind,
            "storagePath": storage_path or previous.get("storagePath") or "",
            "indexed": indexed,
            "availableChunks": len(new_ids) or int(previous.get("availableChunks") or 0),
            "ingestedChunks": len(extracted),
            "extractedChunkIds": extracted,
        }
        lifted = dict(metadata or {})
        for chunk in chunks:
            if not lifted.get("publisher") and chunk.get("publisher"):
                lifted["publisher"] = chunk.get("publisher")
            if not lifted.get("publishedAt") and chunk.get("published_at"):
                lifted["publishedAt"] = chunk.get("published_at")
            if not lifted.get("originalFormat") and chunk.get("original_format"):
                lifted["originalFormat"] = chunk.get("original_format")
            if not lifted.get("title") and (chunk.get("document_title") or chunk.get("doc_name")):
                lifted["title"] = chunk.get("document_title") or chunk.get("doc_name")
            if not lifted.get("documentKey") and chunk.get("document_key"):
                lifted["documentKey"] = chunk.get("document_key")
            if not lifted.get("sourceUrl") and chunk.get("source_url"):
                lifted["sourceUrl"] = chunk.get("source_url")
        for field in ("title", "publisher", "publishedAt", "originalFormat", "documentKey", "sourceUrl"):
            value = str(lifted.get(field) or previous.get(field) or "").strip()
            if field == "sourceUrl" and record.get("sourceUrl"):
                value = record["sourceUrl"]
            if value:
                record[field] = value
        if record.get("documentKey"):
            record["documentKey"] = canonical_document_key(
                document_key=str(record.get("documentKey") or ""),
                source_url=str(record.get("sourceUrl") or ""),
                existing_id=doc_id,
            )
        else:
            record["documentKey"] = doc_id
        self.documents[doc_id] = record
        self.save()

    def register(self, doc_id: str, name: str, source_url: str = "", storage_path: str = "", metadata: Optional[dict] = None) -> None:
        meta = dict(metadata or {})
        current = self.documents.get(doc_id)
        if current is None:
            current = {
                "id": doc_id,
                "name": name or doc_id,
                "sourceUrl": source_url,
                "kind": "corpus" if is_corpus(doc_id) else "managed",
                "storagePath": storage_path,
                "indexed": False,
                "availableChunks": 0,
                "ingestedChunks": 0,
                "extractedChunkIds": [],
                "documentKey": doc_id,
            }
            self.documents[doc_id] = current
        else:
            current["name"] = name or current.get("name") or doc_id
            if source_url:
                current["sourceUrl"] = source_url
            if storage_path:
                current["storagePath"] = storage_path
            if not current.get("documentKey"):
                current["documentKey"] = doc_id
        for field, source in (
            ("title", "title"),
            ("publisher", "publisher"),
            ("publishedAt", "publishedAt"),
            ("originalFormat", "originalFormat"),
            ("documentKey", "documentKey"),
        ):
            value = str(meta.get(source) or "").strip()
            if value:
                current[field] = value
        if current.get("documentKey"):
            current["documentKey"] = canonical_document_key(
                document_key=str(current.get("documentKey") or ""),
                source_url=str(current.get("sourceUrl") or source_url or ""),
                existing_id=doc_id,
            )
        self.save()

    def register_metadata(self, doc_id: str, name: str, storage_path: str = "", metadata: Optional[dict] = None) -> bool:
        """Record a managed document id, name, and storage path.

        Does not read the file, clear extracted chunks, or save. The caller
        saves once when any record changed.
        """
        if not doc_id:
            return False
        current = self.documents.get(doc_id)
        meta = dict(metadata or {})
        if current is None:
            current = {
                "id": doc_id,
                "name": name or doc_id,
                "sourceUrl": str(meta.get("sourceUrl") or ""),
                "kind": "corpus" if is_corpus(doc_id) else "managed",
                "storagePath": storage_path,
                "indexed": False,
                "availableChunks": 0,
                "ingestedChunks": 0,
                "extractedChunkIds": [],
                "documentKey": doc_id,
            }
            for field in ("title", "publisher", "publishedAt", "originalFormat", "documentKey"):
                value = str(meta.get(field) or "").strip()
                if value:
                    current[field] = value
            self.documents[doc_id] = current
            return True
        changed = False
        next_name = name or current.get("name") or doc_id
        if current.get("name") != next_name:
            current["name"] = next_name
            changed = True
        if storage_path and current.get("storagePath") != storage_path:
            current["storagePath"] = storage_path
            changed = True
        source_url = str(meta.get("sourceUrl") or "").strip()
        if source_url and current.get("sourceUrl") != source_url:
            current["sourceUrl"] = source_url
            changed = True
        for field in ("title", "publisher", "publishedAt", "originalFormat", "documentKey"):
            value = str(meta.get(field) or "").strip()
            if value and current.get(field) != value:
                current[field] = value
                changed = True
        return changed

    def mark_extracted(self, doc_id: str, chunk_id: str) -> None:
        with self.lock:
            document = self.documents.setdefault(doc_id, {})
            extracted = list(document.get("extractedChunkIds") or [])
            if chunk_id not in extracted:
                extracted.append(chunk_id)
            document["extractedChunkIds"] = extracted
            document["ingestedChunks"] = len(extracted)
            self.save()

    def mark_document_done(self, doc_id: str) -> None:
        document = self.documents.get(doc_id)
        if not document:
            return
        extracted = list(document.get("extractedChunkIds") or [])
        stored = [chunk_id for chunk_id, chunk in self.chunks.items() if chunk.get("doc_id") == doc_id]
        document["ingestedChunks"] = len(extracted)
        document["indexed"] = bool(extracted) and set(extracted) >= set(stored) and bool(stored)
        # A partial extraction is still searchable. Indexed means every stored passage was sent through extraction.
        if extracted and not document["indexed"]:
            document["indexed"] = True
            document["partial"] = True
        self.save()

    def add_source_map(self, mapping: dict[str, str]) -> None:
        with self.lock:
            self.source_map.update(mapping)
            self.save()

    def delete(self, doc_id: str) -> list[str]:
        canonical = self.canonical_id(doc_id)
        document = self.documents.pop(canonical, None)
        removed = []
        for chunk_id, chunk in list(self.chunks.items()):
            if chunk.get("doc_id") == canonical or chunk_id == canonical:
                self.chunks.pop(chunk_id, None)
                removed.append(chunk_id)
        for key, value in list(self.source_map.items()):
            if value in removed or value == canonical:
                self.source_map.pop(key, None)
        if document:
            removed.extend(document.get("extractedChunkIds") or [])
        self.save()
        return removed

    def set_build(self, **fields) -> None:
        with self.lock:
            self.build.update(fields)
            self.build["updatedAt"] = _now()
            self.save()

    def chunk(self, chunk_id: str) -> Optional[dict]:
        chunk = self.chunks.get(chunk_id)
        return dict(chunk) if chunk else None
