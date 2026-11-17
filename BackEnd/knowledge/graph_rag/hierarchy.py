"""Deterministic document structure from catalog metadata.

LightRAG entities stay cross-document. This module does not call a model and
does not write the semantic projection.
"""

from __future__ import annotations

import hashlib
import re
from typing import Optional

from knowledge.graph_rag.catalog import chunk_document_key
from knowledge.graph_rag.graph_ops import ancestor_ids, enumerate_paths, lexical_score

_ARTICLE = re.compile(
    r"^(?:第\s*[0-9０-９一二三四五六七八九十百千万零〇]+\s*条|article\s+\d+|§\s*\d+)\b",
    re.I,
)
_UNPAGINATED = {"html", "htm", "csv", "json", "xml", "txt", "md", "markdown"}
_SLIDES = {"ppt", "pptx"}
_SHEETS = {"xls", "xlsx"}
_LOCATOR_TEXT = ("anchor", "path", "jsonPointer", "xpath", "sheet")
_LOCATOR_INT = ("rowStart", "rowEnd", "slide")
_LOCATOR_ALIAS = {"row_start": "rowStart", "row_end": "rowEnd", "json_pointer": "jsonPointer"}


def _digest(*parts: str) -> str:
    return hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()[:16]


def _edge_id(source: str, target: str, relation: str) -> str:
    return "e:" + hashlib.sha1(f"{source}\n{target}\n{relation}".encode("utf-8")).hexdigest()[:16]


def _format_name(chunk: dict, document: Optional[dict] = None) -> str:
    document = document or {}
    return str(
        chunk.get("original_format")
        or chunk.get("format")
        or document.get("originalFormat")
        or document.get("format")
        or ""
    ).strip().lower()


def unpaginated(fmt: str) -> bool:
    return str(fmt or "").strip().lower() in _UNPAGINATED


def chunk_locator(chunk: dict, document: Optional[dict] = None) -> Optional[dict]:
    """Native locator. Unpaginated formats do not gain a PDF page."""
    document = document or {}
    raw = chunk.get("locator") if isinstance(chunk.get("locator"), dict) else {}
    merged = dict(raw)
    for source, target in _LOCATOR_ALIAS.items():
        if merged.get(target) in (None, "") and merged.get(source) not in (None, ""):
            merged[target] = merged[source]
    fmt = _format_name(chunk, document)
    try:
        page_number = int(chunk.get("page_number")) if chunk.get("page_number") not in (None, "") else 0
    except (TypeError, ValueError):
        page_number = 0
    label = str(merged.get("label") or chunk.get("heading") or chunk.get("section_title") or "").strip()
    if not label:
        label = str(document.get("title") or document.get("name") or chunk.get("doc_name") or "").strip()
    if not label:
        return None
    locator = {"label": label}
    if fmt in _SLIDES:
        slide = merged.get("slide")
        if slide in (None, "") and page_number > 0:
            slide = page_number
        if slide not in (None, ""):
            try:
                locator["slide"] = int(slide)
            except (TypeError, ValueError):
                pass
    elif fmt in _SHEETS:
        sheet = merged.get("sheet")
        if sheet not in (None, ""):
            locator["sheet"] = str(sheet)
    elif not unpaginated(fmt):
        page = merged.get("page")
        if page in (None, "") and page_number > 0:
            page = page_number
        try:
            page_value = int(page) if page not in (None, "") else 0
        except (TypeError, ValueError):
            page_value = 0
        if page_value > 0:
            locator["page"] = page_value
    for key in _LOCATOR_TEXT:
        if key == "sheet" and "sheet" in locator:
            continue
        value = merged.get(key)
        if value not in (None, ""):
            locator[key] = str(value)
    for key in _LOCATOR_INT:
        if key in locator or key == "slide" and fmt in _SLIDES:
            continue
        value = merged.get(key)
        if value in (None, ""):
            continue
        try:
            locator[key] = int(value)
        except (TypeError, ValueError):
            continue
    return locator


def _section_path(chunk: dict, title: str) -> list[str]:
    raw = chunk.get("section_path")
    parts: list[str] = []
    if isinstance(raw, list):
        parts = [str(item).strip() for item in raw if str(item).strip()]
    elif isinstance(raw, str) and raw.strip():
        parts = [item.strip() for item in re.split(r"\s*(?:>|/)\s*", raw) if item.strip()]
    if not parts:
        parent = str(chunk.get("parent_heading") or "").strip()
        section = str(chunk.get("section_title") or "").strip()
        heading = str(chunk.get("heading") or "").strip()
        if parent and parent not in {title, section, heading}:
            parts.append(parent)
        leaf = section or heading
        if leaf and leaf != title and leaf not in parts:
            parts.append(leaf)
    if title:
        parts = [part for part in parts if part != title]
    return parts


def _explicit_record(chunk: dict, leaf: str) -> Optional[tuple[str, str]]:
    source_type = str(chunk.get("source_type") or "").strip().lower()
    if source_type in {"article", "paragraph", "record"}:
        label = str(chunk.get(source_type) or leaf or source_type).strip()
        return source_type, label or source_type
    for key, kind in (("record_id", "record"), ("paragraph", "paragraph"), ("article", "article")):
        value = str(chunk.get(key) or "").strip()
        if value:
            return kind, value
    if leaf and _ARTICLE.match(leaf):
        return "article", leaf
    return None


def _leaf_type(chunk: dict) -> str:
    source_type = str(chunk.get("source_type") or "").strip().lower()
    if source_type == "table":
        return "table"
    if source_type in {"figure", "image"}:
        return "figure"
    return "chunk"


def _provenance(chunk: Optional[dict], document: dict, method: str, quote: str = "", span: Optional[dict] = None) -> dict:
    chunk = chunk or {}
    doc_id = str(chunk.get("_doc_key") or document.get("id") or chunk.get("doc_id") or "")
    item = {"docId": doc_id, "method": method}
    source_id = str(chunk.get("chunk_id") or "")
    if source_id:
        item["sourceId"] = source_id
    url = str(chunk.get("source_url") or document.get("sourceUrl") or "")
    if url:
        item["sourceUrl"] = url
    locator = chunk_locator(chunk, document) if chunk else None
    if locator is None and document.get("title"):
        locator = {"label": str(document.get("title") or document.get("name") or doc_id)}
    if locator:
        item["locator"] = locator
    if quote:
        item["quote"] = quote[:240]
    if span:
        item["span"] = {"start": int(span["start"]), "end": int(span["end"])}
    return item


def _as_list(value) -> list:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    return [value]


def _lookup_documents(grouped: dict[str, dict]) -> dict[str, str]:
    by_token: dict[str, str] = {}
    for doc_id, document in grouped.items():
        by_token[doc_id] = doc_id
        for field in ("documentKey", "title", "name", "sourceUrl"):
            token = str(document.get(field) or "").strip()
            if token:
                by_token[token.casefold()] = doc_id
    return by_token


def _resolve_target(item, tokens: dict[str, str], documents: dict[str, dict]) -> str:
    if isinstance(item, dict):
        for field in ("docId", "documentKey", "id", "document_key"):
            token = str(item.get(field) or "").strip()
            if token and token in documents:
                return token
            if token and token.casefold() in tokens:
                return tokens[token.casefold()]
        url = str(item.get("sourceUrl") or item.get("source_url") or "").strip()
        title = str(item.get("title") or item.get("name") or "").strip()
    else:
        token = str(item or "").strip()
        if not token:
            return ""
        if token in documents:
            return token
        if token.casefold() in tokens:
            return tokens[token.casefold()]
        url = token if token.startswith("http") else ""
        title = "" if url else token
    if url:
        found = tokens.get(url.casefold())
        if found:
            return found
    if title:
        return tokens.get(title.casefold(), "")
    return ""


def _find_span(text: str, quote: str) -> Optional[dict]:
    start = text.find(quote)
    if start < 0:
        return None
    return {"start": start, "end": start + len(quote)}


_AC_IDENTIFIER = re.compile(
    r"\bAC[\s_-]*(\d{2,3})[\s_-]+(\d{2,3})[\s_-]*([A-Za-z])?(?!\w)",
    re.I,
)


def _ac_identifier_key(number: str, series: str, revision: str) -> str:
    """Keep the revision letter so an unversioned AC number stays distinct."""
    return f"ac-{number}-{series}{revision}".lower()


def _ac_identifiers(text: str) -> list[tuple[str, str, dict]]:
    found: list[tuple[str, str, dict]] = []
    for match in _AC_IDENTIFIER.finditer(str(text or "")):
        number, series, revision = match.group(1), match.group(2), match.group(3) or ""
        found.append(
            (
                _ac_identifier_key(number, series, revision),
                match.group(0),
                {"start": match.start(), "end": match.end()},
            )
        )
    return found


def _storage_filename(document: dict) -> str:
    raw = str(document.get("storagePath") or "").strip().replace("\\", "/")
    return raw.rsplit("/", 1)[-1]


def _known_ac_targets(documents: dict[str, dict]) -> dict[str, str]:
    """Exact AC ids taken from title, name, and storagePath filename only."""
    owners: dict[str, set[str]] = {}
    for doc_id, document in documents.items():
        keys: set[str] = set()
        for value in (document.get("title"), document.get("name"), _storage_filename(document)):
            for key, _quote, _span in _ac_identifiers(str(value or "")):
                keys.add(key)
        for key in keys:
            owners.setdefault(key, set()).add(doc_id)
    return {key: next(iter(doc_ids)) for key, doc_ids in owners.items() if len(doc_ids) == 1}


def build_structure(documents: Optional[dict], chunks: Optional[dict], semantic_nodes: Optional[list] = None) -> dict:
    """Parent chain for every stored chunk. Missing metadata does not invent a level."""
    documents = documents or {}
    chunks = chunks or {}
    grouped: dict[str, list] = {}
    chunk_by_id: dict[str, dict] = {}
    for chunk_id in sorted(chunks):
        chunk = chunks[chunk_id]
        if not isinstance(chunk, dict):
            continue
        if not str(chunk.get("content") or "").strip():
            continue
        doc_id = chunk_document_key(chunk)
        if not doc_id:
            continue
        copied = dict(chunk)
        copied["_doc_key"] = doc_id
        copied["chunk_id"] = str(copied.get("chunk_id") or chunk_id)
        grouped.setdefault(doc_id, []).append(copied)
        chunk_by_id[copied["chunk_id"]] = copied
    if not grouped:
        return {"nodes": [], "edges": []}

    doc_meta: dict[str, dict] = {}
    for doc_id, rows in grouped.items():
        current = dict(documents.get(doc_id) or {})
        for key, document in documents.items():
            if key == doc_id or not isinstance(document, dict):
                continue
            alias = chunk_document_key({"doc_id": key, "document_key": document.get("documentKey") or "", "source_url": document.get("sourceUrl") or ""})
            if alias != doc_id:
                continue
            for field in ("name", "title", "sourceUrl", "publisher", "publishedAt", "originalFormat", "format", "documentKey"):
                if not current.get(field) and document.get(field):
                    current[field] = document[field]
        for row in rows:
            if not current.get("publisher") and row.get("publisher"):
                current["publisher"] = row.get("publisher")
            if not current.get("sourceUrl") and row.get("source_url"):
                current["sourceUrl"] = row.get("source_url")
            if not current.get("originalFormat") and row.get("original_format"):
                current["originalFormat"] = row.get("original_format")
            if not current.get("publishedAt") and row.get("published_at"):
                current["publishedAt"] = row.get("published_at")
            if not current.get("title"):
                current["title"] = row.get("document_title") or row.get("doc_name") or ""
            if not current.get("name"):
                current["name"] = current.get("title") or doc_id
        current["id"] = doc_id
        current["title"] = str(current.get("title") or current.get("name") or doc_id)
        current["name"] = str(current.get("name") or current["title"])
        doc_meta[doc_id] = current

    nodes: dict[str, dict] = {}
    sources: dict[str, dict] = {}

    def add_node(node: dict) -> dict:
        current = nodes.get(node["id"])
        if current is None:
            nodes[node["id"]] = node
            return node
        return current

    for doc_id, document in doc_meta.items():
        publisher = str(document.get("publisher") or "").strip()
        parent_id = ""
        if publisher:
            source_id = "src:" + _digest(publisher.casefold())
            source = sources.get(source_id)
            if source is None:
                source = {
                    "id": source_id,
                    "label": publisher,
                    "type": "source",
                    "kind": "structure",
                    "sourceIds": [],
                    "locator": {"label": publisher},
                }
                sources[source_id] = source
                nodes[source_id] = source
            parent_id = source_id
        doc_node = {
            "id": f"doc:{doc_id}",
            "label": document["title"],
            "type": "document",
            "kind": "structure",
            "docId": doc_id,
            "sourceIds": [],
            "locator": {"label": document["title"]},
        }
        if parent_id:
            doc_node["parentId"] = parent_id
        add_node(doc_node)
        for chunk in grouped[doc_id]:
            title = document["title"]
            parts = _section_path(chunk, title)
            record = None
            if parts and _ARTICLE.match(parts[-1]):
                record = ("article", parts.pop())
            explicit = _explicit_record(chunk, parts[-1] if parts else "")
            if explicit:
                if record is None or explicit[0] != "article":
                    record = explicit
                    if parts and parts[-1] == explicit[1]:
                        parts.pop()
            parent = doc_node
            for index, name in enumerate(parts):
                sec_id = "sec:" + _digest(doc_id, "/".join(parts[: index + 1]))
                section = nodes.get(sec_id)
                if section is None:
                    section = {
                        "id": sec_id,
                        "label": name,
                        "type": "section",
                        "kind": "structure",
                        "docId": doc_id,
                        "parentId": parent["id"],
                        "sourceIds": [],
                        "locator": {"label": name},
                    }
                    nodes[sec_id] = section
                parent = section
            if record:
                rec_id = "rec:" + _digest(doc_id, parent["id"], record[0], record[1])
                article = nodes.get(rec_id)
                if article is None:
                    article = {
                        "id": rec_id,
                        "label": record[1],
                        "type": record[0],
                        "kind": "structure",
                        "docId": doc_id,
                        "parentId": parent["id"],
                        "sourceIds": [],
                        "locator": {"label": record[1]},
                    }
                    nodes[rec_id] = article
                parent = article
            leaf_type = _leaf_type(chunk)
            leaf_id = "chk:" + chunk["chunk_id"]
            locator = chunk_locator(chunk, document)
            preview = " ".join(str(chunk.get("content") or "").split())[:80].rstrip()
            leaf = {
                "id": leaf_id,
                "label": preview or chunk["chunk_id"],
                "type": leaf_type,
                "kind": "structure",
                "docId": doc_id,
                "parentId": parent["id"],
                "sourceIds": [chunk["chunk_id"]],
            }
            if locator:
                leaf["locator"] = locator
            nodes[leaf_id] = leaf

    children: dict[str, list[str]] = {}
    for node in nodes.values():
        parent_id = node.get("parentId")
        if parent_id:
            children.setdefault(parent_id, []).append(node["id"])

    def rollup(node_id: str) -> list[str]:
        node = nodes[node_id]
        if node["type"] in {"chunk", "table", "figure"}:
            return list(node["sourceIds"])
        found: list[str] = []
        for child_id in children.get(node_id, []):
            for source_id in rollup(child_id):
                if source_id not in found:
                    found.append(source_id)
        node["sourceIds"] = found
        return found

    for node in list(nodes.values()):
        if not node.get("parentId"):
            rollup(node["id"])

    edges: dict[str, dict] = {}

    def add_edge(source: str, target: str, relation: str, edge_kind: str, source_ids: list[str], provenance: list[dict]) -> None:
        edge_key = _edge_id(source, target, relation)
        current = edges.get(edge_key)
        if current is None:
            edges[edge_key] = {
                "id": edge_key,
                "source": source,
                "target": target,
                "label": relation,
                "sourceIds": list(source_ids),
                "edgeKind": edge_kind,
                "relationType": relation,
                "directed": True,
                "provenance": list(provenance),
            }
            return
        for source_id in source_ids:
            if source_id not in current["sourceIds"]:
                current["sourceIds"].append(source_id)
        seen = {(item.get("sourceId"), item.get("quote"), item.get("method")) for item in current["provenance"]}
        for item in provenance:
            mark = (item.get("sourceId"), item.get("quote"), item.get("method"))
            if mark not in seen:
                current["provenance"].append(item)
                seen.add(mark)

    # Parent -> child only. The reverse edge would connect sibling chunks through the document.
    for node in nodes.values():
        parent_id = node.get("parentId")
        if not parent_id or parent_id not in nodes:
            continue
        document = doc_meta.get(str(node.get("docId") or ""), {"id": node.get("docId") or ""})
        provenance = []
        for source_id in node.get("sourceIds") or []:
            provenance.append(_provenance(chunk_by_id.get(source_id), document, "structure"))
        if not provenance and node.get("docId"):
            provenance.append(_provenance(None, document, "structure"))
        add_edge(parent_id, node["id"], "CONTAINS", "contains", list(node.get("sourceIds") or []), provenance)

    tokens = _lookup_documents(doc_meta)
    ac_targets = _known_ac_targets(doc_meta)
    for doc_id, document in doc_meta.items():
        doc_node_id = f"doc:{doc_id}"
        for relation, field in (("AMENDS", "amends"), ("SUPERSEDES", "supersedes"), ("REFERS_TO", "references")):
            for item in _as_list(document.get(field)):
                target = _resolve_target(item, tokens, doc_meta)
                if not target or target == doc_id:
                    continue
                add_edge(
                    doc_node_id,
                    f"doc:{target}",
                    relation,
                    "references",
                    [],
                    [_provenance(None, document, "explicit_metadata")],
                )
        for chunk in grouped[doc_id]:
            text = str(chunk.get("content") or "")
            leaf_id = "chk:" + chunk["chunk_id"]
            for relation, field in (("AMENDS", "amends"), ("SUPERSEDES", "supersedes"), ("REFERS_TO", "references")):
                for item in _as_list(chunk.get(field)):
                    target = _resolve_target(item, tokens, doc_meta)
                    if not target or target == doc_id:
                        continue
                    quote = ""
                    span = None
                    if isinstance(item, dict):
                        quote = str(item.get("quote") or "")
                        if quote:
                            span = _find_span(text, quote)
                    add_edge(
                        leaf_id,
                        f"doc:{target}",
                        relation,
                        "references",
                        [chunk["chunk_id"]],
                        [_provenance(chunk, document, "explicit_metadata", quote, span)],
                    )
            for other_id, other in doc_meta.items():
                if other_id == doc_id:
                    continue
                url = str(other.get("sourceUrl") or "")
                title = str(other.get("title") or "")
                if len(url) >= 12 and url in text:
                    add_edge(
                        leaf_id,
                        f"doc:{other_id}",
                        "REFERS_TO",
                        "references",
                        [chunk["chunk_id"]],
                        [_provenance(chunk, document, "explicit_url", url, _find_span(text, url))],
                    )
                if len(title) < 6:
                    continue
                for quoted in (f"《{title}》", f'"{title}"', f"“{title}”"):
                    if quoted in text:
                        add_edge(
                            leaf_id,
                            f"doc:{other_id}",
                            "REFERS_TO",
                            "references",
                            [chunk["chunk_id"]],
                            [_provenance(chunk, document, "quoted_title", quoted, _find_span(text, quoted))],
                        )
                        break
            for identifier, quote, span in _ac_identifiers(text):
                target = ac_targets.get(identifier, "")
                if not target or target == doc_id:
                    continue
                add_edge(
                    leaf_id,
                    f"doc:{target}",
                    "REFERS_TO",
                    "references",
                    [chunk["chunk_id"]],
                    [_provenance(chunk, document, "explicit_identifier", quote, span)],
                )

    for entity in semantic_nodes or []:
        if not isinstance(entity, dict) or not entity.get("id"):
            continue
        by_doc: dict[str, list[str]] = {}
        for source_id in entity.get("sourceIds") or []:
            chunk = chunk_by_id.get(str(source_id))
            if not chunk:
                stored = chunks.get(str(source_id))
                if not isinstance(stored, dict):
                    continue
                doc_key = chunk_document_key(stored)
                chunk = dict(stored)
                chunk["_doc_key"] = doc_key
                chunk["chunk_id"] = str(source_id)
            doc_key = str(chunk.get("_doc_key") or chunk_document_key(chunk))
            if doc_key:
                by_doc.setdefault(doc_key, []).append(str(source_id))
        for doc_key, source_ids in by_doc.items():
            if f"doc:{doc_key}" not in nodes:
                continue
            document = doc_meta.get(doc_key, {"id": doc_key})
            provenance = [
                _provenance(chunk_by_id.get(source_id) or chunks.get(source_id), document, "extracted_sources")
                for source_id in source_ids
            ]
            add_edge(f"doc:{doc_key}", str(entity["id"]), "MENTIONS", "mentions", source_ids, provenance)

    roots = [node_id for node_id, node in nodes.items() if not node.get("parentId") or node["parentId"] not in nodes]
    stack = [(node_id, 0) for node_id in roots]
    seen_level: set[str] = set()
    while stack:
        node_id, level = stack.pop()
        if node_id in seen_level:
            continue
        seen_level.add(node_id)
        node = nodes[node_id]
        if node.get("parentId") not in nodes:
            node.pop("parentId", None)
        node["level"] = level
        for child_id in children.get(node_id, []):
            stack.append((child_id, level + 1))
    ordered = sorted(nodes.values(), key=lambda node: (int(node.get("level") or 0), node["id"]))
    return {"nodes": ordered, "edges": [edges[key] for key in sorted(edges)]}


def select_structure(graph: dict, query: str, limit: int, source_ids: Optional[list[str]] = None) -> dict:
    """Hierarchy browser. Shown nodes keep their ancestors; omitted children stay omitted."""
    nodes = [node for node in graph.get("nodes") or [] if node.get("kind") == "structure"]
    node_ids = {node["id"] for node in nodes}
    edges = [
        edge
        for edge in graph.get("edges") or []
        if edge.get("source") in node_ids
        and edge.get("target") in node_ids
        and edge.get("edgeKind") in {"contains", "references"}
    ]
    query = str(query or "").strip()
    limit = max(int(limit), 0)
    if not query:
        ordered = sorted(nodes, key=lambda node: (int(node.get("level") or 0), node["id"]))
        chosen = [node["id"] for node in ordered[:limit]]
        truncated = len(ordered) > len(chosen)
    else:
        seeded: list[str] = []
        seen_seed: set[str] = set()
        if source_ids:
            leaves = {"chunk", "table", "figure"}
            leaf_for_source: dict[str, str] = {}
            for node in nodes:
                if node.get("type") not in leaves:
                    continue
                for source_id in node.get("sourceIds") or []:
                    leaf_for_source.setdefault(str(source_id), node["id"])
            for source_id in source_ids:
                leaf_id = leaf_for_source.get(str(source_id))
                if leaf_id and leaf_id not in seen_seed:
                    seeded.append(leaf_id)
                    seen_seed.add(leaf_id)
        ranked = []
        for node in nodes:
            score = lexical_score(query, str(node.get("label") or ""))
            locator = node.get("locator") or {}
            if isinstance(locator, dict):
                score = max(score, lexical_score(query, str(locator.get("label") or "")))
            if score > 0:
                ranked.append((score, node["id"]))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        chosen = [node_id for _score, node_id in ranked[:limit]]
        truncated = len(ranked) > len(chosen)
        if seeded:
            chosen = seeded + [node_id for node_id in chosen if node_id not in seen_seed]
    if not chosen:
        return {"nodes": [], "edges": [], "paths": [], "truncated": False}
    keep = set(ancestor_ids(nodes, edges, chosen))
    for edge in edges:
        if edge.get("edgeKind") == "references" and edge.get("source") in keep and edge.get("target") not in keep:
            keep.add(edge["target"])
    keep.update(ancestor_ids(nodes, edges, list(keep)))
    selected = [node for node in sorted(nodes, key=lambda node: (int(node.get("level") or 0), node["id"])) if node["id"] in keep]
    selected_ids = {node["id"] for node in selected}
    kept_edges = [edge for edge in edges if edge.get("source") in selected_ids and edge.get("target") in selected_ids]
    paths, path_truncated = enumerate_paths(selected, kept_edges, chosen[:6], scope="hierarchy")
    return {"nodes": selected, "edges": kept_edges, "paths": paths, "truncated": bool(truncated or path_truncated)}
