"""Pure graph projection, document filters, and path limits.

Paths are structural walks over extracted edges. They are not causal claims.
"""

from __future__ import annotations

import hashlib
import re
from typing import Iterable, Optional

from knowledge.graph_rag.config import CAUSATION_NOTE, MAX_PATH_EDGES, MAX_PATHS

_CJK = re.compile(r"[\u4e00-\u9fff]")
_LATIN = re.compile(r"[a-z0-9]+", re.I)
_CITE = re.compile(r"\[(\d+)\]")
_SEP = "<SEP>"


def causation_note() -> str:
    return CAUSATION_NOTE


def split_source_ids(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        parts: list[str] = []
        for item in value:
            parts.extend(split_source_ids(item))
        return parts
    text = str(value).strip()
    if not text:
        return []
    for separator in (_SEP, "|", ";", ","):
        if separator in text:
            return [part.strip() for part in text.split(separator) if part.strip()]
    return [text]


def tokens(text: str) -> set[str]:
    raw = str(text or "").lower()
    found = {token for token in _LATIN.findall(raw) if len(token) >= 2}
    chars = _CJK.findall(raw)
    found.update(chars)
    found.update("".join(chars[index : index + 2]) for index in range(len(chars) - 1))
    return found


def lexical_score(query: str, text: str) -> float:
    query_tokens = tokens(query)
    if not query_tokens:
        return 0.0
    overlap = query_tokens & tokens(text)
    score = len(overlap) / len(query_tokens)
    compact_query = re.sub(r"\s+", "", str(query or "").lower())
    compact_text = re.sub(r"\s+", "", str(text or "").lower())
    if compact_query and compact_query in compact_text:
        score += 0.35
    return score


def split_windows(text: str, size: int = 1400, overlap: int = 120) -> list[str]:
    """Split one original passage into extraction windows without changing short text."""
    body = str(text or "").strip()
    if not body:
        return []
    if len(body) <= size:
        return [body]
    windows: list[str] = []
    start = 0
    while start < len(body):
        end = min(len(body), start + size)
        if end < len(body):
            newline = body.rfind("\n", start + size // 2, end)
            if newline > start:
                end = newline
        piece = body[start:end].strip()
        if piece:
            windows.append(piece)
        if end >= len(body):
            break
        start = max(end - overlap, start + 1)
    return windows


def _edge_id(source: str, target: str, label: str) -> str:
    digest = hashlib.sha1(f"{source}\n{target}\n{label}".encode("utf-8")).hexdigest()[:16]
    return f"e:{digest}"


def _first(data: dict, keys: tuple[str, ...], default: str = "") -> str:
    for key in keys:
        value = data.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def _source_values(data: dict) -> list[str]:
    """Passage ids stored on a LightRAG row.

    Endpoint fields are also named ``source`` after GraphML is read. Those are
    entity ids, not the passages that support the edge.
    """
    for key in ("source_id", "source_ids", "chunk_id", "chunk_ids"):
        if data.get(key):
            return split_source_ids(data[key])
    for key, value in data.items():
        if str(key).lower() in {"source", "src", "src_id", "target", "tgt", "tgt_id"}:
            continue
        if "source" in str(key).lower() and value:
            return split_source_ids(value)
    for value in data.values():
        if isinstance(value, str) and "chunk-" in value:
            return split_source_ids(value)
    return []


def project_graph(raw_nodes: list[dict], raw_edges: list[dict], source_map: dict[str, str], chunks: dict[str, dict]) -> dict:
    """Keep only nodes and edges whose LightRAG sources resolve to stored passages."""
    nodes: list[dict] = []
    for raw in raw_nodes:
        node_id = str(raw.get("id") or raw.get("entity_name") or "").strip()
        if not node_id:
            continue
        source_ids = _original_sources(_source_values(raw), source_map, chunks)
        if not source_ids:
            continue
        label = _first(raw, ("entity_name", "name", "label"), node_id)
        nodes.append(
            {
                "id": node_id,
                "label": label,
                "type": _first(raw, ("entity_type", "type"), "entity"),
                "sourceIds": source_ids,
                "description": _first(raw, ("description", "summary"), ""),
            }
        )
    node_ids = {node["id"] for node in nodes}
    edges: list[dict] = []
    seen: set[str] = set()
    for raw in raw_edges:
        source = str(raw.get("source") or raw.get("src_id") or "").strip()
        target = str(raw.get("target") or raw.get("tgt_id") or "").strip()
        if source not in node_ids or target not in node_ids or source == target:
            continue
        source_ids = _original_sources(_source_values(raw), source_map, chunks)
        if not source_ids:
            continue
        label = _first(raw, ("keywords", "relation", "label", "description"), "related_to")
        label = re.sub(r"\s+", " ", label)[:80]
        edge_id = _edge_id(source, target, label)
        if edge_id in seen:
            continue
        seen.add(edge_id)
        edges.append(
            {
                "id": edge_id,
                "source": source,
                "target": target,
                "label": label,
                "sourceIds": source_ids,
            }
        )
    return {"nodes": nodes, "edges": edges}


def _original_sources(raw_ids: Iterable[str], source_map: dict[str, str], chunks: dict[str, dict]) -> list[str]:
    found: list[str] = []
    for raw_id in raw_ids:
        original = source_map.get(raw_id, raw_id)
        if original in chunks and original not in found:
            found.append(original)
    return found


def edge_kind(edge: dict) -> str:
    kind = str(edge.get("edgeKind") or "semantic")
    if kind not in {"contains", "mentions", "references", "semantic"}:
        return "semantic"
    return kind


def edge_directed(edge: dict) -> bool:
    if "directed" in edge and edge.get("directed") is not None:
        return bool(edge.get("directed"))
    return edge_kind(edge) != "semantic"


def _chunk_doc(chunk: dict) -> str:
    from knowledge.graph_rag.catalog import chunk_document_key

    return chunk_document_key(chunk) or str(chunk.get("doc_id") or "")


def ancestor_ids(nodes: list[dict], edges: list[dict], selected: Iterable[str]) -> list[str]:
    """Parents of shown structure nodes. Children outside the selection stay out."""
    parent: dict[str, str] = {}
    known = set()
    for node in nodes:
        node_id = str(node.get("id") or "")
        if not node_id:
            continue
        known.add(node_id)
        parent_id = str(node.get("parentId") or "")
        if parent_id:
            parent[node_id] = parent_id
    for edge in edges:
        if edge_kind(edge) != "contains":
            continue
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        if source and target:
            parent.setdefault(target, source)
    keep: list[str] = []
    seen: set[str] = set()
    stack = [str(item) for item in selected if str(item)]
    while stack:
        node_id = stack.pop()
        if node_id in seen:
            continue
        seen.add(node_id)
        keep.append(node_id)
        parent_id = parent.get(node_id, "")
        if parent_id and parent_id not in seen:
            stack.append(parent_id)
    return [node_id for node_id in keep if node_id in known or node_id in parent.values() or node_id in seen]


def filter_graph(graph: dict, doc_ids: Optional[list[str]], chunks: dict[str, dict]) -> dict:
    """Drop anything whose passages are outside the selected documents.

    A kept structure node keeps its ancestors, so a filtered child is not detached.
    """
    allowed = None if doc_ids is None else {str(item) for item in doc_ids}

    def source_ok(source_id: str) -> bool:
        chunk = chunks.get(source_id)
        if not chunk:
            return False
        if allowed is None:
            return True
        doc_id = str(chunk.get("doc_id") or "")
        return doc_id in allowed or _chunk_doc(chunk) in allowed

    def trim_sources(source_ids: list[str]) -> list[str]:
        kept = []
        for source_id in source_ids:
            if source_ok(str(source_id)) and source_id not in kept:
                kept.append(source_id)
        return kept

    raw_nodes = []
    for node in graph.get("nodes") or []:
        copied = dict(node)
        copied["sourceIds"] = trim_sources(list(node.get("sourceIds") or []))
        raw_nodes.append(copied)
    by_id = {node["id"]: node for node in raw_nodes if node.get("id")}

    def initially_kept(node: dict) -> bool:
        kind = node.get("kind") or "semantic"
        if allowed is None:
            return bool(node.get("sourceIds")) or kind == "structure"
        if kind != "structure":
            return bool(node.get("sourceIds"))
        doc_id = str(node.get("docId") or "")
        if doc_id and doc_id not in allowed:
            return False
        if doc_id in allowed:
            return bool(node.get("sourceIds")) or node.get("type") == "document"
        return False

    kept = {node["id"] for node in raw_nodes if node.get("id") and initially_kept(node)}
    kept.update(node_id for node_id in ancestor_ids(raw_nodes, graph.get("edges") or [], list(kept)) if node_id in by_id)
    nodes = [by_id[node["id"]] for node in raw_nodes if node.get("id") in kept]
    node_ids = {node["id"] for node in nodes}
    edges = []
    for edge in graph.get("edges") or []:
        if edge.get("source") not in node_ids or edge.get("target") not in node_ids:
            continue
        kind = edge_kind(edge)
        source_ids = trim_sources(list(edge.get("sourceIds") or []))
        if kind == "semantic" and not source_ids:
            continue
        copied = dict(edge)
        copied["sourceIds"] = source_ids
        if allowed is not None and copied.get("provenance"):
            copied["provenance"] = [
                item
                for item in copied["provenance"]
                if not item.get("docId") or str(item.get("docId")) in allowed
            ]
        edges.append(copied)
    return {"nodes": nodes, "edges": edges}


def _scope_kind(kind: str, scope: str) -> bool:
    if scope == "hierarchy":
        return kind in {"contains", "references"}
    if scope == "semantic":
        return kind == "semantic"
    return kind == "semantic"


def enumerate_paths(nodes: list[dict], edges: list[dict], seeds: list[str], max_edges: int = MAX_PATH_EDGES, limit: int = MAX_PATHS, scope: str = "semantic") -> tuple[list[dict], bool]:
    """Walk edges in their stored direction, at most `max_edges` long.

    Semantic LightRAG edges have no stored direction. CONTAINS and MENTIONS do
    not count as semantic hops, and CONTAINS is not walked upward.
    """
    if scope == "all":
        semantic, semantic_cut = enumerate_paths(nodes, edges, seeds, max_edges=max_edges, limit=limit, scope="semantic")
        remaining = max(int(limit) - len(semantic), 0)
        hierarchy, hierarchy_cut = enumerate_paths(
            nodes, edges, seeds, max_edges=max_edges, limit=remaining, scope="hierarchy"
        )
        combined = semantic + hierarchy
        return combined[:limit], bool(semantic_cut or hierarchy_cut or len(semantic) + len(hierarchy) > limit)
    max_edges = max(1, min(int(max_edges), MAX_PATH_EDGES))
    limit = max(int(limit), 0)
    adjacency: dict[str, list[tuple[str, str, str]]] = {}
    for edge in edges:
        kind = edge_kind(edge)
        if not _scope_kind(kind, scope):
            continue
        adjacency.setdefault(edge["source"], []).append((edge["target"], edge["id"], kind))
        if not edge_directed(edge):
            adjacency.setdefault(edge["target"], []).append((edge["source"], edge["id"], kind))
    paths: list[dict] = []
    seen: set[tuple[str, ...]] = set()
    truncated = False

    def add_path(node_ids: list[str], edge_ids: list[str], kinds: list[str]) -> None:
        nonlocal truncated
        if not edge_ids or len(edge_ids) > max_edges:
            return
        key = tuple(edge_ids)
        if key in seen:
            return
        if len(paths) >= limit:
            truncated = True
            return
        seen.add(key)
        paths.append(
            {
                "nodeIds": list(node_ids),
                "edgeIds": list(edge_ids),
                "hops": len(edge_ids),
                "semanticHops": sum(1 for item in kinds if item == "semantic"),
                "hierarchyDepth": sum(1 for item in kinds if item == "contains"),
            }
        )

    def walk(node_id: str, node_ids: list[str], edge_ids: list[str], kinds: list[str]) -> None:
        nonlocal truncated
        if len(paths) >= limit:
            truncated = True
            return
        if edge_ids:
            add_path(node_ids, edge_ids, kinds)
        if len(edge_ids) >= max_edges or len(paths) >= limit:
            return
        for neighbor, edge_id, kind in adjacency.get(node_id, []):
            if edge_id in edge_ids or neighbor in node_ids:
                continue
            walk(neighbor, node_ids + [neighbor], edge_ids + [edge_id], kinds + [kind])
            if len(paths) >= limit:
                truncated = True
                return

    known = {node["id"] for node in nodes}
    seed_set = [seed for seed in seeds if seed in known]
    if not seed_set:
        seed_set = [node["id"] for node in nodes[:6]]
    for seed in seed_set:
        walk(seed, [seed], [], [])
        if len(paths) >= limit:
            truncated = True
            break
    paths.sort(key=lambda item: (-item["semanticHops"], -item["hops"], item["nodeIds"]))
    return paths[:limit], truncated


def graph_with_paths(graph: dict, seeds: list[str], max_edges: int = MAX_PATH_EDGES, scope: str = "semantic") -> dict:
    paths, truncated = enumerate_paths(
        graph.get("nodes") or [], graph.get("edges") or [], seeds, max_edges=max_edges, scope=scope
    )
    return {
        "nodes": list(graph.get("nodes") or []),
        "edges": list(graph.get("edges") or []),
        "paths": paths,
        "truncated": bool(truncated or graph.get("truncated")),
    }


def _public_locator(raw: dict) -> Optional[dict]:
    label = str(raw.get("label") or "").strip()
    if not label:
        return None
    locator = {"label": label}
    page = raw.get("page")
    try:
        page_value = int(page) if page not in (None, "") else 0
    except (TypeError, ValueError):
        page_value = 0
    if page_value > 0:
        locator["page"] = page_value
    for key in ("anchor", "path", "jsonPointer", "xpath", "sheet"):
        value = raw.get(key)
        if value not in (None, ""):
            locator[key] = str(value)
    for key in ("rowStart", "rowEnd", "slide"):
        value = raw.get(key)
        if value in (None, ""):
            continue
        try:
            locator[key] = int(value)
        except (TypeError, ValueError):
            continue
    return locator


def _public_provenance(items: list, chunks: Optional[dict] = None) -> list[dict]:
    rows = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        doc_id = str(item.get("docId") or "")
        method = str(item.get("method") or "")
        if not doc_id or not method:
            continue
        row = {"docId": doc_id, "method": method}
        if item.get("sourceId"):
            row["sourceId"] = str(item.get("sourceId"))
        if item.get("sourceUrl"):
            row["sourceUrl"] = str(item.get("sourceUrl"))
        locator = item.get("locator") if isinstance(item.get("locator"), dict) else None
        if locator:
            public = _public_locator(locator)
            if public:
                row["locator"] = public
        if item.get("quote"):
            row["quote"] = str(item.get("quote"))
        span = item.get("span") if isinstance(item.get("span"), dict) else None
        if span and span.get("start") is not None and span.get("end") is not None:
            row["span"] = {"start": int(span["start"]), "end": int(span["end"])}
        rows.append(row)
    if rows or not chunks:
        return rows
    return rows


def provenance_for_sources(source_ids: list[str], chunks: dict, method: str = "lightrag") -> list[dict]:
    rows = []
    for source_id in source_ids:
        chunk = chunks.get(source_id)
        if not chunk:
            continue
        doc_id = _chunk_doc(chunk)
        if not doc_id:
            continue
        item = {"docId": doc_id, "sourceId": str(source_id), "method": method}
        url = str(chunk.get("source_url") or "")
        if url:
            item["sourceUrl"] = url
        locator = chunk.get("locator") if isinstance(chunk.get("locator"), dict) else None
        if locator and locator.get("label"):
            item["locator"] = dict(locator)
        rows.append(item)
    return rows


def public_graph(graph: dict, chunks: Optional[dict] = None) -> dict:
    nodes = []
    for node in graph.get("nodes") or []:
        kind = "structure" if node.get("kind") == "structure" else "semantic"
        payload = {
            "id": str(node.get("id") or ""),
            "label": str(node.get("label") or node.get("id") or ""),
            "type": str(node.get("type") or "entity"),
            "sourceIds": [str(item) for item in node.get("sourceIds") or []],
            "kind": kind,
        }
        if kind == "structure" and node.get("level") is not None:
            payload["level"] = int(node.get("level") or 0)
        if kind == "structure" and node.get("parentId"):
            payload["parentId"] = str(node.get("parentId"))
        if node.get("docId"):
            payload["docId"] = str(node.get("docId"))
        locator = node.get("locator") if isinstance(node.get("locator"), dict) else None
        if locator:
            public = _public_locator(locator)
            if public:
                payload["locator"] = public
        nodes.append(payload)
    edges = []
    for edge in graph.get("edges") or []:
        kind = edge_kind(edge)
        source_ids = [str(item) for item in edge.get("sourceIds") or []]
        provenance = _public_provenance(list(edge.get("provenance") or []))
        if not provenance and chunks and kind == "semantic":
            provenance = _public_provenance(provenance_for_sources(source_ids, chunks, str(edge.get("method") or "lightrag")))
        edges.append(
            {
                "id": str(edge.get("id") or ""),
                "source": str(edge.get("source") or ""),
                "target": str(edge.get("target") or ""),
                "label": str(edge.get("label") or ""),
                "sourceIds": source_ids,
                "edgeKind": kind,
                "relationType": str(edge.get("relationType") or edge.get("label") or ""),
                "directed": edge_directed(edge),
                "provenance": provenance,
            }
        )
    paths = []
    for path in graph.get("paths") or []:
        hops = int(path.get("hops") or 0)
        if not 1 <= hops <= MAX_PATH_EDGES:
            continue
        semantic_hops = path.get("semanticHops")
        paths.append(
            {
                "nodeIds": [str(item) for item in path.get("nodeIds") or []],
                "edgeIds": [str(item) for item in path.get("edgeIds") or []],
                "hops": hops,
                "semanticHops": hops if semantic_hops is None else int(semantic_hops),
                "hierarchyDepth": int(path.get("hierarchyDepth") or 0),
            }
        )
    return {"nodes": nodes, "edges": edges, "paths": paths, "truncated": bool(graph.get("truncated"))}


def empty_graph() -> dict:
    return {"nodes": [], "edges": [], "paths": [], "truncated": False}


def merge_graphs(graphs: list[dict], seeds: list[str]) -> dict:
    nodes: dict[str, dict] = {}
    edges: dict[str, dict] = {}
    truncated = False
    for graph in graphs:
        truncated = truncated or bool(graph.get("truncated"))
        for node in graph.get("nodes") or []:
            current = nodes.get(node["id"])
            if current is None:
                nodes[node["id"]] = dict(node)
                continue
            for source_id in node.get("sourceIds") or []:
                if source_id not in current["sourceIds"]:
                    current["sourceIds"].append(source_id)
        for edge in graph.get("edges") or []:
            edges[edge["id"]] = dict(edge)
    merged = graph_with_paths({"nodes": list(nodes.values()), "edges": list(edges.values()), "truncated": truncated}, seeds)
    return merged


def induced_subgraph(graph: dict, node_ids: list[str], seeds: list[str], scope: str = "semantic") -> dict:
    selected = [node for node in graph.get("nodes") or [] if node["id"] in set(node_ids)]
    selected_ids = {node["id"] for node in selected}
    edges = [
        edge
        for edge in graph.get("edges") or []
        if edge.get("source") in selected_ids and edge.get("target") in selected_ids
    ]
    if scope == "hierarchy":
        edges = [edge for edge in edges if edge_kind(edge) in {"contains", "references"}]
    elif scope == "semantic":
        edges = [edge for edge in edges if edge_kind(edge) == "semantic"]
    return graph_with_paths({"nodes": selected, "edges": edges, "truncated": False}, seeds or list(selected_ids), scope=scope)


def _expand_hierarchy(graph: dict, node_ids: list[str], steps: int, requested: int) -> dict:
    from knowledge.graph_rag.config import MAX_EXPAND_NODES

    nodes = list(graph.get("nodes") or [])
    contains = [edge for edge in graph.get("edges") or [] if edge_kind(edge) == "contains"]
    children: dict[str, list[str]] = {}
    parent: dict[str, str] = {}
    cited: dict[str, list[str]] = {}
    for node in nodes:
        if node.get("parentId"):
            parent[node["id"]] = str(node.get("parentId"))
    for edge in contains:
        children.setdefault(edge["source"], []).append(edge["target"])
        parent.setdefault(edge["target"], edge["source"])
    # Stored reference direction only. Parent links stay contains-only so ancestor
    # closure cannot turn around through a document hub into unrelated siblings.
    for edge in graph.get("edges") or []:
        if edge_kind(edge) != "references":
            continue
        cited.setdefault(edge["source"], []).append(edge["target"])
        if not edge_directed(edge):
            cited.setdefault(edge["target"], []).append(edge["source"])
    known = {node["id"] for node in nodes}
    seen = {node_id for node_id in node_ids if node_id in known}
    frontier = set(seen)
    for _ in range(steps):
        nxt: set[str] = set()
        for node_id in frontier:
            nxt.update(children.get(node_id, []))
            nxt.update(target for target in cited.get(node_id, []) if target in known)
        nxt -= seen
        if not nxt:
            break
        seen.update(nxt)
        frontier = nxt
    for node_id in list(seen):
        cursor = node_id
        guard = 0
        while parent.get(cursor) and guard < 32:
            cursor = parent[cursor]
            if cursor in seen:
                break
            if cursor not in known:
                break
            seen.add(cursor)
            guard += 1
    truncated = requested > MAX_PATH_EDGES
    if len(seen) > MAX_EXPAND_NODES:
        levels = {node["id"]: int(node.get("level") or 0) for node in nodes}
        protected: set[str] = set()
        for seed in node_ids:
            cursor = seed if seed in seen else ""
            while cursor:
                protected.add(cursor)
                cursor = parent.get(cursor, "")
                if cursor not in seen:
                    break
        removable = sorted((node_id for node_id in seen if node_id not in protected), key=lambda node_id: (-levels.get(node_id, 0), node_id))
        for node_id in removable:
            if len(seen) <= MAX_EXPAND_NODES:
                break
            seen.discard(node_id)
        truncated = True
    ordered = [node["id"] for node in nodes if node["id"] in seen]
    result = induced_subgraph(graph, ordered, [node_id for node_id in node_ids if node_id in seen], scope="hierarchy")
    result["truncated"] = bool(result.get("truncated") or truncated)
    return result


def expand_hops(graph: dict, node_ids: list[str], hops: int, scope: str = "semantic") -> dict:
    requested = int(hops)
    steps = max(1, min(requested, MAX_PATH_EDGES))
    if scope == "hierarchy":
        return _expand_hierarchy(graph, node_ids, steps, requested)
    if scope == "all":
        semantic = expand_hops(graph, node_ids, hops, scope="semantic")
        hierarchy = expand_hops(graph, node_ids, hops, scope="hierarchy")
        nodes: dict[str, dict] = {}
        edges: dict[str, dict] = {}
        for part in (semantic, hierarchy):
            for node in part.get("nodes") or []:
                nodes[node["id"]] = node
            for edge in part.get("edges") or []:
                edges[edge["id"]] = edge
        merged = graph_with_paths(
            {"nodes": list(nodes.values()), "edges": list(edges.values()), "truncated": False},
            list(node_ids),
            scope="all",
        )
        merged["truncated"] = bool(semantic.get("truncated") or hierarchy.get("truncated") or merged.get("truncated"))
        return merged
    adjacency: dict[str, list[str]] = {}
    for edge in graph.get("edges") or []:
        if edge_kind(edge) != "semantic":
            continue
        adjacency.setdefault(edge["source"], []).append(edge["target"])
        if not edge_directed(edge):
            adjacency.setdefault(edge["target"], []).append(edge["source"])
    known = {node["id"] for node in graph.get("nodes") or []}
    seen = {node_id for node_id in node_ids if node_id in known}
    frontier = set(seen)
    for _ in range(steps):
        nxt: set[str] = set()
        for node_id in frontier:
            nxt.update(adjacency.get(node_id, []))
        nxt -= seen
        if not nxt:
            break
        seen.update(nxt)
        frontier = nxt
    ordered = [node["id"] for node in graph.get("nodes") or [] if node["id"] in seen]
    from knowledge.graph_rag.config import MAX_EXPAND_NODES

    truncated = requested > MAX_PATH_EDGES
    if len(ordered) > MAX_EXPAND_NODES:
        ordered = ordered[:MAX_EXPAND_NODES]
        truncated = True
    result = induced_subgraph(graph, ordered, list(node_ids), scope="semantic")
    result["truncated"] = bool(result.get("truncated") or truncated)
    return result


def citations_from_answer(answer: str, sources: list[dict]) -> list[dict]:
    citations = []
    seen = set()
    for match in _CITE.finditer(answer or ""):
        citation_id = int(match.group(1))
        if citation_id in seen or citation_id < 1 or citation_id > len(sources):
            continue
        seen.add(citation_id)
        source = sources[citation_id - 1]
        citations.append(
            {
                "id": citation_id,
                "sourceIndex": citation_id - 1,
                "chunkId": str(source.get("chunk_id") or ""),
            }
        )
    return citations


def highlight(text: str, query: str, limit: int = 180) -> str:
    body = re.sub(r"\s+", " ", str(text or "")).strip()
    if not body:
        return ""
    if len(body) <= limit:
        return body
    compact = str(query or "").strip()
    if compact and compact[:12] in body:
        index = body.find(compact[:12])
        start = max(0, index - 40)
        return body[start : start + limit]
    return body[:limit]
