"""Lightweight knowledge graph for aerospace document management.

This module provides a minimal, dependency-free knowledge graph that
unifies the structural records, article references, and document
relationships already scattered across the codebase into a single
queryable graph representation.

Design principles (cautious introduction):
- Non-destructive: the graph is an OPTIONAL enhancement, never a
  replacement. Document indexing and retrieval work identically without
  the graph.
- Zero heavy dependencies: pure Python dicts/lists, no Neo4j, no
  NetworkX required. Can be upgraded later if needed.
- JSON persistence: graph is stored as a single JSON file, easy to
  inspect, back up, and version.
- Default-off: controlled via environment variable. If the graph is
  disabled or fails to build, all existing functionality is unaffected.
- Reuses existing logic: entity extraction leverages the existing
  structure.py and _build_structural_metadata rather than reinventing.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

GRAPH_ENABLED = os.environ.get("ENABLE_KNOWLEDGE_GRAPH", "true").lower() in ("1", "true", "yes")
GRAPH_STORAGE_DIR = os.environ.get("KNOWLEDGE_GRAPH_DIR", str(Path(__file__).parent.parent / "data" / "knowledge_graph"))
GRAPH_FILE_NAME = "knowledge_graph.json"

# ============================================================================
# Entity types
# ============================================================================

class EntityType:
    """Known entity types in the aerospace document domain."""
    DOCUMENT = "document"           # A document (CCAR-121, AMM, etc.)
    CHAPTER = "chapter"             # A chapter (第X章)
    SECTION = "section"             # A section (第X节)
    ARTICLE = "article"             # An article/clause (第X条)
    TERM = "term"                   # A defined term (术语)
    REGULATION_CODE = "reg_code"   # A regulation code (CCAR-121.345)
    ORGANIZATION = "organization"   # An organization (CAAC, FAA)
    AIRCRAFT = "aircraft"           # An aircraft type/model
    PROCEDURE = "procedure"         # A procedure or method
    REQUIREMENT = "requirement"     # A requirement or condition
    TABLE = "table"                 # A table or checklist
    UNKNOWN = "unknown"


# ============================================================================
# Relation types
# ============================================================================

class RelationType:
    """Known relation types between entities."""
    CONTAINS = "contains"           # Parent contains child (chapter -> article)
    REFERENCES = "references"       # One entity references another (article -> article)
    DEFINES = "defines"             # Article/term defines a concept
    APPLIES_TO = "applies_to"       # Requirement applies to entity
    AMENDS = "amends"               # Document/article amends another
    EQUIVALENT_TO = "equivalent_to" # Terms/regulations are equivalent
    DERIVED_FROM = "derived_from"   # Entity derived from another
    PART_OF = "part_of"             # Entity is part of another (inverse of contains)
    CATEGORIZED_AS = "categorized_as" # Entity categorized as a type
    UNKNOWN = "unknown"


# ============================================================================
# Data classes
# ============================================================================

@dataclass
class KnowledgeEntity:
    """A node in the knowledge graph."""
    id: str
    type: str
    name: str
    aliases: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    source_doc: str = ""
    source_chunk: str = ""
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeEntity":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class KnowledgeRelation:
    """An edge in the knowledge graph."""
    id: str
    source_id: str
    target_id: str
    type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    source_doc: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeRelation":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ============================================================================
# ID generation helpers
# ============================================================================

def _entity_id(entity_type: str, name: str, doc_name: str = "") -> str:
    """Generate a stable entity ID from type and name."""
    raw = f"{entity_type}:{doc_name}:{name}".lower().strip()
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def _relation_id(source_id: str, target_id: str, rel_type: str) -> str:
    """Generate a stable relation ID."""
    raw = f"{source_id}->{rel_type}->{target_id}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


# ============================================================================
# Knowledge Graph
# ============================================================================

class KnowledgeGraph:
    """A lightweight knowledge graph with basic query capabilities.

    Stores entities and relations in memory with JSON persistence.
    Provides neighbor lookup, subgraph extraction, and entity search.
    """

    def __init__(self):
        self._entities: Dict[str, KnowledgeEntity] = {}
        self._relations: List[KnowledgeRelation] = []
        self._outgoing: Dict[str, List[KnowledgeRelation]] = {}
        self._incoming: Dict[str, List[KnowledgeRelation]] = {}
        self._by_type: Dict[str, Set[str]] = {}
        self._name_index: Dict[str, List[str]] = {}

    # --- Add operations ---

    def add_entity(self, entity: KnowledgeEntity) -> KnowledgeEntity:
        """Add or update an entity. Returns the entity."""
        existing = self._entities.get(entity.id)
        if existing:
            # Merge aliases and properties
            for alias in entity.aliases:
                if alias not in existing.aliases:
                    existing.aliases.append(alias)
            existing.properties.update(entity.properties)
            if entity.confidence > existing.confidence:
                existing.confidence = entity.confidence
            if entity.source_doc and not existing.source_doc:
                existing.source_doc = entity.source_doc
            return existing

        self._entities[entity.id] = entity
        self._by_type.setdefault(entity.type, set()).add(entity.id)

        # Index by name and aliases
        names = [entity.name] + entity.aliases
        for name in names:
            key = name.lower().strip()
            if key:
                self._name_index.setdefault(key, []).append(entity.id)

        return entity

    def add_relation(self, relation: KnowledgeRelation) -> Optional[KnowledgeRelation]:
        """Add a relation. Returns None if either endpoint doesn't exist."""
        if relation.source_id not in self._entities:
            logger.debug("Cannot add relation: source entity %s not found", relation.source_id)
            return None
        if relation.target_id not in self._entities:
            logger.debug("Cannot add relation: target entity %s not found", relation.target_id)
            return None

        # Check for duplicate
        for existing in self._outgoing.get(relation.source_id, []):
            if existing.target_id == relation.target_id and existing.type == relation.type:
                if relation.confidence > existing.confidence:
                    existing.confidence = relation.confidence
                    existing.properties.update(relation.properties)
                return existing

        self._relations.append(relation)
        self._outgoing.setdefault(relation.source_id, []).append(relation)
        self._incoming.setdefault(relation.target_id, []).append(relation)
        return relation

    # --- Query operations ---

    def get_entity(self, entity_id: str) -> Optional[KnowledgeEntity]:
        """Get an entity by ID."""
        return self._entities.get(entity_id)

    def get_entities_by_type(self, entity_type: str) -> List[KnowledgeEntity]:
        """Get all entities of a given type."""
        ids = self._by_type.get(entity_type, set())
        return [self._entities[eid] for eid in ids if eid in self._entities]

    def find_entities_by_name(self, name: str, *, fuzzy: bool = False) -> List[KnowledgeEntity]:
        """Find entities by name (exact or fuzzy substring match)."""
        if not name:
            return []
        key = name.lower().strip()

        # Exact match first
        exact_ids = self._name_index.get(key, [])
        results = [self._entities[eid] for eid in exact_ids if eid in self._entities]

        if fuzzy and not results:
            # Substring match
            seen = set(r.id for r in results)
            for indexed_name, ids in self._name_index.items():
                if key in indexed_name or indexed_name in key:
                    for eid in ids:
                        if eid not in seen and eid in self._entities:
                            results.append(self._entities[eid])
                            seen.add(eid)

        return results

    def get_outgoing_relations(self, entity_id: str, *, rel_type: Optional[str] = None) -> List[KnowledgeRelation]:
        """Get all outgoing relations from an entity."""
        rels = self._outgoing.get(entity_id, [])
        if rel_type:
            return [r for r in rels if r.type == rel_type]
        return list(rels)

    def get_incoming_relations(self, entity_id: str, *, rel_type: Optional[str] = None) -> List[KnowledgeRelation]:
        """Get all incoming relations to an entity."""
        rels = self._incoming.get(entity_id, [])
        if rel_type:
            return [r for r in rels if r.type == rel_type]
        return list(rels)

    def get_neighbors(self, entity_id: str, *, rel_type: Optional[str] = None, direction: str = "both") -> List[KnowledgeEntity]:
        """Get neighboring entities.

        Args:
            entity_id: The entity to find neighbors for.
            rel_type: Optional relation type filter.
            direction: "out", "in", or "both".
        """
        neighbor_ids: Set[str] = set()

        if direction in ("out", "both"):
            for rel in self.get_outgoing_relations(entity_id, rel_type=rel_type):
                neighbor_ids.add(rel.target_id)

        if direction in ("in", "both"):
            for rel in self.get_incoming_relations(entity_id, rel_type=rel_type):
                neighbor_ids.add(rel.source_id)

        return [self._entities[nid] for nid in neighbor_ids if nid in self._entities]

    def get_subgraph(self, entity_ids: Iterable[str], *, max_depth: int = 1) -> "KnowledgeGraph":
        """Extract a subgraph containing the given entities and their neighbors."""
        sub = KnowledgeGraph()
        frontier = set(entity_ids)
        visited: Set[str] = set()

        for depth in range(max_depth + 1):
            next_frontier: Set[str] = set()
            for eid in frontier:
                if eid in visited or eid not in self._entities:
                    continue
                visited.add(eid)
                sub.add_entity(self._entities[eid])

                # Add all relations for this entity
                for rel in self._outgoing.get(eid, []):
                    if rel.target_id in self._entities:
                        sub.add_relation(rel)
                        next_frontier.add(rel.target_id)
                for rel in self._incoming.get(eid, []):
                    if rel.source_id in self._entities:
                        sub.add_relation(rel)
                        next_frontier.add(rel.source_id)

            frontier = next_frontier - visited

        return sub

    # --- Statistics ---

    @property
    def entity_count(self) -> int:
        return len(self._entities)

    @property
    def relation_count(self) -> int:
        return len(self._relations)

    def stats(self) -> Dict[str, Any]:
        """Return graph statistics."""
        type_counts = {t: len(ids) for t, ids in self._by_type.items()}
        rel_type_counts: Dict[str, int] = {}
        for rel in self._relations:
            rel_type_counts[rel.type] = rel_type_counts.get(rel.type, 0) + 1
        return {
            "entity_count": self.entity_count,
            "relation_count": self.relation_count,
            "entities_by_type": type_counts,
            "relations_by_type": rel_type_counts,
        }

    # --- Persistence ---

    def save(self, filepath: str) -> None:
        """Save the graph to a JSON file."""
        data = {
            "version": "1.0",
            "entities": [e.to_dict() for e in self._entities.values()],
            "relations": [r.to_dict() for r in self._relations],
        }
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Saved knowledge graph: %d entities, %d relations -> %s",
                    self.entity_count, self.relation_count, filepath)

    @classmethod
    def load(cls, filepath: str) -> "KnowledgeGraph":
        """Load a graph from a JSON file."""
        path = Path(filepath)
        if not path.exists():
            logger.info("Knowledge graph file not found: %s, returning empty graph", filepath)
            return cls()

        data = json.loads(path.read_text(encoding="utf-8"))
        graph = cls()

        for entity_data in data.get("entities", []):
            try:
                entity = KnowledgeEntity.from_dict(entity_data)
                graph.add_entity(entity)
            except Exception as exc:
                logger.debug("Failed to load entity: %s", exc)

        for relation_data in data.get("relations", []):
            try:
                relation = KnowledgeRelation.from_dict(relation_data)
                graph.add_relation(relation)
            except Exception as exc:
                logger.debug("Failed to load relation: %s", exc)

        logger.info("Loaded knowledge graph: %d entities, %d relations from %s",
                    graph.entity_count, graph.relation_count, filepath)
        return graph


# ============================================================================
# Graph building from document units
# ============================================================================

# Patterns for entity extraction
_ARTICLE_REF_PATTERN = re.compile(r"第\s*([0-9A-Za-z一二三四五六七八九十百千万.]+)\s*条")
_CHAPTER_REF_PATTERN = re.compile(r"第\s*([0-9一二三四五六七八九十百千万]+)\s*章")
_SECTION_REF_PATTERN = re.compile(r"第\s*([0-9一二三四五六七八九十百千万.]+)\s*节")
_CCAR_CODE_PATTERN = re.compile(r"(?:CCAR|FAA|EASA|ATA|AMM|FIM|TSM|IPC|SRM)[- ]?\d+(?:\.\d+)*", re.I)
_DEFINITION_PATTERN = re.compile(r"([\u4e00-\u9fffA-Za-z0-9]{2,20})\s*(?:是指|定义为|系指)")


def _extract_entities_from_unit(unit: Dict[str, Any]) -> List[KnowledgeEntity]:
    """Extract entities from a single document unit.

    Leverages the structural metadata already attached to units by
    _build_structural_metadata, plus regex-based extraction for
    regulation codes and definitions.
    """
    entities: List[KnowledgeEntity] = []
    doc_name = str(unit.get("doc_name", ""))
    content = str(unit.get("content", ""))
    chunk_id = str(unit.get("chunk_id", ""))

    # Document entity
    if doc_name:
        doc_entity = KnowledgeEntity(
            id=_entity_id(EntityType.DOCUMENT, doc_name),
            type=EntityType.DOCUMENT,
            name=doc_name,
            aliases=[Path(doc_name).stem],
            source_doc=doc_name,
            confidence=1.0,
        )
        entities.append(doc_entity)

    # Article entity (from structural metadata)
    article_id = str(unit.get("article_id", ""))
    article_label = str(unit.get("article_label", ""))
    if article_id or article_label:
        article_name = article_label or f"第{article_id}条"
        article_entity = KnowledgeEntity(
            id=_entity_id(EntityType.ARTICLE, article_name, doc_name),
            type=EntityType.ARTICLE,
            name=article_name,
            aliases=[article_id] if article_id else [],
            properties={"article_id": article_id, "article_label": article_label},
            source_doc=doc_name,
            source_chunk=chunk_id,
            confidence=0.95,
        )
        entities.append(article_entity)

    # Chapter/Section entities
    section_title = str(unit.get("section_title", "") or unit.get("heading_text", ""))
    if section_title and len(section_title) <= 100:
        chapter_match = _CHAPTER_REF_PATTERN.search(section_title)
        section_match = _SECTION_REF_PATTERN.search(section_title)

        if chapter_match:
            chap_name = f"第{chapter_match.group(1)}章"
            entities.append(KnowledgeEntity(
                id=_entity_id(EntityType.CHAPTER, chap_name, doc_name),
                type=EntityType.CHAPTER,
                name=chap_name,
                source_doc=doc_name,
                source_chunk=chunk_id,
                confidence=0.9,
            ))

        if section_match:
            sec_name = f"第{section_match.group(1)}节"
            entities.append(KnowledgeEntity(
                id=_entity_id(EntityType.SECTION, sec_name, doc_name),
                type=EntityType.SECTION,
                name=sec_name,
                source_doc=doc_name,
                source_chunk=chunk_id,
                confidence=0.9,
            ))

    # Regulation codes in content
    for code_match in _CCAR_CODE_PATTERN.finditer(content):
        code = code_match.group(0).upper().replace(" ", "-")
        entities.append(KnowledgeEntity(
            id=_entity_id(EntityType.REGULATION_CODE, code),
            type=EntityType.REGULATION_CODE,
            name=code,
            source_doc=doc_name,
            source_chunk=chunk_id,
            confidence=0.85,
        ))

    # Defined terms
    for def_match in _DEFINITION_PATTERN.finditer(content[:500]):
        term = def_match.group(1).strip()
        if 2 <= len(term) <= 20 and not term.isdigit():
            entities.append(KnowledgeEntity(
                id=_entity_id(EntityType.TERM, term),
                type=EntityType.TERM,
                name=term,
                source_doc=doc_name,
                source_chunk=chunk_id,
                confidence=0.7,
            ))

    # Table entity
    if str(unit.get("source_type", "")) == "table" or str(unit.get("is_table", "")) == "True":
        table_name = section_title or f"表格_{chunk_id[:8]}"
        entities.append(KnowledgeEntity(
            id=_entity_id(EntityType.TABLE, table_name, doc_name),
            type=EntityType.TABLE,
            name=table_name,
            source_doc=doc_name,
            source_chunk=chunk_id,
            confidence=0.9,
        ))

    return entities


def _extract_relations_from_unit(unit: Dict[str, Any], entities: List[KnowledgeEntity]) -> List[KnowledgeRelation]:
    """Extract relations between entities found in a unit."""
    relations: List[KnowledgeRelation] = []
    doc_name = str(unit.get("doc_name", ""))
    content = str(unit.get("content", ""))

    # Build a lookup of entities by name
    entity_by_name: Dict[str, KnowledgeEntity] = {}
    for ent in entities:
        entity_by_name[ent.name.lower()] = ent
        for alias in ent.aliases:
            entity_by_name[alias.lower()] = ent

    # Document contains article/chapter/section
    doc_entity = next((e for e in entities if e.type == EntityType.DOCUMENT), None)
    if doc_entity:
        for ent in entities:
            if ent.type in (EntityType.ARTICLE, EntityType.CHAPTER, EntityType.SECTION, EntityType.TABLE):
                relations.append(KnowledgeRelation(
                    id=_relation_id(doc_entity.id, ent.id, RelationType.CONTAINS),
                    source_id=doc_entity.id,
                    target_id=ent.id,
                    type=RelationType.CONTAINS,
                    source_doc=doc_name,
                    confidence=0.95,
                ))

    # Chapter contains article (if both present and article number follows chapter)
    chapter_entity = next((e for e in entities if e.type == EntityType.CHAPTER), None)
    article_entity = next((e for e in entities if e.type == EntityType.ARTICLE), None)
    if chapter_entity and article_entity:
        relations.append(KnowledgeRelation(
            id=_relation_id(chapter_entity.id, article_entity.id, RelationType.CONTAINS),
            source_id=chapter_entity.id,
            target_id=article_entity.id,
            type=RelationType.CONTAINS,
            source_doc=doc_name,
            confidence=0.8,
        ))

    # Article defines term
    term_entities = [e for e in entities if e.type == EntityType.TERM]
    if article_entity and term_entities:
        for term in term_entities:
            relations.append(KnowledgeRelation(
                id=_relation_id(article_entity.id, term.id, RelationType.DEFINES),
                source_id=article_entity.id,
                target_id=term.id,
                type=RelationType.DEFINES,
                source_doc=doc_name,
                confidence=0.7,
            ))

    # Cross-article references in content
    for ref_match in _ARTICLE_REF_PATTERN.finditer(content):
        ref_article = f"第{ref_match.group(1)}条"
        ref_entity = entity_by_name.get(ref_article.lower())
        if ref_entity and article_entity and ref_entity.id != article_entity.id:
            relations.append(KnowledgeRelation(
                id=_relation_id(article_entity.id, ref_entity.id, RelationType.REFERENCES),
                source_id=article_entity.id,
                target_id=ref_entity.id,
                type=RelationType.REFERENCES,
                source_doc=doc_name,
                confidence=0.75,
            ))

    return relations


def build_graph_from_units(
    units: Iterable[Dict[str, Any]],
    *,
    existing_graph: Optional[KnowledgeGraph] = None,
) -> KnowledgeGraph:
    """Build or update a knowledge graph from document units.

    Args:
        units: Iterable of document unit dicts (the same format used
            throughout the indexing pipeline).
        existing_graph: Optional existing graph to update. If None, a
            new empty graph is created.

    Returns:
        The updated KnowledgeGraph.
    """
    graph = existing_graph or KnowledgeGraph()
    unit_count = 0
    entity_count = 0
    relation_count = 0

    for unit in units:
        unit_count += 1
        try:
            entities = _extract_entities_from_unit(unit)
            relations = _extract_relations_from_unit(unit, entities)

            for entity in entities:
                graph.add_entity(entity)
                entity_count += 1

            for relation in relations:
                graph.add_relation(relation)
                relation_count += 1

        except Exception as exc:
            logger.debug("Failed to extract graph data from unit %s: %s",
                         unit.get("chunk_id", "unknown"), exc)

    logger.info("Built graph from %d units: %d entity additions, %d relation additions",
                unit_count, entity_count, relation_count)
    return graph


# ============================================================================
# Query-time entity extraction (for retrieval enhancement)
# ============================================================================

def extract_entities_from_query(query: str) -> List[Dict[str, str]]:
    """Extract entities from a user query for graph-enhanced retrieval.

    Returns a list of {"type": ..., "name": ...} dicts.
    """
    if not query:
        return []

    results: List[Dict[str, str]] = []
    seen: Set[str] = set()

    def _add(ent_type: str, name: str):
        key = f"{ent_type}:{name.lower()}"
        if key not in seen:
            seen.add(key)
            results.append({"type": ent_type, "name": name})

    # Regulation codes
    for match in _CCAR_CODE_PATTERN.finditer(query):
        code = match.group(0).upper().replace(" ", "-")
        _add(EntityType.REGULATION_CODE, code)

    # Article references
    for match in _ARTICLE_REF_PATTERN.finditer(query):
        _add(EntityType.ARTICLE, f"第{match.group(1)}条")

    # Chapter references
    for match in _CHAPTER_REF_PATTERN.finditer(query):
        _add(EntityType.CHAPTER, f"第{match.group(1)}章")

    # Section references
    for match in _SECTION_REF_PATTERN.finditer(query):
        _add(EntityType.SECTION, f"第{match.group(1)}节")

    return results


def graph_expand_query(
    query: str,
    graph: KnowledgeGraph,
    *,
    max_expansions: int = 5,
) -> List[str]:
    """Expand a query using knowledge graph entity neighbors.

    Finds entities matching the query, then returns related entity
    names as query expansion terms. This is useful for recall enhancement
    when the query mentions a specific article or term that has related
    concepts in the graph.

    Args:
        query: The user's search query.
        graph: The knowledge graph to use for expansion.
        max_expansions: Maximum number of expansion terms to return.

    Returns:
        A list of expansion terms (entity names).
    """
    if not query or graph.entity_count == 0:
        return []

    expansions: List[str] = []
    seen: Set[str] = set()

    # Extract entities from query
    query_entities = extract_entities_from_query(query)

    # For each query entity, find matching graph entities and their neighbors
    for qent in query_entities:
        matches = graph.find_entities_by_name(qent["name"], fuzzy=True)
        for match in matches[:3]:
            # Get neighbors (related entities, one hop)
            neighbors = graph.get_neighbors(match.id)
            for neighbor in neighbors:
                name = neighbor.name
                if name and name.lower() not in seen and name.lower() not in query.lower():
                    seen.add(name.lower())
                    expansions.append(name)
                    if len(expansions) >= max_expansions:
                        return expansions

    return expansions


# ============================================================================
# Global graph instance
# ============================================================================

_global_graph: Optional[KnowledgeGraph] = None
_graph_lock = None


def get_global_graph() -> KnowledgeGraph:
    """Get or create the global knowledge graph singleton."""
    global _global_graph, _graph_lock
    if _global_graph is None:
        import threading
        if _graph_lock is None:
            _graph_lock = threading.Lock()
        with _graph_lock:
            if _global_graph is None:
                _global_graph = KnowledgeGraph()
                # Try to load from disk
                if GRAPH_STORAGE_DIR:
                    graph_path = Path(GRAPH_STORAGE_DIR) / GRAPH_FILE_NAME
                    if graph_path.exists():
                        try:
                            _global_graph = KnowledgeGraph.load(str(graph_path))
                        except Exception as exc:
                            logger.warning("Failed to load global knowledge graph: %s", exc)
                            _global_graph = KnowledgeGraph()
    return _global_graph


def save_global_graph() -> None:
    """Save the global graph to disk if storage is configured."""
    if not GRAPH_STORAGE_DIR or _global_graph is None:
        return
    try:
        graph_path = Path(GRAPH_STORAGE_DIR) / GRAPH_FILE_NAME
        _global_graph.save(str(graph_path))
    except Exception as exc:
        logger.warning("Failed to save global knowledge graph: %s", exc)
