"""Graph-backed retrieval: LightRAG storage plus a LangGraph controller.

Importing this package does not load the embedding model or LightRAG.
"""

from __future__ import annotations


def schedule_document_ingest(file_name: str, blocks: list) -> dict:
    """Store managed chunks for a later explicit graph build.

    Indexing a document registers its real chunks. It does not run extraction
    unless GRAPH_AUTO_BUILD is enabled. Unrelated files are never scanned.
    """
    from knowledge.graph_rag.service import get_service

    return get_service().store_managed_blocks(file_name, blocks)


def delete_document_graph(doc_name: str) -> dict:
    """Drop stored passages and graph rows for one managed or corpus document."""
    from knowledge.graph_rag.service import get_service

    return get_service().delete_document(doc_name)


def register_managed_document(record: dict, storage_path: str = "") -> dict:
    """Remember a managed document without reading or extracting its file."""
    from knowledge.graph_rag.service import get_service

    return get_service().register_managed(record, storage_path)


def register_loaded_documents(records: list) -> int:
    """Register metadata for documents the library already loaded."""
    from knowledge.graph_rag.service import get_service

    return get_service().register_persisted_managed_documents(records)


__all__ = [
    "delete_document_graph",
    "register_loaded_documents",
    "register_managed_document",
    "schedule_document_ingest",
]
