import concurrent.futures
import hashlib
import math
import os
import re
import threading
import time
import warnings
import json
import logging
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from search.document_pipeline import (
    _chunk_text_units,
    _persist_document_health_reports,
    _remove_document_health_report,
    build_chain,
    extract_document_units,
    logger,
)
from search.common import (
    DOC_DIR,
    EMBEDDING_MODEL,
    ES_HOST,
    ES_INDEX,
    ES_REQUEST_TIMEOUT_SECONDS,
    IndexOperationResult,
    MAX_CHUNK_SIZE,
    MILVUS_COLLECTION,
    MILVUS_CONTENT_MAX,
    RERANK_DEVICE,
    RERANK_MODEL,
    VECTOR_DIM,
    _acquire_milvus_connection,
    _copy_structural_fields,
    _embedder,
    _embedder_lock,
    _release_milvus_connection,
    _reranker,
    _reranker_lock,
    _truncate_for_milvus_varchar,
    logger,
)

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ["HF_ENABLE_PARALLEL_LOADING"] = os.environ.get("AEROSPACE_HF_ENABLE_PARALLEL_LOADING", "false")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import fitz
import numpy as np
import torch
from docx import Document
from elasticsearch import Elasticsearch, helpers
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility
from sentence_transformers import CrossEncoder, SentenceTransformer

from utils.evidence_scores import calibrate_ranking_score, normalize_structural_score
from utils.es_query_builder import (
    AliasClauseProfile,
    ArticleReferenceProfile,
    CodeVariantProfile,
    EsQueryPlan,
    PrecisionHintProfile,
    WeightedField,
    build_alias_clauses,
    build_article_reference_clauses,
    build_code_variant_clauses,
    build_precision_hint_clauses,
    es_hit_to_channel_result,
    match_clause,
    match_phrase_clause,
    terms_filter,
)
from utils.structure import (
    answer_structural_classification,
    answer_structural_count,
    answer_structural_definition,
    attach_structure_to_blocks,
    build_structural_records,
    clear_structural_records,
    load_all_structural_records,
    remove_structural_records,
    save_structural_records,
)
from utils.retrieval_contract import (
    RetrievalContract,
    apply_contract,
    build_retrieval_contract,
    code_query_variants,
    contract_needs_multi_target_recall,
)
from utils.query_utils import compact_query_text, cross_lingual_query_terms, classify_query_intent

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
def build_document_chunks(units: List[dict], embedder) -> List[dict]:
    if not units:
        return []

    blocks: List[dict] = []
    heading_stack: List[dict] = []
    pending_text_units: List[dict] = []
    pending_chars = 0
    total_units = len(units)
    progress_interval = max(200, total_units // 10) if total_units >= 400 else 0

    def current_section() -> str:
        return heading_stack[-1]["text"] if heading_stack else ""

    def current_parent() -> str:
        return heading_stack[-2]["text"] if len(heading_stack) > 1 else ""

    def flush_pending():
        nonlocal pending_text_units, pending_chars
        if not pending_text_units:
            return
        blocks.extend(_chunk_text_units(pending_text_units, embedder, current_section(), current_parent()))
        pending_text_units = []
        pending_chars = 0

    def should_flush_text(last_unit: dict, current_unit: dict) -> bool:
        if current_unit["page_number"] != last_unit["page_number"]:
            return True
        if current_unit.get("layout_type") != last_unit.get("layout_type"):
            return True
        if pending_chars + len(current_unit["content"]) > int(MAX_CHUNK_SIZE * 1.35):
            return True

        last_article = last_unit.get("article_id", "")
        current_article = current_unit.get("article_id", "")
        if last_article and current_article and last_article != current_article:
            return True

        last_section = last_unit.get("section_id", "")
        current_section = current_unit.get("section_id", "")
        if last_section and current_section and last_section != current_section:
            return True

        # Clause/list/role tags are retrieval metadata, not hard chunk boundaries.
        # Splitting on every tag change makes PDF lines look like sentence chunks.
        if current_unit.get("boundary_conflict"):
            return True
        return False

    for index, unit in enumerate(units, start=1):
        source_type = unit.get("source_type")
        if source_type == "heading":
            flush_pending()
            level = max(int(unit.get("heading_level", 1)), 1)
            while heading_stack and heading_stack[-1]["level"] >= level:
                heading_stack.pop()
            parent_heading = heading_stack[-1]["text"] if heading_stack else ""
            blocks.append(
                {
                    "doc_name": unit["doc_name"],
                    "content": unit["content"],
                    "page_number": unit["page_number"],
                    "page_count": unit["page_count"],
                    "bbox": unit.get("bbox"),
                    "section_title": unit["content"],
                    "parent_heading": parent_heading,
                    "heading_text": unit["content"],
                    "heading_level": level,
                    "is_heading": True,
                    "source_type": "heading",
                    "layout_type": unit.get("layout_type", "single_column"),
                    "parse_confidence": unit.get("parse_confidence", 0.9),
                    "is_table": False,
                    "is_ocr_fallback": unit.get("is_ocr_fallback", False),
                    "highlight_text": unit.get("highlight_text") or unit["content"],
                    "document_type": unit.get("document_type", "AMM"),
                    "cross_page": 0,
                    "boundary_conflict": 0,
                    **_copy_structural_fields(unit),
                }
            )
            heading_stack.append({"level": level, "text": unit["content"]})
            if progress_interval and index % progress_interval == 0:
                logger.info("切片进度：%d/%d 单元，当前证据块数：%d", index, total_units, len(blocks))
            continue

        if source_type == "table":
            flush_pending()
            blocks.append(
                {
                    "doc_name": unit["doc_name"],
                    "content": unit["content"],
                    "page_number": unit["page_number"],
                    "page_count": unit["page_count"],
                    "bbox": unit.get("bbox"),
                    "section_title": unit.get("section_title") or current_section(),
                    "parent_heading": current_parent(),
                    "heading_text": current_section(),
                    "heading_level": 0,
                    "is_heading": False,
                    "source_type": "table",
                    "layout_type": unit.get("layout_type", "table"),
                    "parse_confidence": unit.get("parse_confidence", 0.88),
                    "is_table": True,
                    "is_ocr_fallback": unit.get("is_ocr_fallback", False),
                    "highlight_text": unit.get("highlight_text") or unit.get("section_title") or current_section(),
                    "document_type": unit.get("document_type", "AMM"),
                    "cross_page": 0,
                    "boundary_conflict": 0,
                    **_copy_structural_fields(unit),
                }
            )
            if progress_interval and index % progress_interval == 0:
                logger.info("切片进度：%d/%d 单元，当前证据块数：%d", index, total_units, len(blocks))
            continue

        if pending_text_units:
            last_unit = pending_text_units[-1]
            if should_flush_text(last_unit, unit):
                flush_pending()

        pending_text_units.append(unit)
        pending_chars += len(unit["content"])
        if progress_interval and index % progress_interval == 0:
            logger.info("切片进度：%d/%d 单元，当前证据块数：%d", index, total_units, len(blocks))

    flush_pending()
    return build_chain(blocks)


def _encode_with_progress(embedder, contents: List[str], batch_size: int = 128, stage_label: str = "向量化") -> np.ndarray:
    if not contents:
        return np.empty((0, VECTOR_DIM), dtype=np.float32)

    total = len(contents)
    total_batches = math.ceil(total / batch_size)
    arrays: List[np.ndarray] = []
    started_at = time.time()

    for batch_index, start in enumerate(range(0, total, batch_size), start=1):
        end = min(start + batch_size, total)
        batch_contents = contents[start:end]
        batch_started_at = time.time()
        batch_embeddings = embedder.encode(
            batch_contents,
            batch_size=min(64, len(batch_contents)),
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        arrays.append(np.asarray(batch_embeddings, dtype=np.float32))
        batch_elapsed = time.time() - batch_started_at
        total_elapsed = time.time() - started_at
        logger.info(
            "%s进度：%d/%d 块（%.1f%%），batch %d/%d，本批 %.1fs，累计 %.1fs",
            stage_label,
            end,
            total,
            end / total * 100,
            batch_index,
            total_batches,
            batch_elapsed,
            total_elapsed,
        )

    if len(arrays) == 1:
        return arrays[0]
    return np.vstack(arrays)


def save_metadata_to_es(es_client: Elasticsearch, blocks: List[dict]):
    actions = []
    for block in blocks:
        actions.append(
            {
                "_index": ES_INDEX,
                "_id": block["chunk_id"],
                "_source": {
                    "chunk_id": block["chunk_id"],
                    "content": block["content"],
                    "doc_name": block.get("doc_name", ""),
                    "document_type": block.get("document_type", "AMM"),
                    "section_title": block.get("section_title", ""),
                    "heading_text": block.get("heading_text", ""),
                    "parent_heading": block.get("parent_heading", ""),
                    "heading_level": int(block.get("heading_level", 0)),
                    "is_heading": 1 if block.get("is_heading") else 0,
                    "source_type": block.get("source_type", "text"),
                    "layout_type": block.get("layout_type", "single_column"),
                    "parse_confidence": float(block.get("parse_confidence", 0.0)),
                    "is_table": 1 if block.get("is_table") else 0,
                    "is_ocr_fallback": 1 if block.get("is_ocr_fallback") else 0,
                    "page_number": int(block.get("page_number", 1)),
                    "page_count": int(block.get("page_count", 1)),
                    "bbox": block.get("bbox"),
                    "highlight_text": block.get("highlight_text", ""),
                    "prev_id": block.get("prev_id", ""),
                    "next_id": block.get("next_id", ""),
                    "parent_id": block.get("parent_id", ""),
                    "doc_order": int(block.get("doc_order", 0)),
                    "article_id": block.get("article_id", ""),
                    "article_label": block.get("article_label", ""),
                    "section_id": block.get("section_id", ""),
                    "section_label": block.get("section_label", ""),
                    "clause_id": block.get("clause_id", ""),
                    "list_item_id": block.get("list_item_id", ""),
                    "chunk_role": block.get("chunk_role", "text"),
                    "normalized_heading": block.get("normalized_heading", ""),
                    "has_enumeration": int(block.get("has_enumeration", 0)),
                    "list_density": float(block.get("list_density", 0.0)),
                    "contains_definition": int(block.get("contains_definition", 0)),
                    "contains_steps": int(block.get("contains_steps", 0)),
                    "contains_classification": int(block.get("contains_classification", 0)),
                    "contains_requirements": int(block.get("contains_requirements", 0)),
                    "title_body_consistency": float(block.get("title_body_consistency", 0.0)),
                    "clause_path": block.get("clause_path", ""),
                    "heading_path_text": block.get("heading_path_text", ""),
                    "scope_anchor": block.get("scope_anchor", ""),
                    "scope_key": block.get("scope_key", ""),
                    "retrieval_text": block.get("retrieval_text", ""),
                    "structural_node_id": block.get("structural_node_id", ""),
                    "structural_parent_id": block.get("structural_parent_id", ""),
                    "structural_record_type": block.get("structural_record_type", ""),
                    "structural_level": int(block.get("structural_level", 0)),
                    "structural_number": block.get("structural_number", ""),
                    "structural_path": block.get("structural_path", ""),
                    "structural_source": block.get("structural_source", ""),
                    "direct_child_count": int(block.get("direct_child_count", 0)),
                    "descendant_count": int(block.get("descendant_count", 0)),
                    "cross_page": int(block.get("cross_page", 0)),
                    "boundary_conflict": int(block.get("boundary_conflict", 0)),
                },
            }
        )
    if actions:
        helpers.bulk(es_client, actions)
        es_client.indices.refresh(index=ES_INDEX)


def _es_get_by_id(es_client: Elasticsearch, chunk_id: str) -> Dict:
    try:
        response = es_client.get(index=ES_INDEX, id=chunk_id)
        return response["_source"]
    except Exception:
        return {}


def get_chunk_context(es_client: Elasticsearch, chunk_id: str, context_size: int = 1) -> Dict:
    current = _es_get_by_id(es_client, chunk_id)
    if not current:
        return {}

    prev_blocks = []
    prev_id = current.get("prev_id", "")
    while prev_id and len(prev_blocks) < context_size:
        row = _es_get_by_id(es_client, prev_id)
        if not row:
            break
        prev_blocks.insert(0, row)
        prev_id = row.get("prev_id", "")

    next_blocks = []
    next_id = current.get("next_id", "")
    while next_id and len(next_blocks) < context_size:
        row = _es_get_by_id(es_client, next_id)
        if not row:
            break
        next_blocks.append(row)
        next_id = row.get("next_id", "")

    parent = _es_get_by_id(es_client, current["parent_id"]) if current.get("parent_id") else None
    return {"current": current, "prev": prev_blocks, "next": next_blocks, "parent": parent}


def format_structural_context(ctx: Dict) -> str:
    if not ctx:
        return ""
    parts = []
    if ctx.get("parent"):
        parts.append(f"[PARENT] {ctx['parent'].get('content', '')}")
    for row in ctx.get("prev", []):
        parts.append(f"[PREV] {row.get('content', '')}")
    parts.append(f"[CURRENT] {ctx['current'].get('content', '')}")
    for row in ctx.get("next", []):
        parts.append(f"[NEXT] {row.get('content', '')}")
    return "\n\n".join(parts)


def get_embedder():
    global _embedder
    if _embedder is None:
        with _embedder_lock:
            if _embedder is None:
                started_at = time.time()
                logger.info("加载嵌入模型（本地缓存）")
                _embedder = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
                if torch.cuda.is_available():
                    _embedder = _embedder.to("cuda")
                    logger.info("嵌入模型已加载到 GPU，用时 %.1fs", time.time() - started_at)
                else:
                    logger.info("嵌入模型使用 CPU，用时 %.1fs", time.time() - started_at)
    return _embedder


def get_reranker():
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                try:
                    started_at = time.time()
                    logger.info("加载重排序模型（本地缓存）")
                    rerank_device = "cuda" if RERANK_DEVICE == "cuda" and torch.cuda.is_available() else "cpu"
                    _reranker = CrossEncoder(
                        RERANK_MODEL,
                        device=rerank_device,
                        local_files_only=True,
                    )
                    logger.info("重排序模型加载完成（%s），用时 %.1fs", rerank_device, time.time() - started_at)
                except Exception as exc:
                    logger.warning("重排序模型加载失败，将跳过重排序: %s", exc)
                    _reranker = None
    return _reranker


class DualDBManager:
    def __init__(self):
        _acquire_milvus_connection()
        self._closed = False
        try:
            self.es_client = Elasticsearch(ES_HOST)
            self.embedder = None
        except Exception:
            self._closed = True
            _release_milvus_connection()
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.es_client.close()
        except Exception:
            logger.debug("Elasticsearch client close failed", exc_info=True)
        _release_milvus_connection()

    def __enter__(self) -> "DualDBManager":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _get_embedder(self):
        if self.embedder is None:
            self.embedder = get_embedder()
        return self.embedder

    def _es_mapping(self):
        return {
            "mappings": {
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "content": {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_smart"},
                    "doc_name": {"type": "keyword"},
                    "document_type": {"type": "keyword"},
                    "section_title": {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_smart"},
                    "heading_text": {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_smart"},
                    "parent_heading": {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_smart"},
                    "heading_level": {"type": "integer"},
                    "is_heading": {"type": "integer"},
                    "source_type": {"type": "keyword"},
                    "layout_type": {"type": "keyword"},
                    "parse_confidence": {"type": "float"},
                    "is_table": {"type": "integer"},
                    "is_ocr_fallback": {"type": "integer"},
                    "page_number": {"type": "integer"},
                    "page_count": {"type": "integer"},
                    "bbox": {
                        "properties": {
                            "x": {"type": "float"},
                            "y": {"type": "float"},
                            "width": {"type": "float"},
                            "height": {"type": "float"},
                        }
                    },
                    "highlight_text": {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_smart"},
                    "prev_id": {"type": "keyword"},
                    "next_id": {"type": "keyword"},
                    "parent_id": {"type": "keyword"},
                    "doc_order": {"type": "integer"},
                    "article_id": {"type": "keyword"},
                    "article_label": {"type": "keyword"},
                    "section_id": {"type": "keyword"},
                    "section_label": {"type": "keyword"},
                    "clause_id": {"type": "keyword"},
                    "list_item_id": {"type": "keyword"},
                    "chunk_role": {"type": "keyword"},
                    "normalized_heading": {"type": "keyword"},
                    "has_enumeration": {"type": "integer"},
                    "list_density": {"type": "float"},
                    "contains_definition": {"type": "integer"},
                    "contains_steps": {"type": "integer"},
                    "contains_classification": {"type": "integer"},
                    "contains_requirements": {"type": "integer"},
                    "title_body_consistency": {"type": "float"},
                    "clause_path": {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_smart"},
                    "heading_path_text": {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_smart"},
                    "scope_anchor": {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_smart"},
                    "scope_key": {"type": "keyword"},
                    "retrieval_text": {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_smart"},
                    "structural_node_id": {"type": "keyword"},
                    "structural_parent_id": {"type": "keyword"},
                    "structural_record_type": {"type": "keyword"},
                    "structural_level": {"type": "integer"},
                    "structural_number": {"type": "keyword"},
                    "structural_path": {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_smart"},
                    "structural_source": {"type": "keyword"},
                    "direct_child_count": {"type": "integer"},
                    "descendant_count": {"type": "integer"},
                    "cross_page": {"type": "integer"},
                    "boundary_conflict": {"type": "integer"},
                }
            }
        }

    def _create_milvus_collection(self) -> Collection:
        fields = [
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=MILVUS_CONTENT_MAX),
        ]
        schema = CollectionSchema(fields, f"aerospace semantic index ({EMBEDDING_MODEL}, dim={VECTOR_DIM})")
        collection = Collection(MILVUS_COLLECTION, schema)
        collection.create_index("vector", {"metric_type": "COSINE", "index_type": "IVF_FLAT", "params": {"nlist": 1024}})
        return collection

    def _collection_vector_dim(self, collection: Collection) -> int:
        for field in collection.schema.fields:
            if field.name == "vector":
                try:
                    return int(field.params.get("dim") or 0)
                except Exception:
                    return 0
        return 0

    def ensure_databases(self) -> Collection:
        if not utility.has_collection(MILVUS_COLLECTION):
            collection = self._create_milvus_collection()
        else:
            collection = Collection(MILVUS_COLLECTION)
            current_dim = self._collection_vector_dim(collection)
            if current_dim != VECTOR_DIM:
                logger.warning(
                    "Milvus collection vector dim mismatch: current=%s expected=%s. Recreating collection.",
                    current_dim,
                    VECTOR_DIM,
                )
                utility.drop_collection(MILVUS_COLLECTION)
                collection = self._create_milvus_collection()
        collection.load()
        if not self.es_client.indices.exists(index=ES_INDEX):
            self.es_client.indices.create(index=ES_INDEX, body=self._es_mapping())
        else:
            self.es_client.indices.put_mapping(index=ES_INDEX, body=self._es_mapping()["mappings"])
        return collection

    def reset_databases(self) -> Collection:
        if utility.has_collection(MILVUS_COLLECTION):
            utility.drop_collection(MILVUS_COLLECTION)
        collection = self._create_milvus_collection()
        if self.es_client.indices.exists(index=ES_INDEX):
            self.es_client.indices.delete(index=ES_INDEX)
        self.es_client.indices.create(index=ES_INDEX, body=self._es_mapping())
        return collection

    def _remove_doc_from_es(self, doc_name: str):
        self.es_client.delete_by_query(index=ES_INDEX, body={"query": {"term": {"doc_name": doc_name}}}, refresh=True)

    def _remove_doc_from_milvus(self, collection: Collection, doc_name: str):
        try:
            chunk_ids: List[str] = []
            response = self.es_client.options(request_timeout=ES_REQUEST_TIMEOUT_SECONDS).search(
                index=ES_INDEX,
                body={"query": {"term": {"doc_name": doc_name}}, "_source": ["chunk_id"], "size": 1000},
                scroll="2m",
            )
            scroll_id = response.get("_scroll_id")
            try:
                while True:
                    hits = response.get("hits", {}).get("hits", [])
                    chunk_ids.extend(
                        chunk_id
                        for chunk_id in (str(hit.get("_source", {}).get("chunk_id") or "") for hit in hits)
                        if chunk_id
                    )
                    if not scroll_id or len(hits) < 1000:
                        break
                    response = self.es_client.options(request_timeout=ES_REQUEST_TIMEOUT_SECONDS).scroll(
                        scroll_id=scroll_id,
                        scroll="2m",
                    )
                    scroll_id = response.get("_scroll_id", scroll_id)
            finally:
                if scroll_id:
                    try:
                        self.es_client.clear_scroll(scroll_id=scroll_id)
                    except Exception:
                        pass
            if not chunk_ids:
                return
            delete_batch_size = 1000
            for start in range(0, len(chunk_ids), delete_batch_size):
                batch = chunk_ids[start : start + delete_batch_size]
                quoted_chunk_ids = ", ".join(json.dumps(chunk_id, ensure_ascii=False) for chunk_id in batch)
                collection.delete(f"chunk_id in [{quoted_chunk_ids}]")
        except Exception as exc:
            logger.warning("从 Milvus 删除文档失败 doc_name=%s: %s", doc_name, exc)

    def _insert_blocks_to_milvus(self, collection: Collection, blocks: List[dict], embeddings: List[List[float]], contents: List[str], batch_size: int = 256):
        total = len(blocks)
        if not (len(embeddings) == total == len(contents)):
            raise ValueError("Milvus 插入数据长度不一致")
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_blocks = blocks[start:end]
            batch_embeddings = embeddings[start:end]
            batch_contents = contents[start:end]
            collection.insert(
                [
                    [block["chunk_id"] for block in batch_blocks],
                    batch_embeddings,
                    batch_contents,
                ]
            )

    def _index_document_blocks(self, file_path: str) -> Tuple[List[dict], dict]:
        started_at = time.time()
        units, health = extract_document_units(file_path)
        logger.info("文本单元提取完成：%d 个单元，用时 %.1fs", len(units), time.time() - started_at)
        if not units:
            return [], health
        structure_started_at = time.time()
        structural_records = build_structural_records(units, os.path.basename(file_path))
        save_structural_records(os.path.basename(file_path), structural_records)
        health["structural_record_count"] = len(structural_records)
        logger.info("结构记录构建完成：%d 条，用时 %.1fs", len(structural_records), time.time() - structure_started_at)
        chunk_started_at = time.time()
        logger.info("开始生成证据块")
        blocks = build_document_chunks(units, self._get_embedder())
        logger.info("证据块切片完成：%d 个，用时 %.1fs", len(blocks), time.time() - chunk_started_at)
        attach_started_at = time.time()
        logger.info("开始挂载结构标签")
        blocks = attach_structure_to_blocks(blocks, structural_records)
        logger.info("结构标签挂载完成，用时 %.1fs", time.time() - attach_started_at)
        health["chunk_count"] = len(blocks)
        logger.info("证据块生成完成：%d 个，用时 %.1fs", len(blocks), time.time() - chunk_started_at)
        if blocks:
            low_conf_chunks = sum(1 for block in blocks if block.get("parse_confidence", 0.0) < 0.55)
            health["low_confidence_chunk_ratio"] = round(low_conf_chunks / len(blocks), 4)
        else:
            health["low_confidence_chunk_ratio"] = 0.0
        return blocks, health

    def add_document(self, file_path: str) -> IndexOperationResult:
        file_name = os.path.basename(file_path)
        if not os.path.exists(file_path):
            return IndexOperationResult(
                success=False,
                action="add_document",
                message=f"文件不存在：{file_path}",
                doc_name=file_name,
                error_code="file_not_found",
            )

        try:
            collection = self.ensure_databases()
        except Exception as exc:
            return IndexOperationResult(
                success=False,
                action="add_document",
                message=f"数据库连接失败：{exc}",
                doc_name=file_name,
                error_code="database_unavailable",
            )

        self._remove_doc_from_milvus(collection, file_name)
        self._remove_doc_from_es(file_name)

        logger.info("增量索引：%s，开始解析与结构化", file_name)
        blocks, health = self._index_document_blocks(file_path)
        if not blocks:
            return IndexOperationResult(
                success=False,
                action="add_document",
                message=f"文档 {file_name} 未提取到有效内容",
                doc_name=file_name,
                error_code="no_effective_content",
                health_report=health,
            )

        contents = [_truncate_for_milvus_varchar(block.get("retrieval_text", block["content"])) for block in blocks]
        logger.info("开始向量化：%d 个块", len(contents))
        embedding_started_at = time.time()
        embeddings = _encode_with_progress(self._get_embedder(), contents, stage_label="向量化")
        logger.info("向量化完成，用时 %.1fs", time.time() - embedding_started_at)
        milvus_started_at = time.time()
        logger.info("写入 Milvus")
        self._insert_blocks_to_milvus(collection, blocks, embeddings.tolist(), contents)
        collection.flush()
        collection.load()
        logger.info("Milvus 写入完成，用时 %.1fs", time.time() - milvus_started_at)
        es_started_at = time.time()
        logger.info("写入 Elasticsearch")
        save_metadata_to_es(self.es_client, blocks)
        logger.info("Elasticsearch 写入完成，用时 %.1fs", time.time() - es_started_at)
        _persist_document_health_reports({file_name: health})

        # Optional: build knowledge graph from indexed blocks
        # Disabled by default, controlled via ENABLE_KNOWLEDGE_GRAPH env var
        # Graph building failures never affect indexing results
        try:
            from knowledge.graph import GRAPH_ENABLED, build_graph_from_units, get_global_graph, save_global_graph
            if GRAPH_ENABLED:
                logger.info("构建知识图谱")
                graph_started_at = time.time()
                graph = get_global_graph()
                graph.remove_document(file_name)
                build_graph_from_units(blocks, existing_graph=graph)
                save_global_graph()
                logger.info(
                    "知识图谱构建完成，用时 %.1fs（总计 %d 实体，%d 关系）",
                    time.time() - graph_started_at,
                    graph.entity_count,
                    graph.relation_count,
                )
        except Exception as graph_exc:
            logger.warning("知识图谱构建失败（不影响入库结果）: %s", graph_exc)

        message = f"增量索引成功！文档 {file_name} 共 {len(blocks)} 个证据块"
        return IndexOperationResult(
            success=True,
            action="add_document",
            message=message,
            doc_name=file_name,
            chunk_count=len(blocks),
            health_report=health,
        )

    def remove_document(self, doc_name: str) -> IndexOperationResult:
        try:
            collection = self.ensure_databases()
        except Exception as exc:
            return IndexOperationResult(
                success=False,
                action="remove_document",
                message=f"数据库连接失败：{exc}",
                doc_name=doc_name,
                error_code="database_unavailable",
            )
        self._remove_doc_from_milvus(collection, doc_name)
        self._remove_doc_from_es(doc_name)
        _remove_document_health_report(doc_name)
        remove_structural_records(doc_name)
        try:
            from knowledge.graph import GRAPH_ENABLED, remove_document_from_global_graph
            if GRAPH_ENABLED:
                remove_document_from_global_graph(doc_name)
        except Exception as graph_exc:
            logger.warning("知识图谱删除清理失败（不影响索引删除）: %s", graph_exc)
        return IndexOperationResult(
            success=True,
            action="remove_document",
            message=f"已从索引中删除文档：{doc_name}",
            doc_name=doc_name,
        )

    def build_index(self) -> IndexOperationResult:
        if not os.path.exists(DOC_DIR):
            return IndexOperationResult(
                success=False,
                action="build_index",
                message=f"目录不存在：{DOC_DIR}",
                error_code="doc_dir_missing",
            )

        files = [name for name in os.listdir(DOC_DIR) if name.lower().endswith((".pdf", ".docx"))]
        if not files:
            return IndexOperationResult(
                success=False,
                action="build_index",
                message=f"目录下没有可索引文档：{DOC_DIR}",
                error_code="no_documents",
            )

        try:
            collection = self.reset_databases()
        except Exception as exc:
            return IndexOperationResult(
                success=False,
                action="build_index",
                message=f"数据库连接失败：{exc}",
                error_code="database_unavailable",
            )

        clear_structural_records()
        reports = {}
        all_blocks: List[dict] = []
        for file_name in files:
            path = os.path.join(DOC_DIR, file_name)
            logger.info("解析文档：%s", file_name)
            blocks, health = self._index_document_blocks(path)
            reports[file_name] = health
            all_blocks.extend(blocks)

        if not all_blocks:
            return IndexOperationResult(
                success=False,
                action="build_index",
                message="未提取到任何有效内容",
                error_code="no_effective_content",
                details={"reports": reports},
            )

        contents = [_truncate_for_milvus_varchar(block.get("retrieval_text", block["content"])) for block in all_blocks]
        embeddings = _encode_with_progress(self._get_embedder(), contents, stage_label="全量向量化")
        self._insert_blocks_to_milvus(collection, all_blocks, embeddings.tolist(), contents)
        collection.flush()
        collection.load()
        save_metadata_to_es(self.es_client, all_blocks)
        _persist_document_health_reports(reports)
        message = f"索引构建成功！共 {len(all_blocks)} 个证据块"
        return IndexOperationResult(
            success=True,
            action="build_index",
            message=message,
            chunk_count=len(all_blocks),
            details={"reports": reports, "document_count": len(files)},
        )

