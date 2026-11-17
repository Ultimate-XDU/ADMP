"""HTTP routes for graph status, build, retrieval, and agent answers."""

from __future__ import annotations

from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from knowledge.graph_rag.service import get_service

router = APIRouter()


class LocatorModel(BaseModel):
    label: str
    page: Optional[int] = None
    anchor: Optional[str] = None
    rowStart: Optional[int] = None
    rowEnd: Optional[int] = None
    path: Optional[str] = None
    jsonPointer: Optional[str] = None
    xpath: Optional[str] = None
    slide: Optional[int] = None
    sheet: Optional[str] = None


class SpanModel(BaseModel):
    start: int
    end: int


class ProvenanceModel(BaseModel):
    docId: str
    sourceId: Optional[str] = None
    sourceUrl: Optional[str] = None
    locator: Optional[LocatorModel] = None
    method: str
    quote: Optional[str] = None
    span: Optional[SpanModel] = None


class GraphNodeModel(BaseModel):
    id: str
    label: str
    type: str
    sourceIds: list[str]
    kind: Optional[Literal["structure", "semantic"]] = None
    level: Optional[int] = None
    parentId: Optional[str] = None
    docId: Optional[str] = None
    locator: Optional[LocatorModel] = None


class GraphEdgeModel(BaseModel):
    id: str
    source: str
    target: str
    label: str
    sourceIds: list[str]
    edgeKind: Optional[Literal["contains", "mentions", "references", "semantic"]] = None
    relationType: Optional[str] = None
    directed: Optional[bool] = None
    provenance: Optional[list[ProvenanceModel]] = None


class GraphPathModel(BaseModel):
    nodeIds: list[str]
    edgeIds: list[str]
    hops: int
    semanticHops: Optional[int] = None
    hierarchyDepth: Optional[int] = None


class GraphDataModel(BaseModel):
    nodes: list[GraphNodeModel]
    edges: list[GraphEdgeModel]
    paths: list[GraphPathModel]
    truncated: bool = False


class GraphStatusModel(BaseModel):
    status: str
    documents: int
    nodes: int
    edges: int
    chunks: int
    building: bool
    completedDocuments: int
    totalDocuments: int
    error: Optional[str] = None
    structureNodes: Optional[int] = None
    semanticNodes: Optional[int] = None
    structuredDocuments: Optional[int] = None
    semanticDocuments: Optional[int] = None


class GraphDocumentModel(BaseModel):
    id: str
    name: str
    sourceUrl: str
    chunks: int
    indexed: bool
    format: Optional[str] = None
    publisher: Optional[str] = None


class EvidenceModel(BaseModel):
    chunk_id: str
    content: str
    doc_name: str
    page_number: int
    section_title: str
    source_url: Optional[str] = None
    heading: str
    parent_heading: str
    score: float
    page_count: int
    highlight_text: str
    source_type: str
    locator: Optional[LocatorModel] = None
    doc_id: Optional[str] = None
    document_key: Optional[str] = None
    document_title: Optional[str] = None


class TraceStepModel(BaseModel):
    step: int
    action: str
    query: str
    summary: str
    nodeIds: list[str]
    sourceIds: list[str]


class CitationModel(BaseModel):
    id: int
    sourceIndex: int
    chunkId: str


class DiagnosticSpanModel(BaseModel):
    chunkId: str
    docId: str
    start: int
    end: int
    text: str


class DiagnosticBudgetModel(BaseModel):
    candidates: int
    rerankPool: int
    contextChars: int
    passageChars: int
    toolCalls: int


class DiagnosticsModel(BaseModel):
    budgets: DiagnosticBudgetModel
    candidateSpans: list[DiagnosticSpanModel]
    contextSpans: list[DiagnosticSpanModel]


class GraphAnswerModel(BaseModel):
    answer: str
    sources: list[EvidenceModel]
    citations: list[CitationModel]
    graph: GraphDataModel
    trace: list[TraceStepModel]
    stopReason: str
    diagnostics: Optional[DiagnosticsModel] = None


class BuildBody(BaseModel):
    docIds: list[str] = Field(default_factory=list)


class SearchBody(BaseModel):
    query: str
    docIds: Optional[list[str]] = None
    limit: Optional[int] = None
    scope: Literal["hierarchy", "semantic", "all"] = "semantic"


class ExpandBody(BaseModel):
    nodeIds: list[str] = Field(default_factory=list)
    hops: int
    docIds: Optional[list[str]] = None
    scope: Literal["hierarchy", "semantic", "all"] = "semantic"


class AskBody(BaseModel):
    question: str
    mode: Literal["single", "graph", "agent"]
    docIds: Optional[list[str]] = None
    diagnostics: bool = False


@router.get("/api/graph/status", response_model=GraphStatusModel, response_model_exclude_none=True)
def graph_status():
    return get_service().status_payload()


@router.get("/api/graph/documents", response_model=list[GraphDocumentModel], response_model_exclude_none=True)
def graph_documents():
    return get_service().documents_payload()


@router.post("/api/graph/build", response_model=GraphStatusModel, response_model_exclude_none=True)
def graph_build(body: BuildBody):
    try:
        return get_service().start_build(body.docIds)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/api/graph/search", response_model=GraphDataModel, response_model_exclude_none=True)
def graph_search(body: SearchBody):
    # An empty query is the graph browser. docIds [] means every document.
    return get_service().search(body.query, body.docIds, body.limit, body.scope)


@router.post("/api/graph/expand", response_model=GraphDataModel, response_model_exclude_none=True)
def graph_expand(body: ExpandBody):
    if not body.nodeIds:
        raise HTTPException(status_code=400, detail="nodeIds 不能为空")
    if body.hops < 1:
        raise HTTPException(status_code=400, detail="hops 必须大于 0")
    return get_service().expand(body.nodeIds, body.hops, body.docIds, body.scope)


@router.get("/api/graph/evidence", response_model=EvidenceModel, response_model_exclude_none=True)
def graph_evidence(chunkId: str = Query(...)):
    payload = get_service().evidence_payload(chunkId)
    if payload is None:
        raise HTTPException(status_code=404, detail="原文不存在")
    return payload


@router.post("/api/graph/ask", response_model=GraphAnswerModel, response_model_exclude_none=True)
def graph_ask(body: AskBody):
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question 不能为空")
    return get_service().ask(question, body.mode, body.docIds, body.diagnostics)
