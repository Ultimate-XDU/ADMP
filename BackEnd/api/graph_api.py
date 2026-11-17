"""Standalone graph API for a demo that does not start the legacy search engine.

Run from the repository root:

    GRAPH_OLLAMA_HOST=http://127.0.0.1:11435 ./scripts/run-graph-api

The launcher pins the isolated Ollama port. A normal ADMP process inherits
LLM_BASE_URL (default http://127.0.0.1:11434/v1) instead.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from knowledge.graph_rag.router import router as graph_router
from knowledge.graph_rag.service import get_service

app = FastAPI(title="ADMP Graph API", version="1.0.0")
_cors = [origin.strip() for origin in os.environ.get("AEROSPACE_CORS_ORIGINS", "").split(",") if origin.strip()] or [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)
app.include_router(graph_router)


@app.get("/api/health")
def health():
    graph = get_service().status_payload()
    return {
        "status": "ok",
        "details": {
            "api": "ok",
            "library": "empty",
            "graph": graph["status"],
            "graphBuilding": graph["building"],
        },
    }


@app.get("/api/knowledge-bases")
def empty_knowledge_bases():
    return []


@app.get("/api/documents")
def empty_documents():
    return []


@app.get("/api/chat/conversations")
def empty_conversations():
    return []


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("GRAPH_API_PORT", "8788")))
