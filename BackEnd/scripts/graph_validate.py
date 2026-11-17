"""Ask the running graph API with frozen questions and record the raw result.

Ground truth stays in the questions file. Only the query text is sent.
"""

from __future__ import annotations

import importlib.metadata as metadata
import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QUESTIONS = ROOT / "work" / "corpus" / "questions.jsonl"
OUTPUT = ROOT / "work" / "graph-validation.json"
RAW = ROOT / "work" / "graph-data" / "raw-llm.jsonl"
API = "http://127.0.0.1:8788"
WANTED = ["AV07", "AV14", "AV20", "AV21", "AV22", "AV23"]


def _get(path: str):
    with urllib.request.urlopen(API + path, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _post(path: str, payload: dict):
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        API + path,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read().decode("utf-8"))


def _sourced_path(graph: dict) -> dict:
    evidence = {}
    best = None
    for path in graph.get("paths") or []:
        if int(path.get("hops") or 0) < 2:
            continue
        edge_ids = set(path.get("edgeIds") or [])
        edges = [edge for edge in graph.get("edges") or [] if edge.get("id") in edge_ids]
        if len(edges) < 2:
            continue
        resolved = []
        ok = True
        for edge in edges:
            source_ids = list(edge.get("sourceIds") or [])
            if not source_ids:
                ok = False
                break
            for source_id in source_ids:
                if source_id not in evidence:
                    try:
                        evidence[source_id] = _get("/api/graph/evidence?chunkId=" + urllib.parse.quote(source_id))
                    except Exception as exc:
                        evidence[source_id] = {"error": str(exc)}
                row = evidence[source_id]
                if not str(row.get("content") or "").strip():
                    ok = False
            resolved.append(
                {
                    "id": edge.get("id"),
                    "label": edge.get("label"),
                    "sourceIds": source_ids,
                    "sections": [str(evidence.get(source_id, {}).get("section_title") or "") for source_id in source_ids],
                }
            )
        if ok:
            candidate = {"nodeIds": path.get("nodeIds"), "hops": path.get("hops"), "edges": resolved}
            best = candidate
            break
    return {"found": bool(best), "path": best}


def main() -> int:
    status = _get("/api/graph/status")
    documents = _get("/api/graph/documents")
    graph = _post("/api/graph/search", {"query": "", "docIds": [], "limit": 80})
    provenance = _sourced_path(graph)
    questions = {}
    for line in QUESTIONS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        questions[row.get("query_id") or row.get("id")] = row
    answers = []
    for query_id in WANTED:
        question = str(questions[query_id].get("query") or questions[query_id].get("question") or "")
        result = _post("/api/graph/ask", {"question": question, "mode": "agent", "docIds": []})
        answers.append(
            {
                "query_id": query_id,
                "question": question,
                "answer": result.get("answer"),
                "stopReason": result.get("stopReason"),
                "citations": result.get("citations"),
                "source_ids": [row.get("chunk_id") for row in result.get("sources") or []],
                "trace": result.get("trace"),
                "path_hops": [path.get("hops") for path in (result.get("graph") or {}).get("paths") or []],
            }
        )
        print(query_id, result.get("stopReason"), flush=True)
    versions = {}
    for name in ("lightrag-hku", "langgraph", "fastapi", "networkx"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = ""
    report = {
        "updated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "framework_versions": versions,
        "status": status,
        "documents": documents,
        "search_counts": {
            "nodes": len(graph.get("nodes") or []),
            "edges": len(graph.get("edges") or []),
            "paths": len(graph.get("paths") or []),
        },
        "sourced_two_edge_path": provenance,
        "answers": answers,
        "raw_llm_jsonl": str(RAW),
        "raw_llm_exists": RAW.exists(),
        "notes": [
            "paths are structural walks over extracted edges, not causal proof and not the agent tool trace.",
            "Bibliography pages are excluded from the pilot sample. A path is not treated as an aviation answer unless its passages are technical text.",
            "Frozen question labels were not sent to the model.",
        ],
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("WROTE", OUTPUT, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
