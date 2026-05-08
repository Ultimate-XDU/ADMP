# aerospace_api.py —— FastAPI 服务，统一对接前端
import io
import logging
import os
import re
import sqlite3
import sys
import threading
import time
import json
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import fitz
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# 加载 .env
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

# 确保能导入同目录模块
sys.path.insert(0, str(Path(__file__).parent))

from aerospace_search_main import HybridSearch, DualDBManager, IndexOperationResult
from aerospace_qa import ask_structured, call_qwen

# ==========================================
# FastAPI 应用
# ==========================================
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

logger = logging.getLogger(__name__)
app = FastAPI(title="航空航天知识检索 API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局搜索引擎实例（启动时加载模型，复用）
_engine: Optional[object] = None
_engine_lock = threading.Lock()


def get_engine():
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                try:
                    _engine = HybridSearch(hyde_llm_call=call_qwen)
                except Exception as exc:
                    raise HTTPException(status_code=503, detail=f"检索服务不可用，请先启动 Elasticsearch 和 Milvus：{exc}") from exc
    return _engine


def _ensure_index_action_success(result: IndexOperationResult, success_actions: tuple[str, ...]) -> IndexOperationResult:
    if result.success and result.action in success_actions:
        return result

    status_code = 500
    error_code = str(result.error_code or "").strip().lower()
    if error_code in {"database_unavailable"}:
        status_code = 503
    elif error_code in {"file_not_found"}:
        status_code = 404
    elif error_code in {"doc_dir_missing", "no_documents", "no_effective_content"}:
        status_code = 400
    raise HTTPException(status_code=status_code, detail=result.message or "索引操作失败")


@lru_cache(maxsize=256)
def _render_pdf_page_png(file_path: str, page: int, dpi: int, mtime_ns: int) -> bytes:
    document = fitz.open(file_path)
    try:
        if page < 1 or page > document.page_count:
            raise IndexError(page)
        pdf_page = document.load_page(page - 1)
        zoom = dpi / 72.0
        pix = pdf_page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        return pix.tobytes("png")
    finally:
        document.close()


# ==========================================
# 1. 问答接口
# ==========================================
CHAT_DB_FILE = Path(__file__).parent / "chat_history.sqlite3"
_chat_db_lock = threading.Lock()


def _now_ms() -> int:
    return int(time.time() * 1000)


def _get_chat_db() -> sqlite3.Connection:
    conn = sqlite3.connect(CHAT_DB_FILE, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _init_chat_db() -> None:
    with _chat_db_lock:
        with _get_chat_db() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS chat_conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    payload_json TEXT,
                    created_at INTEGER NOT NULL,
                    FOREIGN KEY(conversation_id) REFERENCES chat_conversations(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_chat_messages_conversation_created
                    ON chat_messages(conversation_id, created_at, id);
                CREATE INDEX IF NOT EXISTS idx_chat_conversations_updated
                    ON chat_conversations(updated_at DESC);
                """
            )


def _shorten_title(question: str) -> str:
    compact = re.sub(r"\s+", " ", str(question or "")).strip()
    if not compact:
        return "Untitled Chat"
    return compact[:40]


def _answer_preview(payload: dict[str, Any], fallback: str = "") -> str:
    answer = str(payload.get("answer") or fallback or "")
    return re.sub(r"\s+", " ", answer).strip()[:160]


def _ensure_chat_conversation(conversation_id: Optional[str], first_question: str) -> tuple[str, int, bool]:
    created = False
    now = _now_ms()
    safe_id = str(conversation_id or "").strip()
    if not safe_id:
        safe_id = uuid.uuid4().hex

    with _chat_db_lock:
        with _get_chat_db() as conn:
            row = conn.execute("SELECT created_at FROM chat_conversations WHERE id = ?", (safe_id,)).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO chat_conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                    (safe_id, _shorten_title(first_question), now, now),
                )
                created = True
                created_at = now
            else:
                created_at = int(row["created_at"])
    return safe_id, created_at, created


def _load_chat_history(conversation_id: str, limit: int = 12) -> list[dict[str, Any]]:
    with _chat_db_lock:
        with _get_chat_db() as conn:
            rows = conn.execute(
                """
                SELECT role, content, payload_json
                FROM chat_messages
                WHERE conversation_id = ?
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (conversation_id, limit),
            ).fetchall()
    return [
        {
            "role": row["role"],
            "content": row["content"],
            "payload": _normalize_cached_payload_sources(row["content"], _parse_chat_payload(row["payload_json"])),
        }
        for row in reversed(rows)
    ]


def _store_chat_turn(conversation_id: str, question: str, qa_payload: dict[str, Any]) -> int:
    now = _now_ms()
    answer = str(qa_payload.get("answer") or "")
    payload = dict(qa_payload)
    payload["conversationId"] = conversation_id
    with _chat_db_lock:
        with _get_chat_db() as conn:
            conn.execute(
                "INSERT INTO chat_messages (conversation_id, role, content, payload_json, created_at) VALUES (?, 'user', ?, NULL, ?)",
                (conversation_id, question, now),
            )
            conn.execute(
                "INSERT INTO chat_messages (conversation_id, role, content, payload_json, created_at) VALUES (?, 'assistant', ?, ?, ?)",
                (conversation_id, answer, json.dumps(payload, ensure_ascii=False), now + 1),
            )
            conn.execute(
                "UPDATE chat_conversations SET updated_at = ? WHERE id = ?",
                (now + 1, conversation_id),
            )
    return now + 1


def _parse_chat_payload(payload_json: Optional[str]) -> Optional[dict[str, Any]]:
    if not payload_json:
        return None
    try:
        payload = json.loads(payload_json)
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _normalize_cached_payload_sources(message_content: str, payload: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not payload:
        return payload
    sources = payload.get("sources")
    if not isinstance(sources, list) or len(sources) < 2:
        return payload

    content = str(message_content or "")
    matched_doc_names = set()
    for source in sources:
        if not isinstance(source, dict):
            continue
        doc_name = Path(str(source.get("doc_name") or "")).name
        if not doc_name:
            continue
        stem = Path(doc_name).stem
        if doc_name in content or (stem and stem in content):
            matched_doc_names.add(doc_name)

    if not matched_doc_names:
        return payload

    ordered_sources = sorted(
        enumerate(sources),
        key=lambda item: (
            0 if Path(str(item[1].get("doc_name") or "")).name in matched_doc_names else 1,
            item[0],
        ),
    )
    normalized_sources = [source for _, source in ordered_sources]
    if normalized_sources == sources:
        return payload

    normalized = dict(payload)
    normalized["sources"] = normalized_sources
    normalized["evidenceIds"] = [source.get("chunk_id", "") for source in normalized_sources if isinstance(source, dict) and source.get("chunk_id")]
    return normalized


_init_chat_db()


class QARequest(BaseModel):
    question: str
    conversationId: Optional[str] = None


class EvidenceBBox(BaseModel):
    x: float
    y: float
    width: float
    height: float


class EvidenceBBoxPage(BaseModel):
    page_number: int
    bbox: EvidenceBBox
    highlight_text: str = ""
    matched_line_count: int = 0
    semantic_line_count: int = 0


class QASourceItem(BaseModel):
    content: str
    doc_name: str
    heading: str
    parent_heading: str
    score: float
    chunk_id: str = ""
    page_number: int = 1
    page_count: int = 1
    section_title: str = ""
    highlight_text: str = ""
    bbox: Optional[EvidenceBBox] = None
    bbox_pages: list[EvidenceBBoxPage] = Field(default_factory=list)
    source_type: str = "text"
    layout_type: str = "single_column"
    parse_confidence: float = 0.0
    document_type: str = "AMM"
    is_ocr_fallback: bool = False
    article_id: str = ""
    article_label: str = ""
    clause_path: str = ""
    chunk_role: str = "text"
    structural_node_id: str = ""
    structural_parent_id: str = ""
    structural_record_type: str = ""


class QAResponse(BaseModel):
    answer: str
    sources: list[QASourceItem] = Field(default_factory=list)
    confidence: float = 0.0
    abstainReason: str = ""
    evidenceIds: list[str] = Field(default_factory=list)
    queryType: str = ""
    conversationId: str = ""
    createdAt: int = 0
    updatedAt: int = 0


@app.post("/api/qa", response_model=QAResponse)
def qa_endpoint(req: QARequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")
    conversation_id, created_at, _ = _ensure_chat_conversation(req.conversationId, question)
    history = _load_chat_history(conversation_id)
    result = ask_structured(question, engine=get_engine(), history=history)
    updated_at = _store_chat_turn(conversation_id, question, result)
    return {
        **result,
        "conversationId": conversation_id,
        "createdAt": created_at,
        "updatedAt": updated_at,
    }


# ==========================================
# 2. 仅检索接口（返回结构化检索结果，不调用 LLM）
# ==========================================
class ChatTitlePayload(BaseModel):
    title: str


@app.get("/api/chat/conversations")
def list_chat_conversations(limit: int = Query(50, ge=1, le=200)):
    with _chat_db_lock:
        with _get_chat_db() as conn:
            rows = conn.execute(
                """
                SELECT
                    c.id,
                    c.title,
                    c.created_at,
                    c.updated_at,
                    (
                        SELECT content
                        FROM chat_messages
                        WHERE conversation_id = c.id AND role = 'user'
                        ORDER BY created_at DESC, id DESC
                        LIMIT 1
                    ) AS latest_question,
                    (
                        SELECT content
                        FROM chat_messages
                        WHERE conversation_id = c.id AND role = 'assistant'
                        ORDER BY created_at DESC, id DESC
                        LIMIT 1
                    ) AS latest_answer,
                    (
                        SELECT payload_json
                        FROM chat_messages
                        WHERE conversation_id = c.id AND role = 'assistant'
                        ORDER BY created_at DESC, id DESC
                        LIMIT 1
                    ) AS latest_payload
                FROM chat_conversations c
                ORDER BY c.updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

    items = []
    for row in rows:
        payload = _normalize_cached_payload_sources(row["latest_answer"], _parse_chat_payload(row["latest_payload"])) or {}
        question = str(row["latest_question"] or "")
        items.append(
            {
                "id": row["id"],
                "title": row["title"],
                "question": question,
                "preview": _answer_preview(payload, str(row["latest_answer"] or "")),
                "createdAt": int(row["created_at"]),
                "updatedAt": int(row["updated_at"]),
            }
        )
    return items


@app.get("/api/chat/conversations/{conversation_id}")
def get_chat_conversation(conversation_id: str):
    with _chat_db_lock:
        with _get_chat_db() as conn:
            conversation = conn.execute(
                "SELECT id, title, created_at, updated_at FROM chat_conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            if conversation is None:
                raise HTTPException(status_code=404, detail="Chat conversation not found")

            message_rows = conn.execute(
                """
                SELECT id, role, content, payload_json, created_at
                FROM chat_messages
                WHERE conversation_id = ?
                ORDER BY created_at ASC, id ASC
                """,
                (conversation_id,),
            ).fetchall()

    messages = [
        {
            "id": str(row["id"]),
            "role": row["role"],
            "content": row["content"],
            "payload": _normalize_cached_payload_sources(row["content"], _parse_chat_payload(row["payload_json"])),
            "createdAt": int(row["created_at"]),
        }
        for row in message_rows
    ]
    latest_question = next((msg["content"] for msg in reversed(messages) if msg["role"] == "user"), "")
    latest_assistant = next((msg for msg in reversed(messages) if msg["role"] == "assistant"), None)
    latest_payload = latest_assistant.get("payload") if latest_assistant else None
    return {
        "id": conversation["id"],
        "title": conversation["title"],
        "question": latest_question,
        "preview": _answer_preview(latest_payload or {}, latest_assistant.get("content", "") if latest_assistant else ""),
        "createdAt": int(conversation["created_at"]),
        "updatedAt": int(conversation["updated_at"]),
        "messages": messages,
    }


@app.patch("/api/chat/conversations/{conversation_id}")
def update_chat_conversation(conversation_id: str, payload: ChatTitlePayload):
    title = _shorten_title(payload.title)
    with _chat_db_lock:
        with _get_chat_db() as conn:
            cursor = conn.execute(
                "UPDATE chat_conversations SET title = ? WHERE id = ?",
                (title, conversation_id),
            )
            if cursor.rowcount == 0:
                raise HTTPException(status_code=404, detail="Chat conversation not found")
    return {"id": conversation_id, "title": title}


@app.delete("/api/chat/conversations", status_code=204)
def clear_chat_conversations():
    with _chat_db_lock:
        with _get_chat_db() as conn:
            conn.execute("DELETE FROM chat_conversations")
    return None


@app.delete("/api/chat/conversations/{conversation_id}", status_code=204)
def delete_chat_conversation(conversation_id: str):
    with _chat_db_lock:
        with _get_chat_db() as conn:
            cursor = conn.execute("DELETE FROM chat_conversations WHERE id = ?", (conversation_id,))
            if cursor.rowcount == 0:
                raise HTTPException(status_code=404, detail="Chat conversation not found")
    return None


class SearchResponse(BaseModel):
    results: list[QASourceItem] = Field(default_factory=list)


@app.post("/api/search", response_model=SearchResponse)
def search_endpoint(req: QARequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")
    engine = get_engine()
    try:
        results = engine.search_for_qa(question)
    except RuntimeError as exc:
        logger.warning("检索接口执行失败 question=%s: %s", question, exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"results": results}


@app.get("/api/evidence/page-image")
def evidence_page_image(
    doc_name: str = Query(..., description="Stored document file name"),
    page: int = Query(1, ge=1),
    dpi: int = Query(144, ge=72, le=220),
):
    resolved_name = _resolve_uploaded_file_name(doc_name, "PDF")
    if not resolved_name:
        raise HTTPException(status_code=404, detail=f"未找到文档: {doc_name}")

    file_path = UPLOAD_DIR / resolved_name
    if not file_path.exists() or file_path.suffix.lower() != ".pdf":
        raise HTTPException(status_code=404, detail=f"无法预览该文档: {resolved_name}")

    try:
        stat = file_path.stat()
        png_bytes = _render_pdf_page_png(str(file_path), page, dpi, stat.st_mtime_ns)
    except IndexError:
        raise HTTPException(status_code=404, detail=f"页码越界: {page}") from None
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"生成预览失败: {exc}") from exc

    return StreamingResponse(
        io.BytesIO(png_bytes),
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@app.get("/api/health")
def health_check():
    details = {
        "api": "ok",
        "uploadDir": UPLOAD_DIR.exists(),
        "stateFile": STATE_FILE.exists(),
        "engineInitialized": _engine is not None,
        "dbManagerInitialized": _db_manager is not None,
    }
    status = "ok"

    try:
        manager = get_db_manager()
        details["elasticsearch"] = bool(manager.es_client.ping())
        try:
            manager.ensure_databases()
            details["milvus"] = True
        except Exception as exc:
            details["milvus"] = False
            details["milvusDetail"] = str(exc)
            status = "degraded"
    except Exception as exc:
        details["elasticsearch"] = False
        details["milvus"] = False
        details["dbDetail"] = str(exc)
        status = "degraded"

    if not details["uploadDir"]:
        status = "degraded"

    return {"status": status, "details": details}


@app.post("/api/reload-modules")
def reload_modules():
    """Hot-reload QA/search modules without restarting the server (keeps model in memory)."""
    import importlib
    import aerospace_search_main
    import aerospace_qa
    import aerospace_structure
    importlib.reload(aerospace_structure)
    importlib.reload(aerospace_search_main)
    importlib.reload(aerospace_qa)
    # Re-import into this module's namespace
    globals()["ask_structured"] = aerospace_qa.ask_structured
    globals()["HybridSearch"] = aerospace_search_main.HybridSearch
    globals()["DualDBManager"] = aerospace_search_main.DualDBManager
    globals()["IndexOperationResult"] = aerospace_search_main.IndexOperationResult
    return {"status": "ok", "message": "Modules reloaded successfully"}


# ==========================================
# 3. 知识库 / 文档管理状态
# ==========================================
UPLOAD_DIR = Path(__file__).parent / "aerospace_docs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

STATE_FILE = Path(__file__).parent / "management_state.json"
HEALTH_REPORT_FILE = Path(__file__).parent / "document_health_reports.json"
_DOCUMENT_TYPE_SUFFIX = {
    "PDF": ".pdf",
    "DOCX": ".docx",
    "PPTX": ".pptx",
    "XLSX": ".xlsx",
}
_SUFFIX_TO_DOCUMENT_TYPE = {suffix: doc_type for doc_type, suffix in _DOCUMENT_TYPE_SUFFIX.items()}
_ALLOWED_DOCUMENT_STATUSES = {"READY", "INDEXING", "CHUNKING", "SYNCED", "FAILED"}

_kb_id_counter = 200
_knowledge_bases: list[dict] = []

# 预置 CAAC 标准分类知识库
_DEFAULT_KNOWLEDGE_BASES = [
    {"id": 101, "name": "民航规章", "description": "中国民航局发布的适航规章，如 CCAR-25、CCAR-33、CCAR-35、CCAR-121 等运输类飞机/发动机/螺旋桨适航标准与运行规则。", "creator": "系统预置", "createdAt": "2026-04-01 00:00"},
    {"id": 102, "name": "规范性文件", "description": "民航局及下属司局发布的管理办法、实施细则、咨询通告（AC）等规范性文件。", "creator": "系统预置", "createdAt": "2026-04-01 00:00"},
    {"id": 103, "name": "法律法规", "description": "《中华人民共和国民用航空法》及相关行政法规、国务院令等上位法律法规。", "creator": "系统预置", "createdAt": "2026-04-01 00:00"},
    {"id": 104, "name": "国际公约", "description": "ICAO 国际民航公约及附件、FAA/EASA 对等适航协议与国际标准参考。", "creator": "系统预置", "createdAt": "2026-04-01 00:00"},
    {"id": 105, "name": "适航指令与通告", "description": "适航指令（AD）、服务通告（SB）、维护通告与适航改进通知。", "creator": "系统预置", "createdAt": "2026-04-01 00:00"},
    {"id": 106, "name": "维修手册与技术文档", "description": "飞机/发动机维修手册（AMM/EMM）、故障隔离手册（FIM/TSM）、结构修理手册等技术文档。", "creator": "系统预置", "createdAt": "2026-04-01 00:00"},
    {"id": 107, "name": "培训与参考资料", "description": "培训课件、答辩汇报、术语词典、缩略语映射与其他参考资料。", "creator": "系统预置", "createdAt": "2026-04-01 00:00"},
]

_KNOWLEDGE_BASE_RULES = [
    (103, "命中上位法律法规关键词", [
        "民用航空法",
        "中华人民共和国飞行基本规则",
        "民用机场管理条例",
        "通用航空飞行管制条例",
        "无人驾驶航空器飞行管理暂行条例",
        "民用航空器国籍登记条例",
        "民用航空器权利登记条例",
        "民用航空安全保卫条例",
        "国务院令",
        "行政法规",
    ]),
    (104, "命中国际公约/条约关键词", [
        "国际民用航空公约",
        "国际公约",
        "公约",
        "议定书",
        "蒙特利尔",
        "华沙",
        "东京",
        "海牙",
        "开普敦",
        "icaoannex",
    ]),
    (106, "命中维修手册与技术文档关键词", [
        "amm",
        "emm",
        "fim",
        "tsm",
        "srm",
        "ipc",
        "维修手册",
        "故障隔离",
        "结构修理",
        "aviationmaintenancetechnician",
        "aircraftinspection",
        "aircraftrepair",
        "aircraftalterations",
        "maintenanceprogram",
        "repairstation",
        "qualitycontrolmanual",
        "ac43.13",
        "ac120-16",
        "ac20-106",
        "ac145-9",
    ]),
    (105, "命中适航指令与通告关键词", ["适航指令", "服务通告", "维护通告", "airworthinessdirective"]),
    (101, "命中民航规章关键词（如 CCAR/部令/适航标准）", ["ccar", "民航规章", "部令", "适航标准", "运行规则"]),
    (102, "命中规范性文件关键词（如运行合格审定规则/管理办法/咨询通告）", [
        "运行合格审定规则",
        "合格审定规则",
        "咨询通告",
        "管理办法",
        "实施细则",
        "程序文件",
        "工作程序",
        "规范性文件",
        "ac-",
    ]),
    (107, "命中培训与参考资料关键词", ["培训", "课件", "答辩", "术语", "缩略语", "参考资料"]),
]

# 启动时加载预置知识库
_knowledge_bases = [dict(kb) for kb in _DEFAULT_KNOWLEDGE_BASES]

_doc_id_counter = 4000
_documents: list[dict] = []


def _ts():
    return time.strftime("%Y-%m-%d %H:%M")


def _ts_from_epoch(timestamp: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(timestamp))


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _format_file_size(size_bytes: int) -> str:
    if size_bytes <= 0:
        return ""
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def _normalize_display_file_name(file_name: str) -> str:
    safe_name = Path(str(file_name or "").strip()).name
    if not safe_name:
        return ""
    return Path(safe_name).stem if Path(safe_name).suffix else safe_name


def _normalize_document_type(doc_type: str, file_name: str = "") -> str:
    normalized_type = str(doc_type or "").strip().upper()
    if normalized_type in _DOCUMENT_TYPE_SUFFIX:
        return normalized_type
    suffix = Path(str(file_name or "")).suffix.lower()
    return _SUFFIX_TO_DOCUMENT_TYPE.get(suffix, "PDF")


def _build_storage_file_name(file_name: str, doc_type: str) -> str:
    safe_name = Path(str(file_name or "").strip()).name
    if not safe_name:
        return ""
    if Path(safe_name).suffix:
        return safe_name
    suffix = _DOCUMENT_TYPE_SUFFIX.get(_normalize_document_type(doc_type, safe_name), "")
    return f"{safe_name}{suffix}" if suffix else safe_name


def _resolve_uploaded_file_name(file_name: str, doc_type: str = "") -> str:
    safe_name = Path(str(file_name or "").strip()).name
    if not safe_name:
        return ""

    direct_path = UPLOAD_DIR / safe_name
    if direct_path.exists():
        return direct_path.name

    built_name = _build_storage_file_name(safe_name, doc_type)
    built_path = UPLOAD_DIR / built_name
    if built_path.exists():
        return built_path.name

    stem = Path(safe_name).stem if Path(safe_name).suffix else safe_name
    for file_path in sorted(UPLOAD_DIR.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True):
        if file_path.is_file() and file_path.stem.lower() == stem.lower():
            return file_path.name
    return ""


def _get_document_storage_file_name(record: dict) -> str:
    stored_name = str(record.get("storageFileName") or "").strip()
    if stored_name:
        return stored_name
    return _build_storage_file_name(str(record.get("fileName") or ""), str(record.get("type") or ""))


def _match_knowledge_base_rule(file_name: str) -> Optional[tuple[int, str]]:
    normalized = re.sub(r"\s+", "", file_name).lower()
    for kb_id, reason, keywords in _KNOWLEDGE_BASE_RULES:
        if any(keyword in normalized for keyword in keywords):
            return kb_id, reason
    return None


def _infer_knowledge_base_id(file_name: str) -> int:
    matched_rule = _match_knowledge_base_rule(file_name)
    if matched_rule is not None:
        matched_kb_id, _ = matched_rule
        if any(kb["id"] == matched_kb_id for kb in _knowledge_bases):
            return matched_kb_id
    return _knowledge_bases[0]["id"] if _knowledge_bases else _DEFAULT_KNOWLEDGE_BASES[0]["id"]


def _normalize_knowledge_base_record(record: dict) -> Optional[dict]:
    if not isinstance(record, dict):
        return None

    kb_id = _safe_int(record.get("id"), 0)
    name = str(record.get("name") or "").strip()
    if kb_id <= 0 or not name:
        return None

    return {
        "id": kb_id,
        "name": name,
        "description": str(record.get("description") or "").strip(),
        "creator": str(record.get("creator") or "he114TnT").strip() or "he114TnT",
        "createdAt": str(record.get("createdAt") or _ts()).strip() or _ts(),
    }


def _normalize_document_record(record: dict) -> Optional[dict]:
    if not isinstance(record, dict):
        return None

    doc_id = _safe_int(record.get("id"), 0)
    if doc_id <= 0:
        return None

    raw_file_name = str(record.get("fileName") or "").strip()
    raw_type = str(record.get("type") or "").strip()
    storage_file_name = str(record.get("storageFileName") or "").strip()
    if not storage_file_name:
        storage_file_name = _resolve_uploaded_file_name(raw_file_name, raw_type)

    storage_path = UPLOAD_DIR / storage_file_name if storage_file_name else None
    display_name = _normalize_display_file_name(raw_file_name or storage_file_name)
    if not display_name:
        return None

    document_type = _normalize_document_type(raw_type, storage_file_name or raw_file_name)
    knowledge_base_id = _safe_int(record.get("knowledgeBaseId"), 0)
    if not any(kb["id"] == knowledge_base_id for kb in _knowledge_bases):
        knowledge_base_id = _infer_knowledge_base_id(storage_file_name or raw_file_name)

    size = str(record.get("size") or "").strip()
    uploaded_at = str(record.get("uploadedAt") or "").strip()
    if storage_path and storage_path.exists():
        if not size:
            size = _format_file_size(storage_path.stat().st_size)
        if not uploaded_at:
            uploaded_at = _ts_from_epoch(storage_path.stat().st_mtime)

    status = str(record.get("status") or "").strip().upper()
    if status not in _ALLOWED_DOCUMENT_STATUSES:
        status = "READY" if storage_path and storage_path.exists() else "FAILED"

    return {
        "id": doc_id,
        "fileName": display_name,
        "knowledgeBaseId": knowledge_base_id,
        "type": document_type,
        "size": size,
        "chunkCount": _safe_int(record.get("chunkCount"), 0),
        "status": status,
        "uploadedAt": uploaded_at or _ts(),
        "summary": str(record.get("summary") or ""),
        "storageFileName": storage_file_name,
    }


def _sort_documents(records: list[dict]) -> list[dict]:
    return sorted(records, key=lambda item: (str(item.get("uploadedAt") or ""), _safe_int(item.get("id"), 0)), reverse=True)


def _persist_management_state():
    payload = {
        "knowledgeBases": _knowledge_bases,
        "documents": _documents,
        "kbIdCounter": _kb_id_counter,
        "docIdCounter": _doc_id_counter,
    }
    temp_path = STATE_FILE.with_suffix(".tmp")
    try:
        temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(STATE_FILE)
    except Exception as exc:
        print(f"[WARN] 保存管理状态失败: {exc}")


def _load_document_health_report_map() -> dict[str, dict]:
    if not HEALTH_REPORT_FILE.exists():
        return {}
    try:
        payload = json.loads(HEALTH_REPORT_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("读取文档健康报告失败: %s", exc)
        return {}

    if isinstance(payload, dict):
        documents = payload.get("documents")
        if isinstance(documents, dict):
            return documents
        return {str(key): value for key, value in payload.items() if isinstance(value, dict)}
    return {}


def _update_document_index_state(storage_file_name: str, chunk_count: int, status: str = "SYNCED") -> bool:
    updated = False
    for record in _documents:
        if _get_document_storage_file_name(record).lower() != str(storage_file_name or "").lower():
            continue
        if chunk_count >= 0:
            record["chunkCount"] = chunk_count
        if status:
            record["status"] = status
        updated = True
        break
    if updated:
        _persist_management_state()
    return updated


def _sync_document_index_state_from_reports():
    reports = _load_document_health_report_map()
    if not reports:
        return

    changed = False
    for record in _documents:
        storage_file_name = _get_document_storage_file_name(record)
        report = reports.get(storage_file_name)
        if not isinstance(report, dict):
            continue
        chunk_count = _safe_int(report.get("chunk_count"), record.get("chunkCount", 0))
        if record.get("chunkCount") != chunk_count:
            record["chunkCount"] = chunk_count
            changed = True
        if record.get("status") in {"READY", "INDEXING", "CHUNKING"}:
            record["status"] = "SYNCED"
            changed = True
    if changed:
        _persist_management_state()


def _sync_documents_with_upload_dir() -> bool:
    global _documents, _doc_id_counter

    changed = False
    normalized_documents: list[dict] = []
    for record in _documents:
        normalized = _normalize_document_record(record)
        if normalized is None:
            changed = True
            continue
        normalized_documents.append(normalized)

    documents_by_storage = {
        str(item.get("storageFileName") or "").lower(): item
        for item in normalized_documents
        if item.get("storageFileName")
    }
    next_doc_id = max([_doc_id_counter] + [item["id"] for item in normalized_documents], default=4000)

    for file_path in sorted(UPLOAD_DIR.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True):
        if not file_path.is_file():
            continue
        document_type = _SUFFIX_TO_DOCUMENT_TYPE.get(file_path.suffix.lower())
        if document_type is None:
            continue

        storage_key = file_path.name.lower()
        existing_record = documents_by_storage.get(storage_key)
        if existing_record is not None:
            if not existing_record.get("size"):
                existing_record["size"] = _format_file_size(file_path.stat().st_size)
                changed = True
            if not existing_record.get("uploadedAt"):
                existing_record["uploadedAt"] = _ts_from_epoch(file_path.stat().st_mtime)
                changed = True
            if existing_record.get("status") == "FAILED":
                existing_record["status"] = "READY"
                changed = True
            continue

        next_doc_id += 1
        normalized_documents.append({
            "id": next_doc_id,
            "fileName": file_path.stem,
            "knowledgeBaseId": _infer_knowledge_base_id(file_path.name),
            "type": document_type,
            "size": _format_file_size(file_path.stat().st_size),
            "chunkCount": 0,
            "status": "READY",
            "uploadedAt": _ts_from_epoch(file_path.stat().st_mtime),
            "summary": "",
            "storageFileName": file_path.name,
        })
        documents_by_storage[storage_key] = normalized_documents[-1]
        changed = True

    sorted_documents = _sort_documents(normalized_documents)
    if sorted_documents != _documents:
        changed = True
    _documents = sorted_documents
    _doc_id_counter = max([4000] + [item["id"] for item in _documents], default=4000)
    return changed


def _load_management_state():
    global _knowledge_bases, _documents, _kb_id_counter, _doc_id_counter

    _knowledge_bases = [dict(kb) for kb in _DEFAULT_KNOWLEDGE_BASES]
    _documents = []
    _kb_id_counter = max([200] + [kb["id"] for kb in _knowledge_bases], default=200)
    _doc_id_counter = 4000

    state_exists = STATE_FILE.exists()
    if state_exists:
        try:
            payload = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[WARN] 读取管理状态失败，将回退到默认配置: {exc}")
        else:
            stored_knowledge_bases = payload.get("knowledgeBases")
            if isinstance(stored_knowledge_bases, list):
                normalized_knowledge_bases = [
                    normalized
                    for normalized in (_normalize_knowledge_base_record(item) for item in stored_knowledge_bases)
                    if normalized is not None
                ]
                if normalized_knowledge_bases:
                    _knowledge_bases = normalized_knowledge_bases

            stored_documents = payload.get("documents")
            if isinstance(stored_documents, list):
                _documents = [
                    normalized
                    for normalized in (_normalize_document_record(item) for item in stored_documents)
                    if normalized is not None
                ]

            _kb_id_counter = max(
                [_safe_int(payload.get("kbIdCounter"), 200), 200] + [kb["id"] for kb in _knowledge_bases],
                default=200,
            )
            _doc_id_counter = max(
                [_safe_int(payload.get("docIdCounter"), 4000), 4000] + [doc["id"] for doc in _documents],
                default=4000,
            )

    skip_upload_dir_sync = os.environ.get("AEROSPACE_SKIP_UPLOAD_DIR_SYNC") == "1"
    if (not skip_upload_dir_sync and _sync_documents_with_upload_dir()) or not state_exists:
        _persist_management_state()


def _delete_document_assets(record: dict):
    storage_file_name = _get_document_storage_file_name(record)
    if not storage_file_name:
        return

    try:
        manager = get_db_manager()
        result = manager.remove_document(storage_file_name)
        _ensure_index_action_success(result, ("remove_document",))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"删除索引失败: {storage_file_name}: {exc}") from exc
    global _engine
    _engine = None

    file_path = UPLOAD_DIR / storage_file_name
    if file_path.exists():
        try:
            file_path.unlink()
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"删除文件失败: {storage_file_name}: {exc}")


_load_management_state()
_sync_document_index_state_from_reports()


class KnowledgeBasePayload(BaseModel):
    name: str
    description: str
    creator: Optional[str] = "he114TnT"


@app.get("/api/knowledge-bases")
def list_knowledge_bases():
    return _knowledge_bases


@app.post("/api/knowledge-bases", status_code=201)
def create_knowledge_base(payload: KnowledgeBasePayload):
    global _kb_id_counter
    _kb_id_counter += 1
    record = {
        "id": _kb_id_counter,
        "name": payload.name,
        "description": payload.description,
        "creator": payload.creator or "he114TnT",
        "createdAt": _ts(),
    }
    _knowledge_bases.insert(0, record)
    _persist_management_state()
    return record


@app.put("/api/knowledge-bases/{kb_id}")
def update_knowledge_base(kb_id: int, payload: KnowledgeBasePayload):
    for kb in _knowledge_bases:
        if kb["id"] == kb_id:
            kb["name"] = payload.name
            kb["description"] = payload.description
            _persist_management_state()
            return kb
    raise HTTPException(status_code=404, detail="知识库不存在")


@app.delete("/api/knowledge-bases/{kb_id}", status_code=204)
def delete_knowledge_base(kb_id: int):
    global _knowledge_bases, _documents
    if not any(kb["id"] == kb_id for kb in _knowledge_bases):
        raise HTTPException(status_code=404, detail="知识库不存在")

    documents_to_delete = [d for d in _documents if d.get("knowledgeBaseId") == kb_id]
    for record in documents_to_delete:
        _delete_document_assets(record)

    _knowledge_bases = [kb for kb in _knowledge_bases if kb["id"] != kb_id]
    _documents = [d for d in _documents if d.get("knowledgeBaseId") != kb_id]
    _persist_management_state()
    return None


# ==========================================
# 4. 文档 CRUD
# ==========================================
class DocumentPayload(BaseModel):
    fileName: str
    knowledgeBaseId: int
    type: str = "PDF"
    size: str = ""
    chunkCount: int = 0
    status: str = "INDEXING"
    summary: str = ""


@app.get("/api/documents")
def list_documents():
    if _sync_documents_with_upload_dir():
        _persist_management_state()
    return _documents


@app.post("/api/documents", status_code=201)
def create_document(payload: DocumentPayload):
    global _doc_id_counter, _documents

    display_name = _normalize_display_file_name(payload.fileName)
    if not display_name:
        raise HTTPException(status_code=400, detail="文档名称不能为空")
    if not any(kb["id"] == payload.knowledgeBaseId for kb in _knowledge_bases):
        raise HTTPException(status_code=400, detail="知识库不存在")

    document_type = _normalize_document_type(payload.type, payload.fileName)
    storage_file_name = _resolve_uploaded_file_name(payload.fileName, document_type)
    if not storage_file_name:
        storage_file_name = _build_storage_file_name(display_name, document_type)

    storage_path = UPLOAD_DIR / storage_file_name if storage_file_name else None
    file_size = payload.size.strip() if payload.size else ""
    uploaded_at = _ts()
    if storage_path and storage_path.exists():
        if not file_size:
            file_size = _format_file_size(storage_path.stat().st_size)
        uploaded_at = _ts_from_epoch(storage_path.stat().st_mtime)

    normalized_status = str(payload.status or "").strip().upper()
    if normalized_status not in _ALLOWED_DOCUMENT_STATUSES:
        normalized_status = "READY"

    existing_record = next(
        (
            item
            for item in _documents
            if _get_document_storage_file_name(item).lower() == storage_file_name.lower()
        ),
        None,
    )

    if existing_record is None:
        _doc_id_counter += 1
        existing_record = {"id": _doc_id_counter}
        _documents.insert(0, existing_record)

    existing_record.update({
        "fileName": display_name,
        "knowledgeBaseId": payload.knowledgeBaseId,
        "type": document_type,
        "size": file_size,
        "chunkCount": payload.chunkCount,
        "status": normalized_status,
        "uploadedAt": uploaded_at,
        "summary": payload.summary,
        "storageFileName": storage_file_name,
    })
    _documents = _sort_documents(_documents)
    _persist_management_state()
    return existing_record


@app.delete("/api/documents/{doc_id}", status_code=204)
def delete_document(doc_id: int):
    global _documents
    target_record = next((d for d in _documents if d["id"] == doc_id), None)
    if target_record is None:
        raise HTTPException(status_code=404, detail="文档不存在")

    _delete_document_assets(target_record)
    _documents = [d for d in _documents if d["id"] != doc_id]
    _persist_management_state()
    return None


# ==========================================
# 5. 文档上传（接收文件，存入 aerospace_docs 目录）
# ==========================================
@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="未提供文件")

    safe_name = Path(file.filename).name
    if not safe_name:
        raise HTTPException(status_code=400, detail="文件名无效")
    if Path(safe_name).suffix.lower() not in _SUFFIX_TO_DOCUMENT_TYPE:
        raise HTTPException(status_code=400, detail="仅支持 PDF、DOCX、PPTX、XLSX 文件")

    dest = UPLOAD_DIR / safe_name
    temp_dest = dest.with_suffix(f"{dest.suffix}.uploading")
    total_bytes = 0
    try:
        if temp_dest.exists():
            temp_dest.unlink()
        with temp_dest.open("wb") as output:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                total_bytes += len(chunk)
        temp_dest.replace(dest)
    except Exception as exc:
        for candidate in (temp_dest,):
            try:
                if candidate.exists():
                    candidate.unlink()
            except OSError:
                pass
        raise HTTPException(status_code=500, detail=f"保存上传文件失败: {exc}") from exc
    finally:
        await file.close()

    size_mb = total_bytes / (1024 * 1024)
    return {
        "fileName": safe_name,
        "size": f"{size_mb:.1f} MB",
        "path": str(dest),
    }


# ==========================================
# 6. 增量索引接口
# ==========================================
_db_manager: Optional[DualDBManager] = None
_db_manager_lock = threading.Lock()


def get_db_manager() -> DualDBManager:
    global _db_manager
    if _db_manager is None:
        with _db_manager_lock:
            if _db_manager is None:
                _db_manager = DualDBManager()
    return _db_manager


class IndexRequest(BaseModel):
    fileName: str


class IndexResponse(BaseModel):
    message: str
    success: bool = True
    action: str = ""
    docName: str = ""
    chunkCount: int = 0
    errorCode: str = ""


def _extract_chunk_count_from_message(message: str) -> int:
    """从索引结果文案中稳定提取块数量。"""
    if not message:
        return 0

    patterns = [
        r"共\s*(\d+)\s*个语义块",
        r"(\d+)\s*个语义块",
        r"共\s*(\d+)\s*个证据块",
        r"(\d+)\s*个证据块",
    ]
    for pattern in patterns:
        match = re.search(pattern, message)
        if match:
            try:
                return int(match.group(1))
            except (TypeError, ValueError):
                return 0
    return 0


@app.post("/api/index", response_model=IndexResponse)
def index_document(req: IndexRequest):
    """增量索引单个文档（不重建整个库）"""
    storage_file_name = _resolve_uploaded_file_name(req.fileName)
    if not storage_file_name:
        raise HTTPException(status_code=404, detail=f"文件不存在: {req.fileName}")
    file_path = UPLOAD_DIR / storage_file_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"文件不存在: {req.fileName}")
    try:
        manager = get_db_manager()
        result = manager.add_document(str(file_path))
        result = _ensure_index_action_success(result, ("add_document",))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"增量索引失败: {exc}") from exc
    # 索引成功后，需要重新初始化搜索引擎以加载新数据
    global _engine
    _engine = None
    # 从结果中提取块数（使用正则，避免文件名中含“共”导致误解析）
    chunk_count = result.chunk_count if result.chunk_count > 0 else _extract_chunk_count_from_message(result.message)
    _update_document_index_state(storage_file_name, chunk_count, status="SYNCED")
    return {
        "message": result.message,
        "success": result.success,
        "action": result.action,
        "docName": storage_file_name,
        "chunkCount": chunk_count,
        "errorCode": result.error_code,
    }


@app.post("/api/rebuild-index")
def rebuild_index():
    """全量重建索引"""
    try:
        manager = get_db_manager()
        result = manager.build_index()
        result = _ensure_index_action_success(result, ("build_index",))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"重建索引失败: {exc}") from exc
    global _engine
    _engine = None
    _sync_document_index_state_from_reports()
    return {
        "message": result.message,
        "success": result.success,
        "action": result.action,
        "docName": "",
        "chunkCount": result.chunk_count,
        "errorCode": result.error_code,
    }


@app.delete("/api/index/{doc_name}")
def remove_document_index(doc_name: str):
    """从索引中删除指定文档"""
    resolved_name = _resolve_uploaded_file_name(doc_name) or doc_name
    try:
        manager = get_db_manager()
        result = manager.remove_document(resolved_name)
        result = _ensure_index_action_success(result, ("remove_document",))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"删除索引失败: {resolved_name}: {exc}") from exc
    global _engine
    _engine = None
    _update_document_index_state(resolved_name, 0, status="READY")
    return {
        "message": result.message,
        "success": result.success,
        "action": result.action,
        "docName": resolved_name,
        "chunkCount": result.chunk_count,
        "errorCode": result.error_code,
    }


# ==========================================
# 7. AI 知识库分类接口
# ==========================================
class ClassifyRequest(BaseModel):
    fileName: str


class ClassifyResponse(BaseModel):
    suggestedKbId: int = 0
    suggestedKbName: str = ""
    newKbName: str = ""
    newKbDescription: str = ""
    reason: str = ""


def _find_kb_by_id(kb_id: int) -> Optional[dict]:
    for kb in _knowledge_bases:
        if kb["id"] == kb_id:
            return kb
    return None


def _rule_based_kb_classify(file_name: str) -> Optional[ClassifyResponse]:
    """规则优先分类，避免关键法规名称被大模型误判。"""
    matched_rule = _match_knowledge_base_rule(file_name)
    if matched_rule is None:
        return None

    kb_id, reason = matched_rule
    kb = _find_kb_by_id(kb_id)
    if kb:
        return ClassifyResponse(
            suggestedKbId=kb["id"],
            suggestedKbName=kb["name"],
            reason=reason,
        )
    return None


@app.post("/api/classify", response_model=ClassifyResponse)
def classify_document(req: ClassifyRequest):
    """根据文件名，用 AI 判断应归入哪个知识库；若无匹配，建议创建新知识库"""
    from aerospace_qa import call_qwen

    rule_hit = _rule_based_kb_classify(req.fileName)
    if rule_hit is not None:
        return rule_hit

    kb_list_text = "\n".join(
        f"  ID={kb['id']}, 名称=\"{kb['name']}\", 描述=\"{kb['description']}\""
        for kb in _knowledge_bases
    )

    system_prompt = "你是一位航空文档知识库管理助手。你的任务是根据文件名判断该文档应归入哪个知识库。请严格按照 JSON 格式回复，不要包含其他内容。"

    user_prompt = f"""当前已有知识库列表：
{kb_list_text}

待分类文件名："{req.fileName}"

请判断该文件应归入哪个已有知识库。回复 JSON 格式（不要 markdown 代码块）：
- 如果能匹配已有知识库：{{"match": true, "kbId": <ID>, "kbName": "<名称>", "reason": "<一句话理由>"}}
- 如果无法匹配，建议创建新知识库：{{"match": false, "newName": "<建议的知识库名称>", "newDescription": "<建议的描述>", "reason": "<一句话理由>"}}"""

    try:
        raw = call_qwen(system_prompt, user_prompt)
        # 从回复中提取 JSON
        json_match = re.search(r'\{[^}]+\}', raw, re.DOTALL)
        if not json_match:
            return ClassifyResponse(reason="AI 未返回有效 JSON")
        data = json.loads(json_match.group())

        if data.get("match"):
            kb_id = int(data.get("kbId", 0))
            kb_name = data.get("kbName", "")
            # 验证 kb_id 是否存在
            if any(kb["id"] == kb_id for kb in _knowledge_bases):
                return ClassifyResponse(
                    suggestedKbId=kb_id,
                    suggestedKbName=kb_name,
                    reason=data.get("reason", ""),
                )
            # ID 不存在，尝试按名称匹配
            for kb in _knowledge_bases:
                if kb["name"] == kb_name:
                    return ClassifyResponse(
                        suggestedKbId=kb["id"],
                        suggestedKbName=kb["name"],
                        reason=data.get("reason", ""),
                    )
            return ClassifyResponse(reason=data.get("reason", "AI 推荐的知识库不存在"))
        else:
            return ClassifyResponse(
                newKbName=data.get("newName", ""),
                newKbDescription=data.get("newDescription", ""),
                reason=data.get("reason", ""),
            )
    except Exception as e:
        return ClassifyResponse(reason=f"分类失败: {str(e)}")


# ==========================================
# 8. 对话标题总结接口
# ==========================================
class SummarizeTitleRequest(BaseModel):
    question: str
    answer: str = ""


@app.post("/api/summarize-title")
def summarize_title(req: SummarizeTitleRequest):
    """根据问答内容生成简短对话标题"""
    from aerospace_qa import call_qwen

    system_prompt = (
        "你是一个对话标题生成器。请根据用户的问题生成一个简短的对话标题，"
        "不超过15个字，不要使用标点符号和引号。只输出标题文本本身。"
    )
    user_prompt = f"问题：{req.question}"
    if req.answer:
        user_prompt += f"\n回答摘要：{req.answer[:200]}"

    try:
        title = call_qwen(system_prompt, user_prompt).strip().strip("\"'")
        if len(title) > 20:
            title = title[:20]
        if not title:
            title = req.question[:15]
        return {"title": title}
    except Exception:
        return {"title": req.question[:15]}


# ==========================================
# 启动入口
# ==========================================
if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("🚀 航空航天知识检索 API 服务")
    print("   地址: http://127.0.0.1:8787")
    print("   文档: http://127.0.0.1:8787/docs")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8787)
