# management_state_store.py — shared management_state.json persistence
from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_STATE_FILE = BASE_DIR.parent / "management_state.json"

_LEGACY_STATE_FILES = (
    BASE_DIR / "management_state.json",
    BASE_DIR.parent / "api" / "management_state.json",
)

_FILE_LOCK_TIMEOUT_SECONDS = 30.0

_process_lock = threading.RLock()


def _acquire_file_lock(handle) -> None:
    if os.name == "nt":
        import msvcrt

        deadline = time.monotonic() + _FILE_LOCK_TIMEOUT_SECONDS
        while True:
            try:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                return
            except OSError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def _release_file_lock(handle) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextlib.contextmanager
def _exclusive_file_lock(path: Path):
    """Best-effort cross-process lock for Windows and POSIX hosts."""
    lock_path = path.with_name(f"{path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        with _process_lock:
            _acquire_file_lock(handle)
            try:
                yield
            finally:
                _release_file_lock(handle)


def _backup_corrupt_state(path: Path) -> Path | None:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = path.with_name(f"{path.stem}.corrupt-{stamp}{path.suffix}")
    try:
        os.replace(path, backup)
    except OSError:
        return None
    return backup


def _read_state_unlocked(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        backup = _backup_corrupt_state(path)
        if backup is not None:
            logger.error("状态文件 %s 解析失败，已备份到 %s，将重新构建：%s", path, backup.name, exc)
        else:
            logger.error("状态文件 %s 解析失败且备份失败，将重新构建：%s", path, exc)
        return {}
    except OSError as exc:
        logger.warning("读取状态文件 %s 失败：%s", path, exc)
        return {}
    if not isinstance(payload, dict):
        backup = _backup_corrupt_state(path)
        logger.error(
            "状态文件 %s 顶层不是 JSON 对象，%s，将重新构建",
            path,
            f"已备份到 {backup.name}" if backup else "备份失败",
        )
        return {}
    return payload


def _merge_state_payloads(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key in ("knowledgeBases", "documents"):
        rows = source.get(key)
        if not isinstance(rows, list):
            continue
        merged_rows = target.setdefault(key, [])
        index_by_id = {
            row.get("id"): index
            for index, row in enumerate(merged_rows)
            if isinstance(row, dict) and row.get("id") is not None
        }
        for row in rows:
            if not isinstance(row, dict) or row.get("id") is None:
                continue
            row_id = row.get("id")
            if row_id in index_by_id:
                merged_rows[index_by_id[row_id]] = row
            else:
                index_by_id[row_id] = len(merged_rows)
                merged_rows.append(row)
    for key in ("kbIdCounter", "docIdCounter"):
        try:
            target[key] = max(int(target.get(key) or 0), int(source.get(key) or 0))
        except (TypeError, ValueError):
            continue


def _merge_legacy_states_unlocked(path: Path) -> None:
    try:
        canonical_mtime = path.stat().st_mtime if path.exists() else 0.0
    except OSError:
        return
    payload = _read_state_unlocked(path)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    merged_sources: list[Path] = []
    for legacy in _LEGACY_STATE_FILES:
        if legacy == path:
            continue
        try:
            if not legacy.exists() or legacy.stat().st_mtime <= canonical_mtime:
                continue
        except OSError:
            continue
        _merge_state_payloads(payload, _read_state_unlocked(legacy))
        merged_sources.append(legacy)
    if not merged_sources:
        return
    _write_state_unlocked(path, payload)
    for legacy in merged_sources:
        retired = legacy.with_name(f"{legacy.name}.migrated-{stamp}")
        try:
            os.replace(legacy, retired)
        except OSError as exc:
            logger.warning("旧状态文件 %s 重命名失败：%s", legacy, exc)
    logger.warning(
        "检测到比 %s 更新的旧路径状态文件（%s），已按 ID 合并（新者胜）并写入规范路径；旧文件已重命名为 .migrated-* 前缀，确认无误后可删除。",
        path,
        ", ".join(item.name for item in merged_sources),
    )


def _write_state_unlocked(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2)
    temp_handle = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        delete=False,
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temp_path = Path(temp_handle.name)
    try:
        with temp_handle:
            temp_handle.write(encoded)
            temp_handle.flush()
            os.fsync(temp_handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def load_state(path: Path = DEFAULT_STATE_FILE) -> dict[str, Any]:
    with _exclusive_file_lock(path):
        _merge_legacy_states_unlocked(path)
        return _read_state_unlocked(path)


def save_state(payload: dict[str, Any], path: Path = DEFAULT_STATE_FILE) -> None:
    if not isinstance(payload, dict):
        raise TypeError("management state payload must be a dict")
    with _exclusive_file_lock(path):
        _write_state_unlocked(path, payload)


def update_state(
    mutator: Callable[[dict[str, Any]], bool | None],
    path: Path = DEFAULT_STATE_FILE,
) -> bool:
    """Run read-modify-write under one file lock.

    The mutator receives the current payload and returns True when it changed
    state. Returning None is treated as changed for convenience.
    """
    with _exclusive_file_lock(path):
        _merge_legacy_states_unlocked(path)
        payload = _read_state_unlocked(path)
        changed = mutator(payload)
        if changed is False:
            return False
        _write_state_unlocked(path, payload)
        return True


def update_document_state(
    storage_file_name: str,
    status: str,
    chunk_count: int | None = None,
    summary: str = "",
    path: Path = DEFAULT_STATE_FILE,
) -> bool:
    storage_key = str(storage_file_name or "").strip()
    if not storage_key:
        return False

    def mutate(payload: dict[str, Any]) -> bool:
        for document in payload.get("documents", []) or []:
            if document.get("storageFileName") != storage_key:
                continue
            document["status"] = status
            if chunk_count is not None:
                document["chunkCount"] = int(chunk_count)
            if summary:
                document["summary"] = summary
            return True
        return False

    return update_state(mutate, path)


class StateStore:
    """Management state persistence with atomic writes and a file lock."""

    def __init__(self, path: Path = DEFAULT_STATE_FILE):
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> dict[str, Any]:
        return load_state(self._path)

    def save(self, payload: dict[str, Any]) -> None:
        save_state(payload, self._path)

    def update(self, mutator: Callable[[dict[str, Any]], bool | None]) -> bool:
        return update_state(mutator, self._path)

    def update_document(
        self,
        storage_file_name: str,
        status: str,
        chunk_count: int | None = None,
        summary: str = "",
    ) -> bool:
        return update_document_state(
            storage_file_name,
            status,
            chunk_count=chunk_count,
            summary=summary,
            path=self._path,
        )
