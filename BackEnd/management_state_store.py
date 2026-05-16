# management_state_store.py — shared management_state.json persistence
from __future__ import annotations

import contextlib
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_STATE_FILE = BASE_DIR / "management_state.json"

_process_lock = threading.RLock()


@contextlib.contextmanager
def _exclusive_file_lock(path: Path):
    """Best-effort cross-process lock for Windows and POSIX hosts."""
    lock_path = path.with_name(f"{path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        with _process_lock:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                try:
                    yield
                finally:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _read_state_unlocked(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


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
