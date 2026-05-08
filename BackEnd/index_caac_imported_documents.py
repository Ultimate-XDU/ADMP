from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

from aerospace_search_main import DOC_DIR, DualDBManager


BASE_DIR = Path(__file__).resolve().parent
REPORT_FILE = BASE_DIR / "caac_import_report.json"
STATE_FILE = BASE_DIR / "management_state.json"
DOC_DIR_PATH = Path(DOC_DIR)


def _configure_console() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _load_json(path: Path, fallback):
    if not path.exists():
        return fallback
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return fallback


def _save_json(path: Path, payload) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def _update_document_state(storage_file_name: str, status: str, chunk_count: int | None = None, summary: str = "") -> None:
    payload = _load_json(STATE_FILE, {})
    changed = False
    for document in payload.get("documents", []):
        if document.get("storageFileName") != storage_file_name:
            continue
        document["status"] = status
        if chunk_count is not None:
            document["chunkCount"] = int(chunk_count)
        if summary:
            document["summary"] = summary
        changed = True
        break
    if changed:
        _save_json(STATE_FILE, payload)


def _selected_report_items(start: int, limit: int) -> list[dict]:
    report = [item for item in _load_json(REPORT_FILE, []) if item.get("file") and not item.get("error")]
    start_index = max(start - 1, 0)
    end_index = start_index + max(limit, 0) if limit else None
    return report[start_index:end_index]


def _ready_state_items(start: int, limit: int) -> list[dict]:
    payload = _load_json(STATE_FILE, {})
    documents = [
        document
        for document in payload.get("documents", [])
        if document.get("status") in {"READY", "FAILED", "INDEXING"} and document.get("storageFileName")
    ]
    documents.sort(key=lambda item: int(item.get("id") or 0))
    start_index = max(start - 1, 0)
    end_index = start_index + max(limit, 0) if limit else None
    return [{"file": document["storageFileName"]} for document in documents[start_index:end_index]]


def main() -> int:
    _configure_console()
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1, help="1-based item number in caac_import_report.json")
    parser.add_argument("--limit", type=int, default=5, help="documents to index in this run")
    parser.add_argument("--sleep", type=float, default=2.0, help="seconds to pause between documents")
    parser.add_argument("--ready", action="store_true", help="index READY/FAILED/INDEXING documents from management_state.json")
    args = parser.parse_args()

    items = _ready_state_items(args.start, args.limit) if args.ready else _selected_report_items(args.start, args.limit)
    if not items:
        print(f"[{_now()}] no report items selected")
        return 0

    print(f"[{_now()}] indexing CAAC imported docs start={args.start} limit={args.limit} count={len(items)}")
    manager = DualDBManager()
    failures: list[tuple[str, str]] = []

    for offset, item in enumerate(items, start=args.start):
        file_name = item["file"]
        file_path = DOC_DIR_PATH / file_name
        print("\n" + "-" * 72)
        print(f"[{_now()}] [{offset}] {file_name}")
        if not file_path.exists():
            message = f"missing file: {file_path}"
            print(f"  FAILED: {message}")
            failures.append((file_name, message))
            _update_document_state(file_name, "FAILED", summary=message)
            continue

        _update_document_state(file_name, "INDEXING", summary=f"CAAC分批索引中：{_now()}")
        started = time.time()
        try:
            result = manager.add_document(str(file_path))
            elapsed = time.time() - started
            if result.success:
                chunk_count = int(result.chunk_count or 0)
                _update_document_state(
                    file_name,
                    "SYNCED",
                    chunk_count=chunk_count,
                    summary=f"CAAC分批索引成功：{_now()}",
                )
                print(f"  OK chunks={chunk_count} elapsed={elapsed:.1f}s")
            else:
                message = result.message or result.error_code or "index failed"
                failures.append((file_name, message))
                _update_document_state(file_name, "FAILED", summary=message)
                print(f"  FAILED: {message} elapsed={elapsed:.1f}s")
        except Exception as exc:
            elapsed = time.time() - started
            message = f"{type(exc).__name__}: {exc}"
            failures.append((file_name, message))
            _update_document_state(file_name, "FAILED", summary=message)
            print(f"  ERROR: {message} elapsed={elapsed:.1f}s")
            traceback.print_exc()

        if args.sleep > 0:
            time.sleep(args.sleep)

    print("\n" + "=" * 72)
    print(f"[{_now()}] batch done failures={len(failures)}")
    for file_name, message in failures:
        print(f"  - {file_name}: {message}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
