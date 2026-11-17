"""Real LightRAG smoke and a disposable delete-after-restart check.

The default command extracts the first two original units of the short icing
advisory into GRAPH_DATA_DIR. --check reloads that store in a new process.
--delete-cycle extracts one original icing paragraph into the current data
directory. --delete-restart opens that directory again and deletes the fixture.
"""

from __future__ import annotations

import json
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from knowledge.graph_rag.corpus import chunks_for
from knowledge.graph_rag.service import get_service


def _print_status(label: str) -> dict:
    status = get_service().status_payload()
    print(label, json.dumps(status, ensure_ascii=False), flush=True)
    return status


def smoke() -> int:
    status = get_service().build_blocking(["caac-icing-sc25-040"], scope="smoke")
    print("SMOKE", json.dumps(status, ensure_ascii=False), flush=True)
    return 0 if status.get("nodes", 0) or status.get("chunks", 0) else 1


def check() -> int:
    status = _print_status("RELOAD")
    return 0 if status.get("chunks", 0) else 1


def delete_cycle() -> int:
    rows = chunks_for("caac-icing-sc25-040", "smoke")
    if not rows:
        print("DELETE_CYCLE no source text", flush=True)
        return 1
    original = dict(rows[0])
    original["chunk_id"] = "c1"
    service = get_service()
    service.store_managed_blocks("disposable-graph-fixture.txt", [original])
    doc_id = "managed:disposable-graph-fixture.txt"
    status = service.build_blocking([doc_id], scope="all")
    print("DELETE_CYCLE", json.dumps(status, ensure_ascii=False), flush=True)
    return 0 if status.get("chunks", 0) else 1


def delete_restart() -> int:
    before = _print_status("BEFORE_DELETE")
    result = get_service().delete_document("disposable-graph-fixture.txt")
    after = _print_status("AFTER_DELETE")
    print("DELETE_RESULT", json.dumps(result, ensure_ascii=False), flush=True)
    removed = after.get("chunks", 0) == 0 and after.get("nodes", 0) <= before.get("nodes", 0)
    return 0 if removed else 1


def main() -> int:
    if "--check" in sys.argv:
        return check()
    if "--delete-cycle" in sys.argv:
        return delete_cycle()
    if "--delete-restart" in sys.argv:
        return delete_restart()
    return smoke()


if __name__ == "__main__":
    raise SystemExit(main())
