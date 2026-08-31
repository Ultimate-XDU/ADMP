import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from utils.state_store import load_state


BASE_DIR = Path(__file__).resolve().parent


def ready_count() -> int:
    payload = load_state()
    return sum(
        1
        for document in payload.get("documents", [])
        if document.get("status") in {"READY", "FAILED", "INDEXING"} and document.get("storageFileName")
    )


def main() -> int:
    batch = 0
    while True:
        remaining = ready_count()
        print(f"\n========== READY remaining: {remaining} ==========", flush=True)
        if remaining <= 0:
            return 0
        batch += 1
        print(f"========== Batch {batch}: indexing next 5 documents ==========", flush=True)
        result = subprocess.run(
            [
                sys.executable,
                "index_caac_imported_documents.py",
                "--ready",
                "--start",
                "1",
                "--limit",
                "5",
                "--sleep",
                "2",
            ],
            cwd=str(BASE_DIR),
        )
        if result.returncode != 0:
            print(f"Batch {batch} failed with exit code {result.returncode}", flush=True)
            return result.returncode
        time.sleep(5)


if __name__ == "__main__":
    raise SystemExit(main())
