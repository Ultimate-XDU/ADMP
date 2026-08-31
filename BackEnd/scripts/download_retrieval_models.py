import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from __future__ import annotations

import os
import sys
import time

from huggingface_hub import snapshot_download


MODELS = [
    (
        "BAAI/bge-m3",
        [
            "pytorch_model.bin",
            "config.json",
            "config_sentence_transformers.json",
            "modules.json",
            "sentence_bert_config.json",
            "sentencepiece.bpe.model",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "1_Pooling/config.json",
        ],
    ),
    (
        "BAAI/bge-reranker-v2-m3",
        [
            "model.safetensors",
            "config.json",
            "sentencepiece.bpe.model",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ],
    ),
]


def configure() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def main() -> int:
    configure()
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] HF_ENDPOINT={endpoint}", flush=True)

    for repo_id, allow_patterns in MODELS:
        started_at = time.time()
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] DOWNLOAD start {repo_id}", flush=True)
        path = snapshot_download(
            repo_id=repo_id,
            allow_patterns=allow_patterns,
            local_files_only=False,
            force_download=False,
        )
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] DOWNLOAD done {repo_id}: {path} "
            f"elapsed={time.time() - started_at:.1f}s",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
