from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

from elasticsearch import Elasticsearch
import torch

from aerospace_search_main import DOC_DIR, EMBEDDING_MODEL, ES_HOST, ES_INDEX, RERANK_MODEL, VECTOR_DIM, DualDBManager
from utils.structure import clear_structural_records
from utils.state_store import update_document_state


BASE_DIR = Path(__file__).resolve().parent
DOC_DIR_PATH = Path(DOC_DIR)


def _configure_utf8_console() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _print_stage_banner(title: str) -> None:
    print("\n" + "-" * 72)
    print(f"[{_now()}] {title}")


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _update_document_state(storage_file_name: str, status: str, chunk_count: int | None = None, summary: str = "") -> None:
    update_document_state(storage_file_name, status, chunk_count=chunk_count, summary=summary)


def _has_chinese_name(path: Path) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in path.stem)


def _select_reindex_files(
    all_files: list[Path],
    *,
    max_size_mb: float | None = None,
    limit: int | None = None,
    prefer_chinese: bool = False,
) -> tuple[list[Path], list[Path]]:
    max_size_bytes = None if max_size_mb is None else max(float(max_size_mb), 0.0) * 1024 * 1024
    eligible: list[Path] = []
    skipped: list[Path] = []
    for path in all_files:
        if max_size_bytes is not None and path.stat().st_size > max_size_bytes:
            skipped.append(path)
        else:
            eligible.append(path)

    if prefer_chinese:
        eligible.sort(key=lambda path: (not _has_chinese_name(path), path.stat().st_size, path.name))
    else:
        eligible.sort(key=lambda path: path.name)

    if limit is not None and limit > 0:
        selected = eligible[:limit]
        skipped.extend(eligible[limit:])
    else:
        selected = eligible
    return selected, skipped


def _print_es_counts() -> None:
    es = None
    try:
        es = Elasticsearch(ES_HOST)
        body = {"size": 0, "aggs": {"docs": {"terms": {"field": "doc_name", "size": 100}}}}
        response = es.search(index=ES_INDEX, body=body)
        print("\n[ES 文档计数]")
        for bucket in response.get("aggregations", {}).get("docs", {}).get("buckets", []):
            print(f"  {bucket['key']}: {bucket['doc_count']}")
    except Exception as exc:
        print(f"[WARN] 读取 ES 计数失败：{exc}")
    finally:
        if es is not None:
            try:
                es.close()
            except Exception:
                pass


def main(
    require_gpu: bool = False,
    max_size_mb: float | None = None,
    limit: int | None = None,
    prefer_chinese: bool = False,
) -> int:
    if require_gpu and not torch.cuda.is_available():
        print(f"[{_now()}] CUDA 不可用，已按 --require-gpu 要求停止重建。")
        return 2
    if require_gpu:
        os.environ["AEROSPACE_RERANK_DEVICE"] = "cuda"

    all_files = sorted(
        path
        for path in DOC_DIR_PATH.iterdir()
        if path.is_file() and path.suffix.lower() in {".pdf", ".docx"}
    )
    files, skipped_files = _select_reindex_files(
        all_files,
        max_size_mb=max_size_mb,
        limit=limit,
        prefer_chinese=prefer_chinese,
    )
    if not files:
        print(f"[{_now()}] 未找到可索引文档：{DOC_DIR_PATH}")
        return 1

    print("=" * 72)
    print(f"[{_now()}] 开始逐个重建索引")
    print(f"文档目录：{DOC_DIR_PATH}")
    print(f"文档数量：{len(files)}")
    if max_size_mb is not None:
        print(f"大文件阈值：{max_size_mb:.1f} MB")
    if limit is not None:
        print(f"本轮数量限制：{limit}")
    if prefer_chinese:
        print("选择策略：中文文件名优先，其次按文件大小")
    print(f"本轮跳过：{len(skipped_files)}")
    print("策略：逐个 add_document，不使用 build_index 的一次性全量汇总入库")
    print(f"嵌入模型：{EMBEDDING_MODEL} / dim={VECTOR_DIM}")
    print(f"重排模型：{RERANK_MODEL}")
    print("=" * 72)

    try:
        manager = DualDBManager()
    except Exception as exc:
        print(f"[{_now()}] 数据库连接失败，无法重建索引：{type(exc).__name__}: {exc}")
        print("请先确认 Docker 中的 Milvus 与 Elasticsearch 已启动，并且 19530 / 9200 端口可访问。")
        return 2

    try:
        print(f"[{_now()}] 重置 Milvus / Elasticsearch 索引，以匹配当前模型维度...")
        manager.reset_databases()
        clear_structural_records()
        print(f"[{_now()}] 索引重置完成")
        for skipped in skipped_files:
            _update_document_state(
                skipped.name,
                "READY",
                chunk_count=0,
                summary="本轮测试子集未入库",
            )
    except Exception as exc:
        print(f"[{_now()}] 索引重置失败：{type(exc).__name__}: {exc}")
        manager.close()
        return 2

    failures: list[tuple[str, str]] = []
    total_chunks = 0
    started_at = time.time()

    for index, file_path in enumerate(files, start=1):
        doc_started_at = time.time()
        print("\n" + "-" * 72)
        print(f"[{_now()}] [{index}/{len(files)}] 开始：{file_path.name}")
        _update_document_state(file_path.name, "INDEXING", summary="正在重新索引")

        try:
            result = manager.add_document(str(file_path))
            elapsed = time.time() - doc_started_at
            if result.success:
                total_chunks += int(result.chunk_count or 0)
                _update_document_state(
                    file_path.name,
                    "SYNCED",
                    chunk_count=int(result.chunk_count or 0),
                    summary=f"最近一次重建成功：{_now()}",
                )
                print(f"[{_now()}] 完成：{file_path.name}")
                print(f"  chunks: {result.chunk_count}")
                print(f"  message: {result.message}")
                print(f"  elapsed: {elapsed:.1f}s")
            else:
                _update_document_state(file_path.name, "FAILED", summary=result.message or "索引失败")
                failures.append((file_path.name, result.message or result.error_code or "索引失败"))
                print(f"[{_now()}] 失败：{file_path.name}")
                print(f"  message: {result.message}")
                print(f"  error_code: {result.error_code}")
                print(f"  elapsed: {elapsed:.1f}s")
        except KeyboardInterrupt:
            _update_document_state(file_path.name, "FAILED", summary="用户中断索引")
            print(f"\n[{_now()}] 用户中断，当前文档标记为 FAILED：{file_path.name}")
            manager.close()
            return 130
        except Exception as exc:
            elapsed = time.time() - doc_started_at
            message = f"{type(exc).__name__}: {exc}"
            _update_document_state(file_path.name, "FAILED", summary=message)
            failures.append((file_path.name, message))
            print(f"[{_now()}] 异常：{file_path.name}")
            print(f"  {message}")
            print(f"  elapsed: {elapsed:.1f}s")
            traceback.print_exc()

    total_elapsed = time.time() - started_at
    print("\n" + "=" * 72)
    print(f"[{_now()}] 逐个重建结束")
    print(f"总 chunks: {total_chunks}")
    print(f"总耗时: {total_elapsed:.1f}s")
    if failures:
        print(f"失败文档: {len(failures)}")
        for file_name, message in failures:
            print(f"  - {file_name}: {message}")
    else:
        print("失败文档: 0")
    _print_es_counts()
    print("=" * 72)
    manager.close()
    return 1 if failures else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebuild the local aerospace retrieval index document by document.")
    parser.add_argument("--require-gpu", action="store_true", help="Abort unless CUDA is available for embedding/reranking.")
    parser.add_argument("--max-size-mb", type=float, default=None, help="Only index files up to this size; larger files are marked READY and skipped.")
    parser.add_argument("--limit", type=int, default=None, help="Only index the first N selected files.")
    parser.add_argument("--prefer-chinese", action="store_true", help="Prefer Chinese-named documents when selecting a limited subset.")
    args = parser.parse_args()
    _configure_utf8_console()
    raise SystemExit(
        main(
            require_gpu=args.require_gpu,
            max_size_mb=args.max_size_mb,
            limit=args.limit,
            prefer_chinese=args.prefer_chinese,
        )
    )
