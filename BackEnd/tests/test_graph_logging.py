import json

from knowledge.graph_rag.runtime import _log_raw


def test_raw_log_disabled_leaves_path_unchanged(tmp_path, monkeypatch):
    target = tmp_path / "raw-llm.jsonl"
    target.write_text("kept\n", encoding="utf-8")
    monkeypatch.setattr("knowledge.graph_rag.runtime.data_dir", lambda: tmp_path)
    monkeypatch.setenv("GRAPH_LOG_RAW_LLM", "0")

    _log_raw({"question": "held-out", "prompt": "secret-prompt", "answer": "secret-answer"})

    assert target.read_text(encoding="utf-8") == "kept\n"
    assert not any("secret-prompt" in line for line in target.read_text(encoding="utf-8").splitlines())


def test_raw_log_default_appends(tmp_path, monkeypatch):
    monkeypatch.setattr("knowledge.graph_rag.runtime.data_dir", lambda: tmp_path)
    monkeypatch.delenv("GRAPH_LOG_RAW_LLM", raising=False)

    _log_raw({"kind": "chat"})

    lines = (tmp_path / "raw-llm.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["kind"] == "chat"
    assert "ts" in record
