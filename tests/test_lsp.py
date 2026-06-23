import time
import tempfile
from pathlib import Path

import my_project_orchestrator.core.lsp as lsp
from my_project_orchestrator.core.lsp import (
    collect_diagnostics,
    find_source_files,
    _to_errors,
)


def test_collect_diagnostics_returns_collected_errors(monkeypatch):
    sentinel = [{"file": "a.py", "line": 3, "message": "undefined name"}]
    monkeypatch.setattr(lsp, "_collect", lambda root, code_lang, files: sentinel)
    assert collect_diagnostics(Path("."), "python", ["a.py"]) == sentinel


def test_collect_diagnostics_times_out_to_none(monkeypatch):
    def _slow(root, code_lang, files):
        time.sleep(5)
        return []

    monkeypatch.setattr(lsp, "_collect", _slow)
    # Hard timeout below the work time -> skip (None), never block.
    assert collect_diagnostics(Path("."), "python", ["a.py"], timeout=0.2) is None


def test_collect_diagnostics_swallows_server_errors(monkeypatch):
    def _boom(root, code_lang, files):
        raise RuntimeError("server crashed")

    monkeypatch.setattr(lsp, "_collect", _boom)
    assert collect_diagnostics(Path("."), "python", ["a.py"]) is None


def test_collect_diagnostics_unsupported_language_returns_none():
    # c/cpp/swift have no multilspy server -> skip without starting anything.
    assert collect_diagnostics(Path("."), "cpp", ["a.cpp"]) is None
    assert collect_diagnostics(Path("."), "swift", ["a.swift"]) is None
    assert collect_diagnostics(Path("."), "", ["a.x"]) is None


def test_collect_diagnostics_no_files_returns_none():
    assert collect_diagnostics(Path("."), "python", []) is None


def test_find_source_files_filters_by_language_and_skips_vendored():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "a.py").write_text("x = 1")
        (td / "b.rs").write_text("fn m() {}")
        (td / "node_modules").mkdir()
        (td / "node_modules" / "c.py").write_text("y = 2")
        py = find_source_files(td, "python")
        assert "a.py" in py
        assert "node_modules/c.py" not in py  # vendored dir skipped
        assert "b.rs" not in py  # wrong language
        assert find_source_files(td, "cpp") == []  # unsupported -> empty


def test_find_source_files_respects_cap():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for i in range(10):
            (td / f"f{i}.py").write_text("x = 1")
        assert len(find_source_files(td, "python", cap=4)) == 4


def test_to_errors_keeps_only_error_severity():
    captured = [
        {
            "uri": "file:///proj/a.py",
            "diagnostics": [
                {
                    "severity": 1,
                    "range": {"start": {"line": 4}},
                    "message": "undefined",
                },
                {
                    "severity": 2,
                    "range": {"start": {"line": 9}},
                    "message": "warn only",
                },
            ],
        }
    ]
    errors = _to_errors(captured)
    assert len(errors) == 1
    assert errors[0]["file"] == "/proj/a.py"
    assert errors[0]["line"] == 5  # 0-based LSP line + 1
    assert errors[0]["message"] == "undefined"
