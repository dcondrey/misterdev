"""Wiring tests: the failure-probe (#4) and verifier-decomposition staging (#5)."""

from pathlib import Path
from types import SimpleNamespace

from misterdev.task_executors.markdown_plan_executor import MarkdownPlanExecutor
from misterdev.agent import ProjectOrchestrator

_CARGO_FAIL = (
    "thread 'scores_a_strike' panicked at src/lib.rs:1:1:\n"
    "assertion `left == right` failed\n  left: 1\n right: 2\n"
    "test result: FAILED. 0 passed; 1 failed"
)


# --- #4: failure probe gated into _build_error_context ---------------------


def _ctx(project, monkeypatch, probe_out="FRESH-TRACE"):
    import misterdev.core.execution.probe as probe

    monkeypatch.setattr(probe, "run_probe", lambda cwd, cmd, timeout=120: probe_out)
    ex = MarkdownPlanExecutor()
    return ex._build_error_context(
        [],
        0,
        _CARGO_FAIL,
        "classified",
        "attributed",
        project=project,
        test_command="cargo test",
        language="rust",
        cwd="/tmp",
    )


def test_probe_off_by_default_leaves_context_unchanged(monkeypatch):
    project = SimpleNamespace(config={}, path=Path("/tmp"))
    out = _ctx(project, monkeypatch)
    assert "Fresh isolated re-run" not in out
    # The exact failing assertion still leads (FailureView is unaffected).
    assert "scores_a_strike" in out


def test_probe_on_appends_fresh_isolated_rerun(monkeypatch):
    project = SimpleNamespace(
        config={"orchestrator": {"failure_probe": True}}, path=Path("/tmp")
    )
    out = _ctx(project, monkeypatch)
    assert "Fresh isolated re-run of the first failing test" in out
    assert "FRESH-TRACE" in out
    assert "cargo test scores_a_strike" in out  # isolate_command built the re-run


def test_probe_no_project_is_noop(monkeypatch):
    # No project/test_command -> no probe, no crash (build/typecheck-failure path).
    ex = MarkdownPlanExecutor()
    out = ex._build_error_context([], 0, _CARGO_FAIL, "c", "a")
    assert "Fresh isolated re-run" not in out


# --- #5: staging hint from the symbol graph --------------------------------


def _sym(name, kind, file_path):
    return SimpleNamespace(name=name, kind=kind, file_path=file_path)


def test_staging_hint_single_file_yields_ordered_stages():
    orch = ProjectOrchestrator()
    syms = {
        "a": _sym("new", "method", "src/lib.rs"),
        "b": _sym("roll", "method", "src/lib.rs"),
        "c": _sym("score", "method", "src/lib.rs"),
    }
    project = SimpleNamespace(
        topography=SimpleNamespace(graph=SimpleNamespace(symbols=syms))
    )
    hint = orch._staging_hint(project)
    assert "Suggested staging" in hint and "Stage 1" in hint


def test_staging_hint_multi_file_is_empty():
    orch = ProjectOrchestrator()
    syms = {
        "a": _sym("new", "method", "src/lib.rs"),
        "b": _sym("helper", "function", "src/util.rs"),
    }
    project = SimpleNamespace(
        topography=SimpleNamespace(graph=SimpleNamespace(symbols=syms))
    )
    assert orch._staging_hint(project) == ""


def test_staging_hint_no_topography_is_empty():
    assert ProjectOrchestrator()._staging_hint(SimpleNamespace()) == ""
