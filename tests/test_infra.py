"""Infra-vs-code failure classification and the self-healing test rerun."""

from pathlib import Path
from types import SimpleNamespace

from misterdev.agent_helpers import worktree_setup_command
from misterdev.core.execution.infra import infra_failure
from misterdev.task_executors.markdown_plan_executor import MarkdownPlanExecutor


def test_infra_failure_flags_environment_faults():
    for out in (
        "Command timed out after 120s: pnpm --filter @countless/server typecheck",
        "Error: Cannot find module 'hono'",
        "sh: tsc: command not found",
        "This is not the tsc command you are looking for",
        "ENOSPC: no space left on device",
        "FATAL ERROR: JavaScript heap out of memory",
        "waiting for the lock on the store",
        "EMFILE: too many open files",
    ):
        assert infra_failure(out), out


def test_infra_failure_ignores_real_code_errors():
    for out in (
        "AssertionError: expected 3 got 4",
        "error TS2345: Argument of type 'string' is not assignable to 'number'",
        "Test Files  1 failed | 20 passed",
        "SyntaxError: Unexpected token )",
        "2 failed, 5 passed in 0.3s",
    ):
        assert infra_failure(out) is None, out


def test_confirm_flaky_self_heals_on_infra_even_with_reruns_zero():
    """A test gate that fails on an environment fault (a timeout) is re-run once
    even when flaky_reruns is 0 — and a clean re-run is treated as a flake, not a
    reverted code failure. A real code error is NOT re-run when reruns is 0."""
    e = MarkdownPlanExecutor()
    project = SimpleNamespace()
    calls = {"n": 0}

    def fake_run(_project, _cmd, timeout=None, cwd=None):
        calls["n"] += 1
        return (True, "ok")  # the re-run passes cleanly

    e._run_command = fake_run  # type: ignore[assignment]

    # Infra signature + reruns=0 → self-heals (re-runs, non-reproducing → flake).
    assert e._confirm_flaky(
        project, "pnpm test", "Command timed out after 180s", 5, None, 0
    )
    assert calls["n"] >= 1

    # A real code failure + reruns=0 → no re-run, treated as a genuine failure.
    calls["n"] = 0
    assert not e._confirm_flaky(
        project, "pnpm test", "AssertionError: 3 != 4", 5, None, 0
    )
    assert calls["n"] == 0


def _gate_executor(outputs, reprimes):
    """A MarkdownPlanExecutor whose gate runs return queued (ok, output) pairs and
    whose worktree re-prime is recorded, not executed."""
    e = MarkdownPlanExecutor()
    queue = list(outputs)

    def fake_run(_project, _cmd, timeout=None, cwd=None):
        return queue.pop(0)

    e._run_command = fake_run  # type: ignore[assignment]
    e._reprime_worktree_deps = lambda _project: reprimes.append(True)  # type: ignore[assignment]
    return e


def test_run_gate_reprimes_and_reruns_once_on_infra():
    """An infra-signatured gate failure re-primes the worktree and re-runs the
    gate exactly once; a green re-run makes the gate pass."""
    reprimes: list = []
    e = _gate_executor(
        [(False, "Command timed out after 120s: tsc"), (True, "ok")], reprimes
    )
    ok, out = e._run_gate(SimpleNamespace(), "npm run typecheck", 120, None)
    assert ok and out == "ok"
    assert reprimes == [True]  # re-primed exactly once


def test_run_gate_does_not_rerun_on_code_error():
    """A plain code failure (a type error) carries no infra signature, so the gate
    is returned as-is with no re-prime and no re-run."""
    reprimes: list = []
    e = _gate_executor(
        [(False, "error TS2345: Argument of type 'string' is not assignable")],
        reprimes,
    )
    ok, out = e._run_gate(SimpleNamespace(), "npm run typecheck", 120, None)
    assert not ok and "TS2345" in out
    assert reprimes == []  # never re-primed on a code failure


def test_run_gate_passes_through_green_without_reprime():
    """A gate that passes on the first run returns immediately, no re-prime."""
    reprimes: list = []
    e = _gate_executor([(True, "ok")], reprimes)
    ok, out = e._run_gate(SimpleNamespace(), "cargo build", 120, None)
    assert ok and out == "ok"
    assert reprimes == []


def test_run_gate_infra_rerun_still_red_returns_rerun_output():
    """When the re-primed re-run still fails, the gate returns the re-run's output
    (not the first), so the retry context reflects the latest state."""
    reprimes: list = []
    e = _gate_executor(
        [(False, "Cannot find module 'hono'"), (False, "error TS2304: name X")],
        reprimes,
    )
    ok, out = e._run_gate(SimpleNamespace(), "npm run typecheck", 120, None)
    assert not ok and "TS2304" in out
    assert reprimes == [True]


def test_worktree_setup_command_resolution(tmp_path: Path):
    """Explicit config wins and "" disables; otherwise the command is auto-detected
    from the project's lockfile, with a bare package.json falling back to install."""
    assert worktree_setup_command({}, tmp_path) is None
    cfg = {"orchestrator": {"worktree_setup_command": "make deps"}}
    assert worktree_setup_command(cfg, tmp_path) == "make deps"
    off = {"orchestrator": {"worktree_setup_command": ""}}
    assert worktree_setup_command(off, tmp_path) is None
    (tmp_path / "package-lock.json").write_text("{}", encoding="utf-8")
    assert worktree_setup_command({}, tmp_path) == "npm ci"
