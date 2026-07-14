"""Infra-vs-code failure classification and the self-healing test rerun."""

from types import SimpleNamespace

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
