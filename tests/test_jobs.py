"""Background job registry and cooperative-stop plumbing."""

import threading
import time

import pytest

from misterdev.agent import ProjectOrchestrator
from misterdev.core.execution.jobs import JobRegistry


def _wait(reg: JobRegistry, run_id: str, want: str, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = reg.status(run_id)
        if state and state["status"] == want:
            return
        time.sleep(0.01)
    raise AssertionError(f"job {run_id} never reached {want!r}: {reg.status(run_id)}")


def test_start_runs_target_and_captures_result():
    reg = JobRegistry()
    run_id = reg.start("build", "/p", lambda: "OK-REPORT")
    _wait(reg, run_id, "succeeded")
    state = reg.status(run_id)
    assert state["kind"] == "build"
    assert state["result"] == "OK-REPORT"
    assert state["error"] is None
    assert state["ended_at"] is not None


def test_exception_marks_job_failed_not_lost():
    reg = JobRegistry()

    def boom() -> str:
        raise ValueError("boom")

    run_id = reg.start("build", "/p", boom)
    _wait(reg, run_id, "failed")
    assert reg.status(run_id)["error"] == "boom"


def test_one_job_per_project_is_refused():
    reg = JobRegistry()
    hold = threading.Event()
    run_id = reg.start("build", "/p", lambda: hold.wait(3) or "x")
    assert reg.status(run_id)["status"] == "running"
    with pytest.raises(RuntimeError):
        reg.start("run", "/p", lambda: "y")
    # A different project is allowed to run concurrently.
    other = reg.start("run", "/q", lambda: "z")
    _wait(reg, other, "succeeded")
    hold.set()
    _wait(reg, run_id, "succeeded")


def test_stop_signals_hook_and_marks_stopped():
    reg = JobRegistry()
    release = threading.Event()
    stopped = {"seen": False}

    def target() -> str:
        release.wait(3)
        return "partial" if stopped["seen"] else "full"

    def stop_hook() -> None:
        stopped["seen"] = True
        release.set()

    run_id = reg.start("build", "/p", target, stop_hook=stop_hook)
    assert reg.status(run_id)["status"] == "running"
    assert reg.stop(run_id) is True
    _wait(reg, run_id, "stopped")
    assert stopped["seen"] is True
    # A stop-labelled result, so the reader isn't misled by the budget mechanism.
    assert reg.status(run_id)["result"].startswith("Stopped by request")


def test_stop_unknown_or_finished_returns_false():
    reg = JobRegistry()
    assert reg.stop("nope") is False
    run_id = reg.start("build", "/p", lambda: "done")
    _wait(reg, run_id, "succeeded")
    assert reg.stop(run_id) is False


def test_stop_on_job_with_no_hook_is_an_honest_no_op():
    """A job kind with nothing to interrupt (e.g. evolve) must not be flagged
    stop_requested: doing so would mislead _runner into relabelling a normal,
    successful completion as 'stopped (partial)' once the target finishes."""
    reg = JobRegistry()
    release = threading.Event()

    def target() -> str:
        release.wait(3)
        return "REAL RESULT"

    run_id = reg.start("evolve", "/p", target)
    assert reg.status(run_id)["status"] == "running"
    assert reg.stop(run_id) is False
    assert reg.status(run_id)["stop_requested"] is False
    release.set()
    _wait(reg, run_id, "succeeded")
    assert reg.status(run_id)["result"] == "REAL RESULT"


def test_list_jobs_reports_all():
    reg = JobRegistry()
    a = reg.start("build", "/a", lambda: "1")
    b = reg.start("run", "/b", lambda: "2")
    _wait(reg, a, "succeeded")
    _wait(reg, b, "succeeded")
    ids = {j["run_id"] for j in reg.list_jobs()}
    assert {a, b} <= ids


def test_status_unknown_run_id_is_none():
    assert JobRegistry().status("missing") is None


def test_finished_jobs_are_evicted_beyond_cap():
    reg = JobRegistry(max_finished=3)
    ids = []
    for i in range(6):
        run_id = reg.start("build", f"/p{i}", lambda: "done")
        _wait(reg, run_id, "succeeded")
        ids.append(run_id)
    # Only the most recent 3 finished jobs are retained; the oldest are gone.
    kept = {j["run_id"] for j in reg.list_jobs()}
    assert len(kept) == 3
    assert kept == set(ids[-3:])
    assert reg.status(ids[0]) is None


def test_running_jobs_are_never_evicted():
    reg = JobRegistry(max_finished=1)
    hold = threading.Event()
    long_running = reg.start("build", "/long", lambda: hold.wait(3) or "x")
    # Churn several finished jobs past the cap while the long one runs.
    for i in range(4):
        done = reg.start("build", f"/p{i}", lambda: "done")
        _wait(reg, done, "succeeded")
    assert reg.status(long_running)["status"] == "running"
    hold.set()
    _wait(reg, long_running, "succeeded")


def test_request_stop_trips_active_client_budget():
    orch = ProjectOrchestrator()

    class _Client:
        _budget = 5.0

    client = _Client()
    orch._active_client = client
    orch.request_stop()
    assert orch._stop_requested is True
    assert client._budget == 0.0


def test_request_stop_without_client_still_sets_flag():
    orch = ProjectOrchestrator()
    orch.request_stop()
    assert orch._stop_requested is True
