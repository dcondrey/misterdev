"""misterdev exposed as an MCP server — thin adapter over ProjectOrchestrator."""

import asyncio
import time

from misterdev import mcp_server
from misterdev.core.execution.jobs import JobRegistry


class _FakeOrch:
    last_build_succeeded = True
    calls: dict = {}

    def build(self, path, args, reference_dir=None, progress_cb=None):
        _FakeOrch.calls["build"] = (path, args, reference_dir)
        if progress_cb is not None:
            progress_cb(done=1, total=1, phase="building")
        return "REPORT-BODY"

    def scan_directory(self, directory):
        _FakeOrch.calls["scan"] = directory

    def list_projects(self):
        return {"projects": ["a", "b"]}

    def get_project_status(self, path):
        return {"path": path, "tasks": []}

    def run_project(self, path, dry_run=False, progress_cb=None):
        _FakeOrch.calls["run_project"] = (path, dry_run)
        if progress_cb is not None:
            progress_cb(done=2, total=4, phase="wave 1")

    def run_task(self, path, task_id):
        _FakeOrch.calls["run_task"] = (path, task_id)

    def request_stop(self):
        _FakeOrch.calls["request_stop"] = True

    def propose_plan(self, path, args):
        _FakeOrch.calls["propose_plan"] = (path, args)
        return {"items": [{"id": "P-001", "title": "t", "approved": False}]}

    def execute_plan(self, path, args):
        _FakeOrch.calls["execute_plan"] = (path, args)
        return "PLAN-EXEC-REPORT"


def _patch(monkeypatch):
    _FakeOrch.calls = {}
    monkeypatch.setattr(mcp_server, "ProjectOrchestrator", _FakeOrch)


def _patch_jobs(monkeypatch):
    """Give the async tools a fresh, isolated registry per test."""
    reg = JobRegistry()
    monkeypatch.setattr(mcp_server, "registry", reg)
    return reg


def _await_status(reg, run_id, want, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = reg.status(run_id)
        if state and state["status"] == want:
            return state
        time.sleep(0.01)
    raise AssertionError(f"{run_id} never reached {want}: {reg.status(run_id)}")


def test_all_expected_tools_registered():
    names = {t.name for t in asyncio.run(mcp_server.mcp.list_tools())}
    assert {
        "scan",
        "list_projects",
        "status",
        "build",
        "run",
        "report",
        "build_async",
        "run_async",
        "job_status",
        "stop_job",
        "list_jobs",
        "propose_plan",
        "get_plan",
        "approve_plan",
        "execute_plan",
    } <= names


def test_tool_definitions_are_well_documented():
    # Guards the Glama TDQS score: every tool needs a substantive description, a
    # title + behavioral annotations, and every parameter needs a description.
    tools = {t.name: t for t in asyncio.run(mcp_server.mcp.list_tools())}
    for t in tools.values():
        assert t.description and len(t.description) > 60, t.name
        assert t.annotations is not None and t.annotations.title, t.name
        for pname, prop in t.inputSchema.get("properties", {}).items():
            assert prop.get("description"), f"{t.name}.{pname} needs a description"
    # Behavioral transparency: honest read-only vs. destructive hints.
    assert tools["status"].annotations.readOnlyHint is True
    assert tools["list_projects"].annotations.readOnlyHint is True
    assert tools["build"].annotations.destructiveHint is True
    assert tools["run"].annotations.destructiveHint is True


def test_build_routes_and_composes_flags(monkeypatch):
    _patch(monkeypatch)
    out = mcp_server.build(
        "/repo", "add rate limiting", budget=5.0, parallel=True, max_tasks=3
    )
    path, args, reference_dir = _FakeOrch.calls["build"]
    assert path == "/repo"
    assert "add rate limiting" in args
    assert "--budget 5.0" in args
    assert "--parallel" in args
    assert "--max-tasks 3" in args
    assert reference_dir is None
    assert "succeeded" in out and "REPORT-BODY" in out


def test_build_dry_run_flag(monkeypatch):
    _patch(monkeypatch)
    mcp_server.build("/repo", "fix tests", dry_run=True)
    _, args, _ = _FakeOrch.calls["build"]
    assert "--dry-run" in args


def test_build_forwards_reference_dir(monkeypatch):
    _patch(monkeypatch)
    mcp_server.build("/repo", "port it", reference_dir="/donor/impl")
    _, _, reference_dir = _FakeOrch.calls["build"]
    assert reference_dir == "/donor/impl"


def test_build_reports_failure(monkeypatch):
    _patch(monkeypatch)
    monkeypatch.setattr(_FakeOrch, "last_build_succeeded", False)
    out = mcp_server.build("/repo", "x")
    assert "did not fully succeed" in out


def test_read_only_tools_route(monkeypatch):
    _patch(monkeypatch)
    assert mcp_server.list_projects() == {"projects": ["a", "b"]}
    assert mcp_server.status("/repo") == {"path": "/repo", "tasks": []}
    assert "/dir" in mcp_server.scan("/dir")
    assert _FakeOrch.calls["scan"] == "/dir"


def test_run_tool_routes(monkeypatch):
    _patch(monkeypatch)
    mcp_server.run("/repo", dry_run=True)
    assert _FakeOrch.calls["run_project"] == ("/repo", True)
    mcp_server.run("/repo", task_id="T-1")
    assert _FakeOrch.calls["run_task"] == ("/repo", "T-1")


def test_report_rejects_non_directory():
    out = mcp_server.report("/no/such/dir/xyz")
    assert "error" in out


def test_report_reads_saved_artifacts(tmp_path):
    reports = tmp_path / ".orchestrator" / "reports"
    reports.mkdir(parents=True)
    (reports / "report_20240101_000000.json").write_text(
        '{"mode": "debug", "project": "p", "completed": ["T-1"], "failed": []}'
    )
    out = mcp_server.report(str(tmp_path))
    assert out["latest_report"]["mode"] == "debug"
    assert out["latest_report"]["completed"] == ["T-1"]
    assert "audit" in out and "models" in out


def test_report_on_unbuilt_project_returns_null_report(tmp_path):
    out = mcp_server.report(str(tmp_path))
    assert out["latest_report"] is None
    assert "audit" in out and "models" in out


def test_build_async_starts_job_and_forwards_args(monkeypatch):
    _patch(monkeypatch)
    reg = _patch_jobs(monkeypatch)
    out = mcp_server.build_async("/repo", "port it", budget=3.0, reference_dir="/ref")
    assert out["status"] == "running"
    run_id = out["run_id"]
    _await_status(reg, run_id, "succeeded")
    state = reg.status(run_id)
    assert state["kind"] == "build"
    assert state["result"] == "REPORT-BODY"
    path, args, reference_dir = _FakeOrch.calls["build"]
    assert path == "/repo" and reference_dir == "/ref"
    assert "--budget 3.0" in args


def test_run_async_starts_job(monkeypatch):
    _patch(monkeypatch)
    reg = _patch_jobs(monkeypatch)
    out = mcp_server.run_async("/repo")
    _await_status(reg, out["run_id"], "succeeded")
    assert _FakeOrch.calls["run_project"] == ("/repo", False)


def test_build_async_refuses_second_job_same_project(monkeypatch):
    _patch(monkeypatch)
    reg = _patch_jobs(monkeypatch)
    hold = {"go": False}

    def _slow(self, path, args, reference_dir=None, progress_cb=None):
        while not hold["go"]:
            time.sleep(0.005)
        return "REPORT-BODY"

    monkeypatch.setattr(_FakeOrch, "build", _slow)
    first = mcp_server.build_async("/repo", "x")
    assert "run_id" in first
    second = mcp_server.build_async("/repo", "y")
    assert "error" in second
    hold["go"] = True
    _await_status(reg, first["run_id"], "succeeded")


def test_job_status_and_list_and_stop(monkeypatch):
    _patch(monkeypatch)
    reg = _patch_jobs(monkeypatch)
    out = mcp_server.build_async("/repo", "x")
    run_id = out["run_id"]
    _await_status(reg, run_id, "succeeded")

    assert mcp_server.job_status(run_id)["status"] == "succeeded"
    assert mcp_server.job_status("nope") == {"error": "unknown run_id: nope"}
    assert run_id in {j["run_id"] for j in mcp_server.list_jobs()["jobs"]}
    # Stopping an already-finished job is a harmless no-op.
    assert mcp_server.stop_job(run_id) == {"run_id": run_id, "stopping": False}


def test_propose_plan_routes(monkeypatch):
    _patch(monkeypatch)
    out = mcp_server.propose_plan("/repo", budget=4.0)
    path, args = _FakeOrch.calls["propose_plan"]
    assert path == "/repo" and "--budget 4.0" in args
    assert out["items"][0]["id"] == "P-001"


def test_execute_plan_routes(monkeypatch):
    _patch(monkeypatch)
    out = mcp_server.execute_plan("/repo", budget=6.0)
    path, args = _FakeOrch.calls["execute_plan"]
    assert path == "/repo" and "--budget 6.0" in args
    assert out == "PLAN-EXEC-REPORT"


def test_get_and_approve_plan_use_the_store(tmp_path):
    from misterdev.core.planning import plan_store

    plan_store.save_plan(
        tmp_path,
        [{"title": "one"}, {"title": "two"}],
    )
    # get_plan reads the persisted proposals.
    got = mcp_server.get_plan(str(tmp_path))
    assert {it["id"] for it in got["items"]} == {"P-001", "P-002"}
    # approve_plan flips the flag and persists.
    approved = mcp_server.approve_plan(str(tmp_path), approve_ids=["P-002"])
    flags = {it["id"]: it["approved"] for it in approved["items"]}
    assert flags == {"P-001": False, "P-002": True}


def test_approve_plan_without_plan_errors(tmp_path):
    out = mcp_server.approve_plan(str(tmp_path), approve_all=True)
    assert "error" in out


def test_get_plan_empty_when_none(tmp_path):
    assert mcp_server.get_plan(str(tmp_path)) == {"items": []}


def test_run_async_reports_task_progress(monkeypatch):
    _patch(monkeypatch)
    reg = _patch_jobs(monkeypatch)
    run_id = mcp_server.run_async("/repo")["run_id"]
    st = _await_status(reg, run_id, "succeeded")
    # Progress reported by the run flowed through the reporter into job status.
    assert st["tasks_done"] == 2 and st["tasks_total"] == 4
    assert st["phase"] == "wave 1"


def test_build_async_reports_task_progress(monkeypatch):
    _patch(monkeypatch)
    reg = _patch_jobs(monkeypatch)
    run_id = mcp_server.build_async("/repo", "do it", budget=1.0)["run_id"]
    st = _await_status(reg, run_id, "succeeded")
    assert (
        st["tasks_done"] == 1 and st["tasks_total"] == 1 and st["phase"] == "building"
    )
