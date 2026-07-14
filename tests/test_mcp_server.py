"""misterdev exposed as an MCP server — thin adapter over ProjectOrchestrator."""

import asyncio

from misterdev import mcp_server


class _FakeOrch:
    last_build_succeeded = True
    calls: dict = {}

    def build(self, path, args, reference_dir=None):
        _FakeOrch.calls["build"] = (path, args, reference_dir)
        return "REPORT-BODY"

    def scan_directory(self, directory):
        _FakeOrch.calls["scan"] = directory

    def list_projects(self):
        return {"projects": ["a", "b"]}

    def get_project_status(self, path):
        return {"path": path, "tasks": []}

    def run_project(self, path, dry_run=False):
        _FakeOrch.calls["run_project"] = (path, dry_run)

    def run_task(self, path, task_id):
        _FakeOrch.calls["run_task"] = (path, task_id)


def _patch(monkeypatch):
    _FakeOrch.calls = {}
    monkeypatch.setattr(mcp_server, "ProjectOrchestrator", _FakeOrch)


def test_all_expected_tools_registered():
    names = {t.name for t in asyncio.run(mcp_server.mcp.list_tools())}
    assert {"scan", "list_projects", "status", "build", "run"} <= names


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
