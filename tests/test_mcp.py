"""MCP tool-host substrate tests.

The live path is exercised against a REAL in-process stdio MCP server (a tiny
FastMCP server exposing `echo` and `add`, launched as a subprocess via the mcp
SDK) — fully offline, no network, no skip. The rest cover graceful degradation
(no config, missing command, hang) and the awareness-injection integration.
"""

import sys
import textwrap
import time
from pathlib import Path

import pytest

import my_project_orchestrator.core.mcp as mcp_mod
from my_project_orchestrator.core.mcp import MCPManager, MCPTool

# A minimal real MCP server: two tools, stdio transport, started as a subprocess
# by the mcp SDK's stdio_client. Written to a temp file and run via the test
# interpreter so the live connect/list/call path runs for real in the suite.
_SERVER_SRC = textwrap.dedent(
    """
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("test-tools")

    @server.tool()
    def echo(text: str) -> str:
        \"\"\"Echo the input text back.\"\"\"
        return text

    @server.tool()
    def add(a: int, b: int) -> int:
        \"\"\"Add two integers.\"\"\"
        return a + b

    if __name__ == "__main__":
        server.run()
    """
)


@pytest.fixture
def server_path(tmp_path) -> Path:
    p = tmp_path / "tiny_mcp_server.py"
    p.write_text(_SERVER_SRC)
    return p


def _stdio_server(server_path: Path) -> dict:
    return {
        "name": "tools",
        "command": sys.executable,
        "args": [str(server_path)],
        "transport": "stdio",
    }


# --- live stdio path (real subprocess server, runs in the suite) -----------


def test_live_discovers_tools(server_path):
    mgr = MCPManager([_stdio_server(server_path)])
    names = {t.name for t in mgr.tools}
    assert {"echo", "add"} <= names
    add_tool = next(t for t in mgr.tools if t.name == "add")
    assert add_tool.server == "tools"
    assert add_tool.qualified_name == "tools.add"
    assert add_tool.description  # FastMCP carries the docstring


def test_live_call_tool_returns_correct_result(server_path):
    mgr = MCPManager([_stdio_server(server_path)])
    assert mgr.call_tool("tools", "add", {"a": 2, "b": 40}) == "42"
    assert mgr.call_tool("tools", "echo", {"text": "pong"}) == "pong"


def test_live_tools_are_cached(server_path):
    mgr = MCPManager([_stdio_server(server_path)])
    first = mgr.tools
    assert first is mgr.tools  # second access returns the same cached list


# --- graceful degradation ---------------------------------------------------


def test_no_servers_is_empty_and_disabled():
    mgr = MCPManager([])
    assert not mgr.enabled
    assert mgr.tools == []
    assert mgr.describe_tools() == ""


def test_no_servers_when_config_is_not_a_list():
    assert MCPManager(None).tools == []
    assert MCPManager({"name": "x"}).tools == []


def test_missing_command_server_is_absent_others_survive(server_path):
    # A server whose command does not exist must be absent without raising; a
    # healthy server in the same manager still discovers its tools.
    mgr = MCPManager(
        [
            {"name": "broken", "command": "/no/such/binary-xyz", "args": []},
            _stdio_server(server_path),
        ],
        connect_timeout=15,
    )
    tools = mgr.tools
    servers = {t.server for t in tools}
    assert "tools" in servers  # healthy one survived
    assert "broken" not in servers  # broken one simply absent


def test_entry_without_name_or_command_is_dropped(server_path):
    mgr = MCPManager(
        [
            {"command": sys.executable},  # no name
            {"name": "nocmd"},  # no command
            _stdio_server(server_path),
        ]
    )
    assert [s["name"] for s in mgr.servers] == ["tools"]


def test_non_stdio_transport_is_skipped():
    mgr = MCPManager([{"name": "remote", "command": "x", "transport": "sse"}])
    assert not mgr.enabled


def test_duplicate_server_name_keeps_first(server_path):
    mgr = MCPManager([_stdio_server(server_path), _stdio_server(server_path)])
    assert len(mgr.servers) == 1


def test_call_tool_unknown_server_returns_none():
    mgr = MCPManager([{"name": "tools", "command": "x"}])
    assert mgr.call_tool("nope", "add", {}) is None


def test_hanging_discovery_is_bounded(monkeypatch, server_path):
    # A server whose startup hangs must be abandoned by the hard timeout; the
    # manager returns what it has (nothing) without blocking the build.
    def _hang(server):
        time.sleep(3600)

    monkeypatch.setattr(mcp_mod, "_list_tools", _hang)
    mgr = MCPManager([_stdio_server(server_path)], connect_timeout=0.3)
    start = time.monotonic()
    tools = mgr.tools
    assert time.monotonic() - start < 10
    assert tools == []


def test_call_tool_error_swallowed(monkeypatch):
    def _boom(server, name, arguments):
        raise RuntimeError("server crashed")

    monkeypatch.setattr(mcp_mod, "_call_tool", _boom)
    mgr = MCPManager([{"name": "tools", "command": "x"}])
    assert mgr.call_tool("tools", "add", {}) is None


# --- describe_tools (awareness text) ----------------------------------------


def test_describe_tools_lists_qualified_names_and_caps(monkeypatch):
    many = [MCPTool("s", f"t{i}", f"desc {i}") for i in range(30)]
    mgr = MCPManager([{"name": "s", "command": "x"}])
    monkeypatch.setattr(type(mgr), "tools", property(lambda self: many))
    text = mgr.describe_tools(cap=5)
    assert "- s.t0: desc 0" in text
    assert "and 25 more" in text
    assert sum(1 for ln in text.splitlines() if ln.startswith("- s.t")) == 5


# --- awareness injection in the executor ------------------------------------


def _executor():
    from my_project_orchestrator.task_executors.markdown_plan_executor import (
        MarkdownPlanExecutor,
    )

    return MarkdownPlanExecutor()


class _FakeMCP:
    def describe_tools(self, cap: int = 25) -> str:
        return "- tools.add: Add two integers."


class _FakeProject:
    def __init__(self, *, enabled: bool, mcp):
        self.config = {"orchestrator": {"mcp_enabled": enabled}}
        self.mcp = mcp


def test_awareness_injected_when_enabled_with_tools():
    ex = _executor()
    out = ex._mcp_awareness(_FakeProject(enabled=True, mcp=_FakeMCP()))
    assert "Available MCP tools" in out
    assert "tools.add" in out


def test_awareness_empty_when_disabled():
    ex = _executor()
    out = ex._mcp_awareness(_FakeProject(enabled=False, mcp=_FakeMCP()))
    assert out == ""


def test_awareness_empty_when_no_mcp_configured():
    ex = _executor()
    out = ex._mcp_awareness(_FakeProject(enabled=True, mcp=None))
    assert out == ""


def test_awareness_empty_when_no_tools_discovered():
    class _EmptyMCP:
        def describe_tools(self, cap: int = 25) -> str:
            return ""

    ex = _executor()
    out = ex._mcp_awareness(_FakeProject(enabled=True, mcp=_EmptyMCP()))
    assert out == ""


# --- project wiring ---------------------------------------------------------


def test_project_mcp_none_without_config(tmp_path):
    from my_project_orchestrator.core.project import Project

    proj = Project(tmp_path, {"name": "p"})
    assert proj.mcp is None


def test_project_mcp_built_from_config(tmp_path, server_path):
    from my_project_orchestrator.core.project import Project

    proj = Project(
        tmp_path, {"name": "p", "mcp": {"servers": [_stdio_server(server_path)]}}
    )
    assert proj.mcp is not None
    assert proj.mcp is proj.mcp  # cached
    assert {t.name for t in proj.mcp.tools} >= {"echo", "add"}
