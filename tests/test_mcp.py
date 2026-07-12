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

import misterdev.core.integration.mcp as mcp_mod
from misterdev.core.integration.mcp import MCPManager, MCPTool
from misterdev.core.integration.mcp_gather import _parse_call, gather_context

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
    from misterdev.task_executors.markdown_plan_executor import (
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
    from misterdev.core.execution.project import Project

    proj = Project(tmp_path, {"name": "p"})
    assert proj.mcp is None


def test_project_mcp_built_from_config(tmp_path, server_path):
    from misterdev.core.execution.project import Project

    proj = Project(
        tmp_path, {"name": "p", "mcp": {"servers": [_stdio_server(server_path)]}}
    )
    assert proj.mcp is not None
    assert proj.mcp is proj.mcp  # cached
    assert {t.name for t in proj.mcp.tools} >= {"echo", "add"}


# --- bounded agentic tool-gathering loop ------------------------------------


class _ScriptedAsk:
    """A fake LLM: returns the next scripted reply per call, recording prompts."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []
        self.calls = 0

    def __call__(self, prompt: str):
        self.prompts.append(prompt)
        self.calls += 1
        if self.replies:
            return self.replies.pop(0)
        return "NO_TOOL"


def test_parse_call_extracts_server_tool_and_args():
    assert _parse_call('CALL tools.add {"a": 2, "b": 40}') == (
        "tools",
        "add",
        {"a": 2, "b": 40},
    )


def test_parse_call_no_args_is_empty_dict():
    assert _parse_call("CALL tools.echo") == ("tools", "echo", {})


def test_parse_call_malformed_args_degrade_to_empty():
    assert _parse_call("CALL tools.add {not json}") == ("tools", "add", {})


def test_parse_call_none_when_no_call_line():
    assert _parse_call("NO_TOOL") is None
    assert _parse_call("I don't need a tool.") is None
    assert _parse_call("") is None


def test_parse_call_multiline_json_args():
    reply = 'CALL tools.add {\n  "a": 2,\n  "b": 40\n}'
    assert _parse_call(reply) == ("tools", "add", {"a": 2, "b": 40})


def test_parse_call_tolerates_markdown_and_prose():
    # Leading prose/markdown and a trailing explanation must not break parsing.
    reply = 'I will compute the sum.\n`CALL tools.add {"a": 1, "b": 2}` then done.'
    assert _parse_call(reply) == ("tools", "add", {"a": 1, "b": 2})


def test_parse_call_object_with_braces_in_string():
    reply = 'CALL tools.run {"cmd": "echo {hi}"}'
    assert _parse_call(reply) == ("tools", "run", {"cmd": "echo {hi}"})


def test_parse_call_dotted_server_name():
    assert _parse_call("CALL my.server.echo") == ("my.server", "echo", {})


def test_parse_call_case_insensitive_keyword():
    assert _parse_call('call tools.add {"a": 1}') == ("tools", "add", {"a": 1})


def test_parse_call_recall_does_not_false_match():
    assert _parse_call("I recall tools.add was useful") is None


def test_parse_call_ignores_stray_brace_in_later_prose():
    # A '{' that is not adjacent to the call must not be slurped as args.
    assert _parse_call("CALL tools.echo\nlater I might use a set {1, 2}") == (
        "tools",
        "echo",
        {},
    )


def test_gather_calls_tool_and_captures_result(server_path):
    # Round 1: model requests add(2, 40); round 2: model says no tool needed.
    mgr = MCPManager([_stdio_server(server_path)])
    ask = _ScriptedAsk(['CALL tools.add {"a": 2, "b": 40}', "NO_TOOL"])
    out = gather_context(mgr, ask, task_description="sum two numbers", max_rounds=3)
    assert "42" in out  # the tool result is captured into the gathered context
    assert "tools.add" in out
    assert ask.calls == 2  # asked twice, then stopped on NO_TOOL
    # The second prompt must carry the gathered result back to the model.
    assert "42" in ask.prompts[1]


def test_gather_is_bounded_by_max_rounds(server_path):
    # A model that requests a tool every round must stop at max_rounds.
    mgr = MCPManager([_stdio_server(server_path)])
    ask = _ScriptedAsk(['CALL tools.add {"a": 1, "b": 1}'] * 100)
    gather_context(mgr, ask, max_rounds=3)
    assert ask.calls == 3  # never exceeds the cap


def test_gather_stops_when_no_tool_first_round(server_path):
    mgr = MCPManager([_stdio_server(server_path)])
    ask = _ScriptedAsk(["NO_TOOL"])
    out = gather_context(mgr, ask, max_rounds=3)
    assert out == ""  # nothing gathered
    assert ask.calls == 1


def test_gather_empty_when_no_manager():
    ask = _ScriptedAsk(['CALL tools.add {"a": 1}'])
    assert gather_context(None, ask, max_rounds=3) == ""
    assert ask.calls == 0  # never asked the model


def test_gather_empty_when_no_tools_discovered():
    class _EmptyMgr:
        def describe_tools(self, cap: int = 25) -> str:
            return ""

    ask = _ScriptedAsk(['CALL tools.add {"a": 1}'])
    assert gather_context(_EmptyMgr(), ask, max_rounds=3) == ""
    assert ask.calls == 0


def test_gather_zero_rounds_disables(server_path):
    mgr = MCPManager([_stdio_server(server_path)])
    ask = _ScriptedAsk(['CALL tools.add {"a": 1}'])
    assert gather_context(mgr, ask, max_rounds=0) == ""
    assert ask.calls == 0


def test_gather_tool_error_does_not_raise(monkeypatch, server_path):
    # A tool that errors (call_tool returns None) must not raise; the loop
    # records a no-result marker and proceeds, then stops on the next NO_TOOL.
    mgr = MCPManager([_stdio_server(server_path)])
    monkeypatch.setattr(mgr, "call_tool", lambda *a, **k: None)
    ask = _ScriptedAsk(['CALL tools.add {"a": 1}', "NO_TOOL"])
    out = gather_context(mgr, ask, max_rounds=3)
    assert "42" not in out  # no usable result; only a no-result marker
    assert "no result" in out
    assert ask.calls == 2


def test_gather_model_error_does_not_raise(server_path):
    mgr = MCPManager([_stdio_server(server_path)])

    def _boom(prompt):
        raise RuntimeError("model down")

    assert gather_context(mgr, _boom, max_rounds=3) == ""


def test_gather_unparseable_request_stops_cleanly(server_path):
    mgr = MCPManager([_stdio_server(server_path)])
    ask = _ScriptedAsk(["I will edit now, no tool."])
    assert gather_context(mgr, ask, max_rounds=3) == ""
    assert ask.calls == 1


# --- on-demand provisioning (FIND) ------------------------------------------
def test_add_server_mounts_live_and_returns_tools(server_path):
    mgr = MCPManager([])  # starts empty
    new = mgr.add_server(_stdio_server(server_path))
    assert any(t.name == "add" for t in new)
    assert any(t.qualified_name == "tools.add" for t in mgr.tools)


def test_add_server_dedups_by_name(server_path):
    mgr = MCPManager([_stdio_server(server_path)])
    assert mgr.add_server(_stdio_server(server_path)) == []  # same name -> no-op


def test_gather_find_provisions_then_calls(server_path):
    # Empty manager: nothing fits, so the model FINDs a server, then calls it.
    mgr = MCPManager([])
    ask = _ScriptedAsk(
        ["FIND add two numbers", 'CALL tools.add {"a": 2, "b": 40}', "NO_TOOL"]
    )
    out = gather_context(
        mgr, ask, max_rounds=4, provide=lambda q: _stdio_server(server_path)
    )
    assert "42" in out and "tools.add" in out
    assert "mounted" in out.lower()  # the FIND result is recorded
    assert ask.calls == 3


def test_gather_find_ignored_without_provide(server_path):
    # With no provider, FIND is not special: it parses as no CALL -> loop stops.
    mgr = MCPManager([_stdio_server(server_path)])
    ask = _ScriptedAsk(["FIND something"])
    assert gather_context(mgr, ask, max_rounds=3) == ""
    assert ask.calls == 1


def test_gather_find_bounded_by_max_provisions(server_path):
    mgr = MCPManager([])
    ask = _ScriptedAsk(["FIND a"] * 10)
    gather_context(
        mgr,
        ask,
        max_rounds=10,
        max_provisions=2,
        provide=lambda q: {"name": "none", "command": "does-not-exist"},
    )
    # 2 FIND rounds consume the budget; the 3rd reply has no FIND handling and no
    # CALL -> the loop stops. So at most 3 asks.
    assert ask.calls <= 3


def test_gather_never_hangs(monkeypatch, server_path):
    # The single tool call is bounded by call_tool's hard timeout; a hanging tool
    # body is abandoned and the loop returns promptly with no result.
    def _hang(server, name, arguments):
        time.sleep(3600)

    monkeypatch.setattr(mcp_mod, "_call_tool", _hang)
    mgr = MCPManager([_stdio_server(server_path)], call_timeout=0.3)
    ask = _ScriptedAsk(['CALL tools.add {"a": 1, "b": 1}', "NO_TOOL"])
    start = time.monotonic()
    out = gather_context(mgr, ask, max_rounds=2)
    assert time.monotonic() - start < 10  # bounded, never hangs
    assert "no result" in out  # the timed-out call left a no-result marker


# --- executor wiring (flag gating + byte-identical off path) ----------------


class _FakeLLMClient:
    def __init__(self, replies):
        self.ask = _ScriptedAsk(replies)

    def generate_code(self, prompt, system_prompt=""):
        return self.ask(prompt)


class _GatherProject:
    def __init__(self, *, tool_use, mcp, llm_replies=(), max_rounds=3):
        self.config = {
            "orchestrator": {
                "mcp_tool_use": tool_use,
                "mcp_max_tool_rounds": max_rounds,
            }
        }
        self.mcp = mcp
        self.llm_client = _FakeLLMClient(llm_replies)


def _gather_task():
    from misterdev.core.models import Task

    return Task(id="t1", description="sum two numbers", project_ref="p")


def test_executor_gather_off_returns_empty_and_never_asks(server_path):
    ex = _executor()
    proj = _GatherProject(
        tool_use=False,
        mcp=MCPManager([_stdio_server(server_path)]),
        llm_replies=['CALL tools.add {"a": 2, "b": 40}'],
    )
    assert ex._mcp_gather(proj, _gather_task()) == ""
    assert proj.llm_client.ask.calls == 0  # flag off => loop not entered at all


def test_executor_gather_on_invokes_tool_and_captures_result(server_path):
    ex = _executor()
    proj = _GatherProject(
        tool_use=True,
        mcp=MCPManager([_stdio_server(server_path)]),
        llm_replies=['CALL tools.add {"a": 2, "b": 40}', "NO_TOOL"],
    )
    out = ex._mcp_gather(proj, _gather_task())
    assert "42" in out
    assert proj.llm_client.ask.calls == 2


def test_executor_gather_no_manager_is_empty():
    ex = _executor()
    proj = _GatherProject(tool_use=True, mcp=None, llm_replies=["CALL x.y"])
    assert ex._mcp_gather(proj, _gather_task()) == ""
    assert proj.llm_client.ask.calls == 0


def test_query_on_failure_frames_gather_around_the_error(monkeypatch):
    # _mcp_gather with error_context must pass the failure into the gather
    # prompt so the model looks up what it needs to FIX it (query-on-failure).
    from types import SimpleNamespace
    import misterdev.task_executors.markdown_plan_executor.context_mixin as cm
    from misterdev.task_executors.markdown_plan_executor import MarkdownPlanExecutor

    captured = {}

    def fake_gather(manager, ask, *, task_description="", **kw):
        captured["desc"] = task_description
        return ""

    monkeypatch.setattr(cm, "gather_context", fake_gather)
    project = SimpleNamespace(
        config={"orchestrator": {"mcp_tool_use": True}, "mcp": {}},
        mcp=object(),  # non-None so the gather path runs
        llm_client=SimpleNamespace(generate_code=lambda p, s: "NO_TOOL"),
    )
    task = SimpleNamespace(description="implement foo()")
    MarkdownPlanExecutor()._mcp_gather(
        project, task, error_context="BOOM: undefined symbol bar"
    )
    assert "BOOM: undefined symbol bar" in captured["desc"]
    assert "FAILED a gate" in captured["desc"]
    assert "implement foo()" in captured["desc"]


def test_gather_no_error_context_is_plain_description(monkeypatch):
    from types import SimpleNamespace
    import misterdev.task_executors.markdown_plan_executor.context_mixin as cm
    from misterdev.task_executors.markdown_plan_executor import MarkdownPlanExecutor

    captured = {}
    monkeypatch.setattr(
        cm, "gather_context",
        lambda m, a, *, task_description="", **k: captured.update(desc=task_description) or "",
    )
    project = SimpleNamespace(
        config={"orchestrator": {"mcp_tool_use": True}, "mcp": {}},
        mcp=object(),
        llm_client=SimpleNamespace(generate_code=lambda p, s: "NO_TOOL"),
    )
    MarkdownPlanExecutor()._mcp_gather(project, SimpleNamespace(description="do X"))
    assert captured["desc"] == "do X"  # no failure framing on the first pass
