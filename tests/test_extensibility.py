"""Pluggable gates and targets, and gather-safe plugin tools in the agentic loop."""

import tempfile
from pathlib import Path

import pytest

from misterdev.plugins import GATES, TARGETS
from misterdev.core.execution.outcomes import GateOutcome, GREEN, RED, SKIP
from misterdev.core.integration.mcp_gather import gather_context


# --- pluggable gates --------------------------------------------------------


@pytest.fixture
def gatekeeper():
    from misterdev.core.verification.gatekeeper import GateKeeper

    with tempfile.TemporaryDirectory() as td:
        yield GateKeeper(Path(td))


def test_plugin_gate_red_blocks(gatekeeper):
    GATES.register("brand", lambda ctx: GateOutcome(RED, "logo missing"))
    try:
        ok, issues, _ = gatekeeper.run_gates({})
        assert not ok
        assert any("G-plugin[brand]" in i and "logo missing" in i for i in issues)
    finally:
        GATES.unregister("brand")


def test_plugin_gate_green_and_skip_do_not_block(gatekeeper):
    GATES.register("g_green", lambda ctx: GateOutcome(GREEN))
    GATES.register("g_skip", lambda ctx: GateOutcome(SKIP, "n/a"))
    try:
        ok, issues, _ = gatekeeper.run_gates({})
        assert ok
        assert not any("G-plugin" in i for i in issues)
    finally:
        GATES.unregister("g_green")
        GATES.unregister("g_skip")


def test_plugin_gate_that_raises_is_skipped(gatekeeper):
    def boom(ctx):
        raise RuntimeError("gate bug")

    GATES.register("g_boom", boom)
    try:
        # A broken third-party gate must not break the pipeline.
        ok, issues, _ = gatekeeper.run_gates({})
        assert ok
    finally:
        GATES.unregister("g_boom")


def test_plugin_gate_receives_context(gatekeeper):
    seen = {}

    def capture(ctx):
        seen["path"] = ctx.project_path
        seen["cmds"] = ctx.commands
        return GateOutcome(GREEN)

    GATES.register("g_ctx", capture)
    try:
        gatekeeper.run_gates({"build_command": "true"})
        assert seen["cmds"] == {"build_command": "true"}
        assert seen["path"] == gatekeeper.project_path
    finally:
        GATES.unregister("g_ctx")


# --- pluggable targets ------------------------------------------------------


class _ElixirTarget:
    markers = ("mix.exs",)

    def commands(self, d):
        return {"build_command": "mix compile", "test_command": "mix test"}


def test_plugin_target_type_discovered():
    from misterdev.core.planning.targets import discover_targets

    TARGETS.register("elixir", _ElixirTarget())
    try:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for sub in ("svc_a", "svc_b"):
                (root / sub).mkdir()
                (root / sub / "mix.exs").write_text("# elixir project\n")
            found = discover_targets(str(root))
            assert {t["name"] for t in found} == {"svc_a", "svc_b"}
            assert all(t["build_command"] == "mix compile" for t in found)
            assert all(t["test_command"] == "mix test" for t in found)
    finally:
        TARGETS.unregister("elixir")


# --- gather-safe plugin tools in the agentic loop ---------------------------


def test_gather_loop_dispatches_local_plugin_tool():
    # A gather-safe tool is callable through the existing loop as local.<name>.
    def echo(args):
        return f"echoed:{args.get('msg')}"

    replies = iter(['CALL local.echo {"msg": "hi"}', "NO_TOOL"])
    ctx = gather_context(
        None,
        lambda prompt: next(replies),
        local_tools={"echo": ("echo a message", echo)},
        max_rounds=3,
    )
    assert "echoed:hi" in ctx


def test_gather_loop_returns_empty_without_any_tools():
    assert gather_context(None, lambda prompt: "NO_TOOL") == ""


def test_gather_loop_local_tool_error_degrades_not_raises():
    def boom(args):
        raise RuntimeError("tool bug")

    replies = iter(["CALL local.bad {}", "NO_TOOL"])
    ctx = gather_context(
        None,
        lambda prompt: next(replies),
        local_tools={"bad": ("broken", boom)},
        max_rounds=3,
    )
    assert "no result / error" in ctx


def test_gather_safe_tools_excludes_non_optin(monkeypatch):
    # Built-in mutating tools (command, file_io) must never be gather-exposed.
    from misterdev.task_executors.markdown_plan_executor.context_mixin import (
        ContextMixin,
    )

    class _Proj:
        config = {"tools": [{"name": "sh", "type": "command"}]}
        path = Path(".")

    local = ContextMixin()._gather_safe_tools(_Proj())
    assert local == {}
