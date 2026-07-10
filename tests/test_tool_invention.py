"""invent_tool: the runtime tool-invention loop (two-timescale P2b).

The model authors a ```tool block, it runs sandboxed, and the output is fed back.
Additive and best-effort; degrades to "" when off, declined, or unsandboxed.
Runner and model call are injected, so these run without Docker or an LLM.
"""

from misterdev.core.evolution.tool_invention import invent_tool
from misterdev.core.evolution.tool_runner import ToolRunResult


class _FakeRunner:
    def __init__(self, *results):
        self._results = list(results)
        self.calls = []

    def run(self, source, stdin=""):
        self.calls.append((source, stdin))
        return self._results.pop(0) if self._results else ToolRunResult("ok", "", "")


def _asker(*replies):
    seq = list(replies)

    def ask(prompt):
        return seq.pop(0) if seq else "NO_TOOL"

    return ask


def test_no_runner_is_noop():
    assert invent_tool(None, _asker("```tool\nprint(1)\n```")) == ""


def test_model_declines_yields_empty():
    runner = _FakeRunner()
    assert invent_tool(runner, _asker("NO_TOOL")) == ""
    assert runner.calls == []  # nothing was run


def test_authored_tool_runs_and_output_is_returned():
    runner = _FakeRunner(ToolRunResult("ok", "inverse=9\n", ""))
    reply = "I need the modular inverse.\n```tool\nprint('inverse=9')\n```"
    out = invent_tool(
        runner, _asker(reply), max_rounds=1, task_description="affine cipher"
    )
    assert runner.calls[0][0] == "print('inverse=9')"
    assert "inverse=9" in out
    assert "you authored and ran" in out.lower()


def test_stdin_block_is_passed_to_the_runner():
    runner = _FakeRunner(ToolRunResult("ok", "echoed", ""))
    reply = "```tool\nimport sys; print(sys.stdin.read())\n```\n```stdin\nhello\n```"
    invent_tool(runner, _asker(reply), max_rounds=1)
    # source is stripped; stdin is kept verbatim (the trailing newline is real input).
    assert runner.calls[0] == ("import sys; print(sys.stdin.read())", "hello\n")


def test_sandbox_skip_stops_gracefully():
    runner = _FakeRunner(ToolRunResult("skip", "", "no sandbox"))
    out = invent_tool(runner, _asker("```tool\nprint(1)\n```"), max_rounds=2)
    assert out == ""  # nothing usable produced


def test_error_output_is_still_surfaced():
    # A tool that errors still returns its diagnostic to the model.
    runner = _FakeRunner(
        ToolRunResult("error", "Traceback: NameError", "Traceback: NameError")
    )
    out = invent_tool(runner, _asker("```tool\nboom\n```"), max_rounds=1)
    assert "error" in out.lower() and "NameError" in out


def test_multi_round_refinement():
    # Round 1 authors a tool; round 2 authors a refined one; round 3 would exceed cap.
    runner = _FakeRunner(
        ToolRunResult("ok", "v1", ""),
        ToolRunResult("ok", "v2-better", ""),
    )
    out = invent_tool(
        runner,
        _asker("```tool\nprint('v1')\n```", "```tool\nprint('v2')\n```"),
        max_rounds=2,
    )
    assert len(runner.calls) == 2
    assert "v1" in out and "v2-better" in out


def test_model_error_degrades():
    def boom(prompt):
        raise RuntimeError("model down")

    assert invent_tool(_FakeRunner(), boom) == ""


def test_seeds_from_past_runs_are_offered_to_the_model():
    # Promoted tools from prior runs appear in the prompt so the model can reuse
    # or adapt them — the compounding payoff.
    seen = {}

    def ask(prompt):
        seen["prompt"] = prompt
        return "NO_TOOL"

    invent_tool(
        _FakeRunner(),
        ask,
        seeds=["def modinv(a, m): return pow(a, -1, m)"],
        max_rounds=1,
    )
    assert "Proven tools from past runs" in seen["prompt"]
    assert "modinv" in seen["prompt"]


def test_max_rounds_zero_disables():
    runner = _FakeRunner(ToolRunResult("ok", "x", ""))
    assert invent_tool(runner, _asker("```tool\nprint(1)\n```"), max_rounds=0) == ""
    assert runner.calls == []
