"""Runtime tool-invention: let the model author a task-specific helper tool and
run it sandboxed, feeding the output back — the runtime half of two-timescale
evolution (``docs/two-timescale-evolution.md``) and the capability that sits #1
open on SWE-bench (live-SWE-agent).

Additive and best-effort, mirroring
:mod:`misterdev.core.integration.mcp_gather`: the model MAY emit a ``tool`` block
(Python source) and an optional ``stdin`` block; the source runs via the hardened,
network-less :class:`~misterdev.core.evolution.tool_runner.ToolRunner` (untrusted
code never touches the host or the repo), and its output is returned as a context
block the executor prepends to the edit context. A model that wants no tool
replies ``NO_TOOL`` and the result is ``""`` — byte-identical to the no-tooling
path. The loop is hard-capped so a model can author, see the output, and refine
once or twice, then stop.

Never raises into the build: no runner, no sandbox, a model error, or an
unparseable reply all degrade to whatever was produced so far (usually nothing).
"""

import re
from typing import Callable, List, Optional, Protocol

from misterdev.logging_setup import setup_logger

logger = setup_logger(__name__)

# The model authors source in a ```tool (or ```python-tool) fence, with optional
# stdin in a ```stdin fence. Searched (not anchored) and DOTALL so the blocks may
# span lines and sit amid prose.
_TOOL_RE = re.compile(r"```(?:tool|python-tool)\s*\n(.*?)```", re.DOTALL)
_STDIN_RE = re.compile(r"```stdin\s*\n(.*?)```", re.DOTALL)

_HEADER = "## Tools you authored and ran (sandboxed) to inform the edit\n"

_INSTRUCTION = (
    "You MAY author ONE small Python helper tool to compute or verify something "
    "for this task before you edit — for example a reference calculation, a "
    "parser, or a checker. It runs SANDBOXED: no network, no filesystem or "
    "repository access, bounded time and memory. It is a pure function — supply "
    "any input it needs yourself.\n\n"
    "To author a tool, emit its source in a fenced ```tool block, and (optionally) "
    "its stdin in a fenced ```stdin block. Its stdout is returned to you so you "
    "can use the result in your edit. To author no tool and proceed straight to "
    "editing, reply with NO_TOOL.\n"
)


class _ToolRunnerLike(Protocol):
    def run(self, source: str, stdin: str = ""): ...


def _render(result) -> str:
    """The tool's output for the model: stdout on success, else stderr/detail."""
    body = (result.stdout or "").strip() or (result.stderr or "").strip()
    return body or "(no output)"


def invent_tool(
    runner: Optional[_ToolRunnerLike],
    ask: Callable[[str], Optional[str]],
    *,
    task_description: str = "",
    error_context: str = "",
    max_rounds: int = 2,
    sink: Optional[List[str]] = None,
    seeds: Optional[List[str]] = None,
) -> str:
    """Run the bounded tool-invention loop; return the invented-tools context block.

    ``runner`` is a :class:`ToolRunner` (may be ``None``). ``ask`` takes a prompt
    and returns the model's reply. Each round the model may author one tool; it is
    run sandboxed and its output fed into the next round's prompt. ``max_rounds``
    hard-caps iterations (< 1 disables). Returns a context string suitable for
    prepending to the edit context, or ``""`` when no tool was authored or run.
    """
    if runner is None or max_rounds < 1:
        return ""
    invented: List[str] = []
    for round_idx in range(max_rounds):
        prompt = _INSTRUCTION
        if seeds:
            prompt += (
                "\n## Proven tools from past runs — reuse or adapt one if it fits\n"
                + "\n\n".join(f"```python\n{s}\n```" for s in seeds)
                + "\n"
            )
        if task_description:
            prompt += f"\n## Task\n{task_description}\n"
        if error_context:
            prompt += (
                "\n## The current failure you may write a tool to diagnose\n"
                f"{error_context}\n"
            )
        if invented:
            prompt += "\n" + _HEADER + "\n\n".join(invented) + "\n"

        try:
            reply = ask(prompt)
        except Exception as e:  # a model/client failure must not sink the build
            logger.debug(
                f"tool-invention: model call failed round {round_idx + 1}: {e}"
            )
            break

        m = _TOOL_RE.search(reply or "")
        if m is None:
            logger.info(
                f"tool-invention: model authored no tool in round {round_idx + 1}; "
                "stopping."
            )
            break
        source = m.group(1).strip()
        if not source:
            break
        sm = _STDIN_RE.search(reply or "")
        stdin = sm.group(1) if sm else ""

        result = runner.run(source, stdin)
        if result.status == "skip":
            logger.info("tool-invention: no sandbox available; tool not run.")
            break
        if sink is not None:
            sink.append(source)  # captured for the tool corpus (P2c)
        logger.info(
            f"tool-invention round {round_idx + 1}/{max_rounds}: ran a tool "
            f"({len(source)}B source) -> {result.status}"
        )
        invented.append(
            f"### tool (round {round_idx + 1})\n```python\n{source}\n```\n"
            f"output [{result.status}]:\n{_render(result)}"
        )

    if not invented:
        return ""
    return (
        "\n\n"
        + _HEADER
        + "You wrote and ran these tools; use their results in your edit.\n\n"
        + "\n\n".join(invented)
    )
