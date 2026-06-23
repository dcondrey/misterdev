"""Bounded, opt-in agentic tool-gathering loop over the MCP substrate.

When ``orchestrator.mcp_tool_use`` is on and an :class:`~my_project_orchestrator.core.mcp.MCPManager`
with discovered tools exists, this runs a *pre-edit* loop in which the model may
request MCP tool calls to gather information before the normal edit-generation
path runs. It is purely ADDITIVE: it produces a context string the executor
prepends to the task context, then proceeds unchanged. When the flag is off the
executor never calls this, so the build is byte-identical to today.

Discipline mirrors :mod:`my_project_orchestrator.core.lsp` and
:mod:`my_project_orchestrator.core.mcp`: every tool call is bounded (it goes
through :meth:`MCPManager.call_tool`, which is timeout-guarded and never raises),
the loop is hard-capped by ``max_rounds``, each call is audited (logged), and any
failure (no manager, no tools, model error, unparseable request, tool error)
degrades to "gather nothing" rather than raising into the build.

Protocol (deterministic, tolerant of a model that wants no tool): each round the
model is shown the available tools and asked to reply with EITHER a single line

    CALL <server>.<tool> {"arg": value, ...}

to request one tool call, OR the literal token ``NO_TOOL`` (or anything that does
not parse as a CALL line) to stop. We parse the first valid CALL line found; if
none is found the loop stops. JSON args are optional (a missing/empty/invalid
args object is treated as ``{}``, logged). The tool result is appended to the
running gathered-context block and fed back into the next round's prompt.
"""

import json
import re
from typing import Callable, List, Optional, Tuple

from my_project_orchestrator.core.mcp import MCPManager
from my_project_orchestrator.logging_setup import setup_logger

logger = setup_logger(__name__)

# One CALL line: "CALL server.tool {json}" — the json object is optional. We keep
# the qualified name and the raw args text; args are parsed separately so a
# malformed object degrades to {} rather than killing the whole match.
_CALL_RE = re.compile(
    r"^\s*CALL\s+([A-Za-z0-9_.\-]+)\.([A-Za-z0-9_\-]+)\s*(\{.*\})?\s*$",
    re.MULTILINE,
)

_GATHER_HEADER = "## Information gathered via MCP tools\n"

_INSTRUCTION = (
    "You may gather information using the MCP tools below BEFORE editing. "
    "To call a tool, reply with EXACTLY one line:\n"
    '    CALL <server>.<tool> {{"arg": value}}\n'
    "(the JSON arguments object is optional). To gather nothing further and "
    "proceed to editing, reply with NO_TOOL. Request at most one tool per reply.\n\n"
    "## Available MCP tools\n{tools}\n"
)


def _parse_call(text: str) -> Optional[Tuple[str, str, dict]]:
    """Parse the first ``CALL server.tool {json}`` line; ``None`` if absent.

    A present-but-malformed JSON args object degrades to ``{}`` (logged), so the
    tool is still attempted with no arguments rather than the round being lost.
    """
    if not text:
        return None
    m = _CALL_RE.search(text)
    if m is None:
        return None
    server, tool, raw_args = m.group(1), m.group(2), m.group(3)
    args: dict = {}
    if raw_args:
        try:
            parsed = json.loads(raw_args)
            if isinstance(parsed, dict):
                args = parsed
            else:
                logger.debug("MCP gather: CALL args not a JSON object; using {}.")
        except (json.JSONDecodeError, ValueError):
            logger.debug("MCP gather: unparseable CALL args; using {}.")
    return server, tool, args


def gather_context(
    manager: Optional[MCPManager],
    ask: Callable[[str], Optional[str]],
    *,
    task_description: str = "",
    max_rounds: int = 3,
    tools_cap: int = 25,
) -> str:
    """Run the bounded tool-gathering loop; return the gathered-context block.

    ``manager`` is the MCP manager (``None`` or one with no tools yields ""),
    ``ask`` is a callable that takes a prompt and returns the model's text reply
    (``None``/empty stops the loop). ``max_rounds`` hard-caps the iterations; a
    value < 1 disables gathering. Returns a context string (empty when nothing
    was gathered) suitable for prepending to the task context. Never raises: any
    failure is logged and degrades to whatever was gathered so far.
    """
    if manager is None or max_rounds < 1:
        return ""
    try:
        tools = manager.describe_tools(cap=tools_cap)
    except Exception as e:  # registry/discovery problem; degrade to no gathering
        logger.debug(f"MCP gather: tool discovery failed: {e}")
        return ""
    if not tools:
        return ""

    gathered: List[str] = []
    for round_idx in range(max_rounds):
        prompt = _INSTRUCTION.format(tools=tools)
        if task_description:
            prompt += (
                f"\n## Task you are gathering information for\n{task_description}\n"
            )
        if gathered:
            prompt += "\n" + _GATHER_HEADER + "\n\n".join(gathered) + "\n"

        try:
            reply = ask(prompt)
        except Exception as e:  # a model/client failure must not sink the build
            logger.debug(f"MCP gather: model call failed in round {round_idx + 1}: {e}")
            break

        parsed = _parse_call(reply or "")
        if parsed is None:
            logger.info(
                f"MCP gather: model requested no tool in round {round_idx + 1}; "
                "stopping."
            )
            break

        server, tool, args = parsed
        logger.info(
            f"MCP gather round {round_idx + 1}/{max_rounds}: CALL {server}.{tool} "
            f"args={sorted(args.keys())}"
        )
        result = manager.call_tool(server, tool, args)
        if result is None:
            # Tool error / unknown server / timeout already logged by call_tool.
            gathered.append(f"### {server}.{tool} -> (no result / error)")
            continue
        gathered.append(f"### {server}.{tool}\n{result}")

    if not gathered:
        return ""
    return (
        "\n\n"
        + _GATHER_HEADER
        + "These results were gathered via MCP tools to inform the edit below.\n\n"
        + "\n\n".join(gathered)
    )
