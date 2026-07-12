"""Bounded, opt-in agentic tool-gathering loop over the MCP substrate.

When ``orchestrator.mcp_tool_use`` is on and an :class:`~misterdev.core.integration.mcp.MCPManager`
with discovered tools exists, this runs a *pre-edit* loop in which the model may
request MCP tool calls to gather information before the normal edit-generation
path runs. It is purely ADDITIVE: it produces a context string the executor
prepends to the task context, then proceeds unchanged. When the flag is off the
executor never calls this, so the build is byte-identical to today.

Discipline mirrors :mod:`misterdev.core.context.lsp` and
:mod:`misterdev.core.integration.mcp`: every tool call is bounded (it goes
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
from typing import Any, Callable, Dict, List, Optional, Tuple

from misterdev.core.integration.mcp import MCPManager
from misterdev.llm.responses import (
    extract_balanced_span as _extract_balanced_object,
)
from misterdev.logging_setup import setup_logger

logger = setup_logger(__name__)

# "CALL server.tool" anywhere in the reply. Tolerant on purpose: searched (not
# anchored) so leading markdown (backticks, ``**``, ``> ``) and trailing prose
# don't break it, and case-insensitive with word boundaries so neither "recall"
# nor a lowercased keyword trips it. The server keeps dots (``a.b.tool`` ->
# server ``a.b``, tool ``tool``); the args object is located and balanced
# separately so it may span multiple lines.
_CALL_RE = re.compile(
    r"\bCALL\b\s+([A-Za-z0-9_.\-]+)\.([A-Za-z0-9_\-]+)",
    re.IGNORECASE,
)

# "FIND <capability>" — request a NEW server be discovered/mounted on demand.
_FIND_RE = re.compile(r"\bFIND\b\s+(.+)", re.IGNORECASE)

_GATHER_HEADER = "## Information gathered via tools\n"

_INSTRUCTION = (
    "You may gather information using the tools below BEFORE editing. "
    "To call a tool, reply with EXACTLY one line:\n"
    '    CALL <server>.<tool> {{"arg": value}}\n'
    "(the JSON arguments object is optional). To gather nothing further and "
    "proceed to editing, reply with NO_TOOL. Request at most one tool per reply.\n\n"
    "## Available tools\n{tools}\n"
)

# Extra line appended to the instruction only when on-demand provisioning is on.
_FIND_INSTRUCTION = (
    "\nIf no available tool fits, you may request a new one: reply with\n"
    "    FIND <capability you need>\n"
    "and a matching server will be discovered and mounted for the next round.\n"
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
    server, tool = m.group(1), m.group(2)
    args: dict = {}
    brace = text.find("{", m.end())
    # Only treat a following object as the args if nothing but whitespace
    # separates it from the call, so a stray ``{`` in later prose isn't grabbed.
    if brace != -1 and text[m.end() : brace].strip() == "":
        raw_args = _extract_balanced_object(text, brace)
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
    local_tools: Optional[
        Dict[str, Tuple[str, Callable[[dict], Optional[str]]]]
    ] = None,
    provide: Optional[Callable[[str], Optional[Any]]] = None,
    max_provisions: int = 2,
) -> str:
    """Run the bounded tool-gathering loop; return the gathered-context block.

    ``manager`` is the MCP manager (may be ``None``). ``local_tools`` maps a
    registered gather-safe tool's name to ``(description, call)`` and is exposed
    to the model as ``local.<name>``, so plugin tools and MCP tools are called
    through the one loop. ``ask`` takes a prompt and returns the model's reply
    (``None``/empty stops). ``max_rounds`` hard-caps the iterations; < 1 disables.
    Returns a context string (empty when nothing was gathered), suitable for
    prepending to the task context. Never raises: any failure is logged and
    degrades to whatever was gathered so far.
    """
    local_tools = local_tools or {}
    if (manager is None and not local_tools) or max_rounds < 1:
        return ""
    tools = ""
    if manager is not None:
        try:
            tools = manager.describe_tools(cap=tools_cap) or ""
        except Exception as e:  # discovery problem; degrade to local tools only
            logger.debug(f"gather: MCP tool discovery failed: {e}")
            tools = ""
    if local_tools:
        local_desc = "\n".join(
            f"- local.{name}: {desc}" for name, (desc, _) in local_tools.items()
        )
        tools = f"{tools}\n{local_desc}" if tools else local_desc

    can_provide = provide is not None and manager is not None
    provisions_left = max_provisions if can_provide else 0
    # Run the loop when there is at least one tool OR the model can FIND one on
    # demand (the "nothing fits, discover one" case). Otherwise nothing to do.
    if not tools and not can_provide:
        return ""

    gathered: List[str] = []
    for round_idx in range(max_rounds):
        prompt = _INSTRUCTION.format(tools=tools)
        if provisions_left > 0:
            prompt += _FIND_INSTRUCTION
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

        if provisions_left > 0 and _CALL_RE.search(reply or "") is None:
            fm = _FIND_RE.search(reply or "")
            if fm:
                capability = fm.group(1).strip()
                provisions_left -= 1
                logger.info(
                    f"gather round {round_idx + 1}/{max_rounds}: FIND {capability!r}"
                )
                cfg = None
                try:
                    cfg = provide(capability)
                except Exception as e:  # provisioning must never sink the build
                    logger.debug(f"MCP gather: provide({capability!r}) error: {e}")
                new_tools = manager.add_server(cfg) if cfg else []
                if new_tools:
                    tools = manager.describe_tools(cap=tools_cap) or tools
                    names = ", ".join(t.qualified_name for t in new_tools)
                    logger.info(f"MCP gather: mounted on demand -> {names}")
                    gathered.append(f"### FIND {capability}\nmounted: {names}")
                else:
                    logger.info(f"MCP gather: no server found for {capability!r}")
                    gathered.append(f"### FIND {capability} -> (no server found)")
                continue

        parsed = _parse_call(reply or "")
        if parsed is None:
            logger.info(
                f"MCP gather: model requested no tool in round {round_idx + 1}; "
                "stopping."
            )
            break

        server, tool, args = parsed
        logger.info(
            f"gather round {round_idx + 1}/{max_rounds}: CALL {server}.{tool} "
            f"args={sorted(args.keys())}"
        )
        if server == "local" and tool in local_tools:
            try:
                result = local_tools[tool][1](args)
            except Exception as e:  # a plugin tool must not sink the build
                logger.debug(f"gather: local tool {tool!r} error: {e}")
                result = None
        elif manager is not None:
            result = manager.call_tool(server, tool, args)
        else:
            result = None
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
        + "These results were gathered via tools to inform the edit below.\n\n"
        + "\n\n".join(gathered)
    )
