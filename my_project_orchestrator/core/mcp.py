"""Optional MCP (Model Context Protocol) tool-host substrate.

MCP servers expose tools the orchestrator could call (search a wiki, query a
database, run a domain-specific check) over a standard protocol. This module is
the *substrate*: it connects to the servers named in ``mcp.servers``, discovers
their tools, and can invoke one — nothing more. It does NOT turn the executor
into an autonomous tool-calling agent; the only integration is awareness
injection (the executor is told which tools exist). The agentic loop that would
let the model actually decide to call a tool mid-build is deliberately left as a
documented seam (see ``MCPManager.call_tool`` and the README).

It mirrors :mod:`my_project_orchestrator.core.lsp`: strictly opt-in (off unless
``orchestrator.mcp_enabled`` and ``mcp.servers`` are set), best-effort, and every
operation is run in a daemon worker thread with a hard timeout so a missing SDK,
a server that fails to start, or one that hangs can NEVER block or slow a build.
A server that misbehaves is simply absent from the registry (logged at debug);
nothing here ever raises into the caller.

The official ``mcp`` Python SDK is imported lazily inside the worker so the
dependency is optional: without it the registry is empty and the manager is a
no-op. stdio transport is supported (the SDK launches the server as a
subprocess); other transports degrade to "server absent".
"""

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from my_project_orchestrator.logging_setup import setup_logger

logger = setup_logger(__name__)

# Hard ceilings (seconds). Discovery may launch several subprocesses, so it gets
# a per-server budget; a single tool call is cheaper. Both are overridable via
# config but always bounded — there is no "wait forever" path.
_DEFAULT_CONNECT_TIMEOUT = 20.0
_DEFAULT_CALL_TIMEOUT = 30.0


@dataclass(frozen=True)
class MCPTool:
    """One tool discovered from an MCP server.

    ``server`` is the configured server name (the routing key for
    :meth:`MCPManager.call_tool`); ``input_schema`` is the JSON Schema the
    server advertises for the tool's arguments (may be empty).
    """

    server: str
    name: str
    description: str = ""
    input_schema: Dict[str, Any] = field(default_factory=dict)

    @property
    def qualified_name(self) -> str:
        """``server.tool`` — unique across servers, used in awareness text."""
        return f"{self.server}.{self.name}"


def _run_bounded(fn, timeout: float, default, what: str):
    """Run ``fn()`` in a daemon thread, returning its result or ``default``.

    Any exception inside ``fn`` is swallowed (logged at debug) and a hung call
    is abandoned to the daemon thread (which dies with the process) once the
    timeout fires — so the caller is guaranteed to get control back within
    ``timeout`` seconds and is never handed an exception. ``what`` names the
    operation for log context.
    """
    box: Dict[str, Any] = {"result": default}

    def _worker() -> None:
        try:
            box["result"] = fn()
        except Exception as e:  # missing SDK / server crash / protocol error
            logger.debug(f"MCP {what} unavailable: {e}")
            box["result"] = default

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        logger.warning(
            f"MCP {what} timed out after {timeout}s; skipping (never blocks)."
        )
        return default
    return box["result"]


def _normalize_servers(servers: Any) -> List[Dict[str, Any]]:
    """Validate and normalize the ``mcp.servers`` config list.

    Each entry must be a mapping with a non-empty ``name`` and ``command``;
    malformed or duplicate-named entries are dropped with a warning rather than
    raised, so a typo in one server never sinks the rest.
    """
    out: List[Dict[str, Any]] = []
    if not isinstance(servers, list):
        if servers:
            logger.warning("mcp.servers must be a list; ignoring.")
        return out
    seen: set = set()
    for entry in servers:
        if not isinstance(entry, dict):
            logger.warning(f"Ignoring non-mapping mcp.servers entry: {entry!r}")
            continue
        name = entry.get("name")
        command = entry.get("command")
        if not name or not command:
            logger.warning(f"Ignoring mcp server without name/command: {entry!r}")
            continue
        transport = (entry.get("transport") or "stdio").lower()
        if transport != "stdio":
            # Only stdio is supported today; anything else is simply absent.
            logger.debug(
                f"MCP server '{name}' uses unsupported transport {transport!r}; skipping."
            )
            continue
        if name in seen:
            logger.warning(
                f"Duplicate mcp server name '{name}'; ignoring the later one."
            )
            continue
        seen.add(name)
        out.append(entry)
    return out


class MCPManager:
    """Connects to configured MCP servers and exposes their tools.

    Discovery (``tools``) connects to every server once, lists its tools, and
    caches the merged result; a server that fails contributes nothing. A
    :meth:`call_tool` connects to the one named server, invokes the tool, and
    returns its textual result (or ``None`` on any failure). Connections are
    per-operation and short-lived: nothing async is held across the synchronous
    orchestrator, which keeps the integration trivially safe.
    """

    def __init__(
        self,
        servers: Any,
        *,
        connect_timeout: float = _DEFAULT_CONNECT_TIMEOUT,
        call_timeout: float = _DEFAULT_CALL_TIMEOUT,
    ):
        self.servers = _normalize_servers(servers)
        self.connect_timeout = float(connect_timeout)
        self.call_timeout = float(call_timeout)
        self._tools: Optional[List[MCPTool]] = None  # lazy, cached

    @property
    def enabled(self) -> bool:
        """True when at least one usable server is configured."""
        return bool(self.servers)

    @property
    def tools(self) -> List[MCPTool]:
        """Merged tool registry across all servers (discovered once, cached).

        Best-effort and bounded: each server's discovery is timeout-guarded, so
        a slow or broken server is skipped rather than blocking the rest. Always
        returns a list (possibly empty); never raises.
        """
        if self._tools is None:
            self._tools = self._discover_all()
        return self._tools

    def _discover_all(self) -> List[MCPTool]:
        tools: List[MCPTool] = []
        for server in self.servers:
            name = server["name"]
            discovered = _run_bounded(
                lambda s=server: _list_tools(s),
                self.connect_timeout,
                default=[],
                what=f"tool discovery ({name})",
            )
            for t in discovered:
                tools.append(
                    MCPTool(
                        server=name,
                        name=t.get("name", ""),
                        description=t.get("description") or "",
                        input_schema=t.get("input_schema") or {},
                    )
                )
        if tools:
            logger.info(
                f"MCP discovered {len(tools)} tool(s) across "
                f"{len({t.server for t in tools})} server(s)."
            )
        return tools

    def call_tool(
        self, server: str, name: str, arguments: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Invoke ``name`` on ``server`` with ``arguments``; return its text.

        Returns the tool's concatenated text content on success, or ``None`` on
        any failure (unknown server, missing SDK, server error, tool error, or
        timeout). Every call is logged. Bounded by ``call_timeout`` — a hanging
        tool can never block a build.

        SEAM: this is the single chokepoint a future agentic tool-calling loop
        would drive. That loop is out of scope here; this method gives it a
        clean, safe, audited entry point (config-gated, timeout-bounded,
        never-raises) to build on without touching the build loop's internals.
        """
        cfg = next((s for s in self.servers if s["name"] == server), None)
        if cfg is None:
            logger.warning(f"MCP call_tool: no configured server named '{server}'.")
            return None
        logger.info(
            f"MCP call_tool: {server}.{name} args={list((arguments or {}).keys())}"
        )
        return _run_bounded(
            lambda: _call_tool(cfg, name, arguments or {}),
            self.call_timeout,
            default=None,
            what=f"tool call ({server}.{name})",
        )

    def describe_tools(self, cap: int = 25) -> str:
        """Render the registry as a concise text block for prompt awareness.

        Empty string when there are no tools (so callers can append
        unconditionally). Bounded by ``cap`` so a server exposing hundreds of
        tools can't blow the prompt budget; each line is one tool.
        """
        tools = self.tools
        if not tools:
            return ""
        lines: List[str] = []
        for t in tools[:cap]:
            desc = " ".join((t.description or "").split())  # collapse whitespace
            if len(desc) > 200:
                desc = desc[:197] + "..."
            suffix = f": {desc}" if desc else ""
            lines.append(f"- {t.qualified_name}{suffix}")
        if len(tools) > cap:
            lines.append(f"- ... and {len(tools) - cap} more")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Worker-thread bodies. Imported lazily so `mcp` stays an optional dependency,
# and each runs its own asyncio event loop in the daemon thread (the orchestrator
# is synchronous; no loop is shared or held across the call).
# ---------------------------------------------------------------------------


def _server_params(server: Dict[str, Any]):
    from mcp import StdioServerParameters

    args = server.get("args") or []
    if not isinstance(args, list):
        args = [str(args)]
    env = server.get("env")
    return StdioServerParameters(
        command=str(server["command"]),
        args=[str(a) for a in args],
        env=dict(env) if isinstance(env, dict) else None,
        cwd=server.get("cwd"),
    )


def _list_tools(server: Dict[str, Any]) -> List[Dict[str, Any]]:
    import asyncio

    async def _main() -> List[Dict[str, Any]]:
        from mcp import ClientSession
        from mcp.client.stdio import stdio_client

        params = _server_params(server)
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_tools()
                out: List[Dict[str, Any]] = []
                for tool in result.tools:
                    out.append(
                        {
                            "name": tool.name,
                            "description": tool.description or "",
                            "input_schema": getattr(tool, "inputSchema", None) or {},
                        }
                    )
                return out

    return asyncio.run(_main())


def _call_tool(
    server: Dict[str, Any], name: str, arguments: Dict[str, Any]
) -> Optional[str]:
    import asyncio

    async def _main() -> Optional[str]:
        from mcp import ClientSession
        from mcp.client.stdio import stdio_client

        params = _server_params(server)
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(name, arguments=arguments)
                if getattr(result, "isError", False):
                    logger.debug(f"MCP tool {name} returned an error result.")
                    return None
                return _result_text(result)

    return asyncio.run(_main())


def _result_text(result) -> Optional[str]:
    """Concatenate the text content of a CallToolResult, or None if none."""
    parts: List[str] = []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts) if parts else None
