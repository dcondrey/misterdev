"""Optional MCP (Model Context Protocol) tool-host substrate.

MCP servers expose tools the orchestrator could call (search a wiki, query a
database, run a domain-specific check) over a standard protocol. This module is
the *substrate*: it connects to the servers named in ``mcp.servers``, discovers
their tools, and can invoke one — nothing more. It does NOT turn the executor
into an autonomous tool-calling agent; the only integration is awareness
injection (the executor is told which tools exist). The agentic loop that would
let the model actually decide to call a tool mid-build is deliberately left as a
documented seam (see ``MCPManager.call_tool`` and the README).

It mirrors :mod:`misterdev.core.context.lsp`: strictly opt-in (off unless
``orchestrator.mcp_enabled`` and ``mcp.servers`` are set), best-effort, and every
operation is run in a daemon worker thread with a hard timeout so a missing SDK,
a server that fails to start, or one that hangs can NEVER block or slow a build.
A server that misbehaves is simply absent from the registry (logged at debug);
nothing here ever raises into the caller.

The official ``mcp`` Python SDK is imported lazily inside the worker so the
dependency is optional: without it the registry is empty and the manager is a
no-op. stdio (subprocess), streamable-http, and sse transports are supported;
the last two connect to a remote ``url`` with optional auth headers, which is how
misterdev reaches a hosted MCP gateway (e.g. Glama) that fronts many servers.
"""

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from misterdev.core.execution.bounded import run_bounded
from misterdev.logging_setup import setup_logger

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
        transport = (entry.get("transport") or "stdio").lower()
        remote = transport in ("http", "streamable-http", "streamable_http", "sse")
        if remote:
            # A remote server (e.g. a hosted MCP gateway) is addressed by url.
            if not name or not entry.get("url"):
                logger.warning(
                    f"Ignoring remote mcp server without name/url: {entry!r}"
                )
                continue
        elif transport == "stdio":
            if not name or not entry.get("command"):
                logger.warning(
                    f"Ignoring stdio mcp server without name/command: {entry!r}"
                )
                continue
        else:
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
        allow_tools: Optional[List[str]] = None,
    ):
        self.servers = _normalize_servers(servers)
        self.connect_timeout = float(connect_timeout)
        self.call_timeout = float(call_timeout)
        # Optional allowlist of callable tools (``server.tool`` or bare ``tool``).
        # None means allow all; an empty/populated set restricts which tools a
        # remote gateway (e.g. Glama, which can front many servers) may expose to
        # the model — the model can only see and call what you allow.
        self.allow_tools: Optional[set] = set(allow_tools) if allow_tools else None
        self._tools: Optional[List[MCPTool]] = None  # lazy, cached

    def _allowed(self, server: str, name: str) -> bool:
        if self.allow_tools is None:
            return True
        return f"{server}.{name}" in self.allow_tools or name in self.allow_tools

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
            discovered = run_bounded(
                lambda s=server: _list_tools(s),
                self.connect_timeout,
                default=[],
                what=f"MCP tool discovery ({name})",
            )
            for t in discovered:
                tool_name = t.get("name", "")
                if not self._allowed(name, tool_name):
                    continue
                tools.append(
                    MCPTool(
                        server=name,
                        name=tool_name,
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
        if not self._allowed(server, name):
            logger.warning(
                f"MCP call_tool: {server}.{name} not in allow_tools; refused."
            )
            return None
        logger.info(
            f"MCP call_tool: {server}.{name} args={list((arguments or {}).keys())}"
        )
        return run_bounded(
            lambda: _call_tool(cfg, name, arguments or {}),
            self.call_timeout,
            default=None,
            what=f"MCP tool call ({server}.{name})",
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


def _auth_headers(server: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Auth headers for a remote server: explicit ``headers`` plus a Bearer token
    read from ``api_key_env`` (so the token stays in the environment, never in
    config on disk). Returns None when there is nothing to send."""
    import os

    headers = {str(k): str(v) for k, v in (server.get("headers") or {}).items()}
    env_var = server.get("api_key_env")
    if env_var:
        token = os.environ.get(str(env_var))
        if token:
            headers.setdefault("Authorization", f"Bearer {token}")
        else:
            logger.warning(
                f"MCP server '{server.get('name')}': api_key_env {env_var!r} is unset."
            )
    return headers or None


@asynccontextmanager
async def _open_session(server: Dict[str, Any]):
    """Yield an initialized MCP ``ClientSession`` over the server's transport.

    stdio launches a subprocess; ``http``/``streamable-http`` and ``sse`` connect
    to a remote endpoint (``url``) with optional auth headers — this is what lets
    misterdev reach a hosted MCP gateway (e.g. Glama) that fronts many servers.
    The ``ClientSession`` handling is identical across transports.
    """
    from mcp import ClientSession

    transport = (server.get("transport") or "stdio").lower()
    if transport in ("http", "streamable-http", "streamable_http"):
        from mcp.client.streamable_http import streamablehttp_client

        async with streamablehttp_client(
            server["url"], headers=_auth_headers(server)
        ) as (read, write, _get_session_id):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session
    elif transport == "sse":
        from mcp.client.sse import sse_client

        async with sse_client(server["url"], headers=_auth_headers(server)) as (
            read,
            write,
        ):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session
    else:
        from mcp.client.stdio import stdio_client

        async with stdio_client(_server_params(server)) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session


def _list_tools(server: Dict[str, Any]) -> List[Dict[str, Any]]:
    import asyncio

    async def _main() -> List[Dict[str, Any]]:
        async with _open_session(server) as session:
            result = await session.list_tools()
            return [
                {
                    "name": tool.name,
                    "description": tool.description or "",
                    "input_schema": getattr(tool, "inputSchema", None) or {},
                }
                for tool in result.tools
            ]

    return asyncio.run(_main())


def _call_tool(
    server: Dict[str, Any], name: str, arguments: Dict[str, Any]
) -> Optional[str]:
    import asyncio

    async def _main() -> Optional[str]:
        async with _open_session(server) as session:
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
