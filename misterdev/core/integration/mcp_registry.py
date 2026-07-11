"""Free, on-the-fly MCP-server discovery over the official registry.

Given a capability query (e.g. ``"fetch web pages"``), search the **official MCP
Registry** (``registry.modelcontextprotocol.io`` — public, no auth, no payment)
and map a locally-runnable, TRUSTED entry to a stdio server config the existing
:class:`~misterdev.core.integration.mcp.MCPManager` can spawn via ``npx``/``uvx``.
The whole MCP ecosystem is free this way: servers ship as npm/PyPI packages that
run as a local subprocess — the paid hosted gateways are only a convenience.

Discipline mirrors the rest of the MCP substrate: read-only HTTP, hard-bounded,
and it NEVER raises (any failure degrades to "discovered nothing"). Provisioning
is deliberately conservative because auto-installing and running a package from
the internet is arbitrary code execution with the build's privileges:

- **Trust gate.** Only entries whose reverse-DNS name starts with a namespace in
  ``trusted_namespaces`` are auto-mapped. The default set is the official
  publishers; widen it in config to reach more of the ecosystem, or pass a
  permissive marker (``"*"``) to trust all — loudly, and never the default.
- **Free/self-contained only.** Paid *remotes* (Smithery et al. needing an API
  key) are skipped, as is any package that REQUIRES an env var / secret we can't
  supply — it could not run anyway, and we will not prompt a server for secrets.
- **Minimal env.** The mapped config carries no ``env``; the MCP SDK then spawns
  the subprocess with a minimal default environment, never the build's secrets.
- **Bounded count.** ``discover_servers`` caps how many servers a build admits.
"""

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

from misterdev.core.execution.bounded import run_bounded
from misterdev.logging_setup import setup_logger

logger = setup_logger(__name__)

_REGISTRY_URL = "https://registry.modelcontextprotocol.io/v0/servers"
_DEFAULT_TIMEOUT = 10.0
_DEFAULT_MAX_SERVERS = 3

# Conservative default: only the official publisher namespaces auto-provision.
# Reaching the *full* ecosystem is a deliberate, config-level opt-in (widen this
# list, or set it to ["*"]) — never the silent default, because each admitted
# server is code we run locally.
DEFAULT_TRUSTED_NAMESPACES = (
    "io.github.modelcontextprotocol",
    "io.modelcontextprotocol",
    "com.modelcontextprotocol",
)

# npm -> npx, PyPI -> uvx. Anything else is not auto-runnable here.
_RUNTIME_BY_REGISTRY = {"npm": "npx", "pypi": "uvx", "pyp/pip": "uvx"}


def _http_get_json(url: str, timeout: float) -> Optional[Dict[str, Any]]:
    """GET ``url`` and parse JSON, or return None. Never raises."""

    def _work() -> Optional[Dict[str, Any]]:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
                if resp.status != 200:
                    return None
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, ValueError, OSError) as e:
            logger.debug(f"MCP registry GET failed ({url}): {e}")
            return None

    return run_bounded(_work, timeout + 2, None, "MCP registry search")


def search_registry(
    query: str, limit: int = 10, timeout: float = _DEFAULT_TIMEOUT
) -> List[Dict[str, Any]]:
    """Search the official registry; return the raw ``server`` objects (or [])."""
    if not query or not query.strip():
        return []
    qs = urllib.parse.urlencode(
        {"search": query.strip(), "limit": max(1, min(limit, 100))}
    )
    payload = _http_get_json(f"{_REGISTRY_URL}?{qs}", timeout)
    if not isinstance(payload, dict):
        return []
    out: List[Dict[str, Any]] = []
    for entry in payload.get("servers") or []:
        server = entry.get("server") if isinstance(entry, dict) else None
        if isinstance(server, dict) and server.get("name"):
            # Carry the registry status forward so mapping can require "active".
            meta = (entry.get("_meta") or {}).get(
                "io.modelcontextprotocol.registry/official"
            ) or {}
            server = {**server, "_status": meta.get("status")}
            out.append(server)
    return out


def _is_trusted(name: str, trusted_namespaces) -> bool:
    if "*" in trusted_namespaces:  # explicit permissive opt-in
        return True
    return any(name == ns or name.startswith(ns + "/") for ns in trusted_namespaces)


def _package_needs_secret(pkg: Dict[str, Any]) -> bool:
    """True if the package REQUIRES an env var / secret we cannot supply."""
    for ev in pkg.get("environmentVariables") or []:
        if ev.get("isRequired") or ev.get("isSecret"):
            return True
    return False


def _positional_values(items) -> List[str]:
    """Extract literal positional argument values, skipping variable inputs."""
    vals: List[str] = []
    for it in items or []:
        # Only concrete literals with no user/variable substitution.
        if (
            it.get("type") in (None, "positional")
            and it.get("value")
            and not it.get("variables")
        ):
            vals.append(str(it["value"]))
    return vals


def to_stdio_config(
    server: Dict[str, Any], trusted_namespaces=DEFAULT_TRUSTED_NAMESPACES
) -> Optional[Dict[str, Any]]:
    """Map a registry ``server`` to a stdio MCPManager config, or None.

    Returns None (skip) unless the server is trusted, active, and exposes a
    locally-runnable npm/PyPI stdio package that needs no secret to start.
    """
    name = server.get("name") or ""
    if not _is_trusted(name, trusted_namespaces):
        return None
    if server.get("_status") not in (None, "active"):
        return None
    for pkg in server.get("packages") or []:
        registry_type = (pkg.get("registryType") or "").lower()
        runtime = _RUNTIME_BY_REGISTRY.get(registry_type)
        transport = (pkg.get("transport") or {}).get("type")
        identifier = pkg.get("identifier")
        if not runtime or transport != "stdio" or not identifier:
            continue
        if _package_needs_secret(pkg):
            continue
        runtime_hint = pkg.get("runtimeHint") or runtime
        args = _positional_values(pkg.get("runtimeArguments"))
        # npx needs -y to run without an install prompt; add it if the registry
        # entry did not already carry it.
        if runtime_hint == "npx" and "-y" not in args:
            args = ["-y", *args]
        args = [*args, identifier, *_positional_values(pkg.get("packageArguments"))]
        return {
            "name": name.replace("/", "."),
            "transport": "stdio",
            "command": runtime_hint,
            "args": args,
            "_discovered": True,
        }
    return None


def discover_servers(
    queries: List[str],
    trusted_namespaces=DEFAULT_TRUSTED_NAMESPACES,
    max_servers: int = _DEFAULT_MAX_SERVERS,
    per_query_limit: int = 10,
    timeout: float = _DEFAULT_TIMEOUT,
) -> List[Dict[str, Any]]:
    """Search each query, trust-map the hits, dedup by name, cap the total.

    Returns stdio server configs ready to hand to :class:`MCPManager`. Loud about
    what it admitted and (when permissive) about the risk it is taking.
    """
    if not queries:
        return []
    if "*" in trusted_namespaces:
        logger.warning(
            "MCP discovery is in PERMISSIVE trust mode: any registry server may "
            "be installed and run locally. This is arbitrary code execution."
        )
    seen: set = set()
    out: List[Dict[str, Any]] = []
    for q in queries:
        if len(out) >= max_servers:
            break
        for server in search_registry(q, limit=per_query_limit, timeout=timeout):
            cfg = to_stdio_config(server, trusted_namespaces)
            if not cfg or cfg["name"] in seen:
                continue
            seen.add(cfg["name"])
            out.append(cfg)
            logger.info(
                f"MCP discovery: provisioning '{cfg['name']}' "
                f"({cfg['command']} {' '.join(cfg['args'])}) for query {q!r}"
            )
            if len(out) >= max_servers:
                break
    if not out:
        logger.info(f"MCP discovery: no trusted runnable server for {queries!r}")
    return out
