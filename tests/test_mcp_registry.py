"""Free MCP-server discovery: trust-gated mapping of registry entries to stdio
configs. Pure mapping + a mocked search (no network, no subprocess)."""

import misterdev.core.integration.mcp_registry as reg
from misterdev.core.integration.mcp_registry import (
    discover_servers,
    search_registry,
    to_stdio_config,
)


def _npm_server(name, identifier, needs_secret=False, status="active"):
    pkg = {
        "registryType": "npm",
        "identifier": identifier,
        "transport": {"type": "stdio"},
        "runtimeHint": "npx",
        "runtimeArguments": [{"value": "-y", "type": "positional"}],
    }
    if needs_secret:
        pkg["environmentVariables"] = [{"name": "API_KEY", "isRequired": True}]
    return {"name": name, "_status": status, "packages": [pkg]}


def _remote_only_server(name):
    # A paid hosted remote (Smithery et al.) — must never auto-provision.
    return {
        "name": name,
        "_status": "active",
        "remotes": [
            {
                "type": "streamable-http",
                "url": "https://server.smithery.ai/x/mcp",
                "headers": [
                    {"name": "Authorization", "value": "Bearer {smithery_api_key}"}
                ],
            }
        ],
    }


def test_maps_trusted_npm_to_npx_stdio():
    s = _npm_server("io.github.modelcontextprotocol/fetch", "server-fetch")
    cfg = to_stdio_config(s)  # default trusted namespaces include official
    assert cfg["transport"] == "stdio"
    assert cfg["command"] == "npx"
    assert cfg["args"] == ["-y", "server-fetch"]
    assert cfg["name"] == "io.github.modelcontextprotocol.fetch"  # '/' -> '.'
    assert cfg["_discovered"] is True
    assert "env" not in cfg  # minimal env -> no build secrets


def test_untrusted_namespace_is_skipped_by_default():
    s = _npm_server("com.randovendor/sketchy", "sketchy-mcp")
    assert to_stdio_config(s) is None  # not in the default trusted set


def test_permissive_star_trusts_any_namespace():
    s = _npm_server("com.randovendor/sketchy", "sketchy-mcp")
    assert to_stdio_config(s, trusted_namespaces=["*"]) is not None


def test_package_requiring_a_secret_is_skipped():
    s = _npm_server("io.github.modelcontextprotocol/db", "db-mcp", needs_secret=True)
    assert to_stdio_config(s) is None  # cannot run without the secret; do not try


def test_inactive_server_is_skipped():
    s = _npm_server("io.github.modelcontextprotocol/old", "old-mcp", status="deleted")
    assert to_stdio_config(s) is None


def test_remote_only_server_is_skipped():
    assert to_stdio_config(_remote_only_server("ai.smithery/x"), ["*"]) is None


def test_pypi_maps_to_uvx():
    s = {
        "name": "io.github.modelcontextprotocol/py",
        "_status": "active",
        "packages": [
            {
                "registryType": "pypi",
                "identifier": "py-mcp-server",
                "transport": {"type": "stdio"},
            }
        ],
    }
    cfg = to_stdio_config(s)
    assert cfg["command"] == "uvx" and cfg["args"] == ["py-mcp-server"]


def test_discover_servers_dedups_and_caps(monkeypatch):
    a = _npm_server("io.github.modelcontextprotocol/a", "a-mcp")
    b = _npm_server("io.github.modelcontextprotocol/b", "b-mcp")
    # Same name appears twice across queries -> deduped; cap holds.
    monkeypatch.setattr(reg, "search_registry", lambda q, **k: [a, b, a])
    got = discover_servers(["x", "y"], max_servers=1)
    assert len(got) == 1 and got[0]["name"] == "io.github.modelcontextprotocol.a"


def test_search_registry_never_raises_on_bad_payload(monkeypatch):
    monkeypatch.setattr(reg, "_http_get_json", lambda url, timeout: None)
    assert search_registry("anything") == []


def test_search_registry_empty_query_is_noop():
    assert search_registry("   ") == []
