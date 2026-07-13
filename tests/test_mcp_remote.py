"""Remote (streamable-http / SSE) MCP transport — e.g. a hosted Glama gateway."""

import types

from misterdev.core.integration import mcp


def test_normalize_accepts_remote_server_with_url():
    servers = mcp._normalize_servers(
        [
            {"name": "glama", "transport": "http", "url": "https://gw.example/mcp"},
            {"name": "local", "command": "run-server"},
        ]
    )
    assert {s["name"] for s in servers} == {"glama", "local"}


def test_normalize_rejects_remote_server_without_url():
    servers = mcp._normalize_servers([{"name": "glama", "transport": "http"}])
    assert servers == []


def test_normalize_rejects_unknown_transport():
    servers = mcp._normalize_servers(
        [{"name": "x", "transport": "carrier-pigeon", "url": "u"}]
    )
    assert servers == []


def test_auth_headers_builds_bearer_from_env(monkeypatch):
    monkeypatch.setenv("GLAMA_API_KEY", "secret-token")
    headers = mcp._auth_headers(
        {"name": "glama", "api_key_env": "GLAMA_API_KEY", "headers": {"X-Env": "prod"}}
    )
    assert headers["Authorization"] == "Bearer secret-token"
    assert headers["X-Env"] == "prod"


def test_auth_headers_none_when_nothing_to_send(monkeypatch):
    monkeypatch.delenv("GLAMA_API_KEY", raising=False)
    assert mcp._auth_headers({"name": "g", "api_key_env": "GLAMA_API_KEY"}) is None
    assert mcp._auth_headers({"name": "g"}) is None


def test_open_session_selects_streamable_http_transport(monkeypatch):
    captured = {}

    class _FakeTransport:
        def __init__(self, url, headers=None):
            captured["url"] = url
            captured["headers"] = headers

        async def __aenter__(self):
            async def _sid():
                return "sid"

            return (None, None, _sid)  # streamable-http yields a 3-tuple

        async def __aexit__(self, *a):
            return False

    class _FakeSession:
        def __init__(self, read, write):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def initialize(self):
            captured["initialized"] = True

        async def list_tools(self):
            return types.SimpleNamespace(tools=[])

    monkeypatch.setattr(
        "mcp.client.streamable_http.streamablehttp_client",
        lambda url, headers=None: _FakeTransport(url, headers),
    )
    monkeypatch.setattr("mcp.ClientSession", _FakeSession)

    out = mcp._list_tools(
        {"name": "glama", "transport": "http", "url": "https://gw.example/mcp"}
    )
    assert out == []
    assert captured["url"] == "https://gw.example/mcp"
    assert captured["initialized"] is True


def _remote(name="glama"):
    return {"name": name, "transport": "http", "url": "https://gw.example/mcp"}


def test_allow_tools_filters_discovery(monkeypatch):
    monkeypatch.setattr(
        mcp, "_list_tools", lambda s, timeout=0: [{"name": "read"}, {"name": "delete"}]
    )
    m = mcp.MCPManager([_remote()], allow_tools=["glama.read"])
    assert {t.name for t in m.tools} == {"read"}


def test_allow_tools_refuses_disallowed_call(monkeypatch):
    seen = {}

    def _fake_call(s, n, a, timeout=0):
        seen["name"] = n
        return "result"

    monkeypatch.setattr(mcp, "_call_tool", _fake_call)
    m = mcp.MCPManager([_remote()], allow_tools=["glama.read"])
    # A disallowed tool is refused before the transport is ever touched.
    assert m.call_tool("glama", "delete", {}) is None
    assert "name" not in seen
    # An allowed tool goes through.
    assert m.call_tool("glama", "read", {}) == "result"


def test_no_allowlist_allows_all(monkeypatch):
    monkeypatch.setattr(mcp, "_list_tools", lambda s, timeout=0: [{"name": "anything"}])
    m = mcp.MCPManager([_remote()])  # allow_tools=None -> unrestricted
    assert {t.name for t in m.tools} == {"anything"}
