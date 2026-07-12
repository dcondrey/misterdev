"""MCP discovery trust ladder: structural runnability, signal scoring, and the
curated/tiered selection. Pure logic + injected signals (no network)."""

import misterdev.core.integration.mcp_registry as reg
from misterdev.core.integration.mcp_registry import (
    TrustSignals,
    discover_servers,
    load_catalog,
    score_server,
    search_registry,
    select_curated,
    to_stdio_config,
)


def _npm_server(name, identifier, needs_secret=False, status="active", is_latest=True):
    pkg = {
        "registryType": "npm",
        "identifier": identifier,
        "transport": {"type": "stdio"},
        "runtimeHint": "npx",
        "runtimeArguments": [{"value": "-y", "type": "positional"}],
    }
    if needs_secret:
        pkg["environmentVariables"] = [{"name": "API_KEY", "isRequired": True}]
    return {"name": name, "_status": status, "_is_latest": is_latest, "packages": [pkg]}


# --- structural runnability ---------------------------------------------------
def test_runnable_package_skips_remote_only():
    remote = {"name": "x", "remotes": [{"type": "streamable-http", "url": "https://x"}]}
    assert reg._runnable_package(remote) is None


def test_runnable_package_skips_secret_requiring():
    assert (
        reg._runnable_package(_npm_server("io.x/db", "db", needs_secret=True)) is None
    )


def test_maps_npm_to_npx_stdio():
    cfg = to_stdio_config(
        _npm_server("io.x/fetch", "server-fetch"),
        trusted_namespaces=["*"],  # permissive -> no scoring, just structure
    )
    assert cfg["command"] == "npx" and cfg["args"] == ["-y", "server-fetch"]
    assert cfg["name"] == "io.x.fetch"


# --- trust scoring ------------------------------------------------------------
def test_score_high_adoption_passes():
    s = _npm_server("io.x/a", "a")
    sig = TrustSignals(
        npm_downloads_month=50000, github_stars=3000, github_pushed_days=30
    )
    assert score_server(s, sig) >= 0.9


def test_score_obscure_is_low():
    s = _npm_server("io.x/a", "a")
    sig = TrustSignals(npm_downloads_month=5, github_stars=2, github_pushed_days=900)
    assert score_server(s, sig) < 0.5


def test_score_inactive_or_archived_is_zero():
    s = _npm_server("io.x/a", "a", status="deleted")
    assert score_server(s, TrustSignals(github_stars=99999)) == 0.0
    s2 = _npm_server("io.x/a", "a")
    assert score_server(s2, TrustSignals(github_stars=99999, archived=True)) == 0.0


def test_namespace_is_a_boost_not_the_only_key():
    s = _npm_server("io.github.modelcontextprotocol/x", "x")
    boosted = score_server(
        s, TrustSignals(), trusted_namespaces=["io.github.modelcontextprotocol"]
    )
    plain = score_server(s, TrustSignals(), trusted_namespaces=[])
    assert boosted > plain and boosted >= reg._NAMESPACE_BOOST


# --- admission via to_stdio_config (scored) -----------------------------------
def test_admits_when_score_meets_threshold():
    s = _npm_server("io.x/good", "good")
    sig = TrustSignals(
        npm_downloads_month=20000, github_stars=1500, github_pushed_days=10
    )
    assert (
        to_stdio_config(s, trusted_namespaces=[], min_trust=0.5, signals=sig)
        is not None
    )


def test_rejects_when_score_below_threshold():
    s = _npm_server("io.x/obscure", "obscure")
    sig = TrustSignals(npm_downloads_month=3, github_stars=1)
    assert to_stdio_config(s, trusted_namespaces=[], min_trust=0.5, signals=sig) is None


def test_permissive_bypasses_quality_bar():
    s = _npm_server("com.rando/anything", "anything")
    assert (
        to_stdio_config(s, trusted_namespaces=["*"], signals=TrustSignals()) is not None
    )


# --- curated / tiered selection ----------------------------------------------
_CATALOG = [
    {
        "name": "core-free",
        "tier": "core",
        "command": "npx",
        "args": ["-y", "a"],
        "requires": [],
    },
    {
        "name": "core-keyed",
        "tier": "core",
        "command": "npx",
        "args": ["-y", "b"],
        "requires": ["B_KEY"],
    },
    {
        "name": "task-free",
        "tier": "task",
        "command": "uvx",
        "args": ["c"],
        "requires": [],
    },
    {"name": "manual-one", "tier": "core", "command": None, "args": [], "requires": []},
]


def test_select_curated_tier_filter():
    got = {c["name"] for c in select_curated(("core",), env={}, catalog=_CATALOG)}
    assert got == {
        "core-free"
    }  # keyed excluded (no env), manual excluded, task excluded


def test_select_curated_activates_keyed_when_env_present():
    got = {
        c["name"]
        for c in select_curated(("core",), env={"B_KEY": "x"}, catalog=_CATALOG)
    }
    assert got == {"core-free", "core-keyed"}


def test_select_curated_all_tiers():
    got = {c["name"] for c in select_curated(("all",), env={}, catalog=_CATALOG)}
    assert got == {"core-free", "task-free"}  # every tier, still config-gated


def test_load_catalog_falls_back_to_module(tmp_path):
    cat = load_catalog(path=tmp_path / "missing.json")
    assert any(e["name"] == "git" for e in cat)  # the in-repo CATALOG


# --- discovery orchestration --------------------------------------------------
def test_discover_dedups_and_caps(monkeypatch):
    a = _npm_server("io.x/a", "a")
    monkeypatch.setattr(reg, "search_registry", lambda q, **k: [a, a])
    monkeypatch.setattr(
        reg,
        "gather_signals",
        lambda s, p, timeout=10.0: TrustSignals(
            npm_downloads_month=50000, github_stars=5000, github_pushed_days=10
        ),
    )
    got = discover_servers(["x", "y"], max_servers=1)
    assert len(got) == 1 and got[0]["name"] == "io.x.a"


def test_search_registry_never_raises(monkeypatch):
    monkeypatch.setattr(reg, "_http_get_json", lambda url, timeout: None)
    assert search_registry("anything") == []
