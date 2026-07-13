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
        lambda s, p, timeout=10.0, cache=None: TrustSignals(
            npm_downloads_month=50000, github_stars=5000, github_pushed_days=10
        ),
    )
    got = discover_servers(["x", "y"], max_servers=1)
    assert len(got) == 1 and got[0]["name"] == "io.x.a"


def test_search_registry_never_raises(monkeypatch):
    monkeypatch.setattr(reg, "_http_get_json", lambda url, timeout, headers=None: None)
    assert search_registry("anything") == []


# --- stemming (curated match reaches plurals / nominalisations) ---------------
def test_stem_normalizes_plurals_and_gerunds():
    assert reg._stem("timezones") == reg._stem("timezone")
    assert reg._stem("searches") == reg._stem("search")
    assert reg._stem("fetching") == reg._stem("fetch")
    assert reg._stem("access") == "access"  # -ss is not a plural


def test_provide_capability_matches_curated_via_stem(monkeypatch):
    # "convert timezones" must resolve to the curated `time` entry (why:
    # "Timezone conversion, current time") instead of falling to discovery.
    monkeypatch.setattr(
        reg, "discover_servers", lambda *a, **k: [{"name": "SHOULD_NOT_REACH"}]
    )
    cfg = reg.provide_capability("convert timezones", env={})
    assert cfg is not None and cfg["name"] == "time"


# --- persistent cache ---------------------------------------------------------
def test_registry_cache_roundtrip_and_ttl(tmp_path):
    c = reg.RegistryCache(tmp_path / "c.json", ttl=1000)
    assert c.get("k") is None
    c.put("k", {"v": 1})
    assert c.get("k") == {"v": 1}
    reg.RegistryCache(tmp_path / "c.json", ttl=1000)  # persisted across instances
    assert reg.RegistryCache(tmp_path / "c.json", ttl=1000).get("k") == {"v": 1}
    assert reg.RegistryCache(tmp_path / "c.json", ttl=-1).get("k") is None  # expired


def test_registry_cache_concurrent_instances_preserve_entries(tmp_path):
    # Two instances on the same file (the manager is shared, the cache is built
    # per call site): each write merges the other's entries and swaps the file
    # atomically, so neither clobbers the other and the JSON is never torn.
    import threading

    path = tmp_path / "c.json"
    a = reg.RegistryCache(path)
    b = reg.RegistryCache(path)

    def _writer(cache, prefix):
        for i in range(50):
            cache.put(f"{prefix}{i}", {"i": i})

    ta = threading.Thread(target=_writer, args=(a, "a"))
    tb = threading.Thread(target=_writer, args=(b, "b"))
    ta.start()
    tb.start()
    ta.join()
    tb.join()
    final = reg.RegistryCache(path)  # re-read from disk: valid JSON, both merged
    assert final.get("a49") == {"i": 49}
    assert final.get("b49") == {"i": 49}


def test_gather_signals_uses_cache(monkeypatch, tmp_path):
    calls = {"n": 0}

    def _gh(url, timeout, headers=None):
        calls["n"] += 1
        return {"stargazers_count": 1234, "archived": False}

    monkeypatch.setattr(reg, "_http_get_json", _gh)
    cache = reg.RegistryCache(tmp_path / "c.json")
    server = {"name": "io.x/a", "repository": {"url": "https://github.com/x/a"}}
    pkg = {"registryType": "pypi", "identifier": "a", "version": "1.0"}
    s1 = reg.gather_signals(server, pkg, cache=cache)
    s2 = reg.gather_signals(server, pkg, cache=cache)
    assert s1.github_stars == 1234 and s2.github_stars == 1234
    assert calls["n"] == 1  # second call served from cache, no network


def test_search_caches_results(monkeypatch, tmp_path):
    calls = {"n": 0}

    def _http(url, timeout, headers=None):
        calls["n"] += 1
        return {"servers": [{"server": {"name": "io.x/a"}}]}

    monkeypatch.setattr(reg, "_http_get_json", _http)
    cache = reg.RegistryCache(tmp_path / "c.json")
    search_registry("web", cache=cache)
    search_registry("web", cache=cache)
    assert calls["n"] == 1


# --- identity dedup against the already-mounted stack -------------------------
def test_discover_skips_known_identity(monkeypatch):
    s = _npm_server("io.x/fetch", "server-fetch")
    monkeypatch.setattr(reg, "search_registry", lambda q, **k: [s])
    monkeypatch.setattr(
        reg,
        "gather_signals",
        lambda *a, **k: TrustSignals(
            npm_downloads_month=50000, github_stars=5000, github_pushed_days=10
        ),
    )
    # Pretend the same package is already mounted (curated) by its launch identity.
    known = {("stdio", "npx", "-y", "server-fetch")}
    assert discover_servers(["fetch"], known_identities=known) == []


# --- search re-ranks by query relevance --------------------------------------
def test_search_reranks_relevant_first(monkeypatch):
    servers = [
        {"server": {"name": "io.x/misc", "description": "an unrelated tool"}},
        {"server": {"name": "io.x/fetcher", "description": "fetch web pages"}},
    ]
    monkeypatch.setattr(
        reg, "_http_get_json", lambda url, timeout, headers=None: {"servers": servers}
    )
    out = search_registry("fetch web pages")
    assert out[0]["name"] == "io.x/fetcher"  # most on-topic first


# --- cgcone discovery backend -------------------------------------------------
def _cgcone_entry(
    name,
    identifier,
    command="npx",
    stars=5000,
    days=10,
    archived=False,
    env=None,
    server_type="stdio",
    tags=None,
    description="fetch web pages and content",
):
    from datetime import datetime, timedelta, timezone

    last = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    args = ["-y", identifier] if command == "npx" else [identifier]
    return {
        "name": name,
        "description": description,
        "category": "docs",
        "tags": tags or ["fetch", "web"],
        "serverType": server_type,
        "stars": stars,
        "isArchived": archived,
        "lastCommit": last,
        "installConfig": {"command": command, "args": args, "env": env or {}, "type": "npm"},
    }


def test_cgcone_config_admits_free_stdio_rejects_keyed_and_remote():
    assert reg._cgcone_config(_cgcone_entry("x/a", "a"))["command"] == "npx"
    assert reg._cgcone_config(_cgcone_entry("x/a", "a", env={"API_KEY": ""})) is None
    assert reg._cgcone_config(_cgcone_entry("x/a", "a", server_type="streamable-http")) is None
    assert reg._cgcone_config({"name": "x", "serverType": "stdio", "installConfig": {}}) is None


def test_cgcone_signals_map_from_index():
    sig = reg._cgcone_signals(_cgcone_entry("x/a", "a", stars=3000, days=5, archived=True))
    assert sig.github_stars == 3000 and sig.archived is True
    assert 0 <= sig.github_pushed_days <= 7


def test_discover_via_cgcone_admits_pins_and_ranks(monkeypatch):
    entries = [
        _cgcone_entry("io.x/fetcher", "server-fetch", stars=8000),
        _cgcone_entry("io.x/obscure", "obscure-pkg", stars=1, description="unrelated"),
    ]
    monkeypatch.setattr(reg, "fetch_cgcone_registry", lambda cache=None, timeout=10.0: entries)
    monkeypatch.setattr(
        reg, "_resolve_version", lambda rt, ident, t, c: "1.2.3"
    )
    out = reg.discover_via_cgcone(["fetch web pages"], max_servers=2)
    assert out and out[0]["name"] == "io.x.fetcher"
    assert out[0]["args"][-1] == "server-fetch@1.2.3"  # version-pinned
    assert out[0]["_source"] == "cgcone"


def test_discover_via_cgcone_drops_bogus_unresolvable_spec(monkeypatch):
    entries = [_cgcone_entry("io.x/bogus", "n8n-monorepo", stars=9000)]
    monkeypatch.setattr(reg, "fetch_cgcone_registry", lambda cache=None, timeout=10.0: entries)
    monkeypatch.setattr(reg, "_resolve_version", lambda rt, ident, t, c: "")  # doesn't resolve
    assert reg.discover_via_cgcone(["fetch web pages"], max_servers=2) == []


def test_discover_via_cgcone_skips_archived_by_score(monkeypatch):
    entries = [_cgcone_entry("io.x/dead", "dead-pkg", stars=9000, archived=True)]
    monkeypatch.setattr(reg, "fetch_cgcone_registry", lambda cache=None, timeout=10.0: entries)
    monkeypatch.setattr(reg, "_resolve_version", lambda rt, ident, t, c: "1.0.0")
    assert reg.discover_via_cgcone(["fetch web pages"], max_servers=2) == []


def test_discover_via_cgcone_dedups_known_identity(monkeypatch):
    entries = [_cgcone_entry("io.x/fetcher", "server-fetch", stars=8000)]
    monkeypatch.setattr(reg, "fetch_cgcone_registry", lambda cache=None, timeout=10.0: entries)
    monkeypatch.setattr(reg, "_resolve_version", lambda rt, ident, t, c: "1.2.3")
    known = {("stdio", "npx", "-y", "server-fetch@1.2.3")}
    assert reg.discover_via_cgcone(["fetch"], known_identities=known) == []


def test_discover_servers_routes_source_cgcone(monkeypatch):
    called = {"cgcone": 0, "official": 0}
    monkeypatch.setattr(
        reg, "discover_via_cgcone",
        lambda *a, **k: called.__setitem__("cgcone", called["cgcone"] + 1) or [],
    )
    monkeypatch.setattr(
        reg, "search_registry",
        lambda *a, **k: called.__setitem__("official", called["official"] + 1) or [],
    )
    discover_servers(["fetch"], source="cgcone")
    assert called == {"cgcone": 1, "official": 0}  # cgcone only, no official search


def test_fetch_cgcone_registry_never_raises(monkeypatch):
    monkeypatch.setattr(reg, "_http_get_json", lambda url, timeout: None)
    assert reg.fetch_cgcone_registry() == []
