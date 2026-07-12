"""Free, on-the-fly MCP-server discovery over the official registry.

Given a capability query (e.g. ``"fetch web pages"``), search the **official MCP
Registry** (``registry.modelcontextprotocol.io`` — public, no auth, no payment)
and map a locally-runnable entry to a stdio server config the existing
:class:`~misterdev.core.integration.mcp.MCPManager` can spawn via ``npx``/``uvx``.
The whole MCP ecosystem is free this way: servers ship as npm/PyPI packages that
run as a local subprocess — the paid hosted gateways are only a convenience.

Because auto-installing and running a package from the internet is arbitrary
code execution with the build's privileges, admission goes through a **trust
ladder** rather than a single on/off gate:

1. **Curated tier.** A shipped, version-pinned allowlist (``curated_servers.json``,
   produced by the reproducible audit in ``scripts/audit_mcp_servers.py``) that
   auto-passes with no network call — the safest, fastest path and a usable
   default. Pinning ``pkg@version`` matters: allowlisting a bare name would not
   stop a malicious *future* release.
2. **Trust-signal scoring.** Anything not curated is scored on real signals —
   npm monthly downloads, GitHub stars, recency, and archived/inactive status —
   and admitted only above ``min_trust``. This reaches the long tail without a
   hand-maintained list. A named ``trusted_namespaces`` prefix is a strong boost.
3. **Permissive.** ``trusted_namespaces=["*"]`` trusts everything (logged
   loudly); still structurally gated but no quality bar. Never the default.

Across all tiers: only free, self-contained stdio packages provision (paid
remotes and secret-requiring packages are skipped — they could not run for
free anyway); the config carries no ``env`` so the subprocess gets a minimal
environment, never the build's secrets; and every network read is bounded and
never raises (any failure degrades to "discovered nothing" / "no signal").
"""

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from misterdev.core.execution.bounded import run_bounded
from misterdev.logging_setup import setup_logger

logger = setup_logger(__name__)

_REGISTRY_URL = "https://registry.modelcontextprotocol.io/v0/servers"
_DEFAULT_TIMEOUT = 10.0
_DEFAULT_MAX_SERVERS = 3
_DEFAULT_MIN_TRUST = 0.5
# Cap network scoring per build so a wide search cannot hammer the GitHub API
# (unauth: 60 req/hr) or stall the build on the long tail.
_DEFAULT_MAX_EVALUATIONS = 12

# A named prefix is a strong signal (the publisher is one we recognise), but no
# longer the ONLY key — signals can admit an unlisted server on their own.
DEFAULT_TRUSTED_NAMESPACES = (
    "io.github.modelcontextprotocol",
    "io.modelcontextprotocol",
    "com.modelcontextprotocol",
)

# npm -> npx, PyPI -> uvx. Anything else is not auto-runnable here.
_RUNTIME_BY_REGISTRY = {"npm": "npx", "pypi": "uvx"}

_CURATED_PATH = Path(__file__).with_name("curated_servers.json")

# Whole-word tokenizer for capability matching (substring match spuriously fires
# — e.g. "and" inside "commander"). Stopwords keep short filler from matching.
_WORD_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "for",
        "with",
        "to",
        "of",
        "in",
        "on",
        "my",
        "me",
        "i",
    }
)


def _tokenize(text: str) -> set:
    return set(_WORD_RE.findall((text or "").lower()))


# --- trust-score weights (transparent + tunable) --------------------------------
_NAMESPACE_BOOST = 0.5
_DOWNLOAD_TIERS = ((10000, 0.5), (1000, 0.35), (100, 0.2))
_STAR_TIERS = ((1000, 0.4), (200, 0.3), (50, 0.15))
_RECENCY_TIERS = ((180, 0.15), (365, 0.1))  # days since last push


# ==============================================================================
# Registry search
# ==============================================================================
def _http_get_json(url: str, timeout: float) -> Optional[Any]:
    """GET ``url`` and parse JSON, or return None. Never raises."""

    def _work() -> Optional[Any]:
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
            meta = (entry.get("_meta") or {}).get(
                "io.modelcontextprotocol.registry/official"
            ) or {}
            server = {
                **server,
                "_status": meta.get("status"),
                "_is_latest": meta.get("isLatest", True),
            }
            out.append(server)
    return out


# ==============================================================================
# Structural runnability (a hard prerequisite for every tier)
# ==============================================================================
def _package_needs_secret(pkg: Dict[str, Any]) -> bool:
    for ev in pkg.get("environmentVariables") or []:
        if ev.get("isRequired") or ev.get("isSecret"):
            return True
    return False


def _positional_values(items) -> List[str]:
    vals: List[str] = []
    for it in items or []:
        if (
            it.get("type") in (None, "positional")
            and it.get("value")
            and not it.get("variables")
        ):
            vals.append(str(it["value"]))
    return vals


def _runnable_package(server: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """First npm/PyPI stdio package that needs no secret to start, or None."""
    for pkg in server.get("packages") or []:
        registry_type = (pkg.get("registryType") or "").lower()
        if registry_type not in _RUNTIME_BY_REGISTRY:
            continue
        if (pkg.get("transport") or {}).get("type") != "stdio":
            continue
        if not pkg.get("identifier") or _package_needs_secret(pkg):
            continue
        return pkg
    return None


def _pinned_identifier(pkg: Dict[str, Any], runtime_hint: str) -> str:
    """``pkg@version`` (npx) / ``pkg==version`` (uvx) when a version is known."""
    identifier = pkg["identifier"]
    version = pkg.get("version")
    if not version:
        return identifier
    return (
        f"{identifier}=={version}"
        if runtime_hint == "uvx"
        else f"{identifier}@{version}"
    )


def _to_config(server: Dict[str, Any], pkg: Dict[str, Any]) -> Dict[str, Any]:
    runtime = _RUNTIME_BY_REGISTRY[(pkg.get("registryType") or "").lower()]
    runtime_hint = pkg.get("runtimeHint") or runtime
    args = _positional_values(pkg.get("runtimeArguments"))
    if runtime_hint == "npx" and "-y" not in args:
        args = ["-y", *args]
    args = [
        *args,
        _pinned_identifier(pkg, runtime_hint),
        *_positional_values(pkg.get("packageArguments")),
    ]
    return {
        "name": server["name"].replace("/", "."),
        "transport": "stdio",
        "command": runtime_hint,
        "args": args,
        "_discovered": True,
    }


# ==============================================================================
# Trust signals + scoring
# ==============================================================================
@dataclass
class TrustSignals:
    npm_downloads_month: Optional[int] = None
    github_stars: Optional[int] = None
    github_pushed_days: Optional[int] = None
    archived: bool = False


def _parse_github(url: str) -> Optional[str]:
    if not url or "github.com" not in url:
        return None
    tail = url.split("github.com/", 1)[1].strip("/")
    parts = tail.split("/")
    if len(parts) < 2 or not parts[0] or not parts[1]:
        return None
    return f"{parts[0]}/{parts[1].removesuffix('.git')}"


def _github_signals(repo_url: str, timeout: float) -> Dict[str, Any]:
    slug = _parse_github(repo_url or "")
    if not slug:
        return {}
    data = _http_get_json(f"https://api.github.com/repos/{slug}", timeout)
    if not isinstance(data, dict):
        return {}
    out: Dict[str, Any] = {
        "stars": data.get("stargazers_count"),
        "archived": bool(data.get("archived")),
    }
    pushed = data.get("pushed_at")
    if isinstance(pushed, str):
        try:
            dt = datetime.fromisoformat(pushed.replace("Z", "+00:00"))
            out["pushed_days"] = (datetime.now(timezone.utc) - dt).days
        except ValueError:
            pass
    return out


def _npm_downloads(pkg_id: str, timeout: float) -> Optional[int]:
    quoted = urllib.parse.quote(pkg_id, safe="@")
    data = _http_get_json(
        f"https://api.npmjs.org/downloads/point/last-month/{quoted}", timeout
    )
    if isinstance(data, dict) and isinstance(data.get("downloads"), int):
        return data["downloads"]
    return None


def gather_signals(
    server: Dict[str, Any], pkg: Dict[str, Any], timeout: float = _DEFAULT_TIMEOUT
) -> TrustSignals:
    """Best-effort live trust signals for a candidate. Never raises."""
    sig = TrustSignals()
    gh = _github_signals((server.get("repository") or {}).get("url", ""), timeout)
    sig.github_stars = gh.get("stars")
    sig.github_pushed_days = gh.get("pushed_days")
    sig.archived = bool(gh.get("archived"))
    if (pkg.get("registryType") or "").lower() == "npm":
        sig.npm_downloads_month = _npm_downloads(pkg["identifier"], timeout)
    return sig


def _tier_score(value: Optional[int], tiers) -> float:
    if value is None:
        return 0.0
    for threshold, points in tiers:
        if value >= threshold:
            return points
    return 0.0


def _in_namespace(name: str, trusted_namespaces) -> bool:
    return any(
        name == ns or name.startswith(ns + "/") for ns in trusted_namespaces or ()
    )


def score_server(
    server: Dict[str, Any], signals: TrustSignals, trusted_namespaces=()
) -> float:
    """Trust score in ``0..1``. Hard-zero on inactive/superseded/archived."""
    if server.get("_status") not in (None, "active"):
        return 0.0
    if server.get("_is_latest") is False or signals.archived:
        return 0.0
    score = 0.0
    if _in_namespace(server.get("name") or "", trusted_namespaces):
        score += _NAMESPACE_BOOST
    score += _tier_score(signals.npm_downloads_month, _DOWNLOAD_TIERS)
    score += _tier_score(signals.github_stars, _STAR_TIERS)
    score += _tier_score(signals.github_pushed_days, _RECENCY_TIERS)
    return min(score, 1.0)


# ==============================================================================
# Curated tier (the vetted, tiered core — see mcp_catalog)
# ==============================================================================
def load_catalog(path: Path = _CURATED_PATH) -> List[Dict[str, Any]]:
    """The curated catalog: the version-pinned ``curated_servers.json`` produced
    by the audit if present, else the in-repo :data:`mcp_catalog.CATALOG`."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        servers = raw.get("servers")
        if isinstance(servers, list) and servers:
            return servers
    except (OSError, ValueError):
        pass
    from misterdev.core.integration.mcp_catalog import CATALOG

    return CATALOG


def select_curated(
    tiers=("core",),
    env: Optional[Dict[str, str]] = None,
    catalog: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Curated stdio configs for the requested tiers.

    Admits only entries that are locally runnable (a ``command`` is set — not a
    manual Go/Docker/OAuth entry) AND whose ``requires`` env vars are all
    present, so a keyed server (e.g. Postgres needing ``DATABASE_URI``) mounts
    only when it can actually run. ``tiers`` accepts ``"all"`` for every tier.
    """
    if env is None:
        import os

        env = dict(os.environ)
    if catalog is None:
        catalog = load_catalog()
    want = None if "all" in tiers else set(tiers)
    out: List[Dict[str, Any]] = []
    for entry in catalog:
        if want is not None and entry.get("tier") not in want:
            continue
        command = entry.get("command")
        if not command or not isinstance(entry.get("args"), list):
            continue  # manual/remote entry — recorded, not auto-mounted
        missing = [v for v in entry.get("requires") or [] if not env.get(v)]
        if missing:
            logger.info(f"MCP curated: '{entry['name']}' needs {missing}; not mounting")
            continue
        out.append(
            {
                "name": entry["name"],
                "transport": "stdio",
                "command": command,
                "args": entry["args"],
                "_curated": True,
            }
        )
    return out


def provide_capability(
    query: str,
    trusted_namespaces=DEFAULT_TRUSTED_NAMESPACES,
    min_trust: float = _DEFAULT_MIN_TRUST,
    env: Optional[Dict[str, str]] = None,
) -> Optional[Dict[str, Any]]:
    """Resolve a capability query to ONE runnable stdio server config, or None.

    The mid-task on-demand path: try the vetted catalog first (keyword match on
    name/category/why, config-gated), then fall back to trust-scored registry
    discovery. Returns a single config the caller mounts live; never raises.
    """
    terms = {t for t in _tokenize(query) if t not in _STOPWORDS}
    if not terms:
        return None
    catalog = load_catalog()
    best, best_score = None, 0
    for cfg in select_curated(("all",), env=env, catalog=catalog):
        entry = next((e for e in catalog if e["name"] == cfg["name"]), {})
        words = _tokenize(
            f"{cfg['name']} {entry.get('category', '')} {entry.get('why', '')}"
        )
        overlap = len(terms & words)  # whole-word overlap, not substring
        if overlap > best_score:
            best, best_score = cfg, overlap
    if best is not None:
        logger.info(f"MCP on-demand: '{best['name']}' matches {query!r} (curated)")
        return best
    found = discover_servers(
        [query],
        trusted_namespaces=trusted_namespaces,
        max_servers=1,
        min_trust=min_trust,
    )
    return found[0] if found else None


# ==============================================================================
# Discovery (the trust ladder)
# ==============================================================================
def to_stdio_config(
    server: Dict[str, Any],
    trusted_namespaces=DEFAULT_TRUSTED_NAMESPACES,
    min_trust: float = _DEFAULT_MIN_TRUST,
    signals: Optional[TrustSignals] = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> Optional[Dict[str, Any]]:
    """Map a registry ``server`` to a stdio config if runnable AND trusted enough.

    Trust is signal-scored (``min_trust``), with ``["*"]`` bypassing the quality
    bar. ``signals`` may be injected (tests / pre-gathered); otherwise gathered
    live for the runnable candidate only.
    """
    pkg = _runnable_package(server)
    if not pkg:
        return None
    permissive = "*" in (trusted_namespaces or ())
    if not permissive:
        if signals is None:
            signals = gather_signals(server, pkg, timeout)
        score = score_server(server, signals, trusted_namespaces)
        if score < min_trust:
            logger.info(
                f"MCP discovery: '{server['name']}' scored {score:.2f} < "
                f"{min_trust:.2f}; skipping (downloads="
                f"{signals.npm_downloads_month}, stars={signals.github_stars})"
            )
            return None
    elif server.get("_status") not in (None, "active"):
        return None
    return _to_config(server, pkg)


def discover_servers(
    queries: List[str],
    trusted_namespaces=DEFAULT_TRUSTED_NAMESPACES,
    max_servers: int = _DEFAULT_MAX_SERVERS,
    min_trust: float = _DEFAULT_MIN_TRUST,
    per_query_limit: int = 10,
    max_evaluations: int = _DEFAULT_MAX_EVALUATIONS,
    timeout: float = _DEFAULT_TIMEOUT,
) -> List[Dict[str, Any]]:
    """Search each query and admit the long-tail servers by trust score.

    Signal-scored up to ``max_evaluations`` network checks; deduped by name;
    capped at ``max_servers``. (The vetted core mounts separately and by name
    via :func:`select_curated`, so discovery is purely the signal tier.)
    """
    if not queries:
        return []
    if "*" in (trusted_namespaces or ()):
        logger.warning(
            "MCP discovery is in PERMISSIVE trust mode: any registry server may "
            "be installed and run locally. This is arbitrary code execution."
        )
    seen: set = set()
    out: List[Dict[str, Any]] = []
    evaluations = 0
    for q in queries:
        if len(out) >= max_servers:
            break
        for server in search_registry(q, limit=per_query_limit, timeout=timeout):
            name = server.get("name") or ""
            if name in seen or evaluations >= max_evaluations:
                continue
            evaluations += 1
            cfg = to_stdio_config(
                server, trusted_namespaces, min_trust, timeout=timeout
            )
            if not cfg:
                continue
            seen.add(name)
            out.append(cfg)
            logger.info(
                f"MCP discovery: provisioning '{cfg['name']}' "
                f"({cfg['command']} {' '.join(cfg['args'])}) for query {q!r}"
            )
            if len(out) >= max_servers:
                break
    if not out:
        logger.info(f"MCP discovery: no admissible server for {queries!r}")
    return out
