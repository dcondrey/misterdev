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
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from misterdev.core.execution.bounded import run_bounded
from misterdev.logging_setup import setup_logger

logger = setup_logger(__name__)

_REGISTRY_URL = "https://registry.modelcontextprotocol.io/v0/servers"
# cgcone's public, key-free registry: ~2200 stdio servers with GitHub stars,
# last-commit, archived status, category/tags, and a heuristic install spec, all
# pre-indexed. Used as a DISCOVERY DATA SOURCE only — its stars/recency feed our
# trust ladder directly (no per-server GitHub call), but its install specs are
# heuristic and unverified, so every admitted candidate is still version-resolved
# (which filters the bogus ones), pinned, and spawned with a minimal env by us.
# We never invoke cgcone's own installer (it writes other CLIs' config files).
_CGCONE_URL = (
    "https://raw.githubusercontent.com/Himanshu507/cgcone/main/public/registry.json"
)
_DEFAULT_TIMEOUT = 10.0
# Bound the body materialized from a (semi-trusted) registry/GitHub/npm/PyPI
# endpoint. Real registry and model-list payloads are well under this; the cap
# only rejects a compromised/misbehaving endpoint slow-dripping an oversized body.
_MAX_RESPONSE_BYTES = 16 * 1024 * 1024
# PEP 503 normalized distribution name; anything else (path separators, spaces)
# is not a real PyPI package and must not be interpolated raw into a request path.
_PYPI_NAME_RE = re.compile(r"^[A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?$")
# npm package name: optional @scope/ prefix then lowercase name.
_NPM_NAME_RE = re.compile(r"^(@[a-z0-9_.-]+/)?[a-z0-9_.-]+$")
_DEFAULT_MAX_SERVERS = 3
_DEFAULT_MIN_TRUST = 0.5
# Cap network scoring per build so a wide search cannot hammer the GitHub API
# (unauth: 60 req/hr) or stall the build on the long tail.
_DEFAULT_MAX_EVALUATIONS = 12
# How long a cached search result / trust-signal reading stays fresh. Trust
# signals (downloads, stars, recency) move slowly; a day of staleness is a fine
# trade for not re-spending the scarce unauth GitHub quota every build.
_CACHE_TTL_SECONDS = 24 * 3600

# A named prefix is a strong signal (the publisher is one we recognise), but no
# longer the ONLY key — signals can admit an unlisted server on their own.
DEFAULT_TRUSTED_NAMESPACES = (
    "io.github.modelcontextprotocol",
    "io.modelcontextprotocol",
    "com.modelcontextprotocol",
)

# npm -> npx, PyPI -> uvx. Anything else is not auto-runnable here.
_RUNTIME_BY_REGISTRY = {"npm": "npx", "pypi": "uvx"}
# The only commands a discovered server may spawn. A registry-supplied runtimeHint
# is honored ONLY if it is one of these — otherwise an untrusted entry could set
# `command` to an arbitrary binary and bypass the npx/uvx launcher sandbox.
_ALLOWED_RUNTIMES = frozenset(_RUNTIME_BY_REGISTRY.values())

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


def _stem(token: str) -> str:
    """Crude, dependency-free suffix stemmer for capability matching.

    Not linguistically correct — just enough that a plural or a gerund in the
    query ("timezones", "searches", "fetching") still matches a vetted catalog
    entry ("timezone", "search", "fetch") instead of missing it and falling
    through to the riskier discovery tier. Longest derivational suffix first,
    then plural normalisation with the usual sibilant ``-es`` exception."""
    for suf in ("ization", "isation", "ations", "ation", "ings", "ing", "ers", "er"):
        if len(token) > len(suf) + 2 and token.endswith(suf):
            return token[: -len(suf)]
    if token.endswith("es") and len(token) > 4:
        base = token[:-2]
        if base[-1:] in ("s", "x", "z") or base[-2:] in ("ch", "sh"):
            return base  # "searches" -> "search"
        return token[:-1]  # "timezones" -> "timezone" (just the plural -s)
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token


def _stemmed_tokens(text: str) -> set:
    return {_stem(t) for t in _tokenize(text)}


# --- trust-score weights (transparent + tunable) --------------------------------
_NAMESPACE_BOOST = 0.5
_DOWNLOAD_TIERS = ((10000, 0.5), (1000, 0.35), (100, 0.2))
_STAR_TIERS = ((1000, 0.4), (200, 0.3), (50, 0.15))
_RECENCY_TIERS = ((180, 0.15), (365, 0.1))  # days since last push


# ==============================================================================
# Registry search
# ==============================================================================
def _github_token() -> Optional[str]:
    """A GitHub API token from the environment, if set.

    Lifts the unauthenticated 60 req/hr ceiling to 5000/hr for signal scoring, so
    a build that scores the long tail is not silently starved into all-zero
    scores. Read at call time (not import) so it works in any process."""
    return os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or None


def _http_get_json(
    url: str, timeout: float, headers: Optional[Dict[str, str]] = None
) -> Optional[Any]:
    """GET ``url`` and parse JSON, or return None. Never raises."""

    def _work() -> Optional[Any]:
        hdrs = {"Accept": "application/json"}
        if headers:
            hdrs.update(headers)
        req = urllib.request.Request(url, headers=hdrs)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
                if resp.status != 200:
                    return None
                raw = resp.read(_MAX_RESPONSE_BYTES + 1)
                if len(raw) > _MAX_RESPONSE_BYTES:
                    logger.debug(f"MCP registry GET oversized (> cap): {url}")
                    return None
                return json.loads(raw.decode("utf-8"))
        except (urllib.error.URLError, ValueError, OSError) as e:
            logger.debug(f"MCP registry GET failed ({url}): {e}")
            return None

    return run_bounded(_work, timeout + 2, None, "MCP registry search")


class RegistryCache:
    """TTL'd on-disk cache for registry searches and trust signals.

    Discovery re-hits the registry, GitHub, and npm every build; without a cache
    a working session exhausts the unauthenticated GitHub quota and every score
    collapses to zero (a SILENT capability loss). This persists both search
    results (keyed by query+limit) and per-server signals (keyed by
    ``name@version``) so repeated builds pay the network cost once per TTL.

    File-backed JSON and never raises: a corrupt/unwritable file degrades to no
    caching. Instances are built per call site but the manager is shared across
    parallel task threads, so all instances on the same path share one
    process-wide lock — a read-modify-write can't interleave and lose entries.
    """

    _locks: Dict[str, threading.Lock] = {}
    _locks_guard = threading.Lock()

    def __init__(self, path: Path, ttl: float = _CACHE_TTL_SECONDS):
        self.path = Path(path)
        self.ttl = float(ttl)
        key = str(self.path.resolve())
        with RegistryCache._locks_guard:
            self._lock = RegistryCache._locks.setdefault(key, threading.Lock())
        self._data: Dict[str, Dict[str, Any]] = self._load()
        try:
            self._mtime: float = self.path.stat().st_mtime
        except OSError:
            self._mtime = 0.0

    def _load(self) -> Dict[str, Dict[str, Any]]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            return raw if isinstance(raw, dict) else {}
        except (OSError, ValueError):
            return {}

    def _flush(self) -> None:
        # Several task threads may each hold their OWN cache instance on the same
        # path (the manager is shared, the cache is built per call site). Merge
        # whatever is on disk under our in-memory entries so a second instance's
        # write does not wipe the first's, then swap the file in atomically
        # (``os.replace`` is an atomic rename) so a reader never sees torn JSON.
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            merged = {**self._load(), **self._data}
            self._data = merged
            tmp = self.path.with_suffix(f".{os.getpid()}.{id(self)}.tmp")
            try:
                tmp.write_text(json.dumps(merged), encoding="utf-8")
                os.replace(tmp, self.path)
                try:
                    self._mtime = self.path.stat().st_mtime
                except OSError:
                    pass
            except Exception:
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass
                raise
        except (OSError, ValueError) as e:
            logger.debug(f"MCP registry cache flush failed: {e}")

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            try:
                mtime = self.path.stat().st_mtime
            except OSError:
                mtime = 0.0
            if mtime != self._mtime:
                self._data = self._load()
                self._mtime = mtime
            entry = self._data.get(key)
            if not entry:
                return None
            if (time.time() - entry.get("ts", 0)) > self.ttl:
                self._data.pop(key, None)
                return None
            return entry.get("value")

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = {"ts": time.time(), "value": value}
            self._flush()


def search_registry(
    query: str,
    limit: int = 10,
    timeout: float = _DEFAULT_TIMEOUT,
    cache: Optional[RegistryCache] = None,
) -> List[Dict[str, Any]]:
    """Search the official registry; return the raw ``server`` objects (or [])."""
    if not query or not query.strip():
        return []
    key = f"search::{query.strip().lower()}::{limit}"
    if cache is not None:
        hit = cache.get(key)
        if hit is not None:
            return hit
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
    # Re-rank by query relevance so the most on-topic servers are evaluated first,
    # within the per-build ``max_evaluations`` budget: the registry's ``search``
    # ordering is opaque and, if the endpoint ever ignores the param, this keeps
    # the best matches at the front instead of scoring an arbitrary page.
    terms = {_stem(t) for t in _tokenize(query) if t not in _STOPWORDS}
    if terms:
        out.sort(key=lambda s: _query_overlap(s, terms), reverse=True)
    logger.info(
        f"MCP registry: query {query!r} returned {len(out)} runnable-shaped result(s)."
    )
    if cache is not None:
        cache.put(key, out)
    return out


def _query_overlap(server: Dict[str, Any], terms: set) -> int:
    """Stemmed whole-word overlap of ``terms`` with a server's name+description."""
    text = f"{server.get('name', '')} {server.get('description', '')}"
    return len(terms & _stemmed_tokens(text))


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
    if runtime_hint not in _ALLOWED_RUNTIMES:
        logger.warning(
            "MCP discovery: ignoring untrusted runtimeHint %r for '%s'; using %s.",
            runtime_hint,
            server.get("name"),
            runtime,
        )
        runtime_hint = runtime
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
    token = _github_token()
    headers = {"Authorization": f"Bearer {token}"} if token else None
    data = _http_get_json(f"https://api.github.com/repos/{slug}", timeout, headers)
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
    if not _NPM_NAME_RE.match(pkg_id):
        return None
    quoted = urllib.parse.quote(pkg_id, safe="@/")
    data = _http_get_json(
        f"https://api.npmjs.org/downloads/point/last-month/{quoted}", timeout
    )
    if isinstance(data, dict) and isinstance(data.get("downloads"), int):
        return data["downloads"]
    return None


def _signal_cache_key(server: Dict[str, Any], pkg: Dict[str, Any]) -> str:
    version = pkg.get("version") or "unversioned"
    return f"signals::{pkg.get('identifier', server.get('name', '?'))}@{version}"


def gather_signals(
    server: Dict[str, Any],
    pkg: Dict[str, Any],
    timeout: float = _DEFAULT_TIMEOUT,
    cache: Optional[RegistryCache] = None,
) -> TrustSignals:
    """Best-effort live trust signals for a candidate. Never raises.

    When a ``cache`` is given, a fresh reading for this ``pkg@version`` short-
    circuits the GitHub + npm round-trips — the scarce unauthenticated quota is
    spent once per TTL, not once per build."""
    key = _signal_cache_key(server, pkg)
    if cache is not None:
        hit = cache.get(key)
        if isinstance(hit, dict):
            return TrustSignals(**hit)
    sig = TrustSignals()
    gh = _github_signals((server.get("repository") or {}).get("url", ""), timeout)
    sig.github_stars = gh.get("stars")
    sig.github_pushed_days = gh.get("pushed_days")
    sig.archived = bool(gh.get("archived"))
    if (pkg.get("registryType") or "").lower() == "npm":
        sig.npm_downloads_month = _npm_downloads(pkg["identifier"], timeout)
    if cache is not None:
        cache.put(key, asdict(sig))
    return sig


def _tier_score(value: Optional[int], tiers) -> float:
    if value is None:
        return 0.0
    for threshold, points in tiers:
        if value >= threshold:
            return points
    return 0.0


def _recency_score(days: Optional[int]) -> float:
    """Recency points from days-since-last-push: SMALLER is better, so tiers are
    an upper bound (unlike the ``>=`` adoption tiers). A repo pushed within 180
    days scores highest; an ancient one scores nothing."""
    if days is None:
        return 0.0
    for threshold, points in _RECENCY_TIERS:
        if days <= threshold:
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
    score += _recency_score(signals.github_pushed_days)
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
    categories: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """Curated stdio configs for the requested tiers.

    Admits only entries that are locally runnable (a ``command`` is set — not a
    manual Go/Docker/OAuth entry) AND whose ``requires`` env vars are all
    present, so a keyed server (e.g. Postgres needing ``DATABASE_URI``) mounts
    only when it can actually run. ``tiers`` accepts ``"all"`` for every tier.
    ``categories``, when given, further restricts to those catalog categories
    (e.g. ``{"docs"}`` to mount only the documentation servers).
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
        if categories is not None and entry.get("category") not in categories:
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
    cache: Optional[RegistryCache] = None,
    known_identities: Optional[set] = None,
    source: str = "official",
) -> Optional[Dict[str, Any]]:
    """Resolve a capability query to ONE runnable stdio server config, or None.

    The mid-task on-demand path: try the vetted catalog first (stemmed whole-word
    match on name/category/why, config-gated), then fall back to trust-scored
    discovery via ``source`` (``official``/``cgcone``/``both``). Returns a single
    config the caller mounts live; never raises.
    """
    terms = {_stem(t) for t in _tokenize(query) if t not in _STOPWORDS}
    if not terms:
        return None
    catalog = load_catalog()
    best, best_score = None, 0
    for cfg in select_curated(("all",), env=env, catalog=catalog):
        entry = next((e for e in catalog if e["name"] == cfg["name"]), {})
        # Stemmed so "convert timezones" matches "Timezone conversion" — a plural
        # or nominalisation should not miss a vetted entry and fall to discovery.
        words = _stemmed_tokens(
            f"{cfg['name']} {entry.get('category', '')} {entry.get('why', '')}"
        )
        overlap = len(terms & words)
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
        cache=cache,
        known_identities=known_identities,
        source=source,
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
    cache: Optional[RegistryCache] = None,
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
            signals = gather_signals(server, pkg, timeout, cache=cache)
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


def _config_identity(cfg: Dict[str, Any]) -> Tuple[str, ...]:
    """Launch identity of a config, matching :func:`mcp._server_identity`.

    Used to skip discovering a package that is already mounted under another
    name (e.g. a curated entry), and to dedup discoveries against each other by
    what they actually RUN rather than by their opaque registry display name."""
    return ("stdio", str(cfg.get("command")), *(str(a) for a in cfg.get("args") or []))


# ==============================================================================
# cgcone discovery backend (a richer, key-free data source for the trust ladder)
# ==============================================================================
def _resolve_version(runtime: str, identifier: str, timeout: float, cache) -> str:
    """Latest published version of ``identifier`` on npm (npx) / PyPI (uvx).

    Doubles as a runnability check: a heuristic/bogus spec (cgcone indexes some)
    resolves to no version and is dropped. Cached so a wide search is cheap."""
    key = f"ver::{runtime}::{identifier}"
    if cache is not None:
        hit = cache.get(key)
        if hit is not None:
            return hit
    if runtime == "npx":
        if not _NPM_NAME_RE.match(identifier):
            version = ""
        else:
            quoted = urllib.parse.quote(identifier, safe="@/")
            data = _http_get_json(
                f"https://registry.npmjs.org/{quoted}/latest", timeout
            )
            version = data.get("version", "") if isinstance(data, dict) else ""
    elif not _PYPI_NAME_RE.match(identifier):
        version = ""
    else:
        quoted = urllib.parse.quote(identifier, safe="")
        data = _http_get_json(f"https://pypi.org/pypi/{quoted}/json", timeout)
        version = (
            (data.get("info") or {}).get("version", "")
            if isinstance(data, dict)
            else ""
        )
    if cache is not None:
        cache.put(key, version)
    return version


def fetch_cgcone_registry(
    cache: Optional[RegistryCache] = None, timeout: float = _DEFAULT_TIMEOUT
) -> List[Dict[str, Any]]:
    """The cgcone ``mcpServers`` list (cached, TTL'd). ``[]`` on any failure."""
    if cache is not None:
        hit = cache.get("cgcone::registry")
        if isinstance(hit, list):
            return hit
    payload = _http_get_json(_CGCONE_URL, timeout)
    servers = payload.get("mcpServers") if isinstance(payload, dict) else None
    servers = servers if isinstance(servers, list) else []
    if servers and cache is not None:
        cache.put("cgcone::registry", servers)
    return servers


def _cgcone_signals(entry: Dict[str, Any]) -> TrustSignals:
    """Trust signals straight from the cgcone index — no live GitHub/npm call."""
    days: Optional[int] = None
    last = entry.get("lastCommit")
    if isinstance(last, str):
        try:
            dt = datetime.fromisoformat(last.replace("Z", "+00:00"))
            days = (datetime.now(timezone.utc) - dt).days
        except ValueError:
            pass
    stars = entry.get("stars")
    return TrustSignals(
        github_stars=int(stars) if isinstance(stars, int) else None,
        github_pushed_days=days,
        archived=bool(entry.get("isArchived")),
    )


def _cgcone_config(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """A free, self-contained stdio config from a cgcone entry, or ``None``.

    Only npx/uvx stdio entries with an EMPTY env admit — a non-empty env means a
    required secret, which cannot run for free (mirrors the official-path rule)."""
    if entry.get("serverType") != "stdio":
        return None
    ic = entry.get("installConfig") or {}
    if ic.get("env"):
        return None
    command = ic.get("command")
    args = ic.get("args")
    if command not in ("npx", "uvx") or not isinstance(args, list) or not args:
        return None
    name = (entry.get("name") or "").replace("/", ".")
    if not name:
        return None
    return {
        "name": name,
        "transport": "stdio",
        "command": command,
        "args": [str(a) for a in args],
        "_discovered": True,
        "_source": "cgcone",
    }


def _pin_cgcone_config(
    cfg: Dict[str, Any], timeout: float, cache
) -> Optional[Dict[str, Any]]:
    """Version-pin a cgcone config's package, or ``None`` if it doesn't resolve.

    cgcone specs are bare (no version) and sometimes bogus; resolving pins to a
    concrete release (our stop-a-malicious-future-release guarantee) AND drops
    anything that isn't actually published."""
    args = list(cfg["args"])
    identifier = args[-1]
    if "@" in identifier[1:] or "==" in identifier:  # already pinned
        return cfg
    version = _resolve_version(cfg["command"], identifier, timeout, cache)
    if not version:
        return None
    args[-1] = (
        f"{identifier}=={version}"
        if cfg["command"] == "uvx"
        else f"{identifier}@{version}"
    )
    return {**cfg, "args": args}


def discover_via_cgcone(
    queries: List[str],
    trusted_namespaces=DEFAULT_TRUSTED_NAMESPACES,
    max_servers: int = _DEFAULT_MAX_SERVERS,
    min_trust: float = _DEFAULT_MIN_TRUST,
    max_evaluations: int = _DEFAULT_MAX_EVALUATIONS,
    timeout: float = _DEFAULT_TIMEOUT,
    cache: Optional[RegistryCache] = None,
    known_identities: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """Admit long-tail servers from the cgcone index via the same trust ladder.

    Relevance-ranked (stemmed overlap on name/description/tags/category), then
    trust-scored from cgcone's pre-indexed stars/recency/archived — so no live
    GitHub call. Each admitted candidate is version-resolved+pinned (which also
    drops cgcone's bogus specs) before it is returned. Deduped by launch identity
    and capped like the official path; never raises."""
    if not queries:
        return []
    entries = fetch_cgcone_registry(cache, timeout)
    if not entries:
        return []
    permissive = "*" in (trusted_namespaces or ())
    terms: set = set()
    for q in queries:
        terms |= {t for t in _stemmed_tokens(q) if t not in _STOPWORDS}
    if not terms:
        return []

    def _relevance(e: Dict[str, Any]) -> int:
        text = (
            f"{e.get('name', '')} {e.get('description', '')} "
            f"{e.get('category', '')} {' '.join(e.get('tags') or [])}"
        )
        return len(terms & _stemmed_tokens(text))

    ranked = sorted(
        (e for e in entries if _relevance(e) > 0),
        key=lambda e: (_relevance(e), e.get("stars") or 0),
        reverse=True,
    )
    seen_ids: set = set(known_identities or ())
    out: List[Dict[str, Any]] = []
    for e in ranked[:max_evaluations]:
        if len(out) >= max_servers:
            break
        cfg = _cgcone_config(e)
        if not cfg:
            continue
        if not permissive:
            synth = {"name": e.get("name"), "_status": "active", "_is_latest": True}
            score = score_server(synth, _cgcone_signals(e), trusted_namespaces)
            if score < min_trust:
                continue
        pinned = _pin_cgcone_config(cfg, timeout, cache)
        if not pinned:  # unresolved / bogus spec
            continue
        identity = _config_identity(pinned)
        if identity in seen_ids:
            continue
        seen_ids.add(identity)
        out.append(pinned)
        logger.info(
            f"MCP cgcone: provisioning '{pinned['name']}' "
            f"({pinned['command']} {' '.join(pinned['args'])})"
        )
    return out


def discover_servers(
    queries: List[str],
    trusted_namespaces=DEFAULT_TRUSTED_NAMESPACES,
    max_servers: int = _DEFAULT_MAX_SERVERS,
    min_trust: float = _DEFAULT_MIN_TRUST,
    per_query_limit: int = 10,
    max_evaluations: int = _DEFAULT_MAX_EVALUATIONS,
    timeout: float = _DEFAULT_TIMEOUT,
    cache: Optional[RegistryCache] = None,
    known_identities: Optional[set] = None,
    source: str = "official",
) -> List[Dict[str, Any]]:
    """Search each query and admit the long-tail servers by trust score.

    Signal-scored up to ``max_evaluations`` network checks; deduped by name AND
    launch identity (so the same package is never mounted twice under two names,
    and anything already in ``known_identities`` — the curated/configured stack —
    is skipped); capped at ``max_servers``. (The vetted core mounts separately
    and by name via :func:`select_curated`, so discovery is purely the signal
    tier.)

    ``source`` selects the backend: ``"official"`` (the MCP registry, default),
    ``"cgcone"`` (the broader cgcone index — stars pre-scored, no GitHub call),
    or ``"both"`` (official first, then cgcone fills the remaining slots).
    """
    if not queries:
        return []
    if source == "cgcone":
        return discover_via_cgcone(
            queries,
            trusted_namespaces,
            max_servers,
            min_trust,
            max_evaluations,
            timeout,
            cache,
            known_identities,
        )
    if "*" in (trusted_namespaces or ()):
        logger.warning(
            "MCP discovery is in PERMISSIVE trust mode: any registry server may "
            "be installed and run locally. This is arbitrary code execution."
        )
    seen: set = set()
    seen_ids: set = set(known_identities or ())
    out: List[Dict[str, Any]] = []
    evaluations = 0
    for q in queries:
        if len(out) >= max_servers:
            break
        for server in search_registry(
            q, limit=per_query_limit, timeout=timeout, cache=cache
        ):
            name = server.get("name") or ""
            if name in seen or evaluations >= max_evaluations:
                continue
            evaluations += 1
            cfg = to_stdio_config(
                server, trusted_namespaces, min_trust, timeout=timeout, cache=cache
            )
            if not cfg:
                continue
            identity = _config_identity(cfg)
            if identity in seen_ids:
                logger.info(
                    f"MCP discovery: '{cfg['name']}' runs an already-known package "
                    f"({cfg['command']} {' '.join(cfg['args'])}); skipping duplicate."
                )
                seen.add(name)
                continue
            seen.add(name)
            seen_ids.add(identity)
            out.append(cfg)
            logger.info(
                f"MCP discovery: provisioning '{cfg['name']}' "
                f"({cfg['command']} {' '.join(cfg['args'])}) for query {q!r}"
            )
            if len(out) >= max_servers:
                break
    if source == "both" and len(out) < max_servers:
        # Fill remaining slots from cgcone, deduped against what official admitted.
        out.extend(
            discover_via_cgcone(
                queries,
                trusted_namespaces,
                max_servers - len(out),
                min_trust,
                max_evaluations,
                timeout,
                cache,
                seen_ids,
            )
        )
    if not out:
        logger.info(f"MCP discovery: no admissible server for {queries!r}")
    return out
