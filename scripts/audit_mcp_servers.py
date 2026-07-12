#!/usr/bin/env python3
"""Reproducible audit that pins the curated MCP catalog to concrete releases.

Reads the vetted, tiered catalog (``mcp_catalog.CATALOG``), resolves each
auto-installable entry to its current npm/PyPI version (so ``npx``/``uvx`` runs a
pinned ``pkg@version`` rather than drifting latest — pinning is what stops a
malicious *future* release), and writes the runtime artifact
``misterdev/core/integration/curated_servers.json``.

Optionally augments the catalog with high-trust servers the audit *discovers*
from the official registry by trust score (npm downloads, GitHub stars, recency).

    python scripts/audit_mcp_servers.py [--augment] [--min-trust 0.75] [--limit 10]

Re-run to refresh; review the JSON diff before committing. Set ``GITHUB_TOKEN``
to lift the unauthenticated 60 req/hr limit when using ``--augment``.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from misterdev.core.integration import mcp_registry as reg  # noqa: E402
from misterdev.core.integration.mcp_catalog import CATALOG  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("audit_mcp_servers")

QUERIES = [
    "fetch web pages",
    "filesystem",
    "sqlite database",
    "git repository",
    "memory knowledge graph",
    "web search",
    "browser automation",
    "documentation search",
    "code search",
]


def _latest_version(registry: str, identifier: str) -> str:
    if registry == "npm":
        data = reg._http_get_json(
            f"https://registry.npmjs.org/{identifier}/latest", 10.0
        )
        return data.get("version", "") if isinstance(data, dict) else ""
    data = reg._http_get_json(f"https://pypi.org/pypi/{identifier}/json", 10.0)
    return (data.get("info") or {}).get("version", "") if isinstance(data, dict) else ""


def _pin(entry: dict) -> dict:
    """Return a copy of a catalog entry with its package pinned to latest."""
    out = dict(entry)
    command, args = entry.get("command"), entry.get("args") or []
    if not command or not args:
        return out  # manual/remote — nothing to pin
    if command == "npx":
        identifier = args[-1]
        registry = "npm"
    else:  # uvx
        identifier = args[-1]
        registry = "pypi"
    if "@" in identifier[1:] or "==" in identifier:
        return out  # already pinned
    version = _latest_version(registry, identifier)
    if not version:
        log.info("  pin  %-20s (no version resolved; leaving unpinned)", entry["name"])
        return out
    pinned = (
        f"{identifier}@{version}" if command == "npx" else f"{identifier}=={version}"
    )
    out["args"] = [*args[:-1], pinned]
    log.info("  pin  %-20s %s", entry["name"], pinned)
    return out


def _augment(min_trust: float, per_query_limit: int, known: set) -> list:
    extra: dict = {}
    for q in QUERIES:
        for server in reg.search_registry(q, limit=per_query_limit):
            name = server.get("name") or ""
            if name in known or name in extra:
                continue
            pkg = reg._runnable_package(server)
            if not pkg:
                continue
            signals = reg.gather_signals(server, pkg)
            score = reg.score_server(server, signals, trusted_namespaces=())
            if score < min_trust:
                continue
            cfg = reg._to_config(server, pkg)
            extra[name] = {
                "name": cfg["name"],
                "category": "discovered",
                "tier": "task",
                "command": cfg["command"],
                "args": cfg["args"],
                "requires": [],
                "repository": (server.get("repository") or {}).get("url"),
                "why": f"signal-discovered (score {score:.2f})",
                "score": round(score, 3),
            }
            log.info("  add  %-28s score=%.2f", name, score)
    return list(extra.values())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--augment", action="store_true", help="add signal-discovered servers"
    )
    ap.add_argument("--min-trust", type=float, default=0.75)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--out", type=Path, default=reg._CURATED_PATH)
    args = ap.parse_args()

    log.info("Pinning curated catalog (%d entries)...", len(CATALOG))
    servers = [_pin(e) for e in CATALOG]
    if args.augment:
        log.info("Augmenting from registry (min_trust=%s)...", args.min_trust)
        servers += _augment(args.min_trust, args.limit, {e["name"] for e in servers})

    args.out.write_text(
        json.dumps(
            {"generated_by": "scripts/audit_mcp_servers.py", "servers": servers},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    runnable = sum(1 for s in servers if s.get("command"))
    log.info(
        "\nWrote %d server(s), %d auto-installable -> %s",
        len(servers),
        runnable,
        args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
