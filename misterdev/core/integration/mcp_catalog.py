"""Curated, tiered catalog of vetted MCP servers.

The trusted **core** of the trust ladder (see :mod:`mcp_registry`): a hand-vetted
set with load tiers, so a build mounts a small, high-signal stack by default and
augments on demand — never "large, universal, always loaded". The long tail of
the ecosystem is reached separately, on demand, through registry discovery +
trust scoring; it does NOT belong in this list.

Each entry:
  ``name``       short id / routing key
  ``category``   grouping (workspace, vcs, docs, data, infra, quality, ...)
  ``tier``       "core" (default-on) | "project" (per-project) | "task" (heavy,
                 only when the task needs it)
  ``command``/``args``  local stdio invocation (npx/uvx); ``None`` command means
                 approved but not auto-installable here (Go binary, Docker, or an
                 OAuth remote) — recorded so the stack is documented, activated
                 manually via ``mcp.servers``.
  ``requires``   env vars that must be set for the server to run; the loader only
                 activates an entry whose vars are all present.
  ``repo``/``why`` provenance + one-line rationale.

Identifiers here are bare; ``scripts/audit_mcp_servers.py`` pins versions into
``curated_servers.json`` (the runtime artifact the loader prefers).
"""

from typing import Any, Dict, List


def _npx(pkg: str):
    return "npx", ["-y", pkg]


def _uvx(pkg: str):
    return "uvx", [pkg]


def _e(name, category, tier, cmd, requires, repo, why) -> Dict[str, Any]:
    command, args = cmd if cmd else (None, [])
    return {
        "name": name,
        "category": category,
        "tier": tier,
        "command": command,
        "args": args,
        "requires": requires,
        "repository": repo,
        "why": why,
    }


# Verified: command present -> resolves on npm/PyPI (checked). None -> manual.
CATALOG: List[Dict[str, Any]] = [
    # ---- core (default-on) ----
    _e(
        "desktop-commander",
        "workspace",
        "core",
        _npx("@wonderwhy-er/desktop-commander"),
        [],
        "https://github.com/wonderwhy-er/DesktopCommanderMCP",
        "Shell + file ops in one surface",
    ),
    _e(
        "serena",
        "workspace",
        "core",
        None,
        [],
        "https://github.com/oraios/serena",
        "LSP-backed symbol navigation (manual: uvx-from-git)",
    ),
    _e(
        "git",
        "vcs",
        "core",
        _uvx("mcp-server-git"),
        [],
        "https://github.com/modelcontextprotocol/servers",
        "Native git operations",
    ),
    _e(
        "context7",
        "docs",
        "core",
        _npx("@upstash/context7-mcp"),
        [],
        "https://github.com/upstash/context7",
        "Version-pinned library docs; anti-hallucination",
    ),
    _e(
        "fetch",
        "docs",
        "core",
        _uvx("mcp-server-fetch"),
        [],
        "https://github.com/modelcontextprotocol/servers",
        "URL fetch + markdown",
    ),
    _e(
        "sentry",
        "quality",
        "core",
        None,
        [],
        "https://github.com/getsentry/sentry-mcp",
        "Error triage, release tracking (OAuth remote)",
    ),
    _e(
        "semgrep",
        "quality",
        "core",
        _uvx("semgrep-mcp"),
        [],
        "https://github.com/semgrep/mcp",
        "Static analysis; find real bugs in a diff",
    ),
    _e(
        "time",
        "util",
        "core",
        _uvx("mcp-server-time"),
        [],
        "https://github.com/modelcontextprotocol/servers",
        "Timezone conversion, current time",
    ),
    _e(
        "sequential-thinking",
        "util",
        "core",
        _npx("@modelcontextprotocol/server-sequential-thinking"),
        [],
        "https://github.com/modelcontextprotocol/servers",
        "Structured multi-step reasoning",
    ),
    # ---- project (load per project) ----
    _e(
        "postgres",
        "data",
        "project",
        _uvx("postgres-mcp"),
        ["DATABASE_URI"],
        "https://github.com/crystaldba/postgres-mcp",
        "Schema introspection, safe queries, plans",
    ),
    _e(
        "vercel",
        "deploy",
        "project",
        None,
        [],
        "https://github.com/vercel/mcp-server",
        "Deployments, domains, env, logs (official)",
    ),
    _e(
        "fly",
        "deploy",
        "project",
        None,
        [],
        "https://github.com/superfly/fly-mcp",
        "Machine lifecycle, multi-region deploy (official)",
    ),
    _e(
        "railway",
        "deploy",
        "project",
        None,
        [],
        "https://github.com/railwayapp/mcp-railway",
        "Rapid full-stack prototyping deploy (official)",
    ),
    _e(
        "crates-io",
        "docs",
        "project",
        None,
        [],
        "https://github.com/rust-lang/cratesio-mcp",
        "Crate search/versions/features (Rust)",
    ),
    _e(
        "docs-rs",
        "docs",
        "project",
        None,
        [],
        "https://github.com/rust-lang/docsrs-mcp",
        "Version-pinned Rust API docs",
    ),
    _e(
        "grafana",
        "observability",
        "project",
        None,
        [],
        "https://github.com/grafana/mcp-grafana",
        "Dashboards/alerts/metric queries (official)",
    ),
    _e(
        "honeycomb",
        "observability",
        "project",
        None,
        [],
        "https://github.com/honeycombio/honeycomb-mcp",
        "Distributed tracing / OTel event exploration",
    ),
    _e(
        "elasticsearch",
        "data",
        "project",
        None,
        [],
        "https://github.com/elastic/mcp-server-elasticsearch",
        "Query/mapping/index management (official)",
    ),
    _e(
        "chroma",
        "data",
        "project",
        _uvx("chroma-mcp"),
        [],
        "https://github.com/chroma-core/chroma-mcp",
        "Local embedded vector store",
    ),
    _e(
        "basic-memory",
        "memory",
        "project",
        _uvx("basic-memory"),
        [],
        "https://github.com/basicmachines-co/basic-memory",
        "File-backed markdown knowledge graph",
    ),
    _e(
        "docker",
        "infra",
        "project",
        _uvx("mcp-server-docker"),
        [],
        "https://github.com/ckreiling/mcp-server-docker",
        "Container lifecycle, logs, compose",
    ),
    # ---- task (heavy; only when the task needs it) ----
    _e(
        "playwright",
        "browser",
        "task",
        _npx("@playwright/mcp"),
        [],
        "https://github.com/microsoft/playwright-mcp",
        "Full browser control",
    ),
    _e(
        "kubernetes",
        "infra",
        "task",
        _npx("mcp-server-kubernetes"),
        [],
        "https://github.com/Flux159/mcp-server-kubernetes",
        "Cluster ops, safe apply",
    ),
    _e(
        "terraform",
        "infra",
        "task",
        None,
        [],
        "https://github.com/hashicorp/terraform-mcp-server",
        "Providers/modules/plans (official, Docker)",
    ),
    _e(
        "figma",
        "design",
        "task",
        None,
        [],
        "https://github.com/figma/figma-mcp-server",
        "Read design specs, exact values (official Dev Mode)",
    ),
    _e(
        "launchdarkly",
        "flags",
        "task",
        None,
        [],
        "https://github.com/launchdarkly/mcp-server",
        "Flag inspection, targeting, audit (official)",
    ),
    _e(
        "posthog",
        "analytics",
        "task",
        None,
        [],
        "https://github.com/posthog/mcp",
        "Product analytics, replays, flags (official)",
    ),
    _e(
        "temporal",
        "workflow",
        "task",
        None,
        [],
        "https://github.com/temporalio/mcp-server-temporal",
        "Workflow/activity/task-queue inspection (official)",
    ),
    _e(
        "mermaid",
        "diagram",
        "task",
        _npx("mcp-mermaid"),
        [],
        "https://github.com/mermaid-js/mermaid-mcp",
        "Diagram generation from text",
    ),
    _e(
        "jupyter",
        "data",
        "task",
        _uvx("jupyter-mcp-server"),
        [],
        "https://github.com/jupyter/mcp-jupyter",
        "Notebook execution and inspection",
    ),
    _e(
        "everything",
        "util",
        "task",
        _npx("@modelcontextprotocol/server-everything"),
        [],
        "https://github.com/modelcontextprotocol/servers",
        "Reference test server (all MCP features)",
    ),
]

CATALOG_BY_NAME = {e["name"]: e for e in CATALOG}
