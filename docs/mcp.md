# MCP (Model Context Protocol)

misterdev speaks MCP in both directions: it can run **as** an MCP server that an AI client drives, and it can **use** MCP servers to give a build access to external tools. Both need the `mcp` extra:

```bash
pip install 'misterdev[mcp]'
```

---

## Part A — misterdev as an MCP server

misterdev ships the `misterdev-mcp` entry point, a thin adapter over the same `ProjectOrchestrator` the CLI uses. Drive it in plain English from Claude Desktop, Claude Code, Cursor, or any MCP client.

The key property: the client only sends a short instruction and receives a short summary. **The entire orchestration — reading the codebase, symbol-graph context, multi-step reasoning, model selection, budget — runs in misterdev's own process with its own LLM key.** Your codebase never enters the client's context window, so the context-scaling misterdev exists to provide is fully preserved.

### Tools

**Core tools:**

| Tool | Kind | What it does |
| --- | --- | --- |
| `list_projects` | read-only | List every project misterdev knows about. Start here. |
| `status` | read-only | Show a project's tasks and their state. |
| `scan` | registers | Discover projects under a directory and register them (idempotent). |
| `report` | read-only | Return the latest build report, per-model ledger, and audit trail for a project. |
| `build` | destructive | Plan **and** execute a goal from scratch, verifying each change through the gates and reverting regressions. The main tool. |
| `run` | destructive | Execute pending tasks for a project. Does not decompose a goal. |

**Async variants** (return immediately with a `run_id`; poll with `job_status`):

| Tool | What it does |
| --- | --- |
| `build_async` | Same as `build`, non-blocking. |
| `run_async` | Same as `run`, non-blocking. |
| `job_status` | Get the current status of an async job by `run_id`. |
| `stop_job` | Request cancellation of a running async job. |
| `list_jobs` | List all jobs (running and completed). |

**Planning tools** (propose a plan before executing):

| Tool | What it does |
| --- | --- |
| `propose_plan` | Analyze the project and return a proposed plan for review before any execution. |
| `get_plan` | Get the current proposed plan for a project. |
| `approve_plan` | Approve specific plan items (by ID) before execution. |
| `execute_plan` | Execute the approved plan. |

**`build` parameters:** `path`, `goal` (plain English, or a mode word `debug` / `complete` / `review`), `budget` (default $10 — conservative for a client-triggered run; raise it explicitly), `dry_run`, `parallel`, `max_tasks`, `reference_dir`.

**`spec_text` — the Claude handoff parameter.** Pass a complete specification written by the calling AI and misterdev skips its own analysis and spec-generation phases, going straight to decompose → execute → verify. This is the intended Claude Code integration pattern: Claude writes the spec with full project context; misterdev executes it using your codebase's own gate suite. The calling AI's context never needs to hold the full codebase:

```python
# From Claude Code or any MCP client
misterdev.build(
    path="/path/to/repo",
    spec_text="""
Feature: token-bucket rate limiter on all /api/v1/* routes
Acceptance: 429 after 100 req/min per IP; X-RateLimit-* headers on every response
Files to modify: api/middleware.py, api/app.py, config.py
...
"""
)
```

`run` takes `path`, optional `task_id`, and `dry_run`.

The mutating tools carry honest MCP annotations. `build` and `run` are destructive (they edit files and make git commits) and refuse a dirty working tree — commit or stash first, or use `dry_run=True` to preview the plan without touching anything.

### Claude Desktop config

Add to `claude_desktop_config.json`:

```jsonc
{
  "mcpServers": {
    "misterdev": {
      "command": "misterdev-mcp",
      "env": { "OPENROUTER_API_KEY": "sk-or-..." }
    }
  }
}
```

Then just ask: *"Have misterdev add rate limiting to the API, keep it under $5."* The client calls `build`; misterdev does the work in-process and returns a compact report of what was done, the gate results, and the cost.

---

## Part B — misterdev using MCP servers

A build can reach out to MCP servers and let the model call their discovered tools mid-task — bounded, opt-in, and constrained by an allowlist. Transports: stdio (subprocess), streamable-http, and sse. The last two connect to a remote `url` with optional auth, which is how misterdev reaches a hosted gateway like Glama that fronts many servers, without running any of them locally.

Everything here is additive and safe by construction: a missing SDK, a server that won't start, or one that hangs is simply absent — never an error, never a hang.

### Configure servers

Declare servers under `mcp.servers`. Each stdio server needs a `name` (the routing key) and a `command`; each remote server needs a `name`, `transport`, and `url`.

```yaml
mcp:
  servers:
    - name: "docs"
      command: "my-mcp-server"
      args: ["--root", "."]
      transport: "stdio"
    - name: "glama"
      transport: "http"                 # http | streamable-http | sse
      url: "https://glama.ai/mcp/instance/..."
      api_key_env: "GLAMA_API_KEY"      # Bearer token read from the environment
      # headers: { X-Custom: "value" }  # optional extra headers
  allow_tools: ["docs.search", "glama.web_fetch"]
```

For a remote gateway, the auth token is read from the environment variable named by `api_key_env` and sent as `Authorization: Bearer <token>` — the token stays out of config on disk. Extra static `headers` can be added alongside.

### The three-layer server model (free — no hosted gateway)

The whole MCP ecosystem is usable for free: nearly every server ships as an npm/PyPI package that runs locally as a stdio subprocess. Reach it through three layers, smallest and most-trusted first:

1. **Curated tier** — a vetted, version-pinned catalog (`mcp_catalog.py`), organised by load tier so a build mounts a *small* high-signal stack and augments on demand. This is the default and the safest path.
2. **Discovery tier** — the long tail, searched on demand from the [official registry](https://registry.modelcontextprotocol.io) and admitted by trust score.
3. **The shell** — every *CLI tool* (`ruff`, `mypy`, `eslint`, `clippy`, `ripgrep`, `jq`, `git`, …) runs through Desktop Commander / the shell, **not** a per-tool MCP. Adding an MCP whose only job duplicates a shell command inflates context and confuses tool selection; don't.

#### Curated tier (`mcp.curated`)

```yaml
mcp:
  curated: "core"        # true | "core" | "project" | "task" | "all" | [list of tiers]
```

Mounts the catalog entries for the requested tier(s). Each is version-pinned (`pkg@1.2.3`, so a malicious *future* release cannot silently apply) and **config-gated**: a keyed server (e.g. Postgres needs `DATABASE_URI`) mounts only when its env vars are present; otherwise it is skipped with a log line. The curated catalog is otherwise key-free by design — servers that require an API key are reached through discovery, not curated defaults. Tiers: `core` (default-on: workspace, git, docs, quality), `project` (per-project: deploy target, database, observability), `task` (heavy, only when needed: browser, design, workflow). Entries that need a Go binary / Docker / OAuth are recorded in the catalog as *manual* and are not auto-mounted. Refresh + re-pin with `python scripts/audit_mcp_servers.py`.

#### On-demand, mid-task (`mcp.discover_on_demand`)

With `orchestrator.mcp_tool_use` on and `mcp.discover_on_demand: true`, the model can request a *new* capability **during** the pre-edit gather loop, not just at build start. When no mounted tool fits, it replies `FIND <capability>`; misterdev resolves it through the same trust ladder (curated catalog match first, then trust-scored discovery), mounts the server live, and the new tools are callable on the next round. Bounded by `mcp.discover_on_demand_max` (default 2) provisions per task, and off by default — this is model-driven local code execution, so it inherits `min_trust`/`trusted_namespaces` and the minimal-env, config-gating guarantees.

#### Discovery tier (`mcp.discover`)

```yaml
mcp:
  discover: ["fetch web pages", "query sqlite"]   # capability queries
  discover_source: "cgcone"                        # cgcone (default) | official | both
  discover_max_servers: 3                          # cap per build (default 3)
  min_trust: 0.5                                   # quality bar (0..1)
  trusted_namespaces: ["io.github.modelcontextprotocol"]  # namespace boost; ["*"] = trust all
```

**`discover_source`** picks the backend. `cgcone` (default) searches the [cgcone](https://cgcone.com) index (~2200 stdio servers) whose GitHub stars / recency / archived-status are *pre-indexed* — so scoring needs **no per-server GitHub call** (this is what sidesteps the unauthenticated 60 req/hr limit) and reaches a much longer tail than the official registry. `official` searches only the [official registry](https://registry.modelcontextprotocol.io). `both` runs official first, then lets cgcone fill any remaining slots. cgcone is used strictly as a **data source**: its install specs are heuristic and mostly unverified (`community`), so every candidate it yields is still trust-scored, **version-resolved + pinned** (which drops the bogus specs, since they don't resolve on npm/PyPI), deduped by launch identity, and spawned with a minimal env by misterdev itself. misterdev never invokes cgcone's own installer (that writes *other* CLIs' config files, not misterdev's). A cgcone entry whose `env` is non-empty (needs a secret) is skipped like any keyed server.

**This runs code from the internet locally.** A discovered server is a package installed and executed with the build's file access and network, so admission goes through a trust score, not a bare on/off gate:

- Each candidate is scored on real signals — npm monthly downloads, GitHub stars, recency, and archived/inactive status — and admitted only at/above `min_trust`. A `trusted_namespaces` match is a strong *boost*, not the only key. `["*"]` trusts everything (logged loudly; remote code execution) and skips the quality bar.
- Paid *remotes* (a hosted gateway needing an API key) and any package that **requires** a secret to start are skipped — they could not run for free anyway.
- Discovered servers spawn with a **minimal environment**, never the build's secrets, and are bounded by the discovery/call timeouts, `discover_max_servers`, and the `allow_tools` allowlist.

### Safety & operational hardening

Discovery and on-demand provisioning run unvetted code from a public registry, so several guarantees back them beyond the trust score:

- **Sandbox respected.** Under `governance.network: none` (what the container env enforces) registry discovery and on-demand `FIND` are **refused** — auto-installing a package on the host would punch through the isolation the project deliberately chose. Explicitly-configured `mcp.servers` and the curated tier still mount (you opted into those by name).
- **Audit trail.** Every provision (`mcp_provision`) and tool call (`mcp_call`) is written to the build's append-only `.orchestrator/audit.jsonl` — package, resolved version, capability, and task id — so *what ran and why* is forensically reconstructable, not just a log line.
- **Untrusted output.** A tool result is external, attacker-influenceable text. It is fenced in the gather context as inert **data** the model must not obey as instructions (prompt-injection / "lethal trifecta" defense); output from a freshly-discovered server is tagged as least-trusted.
- **Off-target unmount.** An on-demand server that mounts but exposes no tool matching the requested capability is unmounted and reported as not-found, so it neither pollutes the tool list nor wastes a provision slot.
- **Signal cache + quota.** Registry searches and trust signals are cached (TTL, under `.orchestrator/`), so repeated builds don't re-spend the scarce unauthenticated GitHub quota (60 req/hr) and collapse every score to zero. Set `GITHUB_TOKEN` (or `GH_TOKEN`) to lift the ceiling to 5000/hr. A package already mounted (curated/configured) is never re-discovered under a second name — dedup is by launch identity, not display name.

### The allowlist

`mcp.allow_tools` is an allowlist of tool names. An entry matches either the qualified `server.tool` form or a bare `tool` name. Omit it to allow every discovered tool. A tool not on the list is refused at both discovery and call time — a hosted gateway may expose a large catalog, so the allowlist is how you scope what a build can actually reach.

### Two levels: awareness vs. tool use

```yaml
orchestrator:
  mcp_enabled: true      # awareness only: discovered tools are described to the model
  mcp_tool_use: true     # bounded gathering loop: the model may CALL tools pre-edit
  mcp_max_tool_rounds: 3 # hard cap on gathering rounds
```

- **`mcp_enabled`** gates awareness injection only: discovered tools are described in the task context. The single-shot edit loop is unchanged.
- **`mcp_tool_use`** layers a bounded agentic pre-edit loop on top (and implies `mcp_enabled`). Each round the model sees the available tools and may reply with one line `CALL <server>.<tool> {json-args}` (or `NO_TOOL` to stop). The call runs through the timeout-guarded `call_tool` and its result is prepended to the task context, so the model gathers information *before* editing. `mcp_max_tool_rounds` (default 3) hard-caps the loop.

With `mcp_tool_use` off, the edit path is byte-identical to a build with no MCP configured; any failure degrades to gathering nothing. Plugin tools marked `gather_safe = True` participate in the same loop — see [plugins.md](plugins.md).

## Stability

The following are stable as of v0.6.0 and follow semantic versioning — breaking changes happen only on a major version bump:

- **Tool names**: `list_projects`, `status`, `scan`, `report`, `build`, `run`, `build_async`, `run_async`, `job_status`, `stop_job`, `list_jobs`, `propose_plan`, `get_plan`, `approve_plan`, `execute_plan`
- **`build` parameters**: `path`, `goal`, `budget`, `dry_run`, `parallel`, `max_tasks`, `reference_dir`, `spec_text`
- **Return shape**: `build` always returns a string report; `build_async`/`run_async` always return `{run_id, status}`; `job_status` always returns `{status, ...}`
- **Behavioral guarantees**: `build` and `run` refuse a dirty working tree; read-only tools never modify files; async tools are idempotent to poll

The MCP server entry point (`misterdev-mcp`) and the `misterdev mcp` subcommand are both stable. Server configuration (Claude Desktop JSON) will not require changes within a major version.
