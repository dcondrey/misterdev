<p align="center">
  <img src="https://raw.githubusercontent.com/dcondrey/misterdev/main/assets/logo.gif" alt="misterdev" width="220">
</p>

<h1 align="center">misterdev</h1>

<p align="center">
  <strong>An autonomous LLM build orchestrator that plans a goal into tasks, edits your code with surgical precision, and verifies every change through correctness gates before it ever reports done.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/misterdev/"><img src="https://img.shields.io/pypi/v/misterdev" alt="PyPI version"></a>
  <a href="https://pypi.org/project/misterdev/"><img src="https://img.shields.io/pypi/pyversions/misterdev" alt="Python versions"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-AGPL--3.0-blue" alt="License: AGPL-3.0"></a>
  <a href="https://github.com/dcondrey/misterdev/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/dcondrey/misterdev/ci.yml?branch=main" alt="CI"></a>
  <a href="https://glama.ai/mcp/servers/dcondrey/misterdev"><img src="https://glama.ai/mcp/servers/dcondrey/misterdev/badges/score.svg" alt="misterdev MCP server"></a>
</p>

<p align="center">
  <a href="#install">Install</a> ·
  <a href="#what-it-does">What it does</a> ·
  <a href="#cli-reference">CLI</a> ·
  <a href="#extending-misterdev">Extending</a> ·
  <a href="#configuration">Configuration</a> ·
  <a href="#development">Development</a> ·
  <a href="#license">License</a>
</p>

---

Point misterdev at a repository and a goal. It reads the codebase as a symbol graph, decomposes the goal into concrete tasks, and works each one in a try-edit-verify loop: it emits an anchored SEARCH/REPLACE edit, applies it against the file on disk, and runs the change through a sequence of correctness gates — build, tests, lint, typecheck, and any optional gates you enable. A gate that fails RED blocks the change; a gate that has nothing to check SKIPs and never blocks. When a change regresses the suite, misterdev reverts it through git. Nothing merges unless it stays green.

```console
$ misterdev build . "add rate limiting to the public API"

  planning   goal → 3 tasks  (model: anthropic/claude-sonnet-4-6, budget $100.00)
  task 1/3   middleware: token-bucket limiter          api/limiter.py
    edit     1 hunk applied · syntax ok
    gates    build GREEN · tests GREEN (142 passed) · lint GREEN · typecheck GREEN
  task 2/3   wire limiter into request pipeline         api/app.py
    edit     2 hunks applied
    gates    build GREEN · tests RED (1 failed) → rolling back, regenerating
    edit     2 hunks applied (attempt 2)
    gates    build GREEN · tests GREEN (145 passed) · lint GREEN · typecheck GREEN
  task 3/3   docs + config surface                      README.md, config.py
    gates    all GREEN

  done       3/3 tasks · 145 tests green · $0.38 over 11 calls
```

Because misterdev only trusts its gates, the loop is honest: "the model said it's done" is never the finish line — the build, the tests, and the diff are.

## Install

```bash
pip install misterdev
# or
uv pip install misterdev
```

Python 3.10 – 3.13. Optional extras add capability without bloating the core install:

```bash
pip install 'misterdev[local-embeddings]'   # offline semantic context ranking (fastembed, no API key)
pip install 'misterdev[lsp]'                 # LSP semantic-diagnostics gate
pip install 'misterdev[web]'                 # headless-browser web verification gate (+ playwright install chromium)
pip install 'misterdev[mcp]'                 # Model Context Protocol tool-host substrate
```

Extras are all opt-in and timeout-bounded. When an extra's runtime dependency is absent, the gate it powers SKIPs rather than failing.

## What it does

### Autonomous build loop

Give misterdev a goal and it drives the whole cycle: analyze the project, plan tasks, edit, and validate — repeating until the goal is met or the budget is spent. Edits are **anchored SEARCH/REPLACE** hunks: the model emits only the changed regions, which are applied against the on-disk file, so a 5,000-line module is edited without reprinting it and without hitting the output-token ceiling. Matching tries exact first, then tolerates whitespace and indentation drift, always requiring a single unique anchor so a partial file is never written.

### Polyglot symbol-graph context

A tree-sitter symbol graph gives misterdev structural understanding of **Python, Rust, TypeScript/JavaScript, Go, Java, C/C++, C#, Swift, and Kotlin**. Per-file outlines plus a whole-project structural map feed planning and editing, and large files are sent as a symbol outline plus verbatim windows of the task-relevant symbols — so context and cost scale with the edit, not with the file.

### Correctness gates

Every change runs through an ordered gate sequence: **build → lint → tests → typecheck**, with optional gates layered on top — an **adversarial critic** (an independent second model that reviews each diff before it is applied), **goal-check**, **claim-verifier**, **mutation** scoring, **runtime-smoke**, **web**, and **vision** verification. A gate that fails **RED** blocks the change; a gate with nothing to check **SKIPs** and never blocks. Regressions are reverted via git, so a working tree only ever moves forward.

### Dynamic model selection

misterdev keeps a per-model **performance ledger** and pairs it with a **cost-aware selector** that picks for quality-per-dollar across the **full breadth of OpenRouter** — routing each task up a **capability ladder** (harvested free / cheap → a strong mid-tier → a frontier tier) and **escalating to a stronger model only when a cheaper one can't clear the gates**. The strongest tier is reserved for the final attempt, so frontier spend is the rare safety net, not the default; a hard task that a mid model stalls on is finished by a frontier model, while easy tasks resolve on free/cheap ones. Quality never drops because a weak model that writes bad code fails the gate and the policy climbs. It runs against **OpenRouter or Anthropic** with automatic failover, caches responses to avoid paying twice, and token budgeting keeps spend inside the ceiling you set.

### Parallel worktrees

Disjoint tasks run concurrently, each in its own **isolated git worktree**, so independent work doesn't contend for the tree. An **integration gate** re-checks each wave against the full suite and reverts any task that regresses it — parallelism without cross-contamination.

### Self-improving

misterdev keeps a durable, fingerprinted stream of its own real failures and runs an **AlphaEvolve-style keep-if-better loop** over its own source: it attributes what breaks, classifies *why* (harness artifact vs observation gap vs capability), proposes a targeted structural self-edit, and promotes it only when it beats the champion on a **held-out task set it never optimized against** — with zero regressions. A reward-hacking guardrail walls off the tests and benchmark. The result is a loop that removes whole failure classes over time **without overfitting** to any one benchmark. See [docs/path-to-100.md](docs/path-to-100.md).

On the correctness side, misterdev works **reproduction-first**: for an issue-driven task it synthesizes a failing test from the acceptance criteria, **validates that the test actually fails on the clean tree** (a test that reproduces nothing is discarded rather than trusted), then drives the fix to turn it green — so the model edits toward a concrete, verified target instead of a description.

**Two-timescale evolution** *(design; see [docs/two-timescale-evolution.md](docs/two-timescale-evolution.md))* takes the self-improvement further than a memoryless runtime agent can: the agent invents task-specific tools at runtime, and the ones that **prove they generalize** are consolidated — through the same held-out promotion gate — into a **persistent, best-per-capability tool library that future runs start from**. Fast loop explores; slow loop keeps the winners; the held-out gate keeps the library general rather than benchmark-overfit. Capability compounds across runs instead of being reinvented each task.

### Extensibility

Tools, gates, and targets **self-register through Python entry points**. `pip install misterdev-plugin-x` adds a capability with **zero edits to the core** — misterdev discovers the entry point at runtime and wires it in. A working example lives at [`examples/misterdev-plugin-hello`](examples/misterdev-plugin-hello). See [Extending misterdev](#extending-misterdev).

### Agentic MCP

misterdev can connect to **Model Context Protocol** servers and let the model call their discovered tools mid-build — bounded, opt-in, and constrained by a tool allowlist. Transports include stdio and **remote streamable-http with auth**, so you can point it at a hosted MCP gateway like **Glama** and give the build access to a whole catalog of tools without running any of them locally.

## Benchmarks

Gate-verified pass@1 on [Aider's polyglot benchmark](https://github.com/Aider-AI/polyglot-benchmark) (Exercism exercises with hidden test suites), `anthropic/claude-sonnet-4-6`:

| Language | Solved | Rate |
| --- | --- | --- |
| JavaScript | 9 / 10 | **90%** |
| Python | 8 / 10 | **80%** |
| Rust | 7 / 10 | **70%** |

A continuous stress run has solved **20/20** across the three languages with zero failures — including the exercises usually cited as hard (bowling, forth, arbitrary-precision decimal). Every solve is judged by the exercise's own hidden tests, not the model's say-so. Full numbers, methodology, and how to reproduce: **[docs/benchmark-results.md](docs/benchmark-results.md)**. Test suite: **1,897 passing** — **[docs/TESTING.md](docs/TESTING.md)**.

## CLI reference

**Don't want to remember flags?** If the first argument isn't a subcommand,
misterdev treats the whole line as plain English, maps it to an action with its
own model, shows you a preview, and asks before anything mutating:

```console
$ misterdev "fix the failing tests but keep it cheap, and run stuff in parallel"
  → I'll run: misterdev build . fix the failing tests --budget 5 --parallel
    proceed? [Y/n]
```

The flag-based commands below still work exactly as written, for scripts and
power users. The `misterdev` command drives everything:

| Command | What it does |
| --- | --- |
| `misterdev scan <dir>` | Discover projects under a directory and register them. |
| `misterdev list` | List all registered projects. |
| `misterdev status [path]` | Show a project's tasks and their state. |
| `misterdev report [path]` | Summarize the latest build's cost/tokens, per-model ledger performance, and the audit trail. Read-only — nothing is re-run. |
| `misterdev run [path]` | Run pending (or a specific `--task`) tasks for a project. `--dry-run`, `--force`, `--status`. |
| `misterdev plan [path]` | Analyze the project, recommend work, and compose a plan interactively. `--budget`, `--no-rollback`. |
| `misterdev build [path] [prompt]` | The autonomous build/debug/complete workflow. See flags below. |

Plain `misterdev` with no subcommand launches interactive planning.

<details>
<summary><strong><code>misterdev build</code> flags</strong></summary>

| Flag | Effect |
| --- | --- |
| `--budget <float>` | Max dollar budget for the run (default 100). |
| `--commit` | Commit after each completed task. |
| `--parallel` | Execute independent tasks concurrently in isolated worktrees. |
| `--dry-run` | Plan only; show tasks without executing. |
| `--interactive`, `-i` | Wait for confirmation between tasks. |
| `--no-verify` | Skip the final validation phase. |
| `--no-suggest` | Skip the suggest scan. |
| `--no-rollback` | Disable auto-bisect/revert of a regressing task. |
| `--focus <area>` | Restrict work to a specific area. |
| `--allow-dirty` | Allow building over uncommitted changes. |
| `--max-tasks <n>` | Cap the tasks this run will plan/execute (bounds cost). |

The `prompt` is free text or a mode word — `debug`, `complete`, `review`, or `new <description>`.
</details>

## Drive it from an AI client (MCP server)

misterdev also ships **as an MCP server** (`misterdev-mcp`), so you can drive it
in plain English from Claude Desktop, Claude Code, Cursor, or any MCP client —
no flags to remember. The client just calls a tool (`build`, `scan`, `status`,
`list_projects`, `run`); the **entire orchestration runs inside misterdev's own
process** with its own model and context budget, and only a short summary
returns to the client — your codebase never enters the client's context window.

```jsonc
// Claude Desktop config (claude_desktop_config.json)
{
  "mcpServers": {
    "misterdev": {
      "command": "misterdev-mcp",
      "env": { "OPENROUTER_API_KEY": "sk-..." }
    }
  }
}
```

Then just ask: *"Have misterdev add rate limiting to the API, keep it under $5."*
Mutating tools (`build`, `run`) refuse a dirty working tree and carry a
conservative default budget. Requires the `mcp` extra: `pip install 'misterdev[mcp]'`.

## Extending misterdev

A plugin is an ordinary Python package that declares entry points in the `misterdev.*` groups. Install it, and misterdev picks it up — no core edits.

A **tool** is a class; a **gate** is a callable returning a `GateOutcome`:

```python
# misterdev_plugin_hello.py
from misterdev.core.execution.outcomes import GateOutcome, GREEN, RED


class HelloTool:
    gather_safe = True  # opt into the agentic gathering loop
    gather_description = "Return a friendly greeting for a name."

    def __init__(self, config: dict):
        self.name = config.get("name", "hello")

    def execute(self, project, name: str = "world", **_ignored):
        return True, f"Hello, {name}!"


def no_shouting_gate(ctx) -> GateOutcome:
    build = (ctx.commands or {}).get("build_command") or ""
    if build and build.isupper():
        return GateOutcome(RED, "build_command is ALL CAPS; please calm down")
    return GateOutcome(GREEN)
```

```toml
# pyproject.toml — the entry points are the whole contract
[project.entry-points."misterdev.tools"]
hello = "misterdev_plugin_hello:HelloTool"

[project.entry-points."misterdev.gates"]
no_shouting = "misterdev_plugin_hello:no_shouting_gate"
```

Targets register the same way through the `misterdev.targets` group. The full, runnable example — tool, gate, `pyproject.toml`, and notes — is at [`examples/misterdev-plugin-hello`](examples/misterdev-plugin-hello).

## Configuration

Drop a `project.yaml` in the repo root. A minimal config names the language and the build/test/lint commands; everything else has a sensible default.

```yaml
name: "My App"
language: "python"
build_command: "python -m compileall -q ."
test_command: "pytest -q"
lint_command: "ruff check ."
llm:
  provider: "openrouter"            # openrouter | anthropic
  model: "anthropic/claude-sonnet-4-6"
  api_key_env_var: "OPENROUTER_API_KEY"
```

Key knobs:

- **Model & budget** — `llm.model`, provider/failover, and the run's dollar ceiling (also `--budget`).
- **Gates** — optional gates (adversarial critic, mutation, runtime-smoke, web, vision, goal-check) are off by default and enabled under the `orchestrator.*` keys.
- **MCP** — declare servers under `mcp.servers` and enable tool use with `orchestrator.mcp_enabled` / `orchestrator.mcp_tool_use`; point at a remote gateway for hosted tool catalogs.
- **Targets** — a `targets:` block gives a polyglot monorepo per-language build/test/lint, routed per task.

**Guides:** [Getting started](docs/getting-started.md) · [Configuration](docs/configuration.md) · [Plugins](docs/plugins.md) · [MCP](docs/mcp.md). `project.yaml.example` documents every configuration key.

## Requirements

- Python **3.10 – 3.13**
- **git** (branch-per-task, worktrees, and rollback all run through it)
- An API key for **OpenRouter** or **Anthropic**
- Optional per-gate toolchains — a Playwright browser for the web gate, a language server for the LSP gate, an MCP SDK for the tool-host substrate (all installed via the matching extra)

## Development

```bash
git clone https://github.com/dcondrey/misterdev
cd misterdev
uv sync
uv run ruff check .
uv run pytest -q
```

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md), and open an issue or a pull request on [GitHub](https://github.com/dcondrey/misterdev).

## License

misterdev is **dual-licensed**:

- **[AGPL-3.0-or-later](LICENSE)** — free for open-source use under the terms of the GNU Affero General Public License.
- **[Commercial license](COMMERCIAL_LICENSE.md)** — for use in a closed-source or proprietary product without AGPL obligations.

Choose the one that fits your project.

---

<p align="center">
  Built by <strong>David Condrey</strong> ·
  <a href="https://github.com/dcondrey/misterdev">github.com/dcondrey/misterdev</a><br>
  <sub>The static mark lives at <code>assets/logo.svg</code>.</sub>
</p>
