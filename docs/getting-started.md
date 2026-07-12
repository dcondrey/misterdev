# Getting Started

misterdev is an autonomous LLM build orchestrator. Point it at a repository and a goal; it reads the codebase as a symbol graph, decomposes the goal into tasks, edits your code with anchored SEARCH/REPLACE hunks, and runs every change through correctness gates (build, tests, lint, typecheck) before it reports done. A change that regresses the suite is reverted through git. Nothing merges unless it stays green.

## Install

```bash
pip install misterdev
# or
uv pip install misterdev
```

Python 3.10 – 3.13. `git` is required (branch-per-task, worktrees, and rollback all run through it).

Optional extras are opt-in and each SKIPs cleanly when its runtime dependency is absent:

```bash
pip install 'misterdev[local-embeddings]'   # offline semantic context ranking (no API key)
pip install 'misterdev[lsp]'                 # LSP semantic-diagnostics gate
pip install 'misterdev[web]'                 # headless-browser web verification gate
pip install 'misterdev[mcp]'                 # Model Context Protocol tool-host substrate
```

### Man page

`man misterdev` works once the install prefix's `share/man` is on your `MANPATH` (typical for system, `--user`, and `pipx` installs). From a source checkout, view it directly with `man ./man/misterdev.1`, or install it for your user:

```bash
install -Dm644 man/misterdev.1 ~/.local/share/man/man1/misterdev.1
```

## Set an API key

misterdev runs against OpenRouter or Anthropic. The key is read from an environment variable named by `llm.api_key_env_var` (default `OPENROUTER_API_KEY`); it never lives in config on disk.

```bash
# OpenRouter (default provider)
export OPENROUTER_API_KEY="sk-or-..."

# or Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
```

If you use Anthropic, set `llm.provider: anthropic` and `llm.api_key_env_var: ANTHROPIC_API_KEY` in your `project.yaml` (see below).

## Create a minimal project.yaml

Drop a `project.yaml` in the repo root. A minimal config names the language and the build/test/lint commands; everything else has a sensible default.

```yaml
name: "My App"
language: "python"
build_command: "python -m compileall -q ."
test_command: "pytest -q"
lint_command: "ruff check ."
llm:
  provider: "openrouter"            # openrouter | anthropic
  model: "anthropic/claude-sonnet-4.6"
  api_key_env_var: "OPENROUTER_API_KEY"
```

The commands are the truth misterdev trusts: a gate is only as good as the command behind it. See [configuration.md](configuration.md) for the full surface.

## Run your first build

Three equivalent ways to drive it.

### Interactive form (nothing to remember)

Just run `misterdev` with no arguments (or `misterdev i`) for a guided menu — pick Build / Run a task list / Debug / Complete / Status / Report, answer a couple of prompts (with sensible defaults), confirm, and it runs:

```console
$ misterdev
  What do you want to do?
    1. Build — describe a goal; misterdev writes/fixes/completes the code
    2. Run a task list — execute a plan file (any format) or the devplan/ dir
    3. Debug — find and fix everything that's broken
    …
  ? Choose (1)
```

### Flag form (scripts, power users)

```bash
misterdev build . "add rate limiting to the public API" --budget 5
```

`.` is the project path; the quoted text is the goal. `--budget` caps the run's dollar spend (default 100).

### Natural-language form (no flags to remember)

If the first argument isn't a known subcommand, misterdev treats the whole line as plain English, maps it to a command with its own model, previews it, and asks before anything mutating:

```console
$ misterdev "add rate limiting to the API but keep it cheap"
  → I'll run: misterdev build . add rate limiting to the API --budget 5
    proceed? [Y/n]
```

## Expected output

A build streams its plan, per-task edits, and gate results, then a summary:

```console
$ misterdev build . "add rate limiting to the public API"

  planning   goal → 3 tasks  (model: anthropic/claude-sonnet-4.6, budget $100.00)
  task 1/3   middleware: token-bucket limiter          api/limiter.py
    edit     1 hunk applied · syntax ok
    gates    build GREEN · tests GREEN (142 passed) · lint GREEN · typecheck GREEN
  task 2/3   wire limiter into request pipeline         api/app.py
    gates    build GREEN · tests RED (1 failed) → rolling back, regenerating
    edit     2 hunks applied (attempt 2)
    gates    build GREEN · tests GREEN (145 passed) · lint GREEN · typecheck GREEN
  task 3/3   docs + config surface                      README.md, config.py
    gates    all GREEN

  done       3/3 tasks · 145 tests green · $0.38 over 11 calls
```

Each gate reports GREEN (passed), RED (failed, blocks the change), or SKIP (nothing to check, never blocks).

## Other commands

| Command | What it does |
| --- | --- |
| `misterdev` / `misterdev i` | Guided interactive menu (the no-argument default) — no flags to remember. |
| `misterdev scan <dir>` | Discover and register projects under a directory. |
| `misterdev list` | List registered projects. |
| `misterdev status [path]` | Show a project's tasks and their state. |
| `misterdev report [path]` | Summarize the latest build's cost, per-model ledger, and audit trail (read-only). |
| `misterdev plan [path]` | Analyze, recommend, and compose a plan interactively. |
| `misterdev run [path]` | Run already-planned pending tasks (`--tasks <file>` for an external list in any format, `--task`, `--dry-run`, `--force`, `--status`). |
| `misterdev build [path] [prompt]` | The autonomous build/debug/complete workflow. |

Plain `misterdev` with no subcommand launches interactive planning.

Useful `build` flags: `--dry-run` (plan only), `--parallel` (independent tasks in isolated worktrees), `--commit` (commit after each task), `--interactive`/`-i`, `--max-tasks <n>` (bound cost), `--focus <area>`, `--allow-dirty`, `--no-rollback`, `--no-verify`. The `prompt` may be free text or a mode word: `debug`, `complete`, `review`, or `new <description>`.

Next: [configuration.md](configuration.md) for the full `project.yaml` surface, [plugins.md](plugins.md) to extend misterdev, and [mcp.md](mcp.md) to drive it from an AI client or give a build access to MCP tools.
