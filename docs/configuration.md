# Configuration

misterdev reads a `project.yaml` from the repo root. A minimal config names the language and the build/test/lint commands; everything else defaults. The shipped `project.yaml.example` documents every key — this page organizes the useful ones by area.

## Project basics

```yaml
name: "My App"
description: "A short description."
language: "python"
repo_path: "."
```

`language` seeds command auto-detection and (when containers are enabled) the default image.

## Build, test, and lint commands

These are the commands the gates run. They apply to the whole repo unless overridden per-target (see [Targets](#targets-polyglot-monorepos)).

```yaml
build_command: "python -m compileall -q ."
test_command: "pytest -q"
lint_command: "ruff check ."
```

Workflow bounds live under `build:`:

```yaml
build:
  max_tasks: 30              # also per-run via --max-tasks N
  max_attempts_per_task: 3
  max_consecutive_failures: 3
  build_timeout: 120
  test_timeout: 180
  lint_timeout: 120
  parallel_analysis: true
  budget: 100.0             # global spend ceiling; the master constraint (also --budget)
```

## LLM: provider, model, key, failover

```yaml
llm:
  provider: "openrouter"                  # openrouter | anthropic
  model: "anthropic/claude-sonnet-4.6"
  api_key_env_var: "OPENROUTER_API_KEY"   # env var name; token stays out of config
  temperature: 0.1
  failover:                               # tried in order if the primary errors
    - { provider: "anthropic", model: "anthropic/claude-opus-4-8" }
```

Dynamic model selection is on by default and self-regulating: `dynamic_selection: "auto"` explores cheap/free models on easy tasks, settles conservative once a task cell matures, and always uses the strongest tier on the final attempt. `use_free_models: true` harvests OpenRouter's rotating free models into the cheapest tier; the gates plus the strong final attempt keep output safe. Set either to `false` to opt out.

## Budget

The run's dollar ceiling is `build.budget` (default 100.0), overridable per run with `--budget`. When misterdev runs as an MCP server the default ceiling is a conservative $10 per call. The budget is the single master constraint that "auto" model selection optimizes against.

## Optional gates (`orchestrator.*`)

Beyond the always-on build/lint/test/typecheck gates, misterdev layers optional gates. All are off by default and enabled under `orchestrator.*`, with their config under sibling top-level keys. Each runs in a daemon thread with a hard timeout; missing config, a missing dependency, an unparseable result, or a timeout is a SKIP (never blocks, never hangs). Only a genuine failure is a RED that blocks.

```yaml
orchestrator:
  adversarial_critic: true      # independent second model reviews each edit pre-apply
  critic_max_rejections: 2
  goal_check: true              # LLM judge: is the goal actually met? (advisory)
  block_on_goal_gap: false      # true => an unmet-goal verdict fails the build
  mutation_gate: true           # assert the suite kills injected faults
  runtime_smoke: true           # launch the artifact and probe it
  web_verify: true              # headless-browser checks (needs [web] extra)
  vision_verify: true           # VLM judges a screenshot against a requirement
  spec_as_tests: true           # red->green TDD test generated per task (advisory)
  governance: true              # risk-classify destructive/paid executor commands
  enable_probes: true           # on by default; set false for a cheaper run
  auto_targets: false           # auto-detect polyglot sub-projects

critic:
  model: "anthropic/claude-opus-4-8"   # use a DIFFERENT model than the generator
mutation:
  command: "mutmut run && mutmut results"
  min_score: 0.8
runtime:
  smoke: { launch: "./app --serve", expect: "pong", probe: "ping", timeout: 30 }
  web:   { url: "http://localhost:5173", checks: ["dom:#app", "no-console-errors"] }
  vision:{ capture: ".orchestrator/web_verify_evidence.png", assert: "shows a chart" }
```

### Flaky-test quarantine (`orchestrator.flaky_reruns`)

A flaky test (a race, a clock, an ordering dependency) makes the test gate fail nondeterministically. Left unchecked it reverts a *correct* edit and burns attempts re-solving code that was never broken — the main hazard when misterdev runs on a repo whose suite it does not control.

```yaml
orchestrator:
  flaky_reruns: 2     # 0 (default) = strict single run; >0 = confirm before trusting a red
```

When `> 0`, a failed test gate is re-run up to this many times with **no code change**. A failure that does not reproduce is a flake: it is quarantined and the gate passes; a failure that reproduces every run stays RED. Applies to both the per-task gate and the integration gate. A rerun costs one extra test run only on an already-red gate. Default `0` preserves the strict single-run behavior.

### No-op test-gate warning (automatic)

Before a run, misterdev validates the resolved `test_command` on the pristine tree. If it exits clean but runs **zero** tests (a wrong path, a marker filter matching nothing, a misinferred runner), the run records a "No-op test gate" warning — the gate would otherwise pass every edit while catching no regression. Advisory; no config.

### Walk-away mode: park tasks that need you (`orchestrator.ask_when_stuck`)

The goal of `misterdev run --tasks <plan>` is: start it, walk away, come back to a finished project. Some tasks can't be finished by the model alone — a step needs a **credential** you must supply (a Cloudflare login, an API token), or it's a **judgment call** (a security review), or a requirement is **too ambiguous** to resolve safely. Rather than fail (and stall the run) or guess, misterdev **parks** such a task with a specific question and keeps going.

```yaml
orchestrator:
  ask_when_stuck: true   # default; set false to restore hard-fail behavior
```

When on, a task the model tries but can't complete/verify — or one where it emits `NEEDS_INPUT: <question>` because only you can decide — is set aside (its work reverted), never counted as a failure, and never aborts the run. Its dependents are parked too (they resume once it does). At the end, the parked tasks and their questions are written to **`.orchestrator/QUESTIONS.md`**:

```markdown
## T004 — Worker Env, constants, wrangler bindings
- Reason: blocked — not authenticated with Cloudflare
- Question: Provide Cloudflare credentials, or say how to proceed.
- Answer: _(write your answer here)_
```

Answer inline (or, for a missing credential, just provide it in your environment), then **re-run the same command**: answered tasks resume with your answer injected as a directive, unanswered ones stay parked, and already-completed tasks are skipped. A run's `--budget` ceiling and the early-abort monitor still bound cost, and because parked tasks don't retry, a broken dependency parks its whole subtree instead of burning the budget on it.

### Requirements preflight: gather inputs up front (`orchestrator.gather_requirements`)

Parking mid-run is the safety net; the preflight is the front door. Before executing, misterdev **reviews the whole plan** for inputs only you can supply — credentials, cloud accounts, tokens (a deterministic scan of task text; add `orchestrator.requirements_llm_review: true` for one extra LLM pass) — and writes them to **`.orchestrator/REQUIREMENTS.md`**, each marked satisfied ✓ / missing ✗ with how to provide it.

```yaml
orchestrator:
  gather_requirements: true      # default; false skips the review
  requirements_llm_review: false # add an LLM pass for non-obvious needs
```

Then the **smart gate** decides whether to spend: it stops before execution **only** when a *missing* input is needed by a **foundational** task (one whose fan-out — transitive dependents — is large), because running would just park that whole subtree. A missing input needed only by **late/leaf** tasks (a real deploy, an npm publish) doesn't stop the run — those proceed and park at the end. Secret *names* the build configures but doesn't need the *value* of (e.g. an app's own `ADMIN_TOKEN`) are listed as advisory and never gate.

Pass `--proceed` to skip the stop and run immediately (parking anything missing). Provide the flagged inputs — set the credential in your environment, or answer a decision in `REQUIREMENTS.md` — then re-run.

## MCP (Model Context Protocol)

Declare servers under `mcp.servers`, then enable awareness and/or the agentic gathering loop under `orchestrator.*`. A remote gateway uses `transport: http` (or `sse`) plus a `url` and a Bearer token from `api_key_env`. `mcp.allow_tools` is an allowlist of `server.tool` (or bare tool) names.

```yaml
orchestrator:
  mcp_enabled: true         # describe discovered tools to the model (additive)
  mcp_tool_use: true        # bounded pre-edit loop: model may CALL tools to gather
  mcp_max_tool_rounds: 3    # hard cap on gathering rounds
mcp:
  servers:
    - name: "docs"                 # routing key for tool calls
      command: "my-mcp-server"     # stdio subprocess
      args: ["--root", "."]
      transport: "stdio"
    - name: "glama"                # hosted gateway fronting many servers
      transport: "http"            # http | streamable-http | sse
      url: "https://glama.ai/mcp/..."
      api_key_env: "GLAMA_API_KEY" # Bearer token read from the environment
  allow_tools: ["docs.search", "glama.web_fetch"]
```

Needs the `mcp` extra. See [mcp.md](mcp.md) for the full flow.

## Targets (polyglot monorepos)

For a monorepo with sub-projects in different languages, declare each with its own toolchain. A task's gate is routed to the target that owns its files; a matched target is self-contained (only its listed commands run — others are skipped, not inherited). The top-level `build_command`/`test_command`/`lint_command` apply only to files outside every target. Omit `targets` entirely for a single-language repo.

```yaml
targets:
  - name: core
    path: emathy-core
    build_command: "cargo build -p emathy-core"
    test_command: "cargo test -p emathy-core --lib"
  - name: web
    path: clients/web
    build_command: "npm run typecheck"
    web: { serve: "npm run serve", url: "http://localhost:8000", checks: ["#app"] }
```

Zero-config alternative: `orchestrator.auto_targets: true` auto-detects sub-projects (best-effort commands). An explicit `targets` list always wins. Custom build systems can be taught via a target plugin — see [plugins.md](plugins.md).

## External task lists (`tasklist` / `run --tasks`)

Point misterdev at a hand-written task list — in whatever shape it is — instead of the `devplan/` directory:

```bash
misterdev run ./my-project --tasks /path/to/PLAN.md   # the list may live in another repo
```

or set it in config: `tasklist: "PLAN.md"` (relative to the project, or an absolute path).

The parser is format-agnostic — **JSON, YAML, Markdown, or plain text**; ordered or unordered lists; one task per line or multi-line with sub-bullets; organized into **phases**; and it reads a **dependency table** (`| Task | Blocked By |`) if present. Field names are alias-mapped (`success_criteria`/`done_when` → acceptance, `blocked_by`/`requires` → dependencies, `relevant_files`/`files` → target files, …), and dependency references resolve by id, title, or task number. Anything too messy for the deterministic parser falls back to LLM normalization.

Parsed tasks flow into the same engine as a devplan: dependency-aware **topological ordering**, **parallel waves** for independent tasks (a dependency table is what unlocks the parallelism), wave-level regression gating, and progress-based **resume** (it tracks which task is active and re-runs only what changed). Preview the plan with `--dry-run`; see which tasks would run vs. skip with `--status`.

## Most useful keys at a glance

| Key | Default | Purpose |
| --- | --- | --- |
| `language` | — | Language for detection and defaults. |
| `build_command` / `test_command` / `lint_command` | auto-detected | Gate commands. |
| `build.budget` | `100.0` | Master dollar ceiling (also `--budget`). |
| `build.max_tasks` | `30` | Task cap (also `--max-tasks`). |
| `llm.provider` | `openrouter` | `openrouter` or `anthropic`. |
| `llm.model` | `anthropic/claude-sonnet-4.6` | Primary model. |
| `llm.api_key_env_var` | `OPENROUTER_API_KEY` | Env var holding the key. |
| `llm.failover` | `[]` | Ordered fallback provider/model list. |
| `llm.dynamic_selection` | `"auto"` | Ledger-driven cost-aware selection. |
| `llm.use_free_models` | `true` | Harvest free models into the cheap tier. |
| `orchestrator.flaky_reruns` | `0` | Re-run a red test gate N times; a non-reproducing failure is a quarantined flake. |
| `orchestrator.ask_when_stuck` | `true` | Park a task that needs input (credential/judgment/ambiguity) with a question in `.orchestrator/QUESTIONS.md` instead of failing; the run keeps going. |
| `orchestrator.gather_requirements` | `true` | Review the plan up front for needed credentials/accounts (`.orchestrator/REQUIREMENTS.md`); stop before spending only if a missing one is foundational. `run --proceed` overrides. |
| `orchestrator.adversarial_critic` | `false` | Independent pre-apply edit review. |
| `orchestrator.goal_check` | `false` | Post-build goal-completion judge (advisory). |
| `orchestrator.mcp_enabled` | `false` | Inject discovered MCP tool awareness. |
| `orchestrator.mcp_tool_use` | `false` | Enable the bounded gathering loop. |
| `orchestrator.mcp_max_tool_rounds` | `3` | Cap on gathering rounds. |
| `orchestrator.auto_targets` | `false` | Auto-detect polyglot targets. |
