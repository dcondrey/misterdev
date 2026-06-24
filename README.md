# Project Orchestrator

**Autonomous, test-gated development framework.**

Project Orchestrator drives an LLM through a deterministic *analyze → spec →
decompose → execute → validate → report* loop, applying surgical code edits and
merging only changes that keep your build, tests, and lint green. It is
polyglot, scales to large files, and verifies correctness before it ever
reports done.

---

## Capabilities

### Surgical editing that scales to large files
- **Anchored SEARCH/REPLACE edits.** The model emits only the changed regions;
  they are applied against the on-disk file, so a 5,000-line module is edited
  without reprinting it (and without truncating past the output-token limit).
- **Forgiving application.** Matching tries exact, then tolerates trailing-space
  / CRLF drift, then wrong indentation (re-indenting the replacement) — always
  requiring a single unique match, so a partial file is never written; a missed
  anchor retries instead.
- **Windowed context.** Large target files are sent as a symbol outline plus
  verbatim windows of the task-relevant symbols, so context (and cost) scales
  with the edit, not the file. Small files are sent whole.

### Polyglot understanding
- **Tree-sitter symbol graph** for Python, Rust, TypeScript/TSX, JavaScript/JSX,
  C, C++, C#, Swift, and Kotlin (best-effort).
- **Per-file outlines** (a symbol table of contents) and a **whole-project
  structural map** feed planning and editing.
- **Build/test/lint detection** for pytest, npm, cargo, SwiftPM, CMake/ctest,
  Meson, Make, and `dotnet`; **error classification** and **test-count parsing**
  across Rust/clang/gcc/swiftc/Roslyn, XCTest, ctest, and VSTest.

### Correctness verification
- **Tree-sitter syntax gate** catches real syntax errors before the expensive
  build, understanding strings/comments (a brace in a literal never false-fails).
- **Gate sequence** G1 build → G2 lint → G3 tests → G3.5 golden suite →
  **G3.6 optional mutation score** → G4 typecheck →
  **G4.5 optional LSP semantic diagnostics** → **G4.6 optional runtime smoke** →
  **G4.7 optional web verification** → **G4.8 optional vision verification** →
  G5 banned-marker → G6 secrets → G9 diff hygiene. Build/test/golden/typecheck
  failures block. An optional **goal-completion check** runs after the gates
  settle (advisory by default).
- **Optional container substrate** (`environment.type: docker`): gate commands
  run inside a throwaway, uid-mapped OCI container against the bind-mounted repo
  so the toolchain is pinned and reproducible. Rootless-first engine detection
  (podman → docker → nerdctl → colima); falls back to local execution when no
  engine is reachable. Git stays host-side.
- **Optional runtime smoke gate** (`runtime.smoke`): launches the built
  artifact, waits for a readiness signal, sends a probe, and asserts the
  expected response — a cheap end-to-end liveness check. Daemon-threaded with a
  hard timeout so it can never block; missing config or timeout is a SKIP.
- **Optional web verification gate** (`orchestrator.web_verify`, `runtime.web`):
  a headless browser (Playwright) optionally starts a dev server, loads a URL,
  and runs declarative `checks` — `dom:<selector>`, `text:<substring>`,
  `no-console-errors`, `axe` (accessibility), `screenshot` (pixel-diff vs a
  seeded baseline) — capturing a real screenshot as evidence. RED only on a
  genuinely failed check. Daemon-threaded with a hard timeout; no config or no
  Playwright/browser is a SKIP. Install with `.[web]` plus `playwright install`.
- **Optional vision verification gate** (`orchestrator.vision_verify`,
  `runtime.vision`): a vision model judges whether a captured screenshot
  satisfies a stated visual requirement (`assert`), affirm→GREEN / deny→RED.
  When the web gate also runs, its captured screenshot is reused automatically
  as the vision input (no need to repeat the path) unless `runtime.vision.capture`
  is set explicitly. Daemon-threaded with a hard timeout; no config / no model /
  no network is a SKIP. Uses the project's LLM client.
- **Optional MCP tool-host substrate** (`orchestrator.mcp_enabled`,
  `mcp.servers`): connects to configured MCP (Model Context Protocol) servers
  over stdio, discovers their tools (`project.mcp.tools`), and can call one
  (`project.mcp.call_tool`). When enabled, the discovered tools are described to
  the model in the task context so it knows they exist (awareness only —
  additive, the single-shot build loop is unchanged). Daemon-threaded with hard
  timeouts so a missing SDK, an unstartable server, or a hang is simply absent,
  never a block or an error. Install with `.[mcp]`.
- **Optional agentic MCP tool use** (`orchestrator.mcp_tool_use`,
  `orchestrator.mcp_max_tool_rounds`): off by default and purely additive on top
  of the substrate above. When on (and an MCP manager with discovered tools
  exists), a BOUNDED pre-edit loop lets the model request MCP tool calls to
  gather information before editing. Each round the model is shown the available
  tools and may reply with one line `CALL <server>.<tool> {json-args}` (or
  `NO_TOOL` to stop); the call runs through the timeout-guarded, never-raising
  `MCPManager.call_tool`, and the result is prepended to the task context. The
  loop is hard-capped by `mcp_max_tool_rounds` (default 3) and stops as soon as
  the model requests no tool. Any failure (no tools, model error, unparseable
  request, tool error) degrades to "gather nothing" — when the flag is off the
  executor path is byte-identical to today.
- **Optional mutation-score gate** (`orchestrator.mutation_gate`, `mutation`):
  runs the project's configured mutation-testing command (tool-agnostic — mutmut,
  cosmic-ray, cargo-mutants, Stryker, ...), parses a score, and RED-blocks (G3.6)
  only when it is below `mutation.min_score`. Proves the suite kills injected
  faults, not just passes. Daemon-threaded with a hard timeout; no config, an
  unparseable score, or a timeout is a SKIP (never a RED).
- **Optional goal-completion check** (`orchestrator.goal_check`): after the gate
  loop settles, an LLM judge reads the goal, the tasks' acceptance criteria, and
  the build's cumulative diff and reports whether the goal is actually met —
  "gates green != goal met". ADVISORY by default (gaps are recorded in the report
  and logged but do not fail the build); set `block_on_goal_gap` to make an unmet
  goal fail. Daemon-threaded with a hard timeout; no goal/criteria/client, an
  unparseable verdict, or a timeout is a SKIP.
- **Optional adversarial critic** (`orchestrator.adversarial_critic`, `critic.model`):
  an **independent second component** that reviews each candidate edit *before it
  is applied* and either approves it or returns concrete objections (misread
  requirements, missed edge cases, leaks, swallowed errors, security holes). A
  rejection feeds those objections back to the generator as the next attempt's
  context — a generate→critique→regenerate loop. Independence is the point: set
  `critic.model` to a **different** model so the reviewer doesn't share the
  generator's blind spots (a same-model critic still runs but is weaker, and that
  is logged). It reviews the **unified diff** of each change (what actually
  changed), and `critic.panel` > 1 runs that many reviewers concurrently through
  different perspective lenses (correctness / edge-cases / safety / requirements),
  rejecting only on a **majority** so a lone false rejection can't block. It is
  advisory, never authoritative — the build/test gates remain
  ground truth, and `critic_max_rejections` (default 2) caps how many
  regenerations it may force before deferring to those gates. Off by default and
  byte-identical when off; daemon-threaded with a hard timeout, so no client / an
  unparseable verdict / a timeout is a SKIP that lets the edit proceed.
- **Spec-as-tests** (`orchestrator.spec_as_tests`, currently DEFERRED): a tested
  generator (`core/spec_tests.py`) turns a task's acceptance criteria into a
  failing pre-implementation test, but it is not yet wired into the build loop
  (doing so would flip the integration-gate baseline red and disable that gate).
  Enabling the flag logs a deferral notice and changes nothing; the wiring seam
  is documented in `core/spec_tests.py`.
- **Regression safety:** branch-per-task, integration gate per wave with
  `git bisect`-style revert of the culprit, test-tamper detection, dirty-tree
  guard.

### Model & context orchestration
- Providers: **OpenRouter** and **Anthropic**, with failover, cost/budget
  ceilings, and **accurate token budgeting via tiktoken**.
- Optional dynamic model selection (UCB cost-aware ladder) and free-model
  harvesting.
- **Semantic context ranking** with pluggable embedding backends —
  `local` (offline [fastembed], no API key) or `openrouter` — blended with a
  lexical identifier signal; degrades to lexical-only if unavailable.

---

## Quick start

### Install
```bash
uv pip install -e .                      # core
uv pip install -e '.[local-embeddings]'  # + offline embeddings (optional)
uv pip install -e '.[lsp]'               # + LSP semantic gate (optional)
uv pip install -e '.[web]' && playwright install chromium  # + web verify gate
uv pip install -e '.[mcp]'               # + MCP tool-host substrate (optional)
```

### Configure — `project.yaml` in the repo root
```yaml
name: "My App"
language: "python"
build_command: "python -m compileall -q ."
test_command: "pytest -q"
lint_command: "ruff check ."
llm:
  provider: "openrouter"
  model: "anthropic/claude-sonnet-4-6"
  api_key_env_var: "OPENROUTER_API_KEY"
  embedding_backend: "auto"   # auto | local | openrouter | none
tools:
  - name: "Git"
    type: "git"
```

### Run
```bash
project-orchestrator scan ./projects                  # discover & register
project-orchestrator build ./projects/my-app "add OAuth2 login"
project-orchestrator build ./projects/my-app --dry-run "..."   # plan only
project-orchestrator                                  # interactive planning
```

---

## Architecture
- **`core/`** — state machine, models, decomposition, gates, topography (symbol
  graph + syntax gate), embeddings, optional LSP gate, container substrate
  (`container.py`), runtime smoke gate (`runtime.py`), web verification gate
  (`web_verify.py`), vision verification gate (`vision_verify.py`), mutation-score
  gate (`mutation_gate.py`), goal-completion check (`goal_check.py`), spec-as-tests
  generator (`spec_tests.py`, deferred), MCP tool-host substrate (`mcp.py`),
  governance layer (`governance.py`, risk classifier + approval gate) and
  append-only audit trail (`audit.py`).
- **`llm/`** — provider clients, failover, token budgeting, SEARCH/REPLACE
  response parsing.
- **`task_executors/`** — the try-test-fix inner loop and edit application.
- **`analyzers/`** — project assessment (structure, completeness, health).
- **`tools/`** — Git, command runner, formatters, file I/O.

---

## Safety & integrity
- Token/dollar **budget ceilings**; per-task and global caps.
- **Validation gates** block regressions; the **golden suite** is model-blind.
- **Branch-per-task** with revert-on-failure; **dirty-tree guard** refuses to
  run over uncommitted work.
- `--interactive` confirms each task; `--dry-run` plans without executing.
- **Governance layer** (opt-in, `orchestrator.governance: true`): a precise risk
  classifier gates executor commands that are destructive/irreversible/paid
  (`rm -rf`, `git push --force`, `DROP TABLE`, `kubectl delete`, `terraform
  destroy`, cloud `delete`, `docker system prune`, deploy/publish, pipe-to-shell)
  while ordinary build/test/lint commands run untouched. In autonomous mode a
  risky command is **blocked** with an escalation record unless
  `governance.auto_approve` is set. **Off by default — the command path is
  byte-identical to today when off.** Extra patterns via
  `governance.approval_required`.
- **Append-only audit trail** at `.orchestrator/audit.jsonl` (on by default,
  gitignored): one structured JSONL line per command run + exit, edit, and gate
  decision. Never raises into the build — an unwritable path degrades to a no-op.
- **Container egress control** (`governance.network: none`): runs gate commands
  with `--network none`. *Honest limit:* this constrains **containerized**
  execution only (`environment.type: docker`); host execution and git keep their
  normal network.
- **Container sandbox limits** (all opt-in, container-only, off path unchanged):
  `environment.memory` / `cpus` / `pids_limit` bound a runaway gate (fork bomb,
  memory hog); `cap_drop: ["ALL"]` drops Linux capabilities and `security_opt`
  (`no-new-privileges`, a `seccomp=` profile) hardens running model-generated
  code. The bind-mounted repo stays writable so build/test still work.

---

## Testing
```bash
uv run pytest -q                       # full suite
RUN_LSP_INTEGRATION=1 uv run pytest tests/test_lsp.py   # opt-in live LSP test
RUN_WEB_INTEGRATION=1 uv run pytest tests/test_web_verify.py     # live browser
RUN_VISION_INTEGRATION=1 uv run pytest tests/test_vision_verify.py  # live VLM
```
