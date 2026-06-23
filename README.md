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
  G4 typecheck → **G4.5 optional LSP semantic diagnostics** →
  **G4.6 optional runtime smoke** → G5 banned-marker → G6 secrets → G9 diff
  hygiene. Build/test/golden/typecheck failures block.
- **Optional container substrate** (`environment.type: docker`): gate commands
  run inside a throwaway, uid-mapped OCI container against the bind-mounted repo
  so the toolchain is pinned and reproducible. Rootless-first engine detection
  (podman → docker → nerdctl → colima); falls back to local execution when no
  engine is reachable. Git stays host-side.
- **Optional runtime smoke gate** (`runtime.smoke`): launches the built
  artifact, waits for a readiness signal, sends a probe, and asserts the
  expected response — a cheap end-to-end liveness check. Daemon-threaded with a
  hard timeout so it can never block; missing config or timeout is a SKIP.
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
  (`container.py`), runtime smoke gate (`runtime.py`).
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

---

## Testing
```bash
uv run pytest -q                       # full suite
RUN_LSP_INTEGRATION=1 uv run pytest tests/test_lsp.py   # opt-in live LSP test
```
