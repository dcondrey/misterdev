# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Entries are grouped by [Conventional Commit](https://www.conventionalcommits.org/)
type.

## [Unreleased]

## [0.6.0] - 2026-07-27

Natural-language-first interface and Claude execution-backend integration.

### Added

- **Zero-friction natural-language CLI.** `misterdev "add rate limiting"` routes
  directly to an autonomous build — no subcommand, no devplan, no confirmation
  prompt. A fast-path heuristic handles coding verbs without an LLM call; query
  words (`what`, `how`, `show`, …) fall through to intent-parsing as before.
- **`spec_text` parameter on `build` (CLI + MCP).** Supply a pre-written
  implementation spec and misterdev skips its own analysis and spec-generation
  phases, going straight to decompose → execute → verify. Intended for Claude
  Code: Claude thinks and writes the spec, misterdev executes it with worktrees,
  gate verification, and parallel task management.
- **Analysis-phase skip for `spec_text`.** When a spec is caller-supplied,
  `_analyze()` is replaced by a zero-cost `ProjectAssessment()` stub, saving
  the full analysis run (LLM calls + build/test execution).
- **Animated Rich progress spinner.** Both `misterdev build` and the
  natural-language path show a live spinner whose text updates as tasks complete
  (`executing [3/7]`), replacing itself with a colour-coded Panel on finish.

### Fixed

- `misterdev build "goal"` — first positional arg is now treated as the goal
  when it is not a real directory path.
- `misterdev run "goal"` — same detection; redirects to the natural-language
  router instead of failing on a missing project path.
- `misterdev run` with no planned tasks now prints a hint to use
  `misterdev build [goal]` instead of silently doing nothing.
- Destructive verbs (`delete`, `remove`, `drop`, `destroy`, `wipe`, `erase`,
  `purge`) still prompt for confirmation even on the fast path.
- Redundant "→ I'll run:" echo suppressed for unambiguous fast-routed requests.
- Projects without a `project.yaml` are now auto-assigned a minimal one on
  first registration, preventing them from being pruned from the registry on
  the next startup.

## [0.5.0] - 2026-07-23

A large hardening + feature run: all planned tiers, the critical/high/medium audit
findings, and the async-job lifecycle over MCP.

### Added

- **Async job lifecycle over MCP.** `build_async`/`run_async` jobs now persist
  across a server restart (a job still `running` when the process died reloads as
  `interrupted`) and expose task-level progress — `done`/`total`/`phase`/`message` —
  surfaced by `job_status`. Persistent by default (`~/.misterdev/jobs.json`,
  `MISTERDEV_STATE_DIR` override).
- **`full_rewrite` escalation rung** — a structurally different retry (rewrite the
  whole target region) between context-widen and model-swap.
- **Escalation sub-steps execute as real gated child tasks** (depth-guarded);
  **proactive keystone splitting** of high-fan-in tasks; a **size/verifiability
  invariant** flags over-large or unverifiable decomposed tasks.
- **Runtime FailureLog read-back** (a task sees its own prior failures within the
  run) and a **schedulable, benchmark-gated evolution pass**.
- **compile_view per-language adapter registry** — real rust/typescript/go/swift/
  csharp diagnostic parsers — and **named-symbol definitions surfaced** on
  "cannot find X" / type-mismatch diagnostics.
- **Documentation tool (context7/fetch) mounted by default** (isolation-gated,
  opt-out via `mcp.docs_tool`).

### Changed

- **spec-as-tests is default-on** (advisory). The **mutation gate scores every
  changed source file**, not just the largest. The **edit region is never
  truncated** by the context budget. The **independent judge** detects a
  same-model config as non-independent instead of silently claiming independence.

### Fixed

- **CRITICAL — the integration gate reverts a wave that breaks the build.** A
  post-wave suite that no longer compiles/collects was waved through as unparseable
  (a false-GREEN); it is now reverted while genuinely ambiguous failures are still
  left alone.
- A **zero-test gate** and a **MANIFEST acceptance pass-through with no prior
  objective gate** are hard-rejected. **dotnet/VSTest counts sum across all
  projects**; the **C# classifier anchors on the Roslyn `error CSxxxx:` prefix**.
  The decomposer **prunes dependencies pointing at trimmed tasks**; model selection
  **widens instead of silently defaulting** when the top tier is empty;
  `RealTimeAligner` tolerates a malformed `consensus.json`; an unchanged file is no
  longer treated as fully mutated; deferral task ids containing ` - ` are preserved.

### Security

- **Reject an untrusted MCP `runtimeHint`** outside the npx/uvx allowlist, closing
  arbitrary-binary execution via a discovered server.

### Performance

- **SymbolGraph per-file index** (file-scoped queries are no longer
  O(all-symbols)); the **Gatekeeper git-diff is memoized** within one gate run
  (3 subprocess passes → 1).

### Removed / robustness

- **web-verify no longer leaks a dev-server** on a silent readiness wait;
  **worktrees are torn down** if batch prep raises; **venv setup is
  timeout-bounded**.
- Orphaned dead modules `held_out_oracle` and `early_abort`, and the unused
  `ModelSelector.is_ready`.

### Changed

- **Integration gate: identity-based regression detection on a red baseline.** The
  post-wave gate previously ran in COUNT mode when the suite started red —
  reverting a wave only if the failure *count* rose. That is blind to an
  offsetting change that fixes test A but breaks test B (net-zero count), which
  slips through as a real regression; it also can't tell a genuine fix from a
  no-op that merely kept the count. The gate now prefers IDENTITY mode when the
  baseline's failing-test set is parseable (via the existing FailureView
  parsers): it reverts a wave that introduces any test not failing at baseline,
  regardless of count, and surfaces a "no progress" signal when a wave resolves
  none of the baseline failures. Falls back to COUNT mode when the output can't be
  parsed, so behavior is unchanged where identities are unavailable. Surfaced by a
  dogfooding run where the count-mode gate had to catch a destructive stub.

### Added

- **Two-timescale evolution — a self-improving loop that compounds across runs.**
  The capability the current #1 open-source scaffolds lack: they invent
  task-specific tools at runtime and discard them every task; misterdev keeps the
  ones that generalize.
  - **Runtime tool-invention** (opt-in, `orchestrator.runtime_tooling`): mid-task
    the model may author a small Python helper tool; it runs **sandboxed** — a
    hardened, network-less container (`ToolRunner` over `ContainerEngine`: all
    Linux capabilities dropped, `no-new-privileges`, memory/CPU/PID caps, an
    isolated non-repo workdir, `--rm`). Untrusted model code never touches the
    host or the repo, and with no container engine the capability degrades to a
    no-op. The tool's output feeds the edit context.
  - **Held-out consolidation**: every invented tool is captured with its task's
    outcome into a persistent **tool corpus** (a free byproduct of normal runs); a
    deliberate promotion pass
    (`python -m misterdev.core.evolution.tool_promotion`) admits the tools whose
    with-tool success **generalizes** on a held-out task split — baseline drawn
    per-niche from the reproduction corpus — into a best-per-capability
    **`ToolLibrary`**, through the same anti-overfit gate scaffold evolution uses.
    Promoted tools then **seed future runs**, so capability compounds instead of
    being reinvented. Design: `docs/two-timescale-evolution.md`.

### Fixed

- **A no-usable-edit response no longer burns a solve attempt.** A response that
  is not code, whose anchored edit does not apply, or that contains no edit at all
  changed nothing on disk — it is a formatting failure, not a solve attempt — so
  it now grants a bounded extra iteration (the model still escalates a tier each
  pass; frontier stays reserved for the true final attempt) instead of consuming
  one of the retry budget's real attempts.

## [0.3.1] - 2026-07-09

### Added

- **Full-breadth model routing with a frontier escalation ladder** — the selector
  was on by default but its capability ladder was empty, so it could only route
  *down* to free/cheap models and never *up*. It now ships a default ladder
  grounded in the live OpenRouter catalog (harvested-free / cheap → mid → frontier)
  and routes each task by quality-per-dollar across the full breadth of models. The
  strongest tier is reserved for the **final attempt only**, so on a cold cell a
  hard task starts at the mid tier and escalates to a frontier model as the safety
  net — not on the first try. Verified live: a task cheaper tiers stalled on was
  resolved by the frontier tier on the final attempt, for pennies. The gate quality
  floor is unchanged, so routing only ever moves cost/latency, never shipped
  quality.
- **Reproduction-first with pre-patch validation** — the spec-as-tests workflow now
  runs its generated reproduction test on the **clean tree first** and keeps it only
  if it actually fails there; a test that reproduces nothing is a false gate (a
  wrong edit would also pass it) and is discarded rather than trusted as the target.
  Engaged automatically on both SWE-bench execution paths (host and containerized),
  where the judged tests are hidden and a validated reproduction is the one gate
  that targets the graded behavior.
- **Two-timescale evolution — self-authored tool library (consolidation layer)** — a
  new `ToolLibrary` over the existing MAP-Elites + held-out promotion machinery: a
  runtime-invented tool is admitted to a persistent, best-per-capability library
  **only if it passes the same held-out generalization gate** that guards scaffold
  self-edits, and promoted tools seed future runs. This is the memory a memoryless
  runtime agent lacks — capability compounds across runs instead of being reinvented
  each task. Pure and unit-tested; the runtime-invention surface is a follow-up.
  Design: `docs/two-timescale-evolution.md`.
- **Reproduction corpus + micro-eval screen (dense evolution fitness)** — the code
  evolver was starved by its fitness signal: one whole-suite benchmark run per
  candidate (minutes, dollars), where a single-edit mutation usually moves the
  aggregate rate by less than the noise band and registers as nothing. A new
  reproduction corpus (`.orchestrator/evolution/reproduction.json`) accumulates
  per-case pass/fail history across runs — the growing ground truth of what
  misterdev can and cannot do. A micro-evaluator then screens a candidate by
  running ONLY the cases it targets (currently-failing, in the blamed niche) plus
  a guard sample of currently-passing cases, in seconds instead of the whole
  suite: accepted iff it flips a target red→green and breaks no guard. `--beam N`
  proposes N candidates per step and spends the full-benchmark oracle only on the
  best screened survivor — widening the search without widening its cost. Opt-in
  (`--screen` / `--beam`); the full benchmark remains the promotion oracle, so the
  cheap screen only ever *pre-filters*, never promotes. Everything decision-side is
  pure and unit-tested with fakes; the corpus/selective-run adapters are thin.
- **Real-build failure stream** — every finished build now appends what actually
  broke (error, language, classified category) to `.orchestrator/failures.jsonl`
  in the shape failure attribution already consumes. Previously the saved report
  kept only failed-task ids, so the code evolver could learn only from the
  synthetic benchmark; it can now be aimed at real weaknesses. Errors are
  fingerprinted (line numbers / addresses / temp paths normalized away) so the
  same failure recurring across runs collapses to one high-value target, and
  records carry a recency weight. Best-effort: a learning stream never fails a
  build.
- **Evolution from real failures** — `python -m misterdev.core.evolution
  --from-failures` aims a self-edit at the highest-*weight* niche in the failure
  stream (recency × recurrence, so a nagging current failure outranks a pile of
  stale one-offs) instead of the benchmark's worst niche. The benchmark still
  runs as the promotion gate, so a real-failure-targeted edit still cannot be
  adopted unless it holds or improves benchmark capability with zero regressions.
  The proposer's wording is tagged with where the target came from.
- **Lesson efficacy attribution** — the scored lesson memory now reinforces on
  MEASURED help, not mere recurrence. Each lesson carries a stable id (surviving
  reword/refresh) and an efficacy EWMA of the outcome delta (run success rate vs
  the project's trailing baseline) of the runs it was injected into. Retrieval
  boosts proven-helpful lessons, damps freeloaders, and quarantines a lesson with
  enough evidence of correlating with worse-than-baseline runs (kept on disk, out
  of the prompt). Correlational credit, not an isolated A/B — but it separates a
  lesson that pulls its weight from one that just rides along, which recurrence
  alone cannot.
- **Semantic lesson retrieval + warm-start** — lesson ranking blends dense
  embedding similarity with lexical overlap when an embedder is available (reusing
  the project's embedding backend, which prefers a free offline model), so a
  lesson relevant by meaning surfaces even with no shared tokens; it degrades to
  lexical-only otherwise. A new solved-task index seeds each build's spec with how
  the most similar previously-solved tasks were approached, so a recurring shape
  starts from a proven approach instead of cold. Both best-effort, deduplicated,
  and bounded.
- **Per-run benchmark cost** — the polyglot harness now records misterdev's actual
  spend per instance and aggregates it into the suite report, so the evolution
  fitness function's cost tie-breaker (equal capability for less money wins) is
  live on real runs instead of always reading zero.
- **Self-reflection on failure (Reflexion loop)** — a failed gate now triggers a
  short root-cause reflection, accumulated across attempts and fed into the next
  one, so a retry debugs the underlying problem instead of re-patching the
  symptom. On by default (`orchestrator.reflection`); independent and
  timeout-bounded, SKIP-on-error so it only ever adds guidance.
- **Reproduction-first repair** — when `spec_as_tests` is on, the generated
  failing test's source is injected into the edit context as the concrete target
  ("make this pass"), so each attempt aims at an executable objective and
  converges via real feedback rather than the model's self-report.

### Fixed

- Acceptance-command extraction no longer runs a mangled command built from an
  LLM's trailing prose clause (which rejected correct code); it cuts at the first
  prose connective and strips a stray quote, keeping balanced quoted args.
- A context-assembly overflow can no longer hard-fail a task: `generate()` now
  middle-elides any prompt over a generous character ceiling for every caller.
- The decomposer right-sizes work — fewest tasks that cover it, never splitting
  one small file across several tasks.

## [0.2.2] - 2026-07-06

### Changed

- Enriched the MCP tool definitions for tool-definition quality: every parameter
  now carries a concrete example and constraints, each tool documents when to
  use it (and when not, and related tools), and titles describe the action with
  an explicit side-effects note.

## [0.2.1] - 2026-07-06

### Fixed

- Use an absolute logo URL in the README so it renders on the PyPI project page
  (relative image paths only resolve on GitHub).

## [0.2.0] - 2026-07-06

### Added

- **SWE-bench evaluation harness** (`evaluation/swebench/`) — run misterdev on
  real GitHub-issue tasks and grade the patch against the task's own hidden
  tests. Includes a Docker runner that executes the build/test gates inside each
  instance's official image, so a task runs in its exact environment.
- **Complete reference sites** — every external call site of a symbol being
  edited is surfaced up front, so a delete/rename/refactor updates them all in
  one attempt instead of chasing missed callers one build-error at a time.
- **Dangling-reference gate** — a deterministic pre-build check that rejects an
  edit which removes or renames a symbol while leaving references to it.
- **Prompt caching** — the stable context prefix is marked cacheable (Claude),
  so a task's retries re-read it at a fraction of the input cost; cache reads are
  priced accordingly in the budget and ledger.
- **Smarter model ledger** — hard-avoids models proven incompetent on a task
  cell, skips models proven too slow, warm-starts a cold cell from a model's
  global record (empirical-Bayes shrinkage), and reserves free models for the
  easiest tasks.
- **Adversarial critic** now also checks for symptom-vs-root-cause fixes and
  code duplication (DRY), and auto-enables for refactor/fix/integration tasks.

### Fixed

- The symbol graph now refreshes after each task instead of going stale for the
  rest of the run.
- A completed task's `status: completed` is committed into its source markdown,
  so a finished devplan is no longer re-run.
- An acceptance-command manifest error no longer false-fails a task whose real
  gates already passed.
- A reverted task's untracked orphan files are cleaned up (bounded to files the
  task created).
- Multi-file edits apply atomically, rolling back on a mid-batch write failure.
- An out-of-credits (HTTP 402) response halts the run gracefully instead of
  crashing with a stack trace.

## [0.1.0] - 2026-07-05

First public release.

### Features

- **Autonomous polyglot build orchestrator** — drives an LLM through an
  analyze → spec → decompose → execute → validate → report loop, applying
  surgical anchored edits across Python, Rust, TypeScript/JavaScript, C, C++,
  C#, Swift, and Kotlin.
- **Correctness gates** — a build → lint → tests → typecheck gate sequence
  merges only changes that keep the project green, with a tree-sitter syntax
  gate ahead of the expensive build, plus optional critic/goal-check/
  claim-verifier/mutation/runtime-smoke/web/vision gates.
- **Dynamic model selection** — a cost-aware ledger and selector pick models
  per task against a quality floor, with a self-assembling model ladder and
  response caching, over OpenRouter or Anthropic with failover.
- **Parallel worktree execution** — independent work runs concurrently in
  isolated git worktrees, with an integration gate that reverts regressions.
- **Pluggable tools, gates, and targets** — third parties extend misterdev with
  zero core edits via the `misterdev.tools`, `misterdev.gates`, and
  `misterdev.targets` entry-point groups.
- **Agentic MCP tool use** — a bounded gathering loop can call MCP tools,
  including remote gateways (e.g. Glama) over streamable-http with auth,
  constrained by a tool allowlist.
- **misterdev as an MCP server** — the `misterdev-mcp` entry point exposes
  build/scan/status/list/run as MCP tools so any AI client can drive misterdev
  conversationally; orchestration runs in-process, keeping the codebase out of
  the client's context window.
- **Natural-language CLI** — `misterdev "<plain English>"` maps intent to an
  action with misterdev's own model, previews it, and confirms before mutating —
  no flags to memorize. Subcommands still work unchanged.

[Unreleased]: https://github.com/dcondrey/misterdev/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/dcondrey/misterdev/releases/tag/v0.1.0
