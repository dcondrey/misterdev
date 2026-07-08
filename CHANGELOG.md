# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Entries are grouped by [Conventional Commit](https://www.conventionalcommits.org/)
type.

## [Unreleased]

### Added

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
