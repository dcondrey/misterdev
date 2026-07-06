# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Entries are grouped by [Conventional Commit](https://www.conventionalcommits.org/)
type.

## [Unreleased]

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
