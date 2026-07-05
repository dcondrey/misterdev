# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Entries are grouped by [Conventional Commit](https://www.conventionalcommits.org/)
type.

## [Unreleased]

The forthcoming **0.1.0** — first public release.

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

[Unreleased]: https://github.com/dcondrey/misterdev/commits/main
