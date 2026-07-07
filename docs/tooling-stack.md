# Tooling stack: the sensors and effectors misterdev has per language

Tree-sitter is the floor, not the ceiling. An autonomous editor is only as good
as the ground-truth it can observe. This maps the 10-layer target against what
misterdev has today, and sequences the remaining work.

## Current status (verified against the code)

| # | Layer | misterdev today | Status |
|---|-------|-----------------|--------|
| 2 | Compiler/test ground-truth | `GateKeeper` G1/G3/G4: build, test, typecheck; auto-detected per language (`detection.py`) | **Strong** |
| 4 | Tree-sitter | 10-grammar symbol graph (`context/topography`), syntax gating, contract extraction (`context/contracts`) | **Strong** |
| 10 | Commit/change discipline | git rollback, isolated worktrees, integration gate, save-time format/lint hooks | **Strong** |
| 3 | Formatter/linter | G2 `lint_command` gate; formatters via hooks | **Partial** |
| 6 | Documentation | MCP tool-host (context7) callable mid-build | **Partial** |
| 8 | Security scanning | **G2.5 audit gate** — `cargo/npm/pip/dotnet` audit, advisory, SKIP-if-absent | **Shipped** |
| 7 | Static analysis | clippy + lint gate only; not miri/bandit/detekt/knip/periphery | **Shallow** |
| 1 | LSP semantic | `context/lsp.py` — off-by-default diagnostics **gate**, not a context source | **Partial** |
| 5 | Package/dependency mgmt | reads manifests to pick commands; no add / lock-file edit | **Weak** |
| 9 | Build-system understanding | detects build/test incl. meson/cmake/gradle | **Partial** |

Guidance (`context/guidance`, 11 languages, relevance-selected) is the *advisory*
form of layers 7–8: it tells the model to run clippy/miri/audit and use
constant-time comparison. G2.5 begins converting that advice into enforcement —
ground truth beats a prompt.

## Remaining build targets (leverage order)

### A. Static-analysis deepening (layer 7) — small
Auto-configure strict per-language linters as `lint_command` when unset, so the
tools the guidance recommends actually run. Per-language matrix:

| Language | lint (strict) | audit (shipped) |
|----------|---------------|-----------------|
| Rust | `cargo clippy -- -D warnings` | `cargo audit` |
| Python | `ruff check` | `pip-audit` |
| TS/JS | `eslint --max-warnings 0` | `npm audit --omit=dev` |
| C# | `dotnet format --verify-no-changes` | `dotnet list package --vulnerable` |
| Swift | `swiftlint --strict` | — |
| Kotlin | `detekt` | — |
| C/C++ | `clang-tidy` | — |

Files: `detection.py` (`detect_lint_command`), wire like `detect_audit_command`.

### B. Package/dependency management (layer 5) — medium
Let misterdev add dependencies and edit manifests instead of only reading them:
`cargo add`, `npm/pnpm add`, `uv add`, `dotnet add package`. Respect lock files
(a refactor must not churn them; a dependency add must). Surface as a bounded
tool the editor can call, gated behind manifest-diff review.

### C. LSP as semantic context (layer 1) — large, highest capability jump
Promote `context/lsp.py` from an off-by-default diagnostics gate to a semantic
*context source*: type info, hover, go-to-definition, and find-references fed
into planning and editing (not just post-hoc diagnostics). This is what closes
the gap between "syntactically plausible" and "semantically correct" edits.
Run the language server per project language as a subprocess over JSON-RPC;
cache resolved symbols; bound every request.

## Design notes
- **Advisory vs blocking.** G2.5 is advisory by construction — a security audit
  must not stall a build on an unfixable transitive CVE, and a missing tool must
  SKIP, not false-RED. A future `audit_blocking` config flag can promote it.
- **Uniformity is the trust lever.** An agent excellent at Rust and mediocre at
  Python is distrusted across the board. Every layer should reach every language
  in the stack (Rust, Swift, Kotlin, C#, C/C++, Python, TS, React, Elixir), not
  just the common ones.
- **MCP shape.** The end state is one server per concern (LSP-backed semantic,
  build, test, audit, docs, package-manager), orchestrated by the agent. G2.5,
  the lint/build/test gates, and the guidance registry are steps toward it.
