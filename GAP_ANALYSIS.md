# GAP_ANALYSIS — project-completer (donor) → misterdev (canonical)

Phase 1, read-only. Verdict up front, evidence below. No code changed.

**Scope:** project-completer is a ~10k-LOC JavaScript MCP server (`/Volumes/A/autocoder/project-completer`) covering the same domain as misterdev (autonomously audit/fix/complete/build projects). misterdev is the maintained, published product (PyPI + MCP registry, `dcondrey`). Goal: reimplement the donor's *good ideas* as idiomatic Python, not merge the JS.

---

## TL;DR recommendation

**Port #1 — Reference-guided build (`reference_dir` digest).** Highest value-per-unit-effort. misterdev already solves the hard part (multi-language symbol extraction via tree-sitter topography across 9 languages); the donor's `reference-analyzer` is a thin digest layer on top of capability misterdev *already has*. It's additive, offline-testable, self-contained, fits "one reviewable commit + tests," and is directly the workflow you're using right now (porting a reference implementation into a target). Effort **S–M**, risk **MED**.

**Port #2 (strategic, not a quick port) — Async job lifecycle over MCP.** The donor's single most differentiated architecture: `start → run_id → status / pause / resume / stop`. misterdev's MCP `build`/`run` are **fully synchronous** — they block until the entire build finishes and return a string. For any non-trivial autonomous build over MCP, the client can't monitor, pause, or survive a timeout. Highest raw value, but effort **L** and risk **HIGH** (deep orchestrator rework, hard to test offline). Deserves its own dedicated effort, not folded into a quick port. **Critically: do NOT copy the donor's concurrency model — that's exactly where its 5 critical bugs live** (see "Bugs not to port"). misterdev's existing worktree/parallel scheduler is already race-safe; reuse it.

Everything else the donor does, misterdev already does at equal-or-better quality. Details below.

---

## Inventories (verified against source)

### project-completer — 18 MCP tools, 3 workflows

| Workflow | Tools |
|---|---|
| **Completion** (audit→fix→verify) | `start_completion`, `get_completion_status`, `get_completion_findings`, `get_completion_report`, `pause_completion`, `resume_completion`, `stop_completion` |
| **Evolution** (analyze→plan→approve→execute) | `analyze_project`, `create_evolution_plan`, `get_evolution_plan`, `approve_plan`, `execute_plan`, `get_execution_status` |
| **Spec build** (decompose→scaffold→implement→converge) | `build_from_spec` (supports `reference_dir` porting), `get_spec_build_status`, `stop_spec_build` |

Key `lib/*.js`: `orchestrator`, `evolution-orchestrator`, `spec-build-orchestrator`, `scheduler` (concurrency + file-lock + budget), `model-router` (complexity→model), `convergence` (diminishing-returns detector), `strategies` (per-stack audit prompts), `knowledge` (approve/reject learning), `library-intel` (ecosystem detect + `npm/pip audit`), `reference-analyzer` (extract interfaces/data-models/module-graph from a donor impl), `sandbox` (native/Docker), `write-queue` (serialize SQLite). Persistence: `better-sqlite3`.

### misterdev — 5 MCP tools, 8 CLI subcommands, deep subsystems

- **MCP tools:** `list_projects`, `status`, `scan`, `build` (plan+execute a goal, synchronous), `run` (execute planned tasks, synchronous).
- **CLI:** `scan`, `list`, `status`, `report`, `run`, `build`, `plan`, `interactive` (+ `mcp`), plus natural-language routing.
- **Subsystems:** planning (`decompose_spec`, `advisor.recommend_work`, `sovereign` strategies), verification (build/test/lint/typecheck/mutation/smoke/web/held-out-oracle gates), learning (`lesson_store`, `failure_taxonomy`, `warm_start`), evolution (`EvolutionLoop`, tool invention), economics (ledger-based UCB `model_selector`), context (tree-sitter topography for 9 languages), integration (`MCPManager`).

---

## Gap table

Legend — **Have**: equivalent / partial / missing. **Value/Effort/Risk** apply only to *missing* or *partial* rows worth porting.

| # | Donor capability | misterdev today | Have | Port value | Effort | Risk | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | **Reference-guided build** (`reference-analyzer`: extract interfaces / data-models / module graph from a donor impl, feed the builder) | `decompose_spec` + tree-sitter topography (9 langs) exist, but **no reference-porting layer**; `build` takes a goal string only | **missing** (spec decomposition partial) | **HIGH** | **S–M** | MED | **PORT FIRST.** Topography already does the hard multi-language extraction; this is a digest + prompt-wiring layer. |
| 2 | **Async job lifecycle over MCP** (`start`→`run_id`→`status`/`pause`/`resume`/`stop`) | MCP `build`/`run` are **synchronous, blocking, return a string** (verified: no `run_id`/job/thread/asyncio in `mcp_server.py`) | **missing** | **HIGH** | **L** | HIGH | **PORT SECOND, standalone.** Real defect for MCP use. Big lift; do NOT copy donor concurrency (buggy). |
| 3 | **Human-in-loop plan approval over MCP** (`create_plan`→`approve_items`/`reject_items`→`execute`) | Interactive `plan` CLI + `advisor.recommend_work`; **no MCP approve/reject surface** | **partial** | MED | M | MED | Defer. Overlaps existing planning; mostly an MCP-surface exposure of what CLI already does. |
| 4 | **Structured findings query** (`get_completion_findings` by severity/file/status) | Produces build *reports* + audit trail; **no queryable findings store** | **partial** | LOW–MED | M | LOW | Defer. Requires a findings store misterdev's model doesn't currently keep. |
| 5 | **Ecosystem security audit** (`library-intel`: `npm audit` / `pip-audit` / `cargo audit`) | `detection.py:139-148` **already** resolves `cargo audit` / `npm audit --omit=dev` / `pip-audit` as an advisory gate | **equivalent** | — | — | — | **SKIP — would duplicate.** |
| 6 | **Preference learning** (`knowledge`: approve/reject/success/failure) | `lesson_store`, `failure_taxonomy`, `warm_start`, metacognition | **equivalent** | — | — | — | Skip. |
| 7 | **Model routing** (`model-router`: complexity→opus/sonnet/haiku) | `model_selector` (ledger-driven UCB, quality-per-dollar under a validation floor) | **equivalent (better)** | — | — | — | Skip — misterdev's is superior. |
| 8 | **Convergence detector** (`convergence`) | `metacognition` / `advisor` diminishing-returns handling | **equivalent** | — | — | — | Skip (donor's has div-by-zero bugs anyway). |
| 9 | **Concurrency scheduler + file lock** (`scheduler`) | Parallel isolated git worktrees + integration gate + auto-bisect revert | **equivalent (safer)** | — | — | — | Skip. |
| 10 | **Sandbox** (native/Docker) | `container_env` + `evolution/sandbox` | **equivalent** | — | — | — | Skip. |
| 11 | **Per-stack audit prompts** (`strategies`) | `guidance/*` per-language conventions + analyzer prompts | **equivalent** | — | — | — | Skip. |
| 12 | **SQLite persistence + write-queue** | YAML/project registry + `.orchestrator/` progress (content-hash replay) | **equivalent (different model)** | — | — | — | Skip — no reason to adopt SQLite. |

**Net:** of 12 donor capability clusters, 8 are already equal-or-better in misterdev (skip), 2 are partial/low-priority (defer), and **2 are genuine high-value gaps** (rows 1 and 2).

---

## Bugs not to port (from donor `todo.md`: 5 critical / 14 high)

The donor's open criticals cluster almost entirely in its **async/concurrency machinery** — the exact subsystem behind Port #2. Reimplementing that surface on misterdev's existing race-safe worktree scheduler *avoids* these by construction; copying the JS structure would import them:

- **C-006 / C-009** scheduler: file lock not atomic with CLI exec; budget check reads stale value under concurrent dispatch → overruns.
- **C-007 / C-008** CLI wrapper: double-resolve on exit+error race; SIGKILL timer not cleared on normal exit (leaks event loop).
- **C-010 / H-020** server: no guard against concurrent `start_completion` on same dir (DB/file corruption); `project_dir` unvalidated (path traversal).
- **CLU-003 / CLU-004** systemic: concurrent-run corruption; resource exhaustion (DB conns / orchestrators / AbortControllers never released).
- **H-021 / H-022** convergence: div-by-zero and off-by-one → premature convergence (another reason to skip row 8).
- **SYS-001** silent error swallow across `git.js` / `fix.js` / `report.js` (catch-return-empty) — violates misterdev's "no silent swallow" baseline; do not reproduce.

---

## Recommended port order

1. **Reference-guided build** — `reference_dir` digest built on existing topography; new optional param on `build` (+ optional CLI flag), offline-testable against a fixture reference tree. **← start here.**
2. **Async MCP job lifecycle** — dedicated effort; new `run_id`-based tools alongside (not replacing) the synchronous ones; reuse misterdev's worktree scheduler; never copy the donor's concurrency code.
3. *(defer)* MCP plan-approval surface (row 3).
4. *(defer)* Findings query store (row 4).

Rows 5–12: **do not port** (duplicate existing, equal-or-better misterdev capability).

---

*Phase 2 will implement exactly one capability — your pick — as a single reviewable commit with tests, `ruff` + `pytest` green, no heavy deps, public surface intact.*
