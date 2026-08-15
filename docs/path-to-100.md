# Path to 100%: self-improving, non-overfitting orchestration

The north-star plan. A single reference so we build the whole system without
losing the goal. Written 2026-07-08.

## The goal

misterdev solves **100%** of a fixed, verifier-equipped benchmark of solvable
tasks, and does so by *improving itself over time* with fixes that **generalize**
(never overfit). 100% is a convergence target, not a one-shot claim.

## Why 100% is reachable (the core claim, stated precisely)

With a verifier (the test gate), correctness is a **search** problem, not a
one-shot capability problem. If the model has nonzero probability `p` of emitting
a passing solution per attempt, verified resampling reaches pass with probability
`1 − (1−p)^k → 1`. So a verifier-equipped task fails in exactly three ways:

1. **Artifact** — a correct edit was *rejected* by a harness bug (effective `p=0`
   while true `p>0`). Removable. *(e.g. the dangling-ref guard false-positive.)*
2. **Search / budget** — true `p>0` but too small to hit in the attempts allowed.
   Addressable: guide the search (raise `p`), diversify, or spend more.
3. **True saturation** — true `p=0`: unreachable at the current model/tool tier.
   The *only* real 100% blocker.

**Empirical basis:** across this project's runs, every apparent "capability wall"
was category 1 or 2, never 3 — bowling/forth/decimal all passed once the artifact
was removed or the task re-structured; bowling passes in three languages. So for
this benchmark class, 100% is barred only by artifacts we haven't removed and
searches we haven't guided — both addressable.

## Invariants (non-negotiable)

- **I1 — Never reject a correct solution.** Every harness guard/gate must pass any
  solution the verifier would accept. Guards are structural and unit-tested
  against real correct+incorrect samples.
- **I2 — Never overfit.** A self-improvement is accepted only if it lifts a
  **held-out** task set it was *not* derived from, with zero regressions. Storing
  task→answer maps, editing tests/benchmarks/held-out data, or special-casing task
  identities is forbidden by construction.
- **I3 — Never mislabel saturation.** A "saturation / give up" verdict requires
  ruling out artifact, observation, search, and convergence *with evidence*.
  Default assumption for any failure is "removable," because assuming otherwise
  was wrong every time it was tried.
- **I4 — The verifier is ground truth.** Nothing completes unless the gate is
  green (or an objective build/typecheck gate verifies a no-test target).

## Architecture (layers → modules → status)

Legend: ✓ exists · ~ partial · ✗ to build.

- **L0 Verified-search core** — try→verify→retry until green, within budget.
  Gates (build/typecheck/test/acceptance) ✓ · structural guards (dangling,
  tamper) ✓ · strategy escalation + stall detector ✓ · staged decomposition ✓.
  Gap: **search diversity** — on repeat failure, vary the *approach*
  (re-decompose / different data model / different algorithm), not re-sample. ~
- **L1 Faithful observation** — the model sees exact ground truth on failure.
  `FailureView` (pytest/jest/cargo → exact expected/actual) ✓. Gap: un-truncated
  diffs (pytest `-vv`), more runners (go/junit/ctest), and **failure-triggered,
  single-shot execution probes** (run the one failing test, capture real output). ~
- **L2 Failure taxonomy (self-awareness)** — classify *each* residual failure by
  **cause**: artifact / observation / search-budget / convergence / saturation.
  `classify_failure` (`core/learning/failure_taxonomy.py`) wired into `driver.py`
  after blame attribution. ✓
- **L3 Structural self-repair** — turn a classified failure into a *structural*
  fix (a guard, a seam, a strategy — removes the whole class), evaluated in a
  sandbox. Scaffold exists: `proposer` (blame→edit) ✓ · `sandbox` score-without-
  trust ✓ · `archive` MAP-Elites ✓ · `prior` mutation meta-learning ✓ · `loop`
  keep-if-better ✓ · `guardrail` reward-hacking wall ✓ · `failure_log` ✓ ·
  `warm_start` ✓. Gap: the proposer must be constrained to **structural** edits,
  and fed the L2 cause. ~
- **L4 Generalization gate (anti-overfit ratchet)** — the crux. DERIVE/HOLDOUT
  split (`holdout.split_tasks`/`decide_promotion`) plus a paired McNemar advisory
  check (`paired.py`) wired into `driver.py`'s live promotion decision. ✓ Gap:
  the split is **in-distribution** (drawn from the same benchmark) — see D1 below,
  still open.
- **L5 Saturation escape hatch** — confirmed-saturation tasks route to capability
  escalation (stronger tier / tools / finer decomposition / flag), not endless
  mutation. ✗
- **L6 Convergence meter** — track held-out pass rate across self-improvement
  iterations; each accepted fix should remove ≥1 failure class and ratchet the
  rate up. Plateau ⇒ residual is saturation (escalate) or a gate too strict
  (loosen under I1). ✗

## The generalization gate (L4) — detailed, because it is the whole ballgame

Split the benchmark into three disjoint pools, fixed by seed:
- **DERIVE** — the failures the proposer is allowed to *look at* to design a fix.
- **HOLDOUT** — never shown to the proposer; the fix must *raise* its pass rate
  (or hold it while raising DERIVE) to be accepted. This is what distinguishes a
  general capability from a memorized answer.
- **REGRESSION** — the current champion's passing set; any drop is a hard reject
  (already in `fitness`).

Accept a mutation iff: `regressions == 0` **and** `holdout_delta ≥ 0` **and**
`derive_delta > noise_band` **and** the diff touches *structural* code
(gates/seams/strategy/guidance mechanisms) not answer-keyed data. Overfitting
raises DERIVE while HOLDOUT stays flat → rejected by construction. This is
train/dev/test discipline applied to self-modification.

Anti-gaming (extends `guardrail`): forbid edits to test files, benchmark data,
the held-out manifest, or the scoring code; forbid persisting task-id→solution
maps; require the reproduction corpus (`learning/reproduction.py`) to stay green.

## Build sequence (milestones, each with an acceptance metric)

- **M0 — Run the loop end-to-end once.** Confirm the existing scaffold produces a
  scored mutation against the live benchmark with the guardrail active.
  *Accept:* one full `run_evolution` cycle emits a champion + archive on disk with
  a real (even zero) delta. Removes the "never actually run" risk before extending.
- **M1 — Failure taxonomy (L2). DONE.** `classify_failure` runs on the blamed
  niche's sample error in `driver.py`, recording cause + evidence on `Blame`
  before proposal. Not yet independently re-verified against the ≥90%
  hand-label agreement bar this milestone specifies — that check is still open.
- **M2 — Held-out generalization gate (L4). DONE.** `holdout.split_tasks` +
  `decide_promotion` gate live promotion in `driver.py`; `paired.py`'s McNemar
  test runs alongside as an advisory (logged, not yet gating). The synthetic
  overfit/generalize acceptance-criteria check this milestone specifies has not
  been run as a dedicated test — worth adding before trusting the gate under
  adversarial pressure.
- **M3 — Structural-fix proposer (L3).** Feed L2 cause to `proposer`; constrain it
  to structural edits; reject non-structural diffs pre-sandbox.
  *Accept:* proposer, given an artifact-class blame, emits a guard/seam edit (not a
  prompt-keyword tweak) that passes M2's gate.
- **M4 — Observation completeness (L1).** pytest `-vv` un-truncation; go/junit/ctest
  parsers; failure-triggered single-shot probe.
  *Accept:* per-attempt pass rate on a fixed sample rises vs the M0 baseline (more
  guided search = higher `p`), measured, not assumed.
- **M5 — Search diversity (L0).** On repeated identical failure, force a different
  approach (re-decompose / alternate data model), bounded.
  *Accept:* a task that stalls under repetition converges under diversity on a
  captured case.
- **M6 — Saturation escape hatch (L5).** Route confirmed-saturation to escalation.
  *Accept:* a genuinely out-of-tier task is escalated, not mutated forever.
- **M7 — Convergence run (L6).** Iterate the loop across many cycles on the full
  benchmark; plot held-out pass rate.
  *Accept:* held-out rate is monotone non-decreasing across accepted mutations and
  approaches the ceiling; residual failures are all M1-classified saturation with
  an escalation route.

## Risks & kill-criteria

- **Metric-gaming is the real risk, not capability.** A score-optimizing loop
  finds degenerate maxima. Mitigation: I2 held-out gate + guardrail + structural-
  only + reproduction corpus. *Kill:* if an accepted mutation ever raises DERIVE
  while lowering HOLDOUT undetected, the gate is broken — stop and fix L4.
- **Held-out leakage.** If proposer context ever includes HOLDOUT, the gate is a
  lie. Enforce the split at the data boundary, not by convention.
- **Saturation over-assignment (I3).** The failure mode of *this project's own
  reasoning*: calling artifacts "capability." Every saturation label is adversarial-
  reviewed against the other four classes before it routes to give-up.
- **True out-of-tier tasks** cap 100% until L5 escalation exists; that's a
  capability lever (model/tools), not an orchestration bug — name it, don't hide it.

## Deepening (v2): where the first cut was too conventional

The v1 layers above are necessary but settle for textbook answers in five places.
The real leverage:

- **D1 — Held-out must be CROSS-DISTRIBUTION, not in-distribution.** In-distribution
  held-out catches task-memorization but not *benchmark-specialization*: since the
  benchmark is the target, "generalizes across it" and "overfits to it" coincide. A
  fix derived from a kata must lift a held-out pool of a *different kind* (real-repo
  bug / SWE-bench instance), or it is rejected as a benchmark trick. This replaces
  L4's HOLDOUT with a cross-domain HOLDOUT. It is the actual meaning of "never
  overfit."
- **D2 — The taxonomy (L2) must SELF-CORRECT from outcomes.** A static classifier
  inherits our own bias (we called artifacts "saturation" every time). Every
  "saturation/give-up" verdict is periodically **re-attempted under a changed
  condition**; a later pass is a labeled counterexample that moves the classifier's
  boundary. The cause-model learns its own error from a positive signal (the
  re-attempt outcome), not from hand-labels alone. Directly enforces I3.
- **D3 — Verifier decomposition (dense reward) is the primary `p`-raiser.** At 100%
  the binding constraint is tasks with tiny per-attempt `p`; you cannot out-search
  that. Raise `p` structurally by **synthesizing intermediate property/invariant
  checks** from the task's own tests and types, turning one low-`p` verified goal
  into a chain of high-`p` verified subgoals. Staged decomposition (already shipped)
  is the primitive; the general form densifies a sparse reward gradient. This is
  central, not a diversity tactic — it is what makes the hard tail reachable.
- **D4 — Run before refine.** M0 is the validation gate for the entire plan, not a
  warm-up. No run artifacts exist yet; the loop may not complete a cycle. Weight
  building over planning until one real accepted mutation exists.
- **D5 — Unify benchmark and open-ended under verifier-synthesis.** One engine
  (verified search), two verifier sources: *given* (benchmark tests) or *synthesized*
  (grounded acceptance criteria from a vague goal — `_ground_completion_spec` is the
  primitive). Open-ended work becomes tractable the moment a verifier is synthesized;
  then the same search + self-repair machinery applies. This collapses the two goals
  into one architecture.

Revised through-line: the two levers that actually reach 100% without overfitting
are **D3 (dense reward via verifier decomposition)** to raise effective `p` on the
hard tail, and **D1 (cross-distribution held-out)** to prove every fix is a real
capability. L2/L4 support them; everything else is plumbing. D4 still comes first.

## Deepening (v3): freedom, scope, and reachability — 2026-08-15

L2 and L4 shipped since v2 was written (see Architecture above), closing the two
gaps v2's one-line status called out. What's left is not "does the loop work" —
`run_evolution` end-to-end, gated, held-out-checked — but **how far it's allowed
to reach**, and **whether anyone ever runs it**. Six findings, from a direct read
of `core/evolution/`'s current source (not the v1/v2 design intent):

- **E1 — Reachability.** The scaffold has zero invocation surface beyond
  `python -m misterdev.core.evolution` — no CLI subcommand, no MCP tool, no
  `project.yaml` config section, no scheduled trigger. A correctly-designed loop
  nobody runs converges to nothing. Being wired now: a `misterdev evolve` CLI
  subcommand, an `evolve_async`/`job_status` MCP pair (reusing the existing
  `JobRegistry`), an `evolution:` config block, and an opt-in nightly CI job
  (`run_scheduled_evolution`, which already has the lock + circuit breaker this
  needs — it just has zero callers today).
- **E2 — Tool-invention and code-patch evolution are fully siloed.**
  `tool_invention.py`/`tool_promotion.py` (container-sandboxed helper-tool
  synthesis, gated behind `runtime_tooling`) and `driver.py`'s benchmark-gated
  code-patcher share no imports in either direction — two disconnected
  self-improvement mechanisms with two disconnected trust models. A blamed niche
  should be able to trigger *either* a code-patch proposal or a tool-invention
  proposal as two mutation kinds under the same guardrail/fitness/archive spine,
  not force every niche through one fixed mechanism.
- **E3 — The guardrail denylist walls off exactly the machinery most in need of
  improvement.** `core/verification/`, `core/learning/`, `llm/responses/` are
  correctly forbidden — letting evolution edit its own judge is the textbook
  reward-hacking hole. But this also means the critic's prompt, the
  acceptance-judge's framing, and the failure-taxonomy classifier can *never* be
  improved by evolution, and those are exactly the parts a same-model-review
  audit (see CHANGELOG `[Unreleased]`, 2026-08-14) found weakest. Proposed
  carve-out, not a loosened denylist: allow PROMPT-TEXT-only mutations (never
  control-flow) to specific judge files, gated by a second, independent model
  blessing the prompt-diff itself pre-sandbox, on top of the existing benchmark
  gate. Two structural gates on the one class of edit that's genuinely risky to
  allow, rather than a blanket unban.
- **E4 — Archive lineage is recorded, never mined.** `Candidate.parent_id` exists
  "so stepping-stone chains... can be mined from the archive" (archive.py's own
  docstring) — but the best-per-niche archive discards every non-winning
  candidate, and nothing reads `parent_id` back into a proposal. A locally-worse
  mutation that would have been a stepping stone toward a later win is lost the
  moment it loses once.
- **E5 — `favored_kinds()` is purely exploitative.** Mutation-kind selection
  weights by historical win rate with no explicit exploration term, so an
  under-tried kind (e.g. `contract-extraction`) can be starved indefinitely by an
  early lucky win in another kind. An epsilon-greedy or UCB term would keep the
  open-vocabulary kind space (tag-based, not a closed enum — proposer.py) genuinely
  explored instead of collapsing to whatever won first.
- **E6 — D1 (cross-distribution holdout) is concretely buildable now, not
  hypothetical.** D1 called for a holdout pool of "a different kind (real-repo
  bug / SWE-bench instance)" in the abstract; `evaluation/swebench/` (harness.py,
  grader.py, docker_runner.py, instance.py) is a complete, real-GitHub-issue
  harness that already exists as a separate evaluation tool, entirely unwired
  from `driver.py`. `run_evolution`'s `run_bench` parameter is already injectable
  — this is an adapter function (`evaluation.swebench.harness` in place of
  `evaluation.polyglot.harness`), not a new benchmark integration from scratch.
  Wiring it as the cross-distribution pool D1 specifies would be the single
  highest-leverage next step for the "never overfit" claim in this document's
  title, since the current L4 holdout is in-distribution and can't yet
  distinguish benchmark-specialization from real generalization.

E1 is being implemented now. E2–E6 are prioritization decisions, not autonomous
fixes: E3 and E2 change what an unattended, self-modifying loop is allowed to
touch, and E6 is real build effort against a harness that itself has open,
unresolved test failures (`test_swebench_harness.py`).

## One-line status

Verified-search core, structural guards, faithful observation, and the full
evolution scaffold — including L2 (cause-taxonomy) and L4 (held-out gate) — now
exist and are wired end to end in `driver.py`. What's missing has shifted from
"does the loop work" to "how far can it reach": cross-distribution holdout (D1 /
E6), a proposer that can choose tool-invention over a code patch (E2), and an
operational surface a human or agent can actually trigger (E1, in progress). L5
(saturation escape hatch) and L6 (convergence meter) remain unbuilt.
