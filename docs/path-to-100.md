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
  Today: `attribution.Blame` ranks by *niche*, not cause. ✗ **(build first.)**
- **L3 Structural self-repair** — turn a classified failure into a *structural*
  fix (a guard, a seam, a strategy — removes the whole class), evaluated in a
  sandbox. Scaffold exists: `proposer` (blame→edit) ✓ · `sandbox` score-without-
  trust ✓ · `archive` MAP-Elites ✓ · `prior` mutation meta-learning ✓ · `loop`
  keep-if-better ✓ · `guardrail` reward-hacking wall ✓ · `failure_log` ✓ ·
  `warm_start` ✓. Gap: the proposer must be constrained to **structural** edits,
  and fed the L2 cause. ~
- **L4 Generalization gate (anti-overfit ratchet)** — the crux. Today the accept
  gate (`fitness`) is **regression-only** on the optimized set. ✗ **Add a held-out
  split** the fix is validated on but never derived from; accept only on held-out
  gain + zero regressions. (Spec below.)
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
- **M1 — Failure taxonomy (L2).** Deterministic classifier: run artifacts
  (edits produced, guard rejections, gate outputs, attempt history, budget state)
  → {artifact, observation, search, convergence, saturation} with evidence.
  *Accept:* on captured real failures, ≥90% agree with hand labels; every
  saturation label carries a ruled-out-others record (I3).
- **M2 — Held-out generalization gate (L4).** Add DERIVE/HOLDOUT/REGRESSION split
  and the accept rule above; wire into `loop`/`fitness`/`driver`.
  *Accept:* a synthetic overfit mutation (special-cases a DERIVE task) is
  *rejected*; a synthetic structural fix that generalizes is *accepted*.
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

## One-line status

Verified-search core, structural guards, faithful observation, and the full
evolution scaffold exist. The two missing pieces that convert "a loop that tweaks
and plateaus" into "a loop that removes failure classes and converges to 100%
without overfitting" are **L2 the cause-taxonomy (self-awareness)** and **L4 the
held-out generalization gate (anti-overfit ratchet)**. Build M0→M2 first.
