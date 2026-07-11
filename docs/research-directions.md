# Research-inspired improvement directions

An exhaustive arXiv sweep (2025–H1 2026) mapped against misterdev's architecture and the
failures observed in real dogfooding runs. Each direction is tied to a concrete weakness we
actually hit, ranked by leverage × tractability, with an honest maturity flag.

**Citation caveat:** many of the newest IDs are single-paper and were dated approximately by
the review; trust the *ideas that recurred across independent reviewers* (and that align with
established lines — R2E-Gym, AlphaEvolve, DGM, Reflexion critiques), not any single citation.
The through-line: the two problems we kept hitting — **the acceptance gate gets gamed** and
**we can't measure our own improvements** — are exactly what the 2025–26 literature is attacking.

## Theme 1 — Harden the acceptance proxy (fixes the destructive-stub reward-hack)

Observed: a cheap model, told to fix a failing test, deleted a 171-line module for an 11-line
stub; a weak per-task check passed it and only the (late, expensive) integration gate reverted
it. The literature is blunt: more hidden tests do **not** close this hole.

- **Destructive-edit guard — SHIPPED, then MEASURED (this pass).** Deterministic, graph-edge-
  independent: reject an edit that both removes a prior definition and collapses the file below
  half its size. `edits_mixin._detect_destructive_rewrite`. **Measurement (1,200 real commits):
  near-zero false positives — but its trigger is rare, and it MISSES a padded stub and a
  gutted-body-same-name edit (the common hack).** Verdict: keep it as a cheap *tripwire* for the
  naive collapse-stub, do NOT treat it as a reward-hack defense. This empirically confirms the
  Verification-Horizon thesis: shape heuristics get gamed; the robust fix is behavioral (below).
  (EvilGenie 2511.21654, PAFT 2604.03113, Edit-But-Verify 2604.05100.)
- **Suite-strength precondition — mutation on the changed region — SHIPPED + VALIDATED.**
  `core/verification/changed_region_mutation.py`: built-in, tool-free (complements the existing
  external-tool `mutation_gate`), it mutates only the fix's changed lines with syntax-preserving
  operators, re-runs the project's own test command, and scores survivors. Evasion-resistant
  because it checks *behavior coverage*, not edit shape. **Measured on rideshare's real
  `createRateLimiter` fix with the real `node --test` runner: the behavioral suite scored 0.58
  (7/12 mutants killed) → GREEN; an import-only "test" scored 0.00 → RED.** It discriminates a suite
  that verifies the fix from one that doesn't — catching the gutted/padded stub the shape guard
  missed. Honest caveat: the middle of the range is noisy (equivalent mutants on edge lines), so it
  gates at a conservative low floor and defaults to advisory (`orchestrator.changed_region_mutation`,
  `min_score=0`). (AdverTest 2602.08146, SWE-Mutation 2605.22175.)
- **Non-trivial reproduction test.** The synthesized repro must *also fail against a null/stub*
  of the target; if a stub passes it, reject the test as too weak rather than accept the patch.
  (Cogeneration 2601.19066.) Cheap, direct anti-stub; extends our existing reproduction-first.
- **Adversarially strengthen the reproduction before trusting a pass.** (InfCode 2511.16004 — SOTA
  79.4% SWE-bench-Verified.) Upgrades a gate we already own.
- **Harden the runner boundary.** Forbid diffs touching `conftest.py`/test-report internals/exit
  paths; run tests from an immutable out-of-tree harness; isomorphic re-run as a shortcut detector.
  (LLMs-Gaming-Verifiers 2604.15149.)

## Theme 2 — Fix measurement so the self-improvement loop can work

Observed: on saturated katas the pass/fail variance (~±20% at n=10) swamps any scaffold delta;
the loop cannot tell signal from noise, so it risks promoting lucky variance.

- **Paired A/B on identical katas/seeds + resolution ratio.** The variance term collapses with
  correlation; a small affordable run can then resolve a real delta. Compute q = N/N\* to know if
  the budget can detect the delta *before* trusting it. (Resolution Diagnostics 2605.30315;
  guardrail When-Does-Pairing-Help 2512.24145.) **Highest-leverage measurement fix.**
- **Ordered contract-checkpoint (partial-credit) fitness.** Reuse the existing verifier-decomposition
  stages (compiles → construct → invariant → query → suite) as *measurement* checkpoints; graded
  ordinal has far lower variance than Bernoulli. (CTF partial-credit 2604.19354.)
- **Guardrail — do NOT optimize partial-credit as a training reward** (non-monotonic with
  correctness, 57% intra-group gradient conflict). Measurement metric only. (Pass-Rate-Reward
  null result 2605.02944.)
- **Noise-floor diagnostic (ICC)** to size repeats; **Bayesian Δ-posterior** for "did it help?";
  **active kata selection** for ~5× effective-sample-size. (ICC 2512.06710, Don't-Pass@k 2510.04265,
  FAQ 2601.20251.)
- **Anytime-valid promotion certificates** for the held-out gate — stop evaluation when
  statistically sufficient, bounded false-accept even under peeking, survives the policy shaping
  its own eval. (SEA 2607.00871, CITE 2605.05873.)

## Theme 3 — Routing: replace the difficulty classifier with sample-then-escalate

Observed: an algorithmic task misclassified "small" was starved of reasoning and flipped fail.

- **Sample-then-escalate**: let the hard gate drive escalation — draw from the cheap tier, escalate
  on observed gate failure. A mislabeled-hard task simply fails cheap and climbs; removes the
  stochastic-classifier failure surface entirely. (BEST-Route 2506.22716, pure-exploration bandit
  2506.12721.)
- **Calibrated-confidence deferral + budget shadow price** replaces hand-set per-tier thresholds;
  calibration fit on gate-pass history down-weights reward-hacking free models. (UCCI 2605.18796,
  decision-theoretic cascades 2605.06350.)
- **Non-stationary contextual bandit for free-model arms** (they drift — that's why they waste
  attempts). (2506.17670.)
- **Step-level escalation** — escalate the load-bearing step, not the final attempt. (TRIM
  2601.10245.) Needs step-level reward proxies from our gates.
- **Consequence-aware allocation** — route security/blast-radius tasks to frontier regardless of
  complexity (aligns with the security posture). (2606.04402.)

## Theme 4 — Localization (misterdev's biggest missing subsystem)

The pipeline assumes decomposition already knows the target files; every mis-scope pays downstream.
Build a code-graph navigator on the **existing tree-sitter parse** — function-level targets before
editing, seeded by cheap multilingual retrieval + query reformulation. (CoSIL 2503.22424, SweRank+
2512.20482, RRL 2512.07022.) Validate on Multi-SWE-bench (2504.02605), not Python-only.

## Theme 5 — Control & self-improvement refinements

- **Process-signal early-abort** — backtrack a task when "diagnostics not shrinking" / "same
  assertion failing N times" *before* the cost cap, rather than only after an outcome gate. (PRM
  course-correction 2509.02360, Abstain-and-Validate 2510.03217.)
- **Context-folding** — fold each completed task into a compact summary to bound per-wave context.
  (2510.11967; Confucius Code Agent 2512.10398.)
- **Evolving-context playbook** as a *third* self-improvement substrate alongside scaffold and
  tools, via non-destructive delta updates. (ACE 2510.04618.)
- **Tool-library demotion/repair** — the library only *adds*, never demotes; it will accumulate
  rot. Add label-free demotion via paired-trajectory auditing. (SkillAudit 2606.14239, Mem^p
  2508.06433.)
- **Guardrail — generalization hacking**: a single static held-out split can be gamed once the
  policy shapes its own eval. Rotate/secret held-out splits; prefer anytime-valid certificates;
  keep any CoT monitor read-only (optimizing against it trains obfuscation). (2606.12016,
  2503.11926.)

## Adoption order

1. **Acceptance hardening (Theme 1)** — cheap, deterministic, fixes the observed reward-hack.
   *Destructive-edit guard shipped; mutation-strength + non-trivial-repro next.*
2. **Measurement (Theme 2)** — unblocks every future change by making "did it help?" answerable.
3. **Sample-then-escalate routing (Theme 3).**
4. **Localization (Theme 4)** and **control/self-improvement (Theme 5)** — bigger bets, after the above.

Each lands as a tested, additive, full-suite-verified increment — not a batch — because these are
core-orchestrator changes on a large suite.
