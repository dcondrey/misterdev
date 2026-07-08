# Continuous Learning & Self-Improvement: State and Plan

Analysis of misterdev's ability to learn across tasks and runs, and to analyze and
improve its own code. Every claim below is cited to source read on 2026-07-07.

> **Status (2026-07-07): all four tiers implemented and wired.** Tier 1 —
> `core/learning/failure_log.py` (real-build failure stream, fingerprinted +
> recency-weighted), wired at build Phase 6. Tier 2 — `core/learning/targeting.py`
> + `evolution --from-failures` + per-run benchmark cost. Tier 3 — lesson efficacy
> EWMA + quarantine in `core/planning/lesson_store.py`, credited from run outcomes
> in `metacognition.py`. Tier 4 — semantic lesson retrieval + `core/learning/
> warm_start.py`, both behind the existing embedding backend with lexical
> fallback. 33 new tests; full suite green except three pre-existing
> `python`-not-on-PATH grader tests. The gap analysis below is retained as the
> rationale; the plan tiers describe what was built.

---

## Verdict

misterdev is well past "an LLM in a while loop." Three distinct learning layers
already exist, each with real design thought — scored/decaying lesson memory, a
MAP-Elites archive of self-edits, a keep-if-better fitness rule with a measured
noise band, a mutation prior that meta-learns which edit kinds pay off. The pure
decision logic is seamed and testable.

The cap on how much smarter it gets is not the mechanisms. It is that **the two
persistent learners are disconnected, the code-evolver never runs from real usage,
and nothing measures whether any learned artifact actually helped.** Learning is
asserted (recurrence of an LLM rule) rather than demonstrated (measured effect on
outcomes). Closing those gaps is structural, not prompt-tuning.

---

## Current-state map

### Layer 1 — Reflexion (within one task)
`core/verification/reflection.py`

Accumulates root-cause reflections across retry attempts of a single task; attempt
N sees why 1..N-1 failed (`reflection.py:50-95`). Best-effort, timeout-bounded,
skips on any error. **Scope: ephemeral.** Dies when the task ends; nothing persists.

### Layer 2 — Lesson store (across runs, per project)
`core/planning/lesson_store.py` + `core/planning/metacognition.py`

The only persistent learner wired into the normal build loop. After each build an
LLM audits the session traces and emits 1-5 project rules
(`metacognition.py:34-71`), folded into a scored memory that:

- reinforces on recurrence, dedups rewordings by overlap coefficient, decays the
  untouched, evicts by *value* not age (`lesson_store.py:122-159`);
- retrieves the top-12 lessons relevance-weighted to the build goal
  (`lesson_store.py:162-180`), injected into the spec (`agent.py:650-658`).

The engineering here is genuinely good. **Scope: `project.path/.orchestrator/
lessons.json` — project-local** (`metacognition.py:27`). No cross-project transfer.

### Layer 3 — Evolution loop (self-improving its own code)
`core/evolution/*`

AlphaEvolve / Darwin-Gödel style. `run_evolution` (`driver.py:51`):
benchmark → attribute failures to the highest-blame niche (`attribution.py:82`) →
propose the smallest targeted self-edit (`proposer.py:87`) → apply in an isolated
git worktree → run the full gate suite as a hard precondition → benchmark → keep
only if it beats the champion past a measured noise band with zero regressions
(`fitness.py:59-79`, `loop.py:67-127`). Supporting pieces:

- **MAP-Elites archive** — keeps the elite per behavioral niche so stepping-stones
  survive (`archive.py`).
- **Mutation prior** — mines the archive for which *kinds* of edit land as elites
  and biases the proposer toward them (`prior.py`).

**Scope: dormant.** `run_evolution` is referenced *only* by
`core/evolution/__main__.py` — a manual CLI, dry-run by default. The normal build
loop never triggers it. Verified: grep for `run_evolution|evolve` outside
`evolution/` and tests returns nothing.

---

## Gap analysis

### G1 — The code-evolver never runs from real usage (highest leverage)

Attribution mines the synthetic polyglot benchmark
(`adapters.py:93-139`, `attribution.py:46`). Real builds produce exactly the signal
attribution wants — failed tasks, gate errors, classified error categories — but
nothing feeds them in. So misterdev improves its own code *only* in a lab, *only*
when a human runs the CLI. "Smarter the more it runs" is currently false for the
code layer: more real builds teach it nothing about its own weaknesses.

Compounding this: **the saved report discards the failure signal.** `to_dict`
persists failed tasks as bare ids (`report.py:117-118`) — no error text, no
language, no category. The rich `Task` error data exists in memory during the run
and is dropped at save. Closing the loop therefore requires *persisting* the
failure artifacts first, then feeding them to attribution.

### G2 — Lessons are validated by recurrence, not efficacy

The lesson store's quality signal is "an audit re-derived this rule"
(`lesson_store.py:1-18`). Recurrence correlates with usefulness but also with the
LLM re-emitting the same generic rule ("always run black") regardless of effect.
There is **zero counterfactual measurement** — grep for
`counterfactual|efficacy|ablat|helped` across the package is empty. A lesson is
never scored against whether the run it was injected into actually did better. This
is precisely the differential test ("would the outcome be worse without this?")
that the store does not apply to itself.

### G3 — No outcome attribution anywhere

Injected lessons, accumulated reflections, and archived candidates are never tied
back to the outcome of the run they influenced. Without that link, every "learning
makes it better" claim is an assumption. The infrastructure to measure it (per-run
reports with pass/fail counts and cost) already exists and is thrown away
un-joined.

### G4 — Retrieval is bag-of-words while embeddings sit unused

Lesson similarity and retrieval use destopped token-overlap
(`lesson_store.py:51-62`). A working embeddings subsystem already exists
(`core/economics/embeddings.py`) and is used only for model selection. Semantic
retrieval would surface a relevant past lesson that shares no literal tokens with
the current goal.

### G5 — No warm-start from past solutions

A new task similar to one already solved starts cold. The archive stores winning
self-edits by niche but there is no task → past-solution retrieval for ordinary
builds, so solved shapes are re-derived from scratch (a "smarter/faster" lever, not
just "smarter").

### Secondary findings

- **Cost objective is inert in live evolution.** `run_benchmark` always returns
  cost `0.0` (`adapters.py:110-139`), so the cost tie-breaker in `FitnessScore.beats`
  never fires on real runs. The load-bearing objectives (resolved-rate,
  regressions) are unaffected, but the "equal capability for less money wins" rule
  is currently a no-op live.
- **No feedback that a lesson caused a regression.** Decay is time-based, not
  outcome-based; a lesson that actively misleads only fades by disuse, never by
  evidence of harm.

---

## Sequenced plan

Ordered by leverage per unit risk. Each tier is independently shippable and behind
an opt-in flag until measured. Every tier states how success is *measured*, not
asserted.

### Tier 1 — Persist the real failure signal (unblocks everything)

**Change.** Extend the saved report to keep, per failed task: error text (bounded),
language, and `classify_error` category. Write a durable append-only
`.orchestrator/failures.jsonl` in the `BenchResult`-compatible shape attribution
already consumes (`adapters.py:38-61`).

**Why first.** G1 and G3 both need this; it is pure data capture with no behavior
change and no new spend. It is the seam the rest hangs off.

**Success criteria.** After a build with ≥1 failed task, `failures.jsonl` contains
one record per failure with non-empty error + language + category;
`attribute(load_failures())` returns a ranked blame map. Zero change to build
outcomes (data-only).

### Tier 2 — Close the real-build → evolution loop

**Change.** Add an opt-in trigger (config flag + CLI) that runs `run_evolution`
with attribution sourced from accumulated `failures.jsonl` instead of the benchmark.
Keep dry-run default and the full gate/worktree sandbox unchanged — the guardrails
are already correct (`loop.py:84-99`, `adapters.py:197-225`).

**Why.** Turns "self-improves in a lab" into "self-improves from what actually
breaks." The dangerous machinery (apply/gate/promote) already exists and is
guardrailed; this only changes the *targeting source*.

**Success criteria.** On a repo with a known recurring real failure mode, a live
run proposes an edit targeted at that mode, the sandbox gates gate it, and
promotion happens only on a measured resolved-rate gain past the noise band with
zero regressions. Reuse the existing benchmark as the promotion gate so a real-data
target still can't regress lab capability. Also fix the inert cost objective
(aggregate per-run spend into `run_benchmark`) so the tie-breaker is live.

### Tier 3 — Efficacy attribution for the lesson store

**Change.** Tag each injected lesson set with the run id it influenced; on the next
report, join outcome (resolved-rate / regressions delta vs the project's trailing
baseline) back onto those lessons. Reinforce on *measured help*, decay — or
actively demote — lessons that ride along without ever correlating with success, or
that co-occur with regressions. Recurrence becomes a prior, evidence becomes the
posterior.

**Why.** Converts G2/G3 from assumption to measurement, applying the differential
test to misterdev's own memory. Depends on Tier 1's persisted signal.

**Success criteria.** A synthetically-injected useless lesson decays below the
retention floor within N runs where it never correlates with a win; a genuinely
useful one accrues score faster than pure recurrence would give it. Measured via an
A/B: same task set with and without the efficacy weighting.

### Tier 4 — Semantic retrieval + warm-start (smarter *and* faster)

**Change.** Back lesson retrieval with the existing embeddings subsystem
(`economics/embeddings.py`), token-overlap as the graceful fallback. Add a
task → nearest-solved-task index so a new task warm-starts from the diff/approach of
its closest solved neighbor.

**Success criteria.** Retrieval surfaces relevant lessons with zero literal token
overlap (impossible today); warm-started tasks reach first-green in fewer
attempts/tokens than cold on a held-out set. Measure attempts-to-green and
tokens-to-green, not vibes.

---

## The continuous-state north star, positioned

The most promising direction on record is a never-off model with persistent,
evolving internal state — the server as its body — not an LLM re-invoked in a loop.
This existing machinery is the *substrate* that direction needs, not a competitor
to it:

- The lesson store + archive + prior are the **persistent state** such a system
  would evolve continuously rather than fold in at run boundaries.
- The fitness rule + gate sandbox are the **safety envelope** that keeps continuous
  self-modification from drifting into regression — the hardest part of an always-on
  self-editor, already built and guardrailed.
- Tier 1's failure stream is the beginning of **real senses**: the system feeling
  what actually breaks, continuously, instead of at benchmark time.

The honest gap between here and there: everything today is **batch and
boundary-triggered** (audit after a build, evolve on manual invocation). Continuous
state means the fold-in becomes incremental and always-on, and attribution runs on
a live stream rather than a saved report. Tiers 1-3 are the prerequisites that make
that transition an evolution of this codebase rather than a greenfield rewrite —
they turn the boundary-triggered learners into stream-consumers.

---

## What not to do

- **Do not tune audit/lesson prompts to raise lesson counts.** More lessons of
  unmeasured value is noise; the store already fights that with decay/eviction.
  Efficacy measurement (Tier 3) is the real fix, per prior guidance that structural
  gates beat keyword-matching prompt tweaks.
- **Do not auto-enable live evolution.** Off-by-default via explicit invocation is
  the correct gate for a system that self-edits and spends budget; keep the human
  in the loop for promotion until efficacy is demonstrated on real data.
- **Do not build cross-project lesson transfer before efficacy exists.** Propagating
  unvalidated lessons across projects multiplies unvalidated noise. Global transfer
  is worth doing *after* Tier 3 can tell a durable lesson from a lucky one.
