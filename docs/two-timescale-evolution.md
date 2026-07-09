# Two-timescale evolution: runtime tool invention + offline consolidation

## The idea in one line
The agent invents task-specific tools at runtime (fast, cheap, ephemeral), and the
tools that *prove they generalize* are consolidated — through the existing
held-out promotion gate — into a persistent, self-authored tool library that
future runs start from. Fast loop explores; slow loop keeps the winners.

## Why this is a step past the current #1 systems
Read from source (July 2026), not papers:

- **live-SWE-agent** (#1 open on SWE-bench Verified, 79.2%) is `mini-swe-agent`
  plus a config/prompt. Its entire edge is: (a) the agent writes its own Python
  helper tools *at runtime* to aid the current issue, and (b) a per-step nudge
  ("decide if there are any tools you can create") that keeps that pressure on.
  temperature 0, single trajectory, no best-of-N. **But every tool it invents is
  thrown away when the task ends.** It re-derives the same edit tool, the same
  reproduce-and-minimize script, on *every* instance. It has no memory.
- **Darwin-Gödel Machine / offline scaffold evolution** is the opposite: it
  accumulates, but only at the coarse grain of whole-scaffold code edits, and only
  across full generations (80 iterations, ~2 weeks). Too slow and too blunt to
  harvest the fine-grained tools an agent naturally invents mid-task.

Neither system has the bridge. **The novel contribution is the bridge: runtime
invention feeds offline consolidation.** live-SWE-agent has invention without
memory; DGM has memory without runtime invention. The synthesis is
working-memory → long-term-memory for an agent's own capabilities. It *compounds*
— each run makes the next start stronger — which is the only path to eventually
*exceeding* the leaders rather than matching them.

## How it maps onto machinery misterdev already owns
Nothing here is a new evolution engine; it is a new *substrate* for the existing one.

| Existing seam | Today | Under two-timescale |
| --- | --- | --- |
| `evolution/archive.py` `Candidate` | `patch` = a scaffold self-edit diff | also: `patch` = a self-authored tool's source |
| `EvolutionArchive` (MAP-Elites) | best scaffold-edit per behavioral niche | also: best *tool* per capability niche ("reproduce/minimize pytest failures", "AST-safe rename", ...) |
| `evolution/holdout.py` `decide_promotion` | rejects derive-up / holdout-down as overfit | **the same gate** decides which tools become permanent |
| `evolution/fitness.py` `FitnessScore` | resolved/total/cost/regressions | identical — a tool scores by tasks solved *with it available*, net of cost and regressions |
| run seeding | scaffold config | future runs load promoted tool elites into the agent's starting toolbelt |

The held-out gate is the credibility keystone: a tool is promoted only if it lifts
resolve rate on tasks the invention never saw. That is exactly the guard that
stops the tool library becoming a benchmark-overfit grab-bag — the failure mode
every SWE-bench system is criticized for.

## Phased plan
- **P1 — Consolidation substrate (offline, pure, $0 to test).** A `ToolLibrary`
  over the archive's MAP-Elites semantics: a tool candidate (id, source,
  capability-niche, provenance, fitness), best-per-niche promotion via
  `decide_promotion`, JSON persistence that degrades to empty, and a `seed()` that
  loads promoted elites for a run. Bootstrap with 1-2 hand-authored seed tools so
  the pipeline is exercised before runtime invention exists. Fully unit-testable.
- **P2 — Runtime invention surface.** Give the executor's model a sandboxed
  scratch space to author + run task-specific Python tools mid-task, with the
  per-step reflection nudge. Sandboxed via the existing `evolution/sandbox.py` +
  `core/execution/container.py` (model-authored code is untrusted and must not
  touch the host). Each tool that contributes to a gate-passing task is captured
  as a P1 candidate.
- **P3 — Close the loop.** Captured runtime tools flow into the offline loop as
  candidates; promoted tools seed subsequent runs; measure compounding (does run N
  start stronger than run 1 on held-out tasks?).

## Security (P2, non-negotiable)
Runtime tools are model-authored code = untrusted. They execute only inside the
sandbox/container boundary, never on the host; no network unless explicitly
granted; bounded CPU/time; the tool source is captured and reviewable. A promoted
tool is re-validated in the sandbox before it can seed a run.

## What success looks like for the pitch
"An agent that invents its own tools at runtime *and* consolidates the winners
into permanent, generalization-verified capability" is a genuinely novel research
claim — runtime exploration + offline consolidation with a held-out promotion gate
— and it is defensible precisely because the promotion gate is an anti-overfitting
mechanism, not a benchmark-chasing one.
