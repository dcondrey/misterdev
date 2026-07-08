"""The evolution driver — one full self-improvement run, end to end.

Composes the spine and adapters into a run:

    baseline benchmark → attribute failures → (dry-run) propose ONE targeted edit,
    or (live) apply/gate/benchmark/promote for N steps, archiving every candidate.

Two modes, safe-by-default:

* **dry-run** (default): measure the baseline, find the highest-blame niche, and
  propose a single edit WITHOUT applying it or promoting anything. No source is
  touched. This is what validates the pipeline on the first benchmark cheaply.
* **live**: run the full keep-if-better loop with the real worktree/gate/benchmark
  sandbox. Opt-in, because it applies self-edits and spends real benchmark budget.

The benchmark/proposer/sandbox steps are injectable so the composition is unit-
tested with fakes; production wires the real adapters.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional

from misterdev.logging_setup import setup_logger

from .adapters import baseline_passed, make_proposer, run_benchmark, score_of
from .archive import Candidate, EvolutionArchive
from .attribution import Blame, top_target
from misterdev.core.learning.reproduction import ReproductionCorpus

from .fitness import FitnessScore
from .loop import EvolutionLoop, Mutation, StepResult
from .prior import MutationPrior
from .proposer import LLMProposer
from .sandbox import SandboxEvaluator
from .screen import MicroEvaluator

logger = setup_logger(__name__)

_DEFAULT_NOISE_BAND = 0.05


@dataclass
class EvolutionResult:
    """Outcome of a run: the baseline, where the blame fell, and what was done."""

    baseline: FitnessScore
    blame: Optional[Blame]
    proposals: List[Mutation] = field(default_factory=list)  # dry-run
    steps: List[StepResult] = field(default_factory=list)  # live
    champion: Optional[Candidate] = None
    note: str = ""


def run_evolution(
    project,
    benchmark_dir: str,
    workdir: str,
    *,
    steps: int = 1,
    noise_band: float = _DEFAULT_NOISE_BAND,
    limit: Optional[int] = None,
    languages: Optional[List[str]] = None,
    model: Optional[str] = None,
    live: bool = False,
    archive_path=None,
    gate_commands: Optional[dict] = None,
    run_bench: Optional[Callable] = None,
    proposer: Optional[LLMProposer] = None,
    sandbox: Optional[object] = None,
    target: Optional[Blame] = None,
    screen: bool = False,
    beam: int = 1,
    targets: int = 12,
    guards: int = 8,
    corpus_path=None,
) -> EvolutionResult:
    """Run one evolution pass. See module docstring for dry-run vs live.

    ``run_bench(cwd)`` -> ``(results, cost, raw)`` and ``proposer`` / ``sandbox``
    default to the real adapters but are injectable for tests. ``gate_commands``
    (required for live) are misterdev's own build/test/lint commands, run in the
    sandbox to prove a self-edit did not break misterdev before it is scored.

    ``target`` overrides where the mutation is aimed: when supplied (e.g. the
    highest-weight niche from the REAL-build failure stream), evolution improves
    what actually breaks in use instead of the benchmark's worst niche. The
    benchmark still runs — it supplies the baseline, the regression reference set,
    and the promotion gate — so a real-failure-targeted edit still cannot be
    promoted unless it holds or improves benchmark capability with zero
    regressions. This is the cross-gate that keeps real-data targeting safe.
    """
    bench = run_bench or (
        lambda cwd: run_benchmark(
            cwd, benchmark_dir, workdir, limit=limit, languages=languages, model=model
        )
    )
    results, cost, _raw = bench(str(project.path))
    baseline = score_of(results, cost=cost)
    # Accumulate this run's per-case outcomes into the reproduction corpus — the
    # growing ground truth that the micro-eval screen draws its targets and guard
    # from. Best-effort: a corpus write must never abort an evolution run.
    corpus = ReproductionCorpus(
        corpus_path
        or (project.path / ".orchestrator" / "evolution" / "reproduction.json")
    )
    try:
        corpus.update(results)
    except Exception as e:
        logger.warning(f"Evolution: corpus update failed (non-fatal): {e}")
    # A real-failure target overrides the benchmark's worst niche; the benchmark
    # blame is the fallback when no target is supplied.
    blame = target or top_target(results)
    logger.info(
        f"Evolution: baseline {baseline.resolved}/{baseline.total}; "
        f"target = {blame.niche if blame else 'none (all passed)'} "
        f"(source: {blame.source if blame else 'n/a'})."
    )
    if blame is None:
        return EvolutionResult(baseline=baseline, blame=None, note="nothing to improve")

    # L2 self-awareness: classify WHY this niche fails so the proposer aims the
    # right KIND of structural fix (and a capability-wall is flagged, not blindly
    # mutated). Best-effort — a classification failure must not abort the run.
    try:
        from types import SimpleNamespace
        from misterdev.core.learning.failure_taxonomy import classify_failure

        sample = blame.examples[0] if blame.examples else ""
        cls = classify_failure(SimpleNamespace(error=sample, category=""))
        blame.cause = cls.cause.value
        blame.cause_evidence = cls.evidence
        logger.info(
            f"Evolution: blame cause = {blame.cause} ({blame.cause_evidence}); "
            f"{'removable' if cls.removable else 'possible capability wall'}."
        )
    except Exception as e:
        logger.warning(f"Evolution: cause classification failed (non-fatal): {e}")

    archive = EvolutionArchive(
        archive_path or (project.path / ".orchestrator" / "evolution" / "archive.json"),
        noise_band=noise_band,
    )
    prior = MutationPrior(archive)
    proposer = proposer or make_proposer(project)

    if not live:
        # Safe path: propose one targeted edit, do not apply or promote.
        try:
            mutation = proposer.propose(blame, favored_kinds=prior.favored_kinds())
            proposals = [mutation]
            note = "dry-run: proposed edit, not applied"
        except Exception as e:  # a failed proposal is a valid, reportable outcome
            logger.warning(f"Evolution: dry-run proposal failed: {e}")
            proposals, note = [], f"dry-run: proposal failed: {e}"
        return EvolutionResult(
            baseline=baseline, blame=blame, proposals=proposals, note=note
        )

    # Live path: full keep-if-better loop over the real sandbox.
    if sandbox is None:
        from .adapters import RealSandbox

        if not gate_commands:
            raise ValueError("live evolution requires gate_commands for the sandbox")
        sandbox = RealSandbox(
            project,
            benchmark_dir,
            workdir,
            gate_commands,
            limit=limit,
            languages=languages,
            model=model,
        )
    evaluate = SandboxEvaluator(
        apply=sandbox.apply,
        gates=sandbox.gates,
        benchmark=sandbox.benchmark,
        baseline_passed=baseline_passed(results),
    )
    # Optional cheap screen: derive the targeted (currently-failing, in-niche) and
    # guard (currently-passing) cases from the corpus and build a micro-evaluator
    # over the sandbox's selective benchmark. Only armed when there are real
    # targets AND the sandbox can run a case subset; otherwise the loop runs
    # single-candidate straight to the oracle, exactly as before.
    screen_fn = None
    if (screen or beam > 1) and hasattr(sandbox, "benchmark_only"):
        target_ids = [c.id for c in corpus.failing(niche=blame.niche, limit=targets)]
        if target_ids:
            guard_ids = [
                c.id for c in corpus.guard_sample(guards, exclude=set(target_ids))
            ]
            screen_fn = MicroEvaluator(
                apply=sandbox.apply,
                gates=sandbox.gates,
                run_only=sandbox.benchmark_only,
                target_ids=target_ids,
                guard_ids=guard_ids,
            ).screen
            logger.info(
                f"Evolution: screen armed ({len(target_ids)} targets, "
                f"{len(guard_ids)} guards, beam {max(1, beam)})."
            )
    # L4 held-out gate: split the baseline tasks into disjoint DERIVE/HOLDOUT pools
    # and promote a candidate only when it gains on DERIVE without dropping HOLDOUT,
    # rejecting a gain that does not generalize. Falls back to the single-set rule
    # when there is no holdout signal (too few tasks). The split is on RESULTS, not
    # runs, so this adds no benchmark cost.
    from .holdout import decide_promotion, split_tasks

    def _pool_score(pool_results, base_passed) -> FitnessScore:
        passed_now = {r.name for r in pool_results if getattr(r, "resolved", False)}
        regressions = sum(1 for n in base_passed if n not in passed_now)
        return FitnessScore(
            resolved=len(passed_now),
            total=len(pool_results),
            cost=0.0,
            regressions=regressions,
        )

    derive_slugs, holdout_slugs = split_tasks(
        [getattr(r, "name", "") for r in results], holdout_fraction=0.3
    )
    derive_set, holdout_set = set(derive_slugs), set(holdout_slugs)

    def _partition(res):
        return (
            [r for r in res if getattr(r, "name", "") in derive_set],
            [r for r in res if getattr(r, "name", "") in holdout_set],
        )

    base_d, base_h = _partition(results)
    derive_base = _pool_score(base_d, set())
    holdout_base = _pool_score(base_h, set())
    derive_base_passed = {r.name for r in base_d if getattr(r, "resolved", False)}
    holdout_base_passed = {r.name for r in base_h if getattr(r, "resolved", False)}

    promote_decider = None
    if holdout_set:

        def promote_decider(score):
            rep = getattr(evaluate, "last_report", None)
            if rep is None:
                return score.beats(baseline, noise_band), ""
            mut_d, mut_h = _partition(list(getattr(rep, "results", [])))
            decision = decide_promotion(
                _pool_score(mut_d, derive_base_passed),
                derive_base,
                _pool_score(mut_h, holdout_base_passed),
                holdout_base,
                noise_band,
            )
            return decision.promote, decision.reason

        logger.info(
            f"Evolution: held-out gate armed "
            f"({len(derive_set)} derive, {len(holdout_set)} holdout)."
        )

    loop = EvolutionLoop(
        archive=archive,
        evaluate=evaluate,
        propose=lambda _target: proposer.propose(
            blame, favored_kinds=prior.favored_kinds()
        ),
        noise_band=noise_band,
        champion=baseline,
        screen=screen_fn,
        beam=max(1, beam),
        promote_decider=promote_decider,
    )
    step_results: List[StepResult] = []
    for i in range(max(1, steps)):
        res = loop.step(blame.niche, blame.niche)
        step_results.append(res)
        logger.info(f"Evolution: step {i + 1}/{steps} -> {res.reason}")
    return EvolutionResult(
        baseline=baseline,
        blame=blame,
        steps=step_results,
        champion=archive.champion(),
        note="live: full loop",
    )
