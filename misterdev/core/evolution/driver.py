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
from .fitness import FitnessScore
from .loop import EvolutionLoop, Mutation, StepResult
from .prior import MutationPrior
from .proposer import LLMProposer
from .sandbox import SandboxEvaluator

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
) -> EvolutionResult:
    """Run one evolution pass. See module docstring for dry-run vs live.

    ``run_bench(cwd)`` -> ``(results, cost, raw)`` and ``proposer`` / ``sandbox``
    default to the real adapters but are injectable for tests. ``gate_commands``
    (required for live) are misterdev's own build/test/lint commands, run in the
    sandbox to prove a self-edit did not break misterdev before it is scored.
    """
    bench = run_bench or (
        lambda cwd: run_benchmark(
            cwd, benchmark_dir, workdir, limit=limit, languages=languages, model=model
        )
    )
    results, cost, _raw = bench(str(project.path))
    baseline = score_of(results, cost=cost)
    blame = top_target(results)
    logger.info(
        f"Evolution: baseline {baseline.resolved}/{baseline.total}; "
        f"top blame = {blame.niche if blame else 'none (all passed)'}."
    )
    if blame is None:
        return EvolutionResult(baseline=baseline, blame=None, note="nothing to improve")

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
    loop = EvolutionLoop(
        archive=archive,
        evaluate=evaluate,
        propose=lambda _target: proposer.propose(
            blame, favored_kinds=prior.favored_kinds()
        ),
        noise_band=noise_band,
        champion=baseline,
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
