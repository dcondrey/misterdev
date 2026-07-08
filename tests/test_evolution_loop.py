import tempfile
from pathlib import Path

from misterdev.core.evolution import (
    EvolutionArchive,
    EvolutionLoop,
    FitnessScore,
    Mutation,
)


def _loop(evaluate, propose, baseline=None, noise_band=0.05):
    arc = EvolutionArchive(
        Path(tempfile.mkdtemp()) / "archive.json", noise_band=noise_band
    )
    return EvolutionLoop(
        archive=arc,
        evaluate=evaluate,
        propose=propose,
        noise_band=noise_band,
        champion=baseline or FitnessScore(50, 100, 1.0),
    )


def _mut(paths=("misterdev/config.py",), patch="diff"):
    return lambda target: Mutation(target=target, paths=list(paths), patch=patch)


def test_real_gain_is_promoted_and_advances_the_champion():
    loop = _loop(evaluate=lambda m: FitnessScore(60, 100, 1.0), propose=_mut())
    res = loop.step("misterdev/config.py", "cost")
    assert res.promoted and res.reason == "promoted"
    assert loop.champion.resolved == 60  # incumbent advanced
    # A second candidate must now beat 60, not the original 50.
    loop.evaluate = lambda m: FitnessScore(58, 100, 1.0)
    res2 = loop.step("misterdev/config.py", "cost")
    assert not res2.promoted


class _Verdict:
    def __init__(self, accepted, targeted_resolved=1):
        self.accepted = accepted
        self.rank_key = (targeted_resolved, 0)


def test_beam_screen_picks_best_survivor_for_the_oracle():
    # Propose 3 candidates distinguished by patch; screen accepts two, ranking the
    # one that fixes more targets first. Only that one reaches the oracle.
    patches = iter(["worst", "best", "rejected"])
    evaluated = []

    def propose(target):
        return Mutation(
            target=target, paths=["misterdev/config.py"], patch=next(patches)
        )

    def screen(m):
        return {
            "worst": _Verdict(True, targeted_resolved=1),
            "best": _Verdict(True, targeted_resolved=3),
            "rejected": _Verdict(False),
        }[m.patch]

    def evaluate(m):
        evaluated.append(m.patch)
        return FitnessScore(60, 100, 1.0)

    loop = _loop(evaluate=evaluate, propose=propose)
    loop.screen = screen
    loop.beam = 3
    res = loop.step("misterdev/config.py", "cost")
    assert res.promoted
    assert evaluated == ["best"]  # oracle ran exactly once, on the top survivor


def test_all_screened_out_skips_the_oracle():
    evaluated = []

    def evaluate(m):
        evaluated.append(m)
        return FitnessScore(60, 100, 1.0)

    loop = _loop(evaluate=evaluate, propose=_mut())
    loop.screen = lambda m: _Verdict(False)
    loop.beam = 3
    res = loop.step("misterdev/config.py", "cost")
    assert not res.promoted and res.reason == "all candidates screened out"
    assert evaluated == []  # never spent the expensive oracle


def test_within_noise_gain_is_archived_but_not_promoted():
    loop = _loop(evaluate=lambda m: FitnessScore(52, 100, 1.0), propose=_mut())
    res = loop.step("misterdev/config.py", "cost")
    assert not res.promoted
    assert res.reason == "within noise band"


def test_regression_is_never_promoted():
    loop = _loop(
        evaluate=lambda m: FitnessScore(90, 100, 1.0, regressions=2), propose=_mut()
    )
    res = loop.step("misterdev/config.py", "cost")
    assert not res.promoted and res.reason == "regression"


def test_guardrail_refuses_a_candidate_touching_the_judge():
    called = {"evaluated": False}

    def evaluate(m):
        called["evaluated"] = True
        return FitnessScore(99, 100, 0.1)

    loop = _loop(
        evaluate=evaluate, propose=_mut(paths=["evaluation/polyglot/grader.py"])
    )
    res = loop.step("evaluation/polyglot/grader.py", "cost")
    assert not res.promoted
    assert res.reason.startswith("guardrail:")
    # The candidate never reached the fitness function — refused before scoring.
    assert called["evaluated"] is False


def test_gates_failure_discards_before_scoring():
    # evaluate() returns None when the sandbox gate suite fails.
    loop = _loop(evaluate=lambda m: None, propose=_mut())
    res = loop.step("misterdev/config.py", "cost")
    assert not res.promoted and res.reason == "gates failed"


def test_proposer_failure_is_survived():
    def boom(target):
        raise RuntimeError("model timed out")

    loop = _loop(evaluate=lambda m: FitnessScore(99, 100, 0.1), propose=boom)
    res = loop.step("misterdev/config.py", "cost")
    assert not res.promoted and "proposal failed" in res.reason


def test_evaluator_failure_is_survived():
    def boom(m):
        raise RuntimeError("worktree apply failed")

    loop = _loop(evaluate=boom, propose=_mut())
    res = loop.step("misterdev/config.py", "cost")
    assert not res.promoted and "evaluation failed" in res.reason


def test_promoted_candidate_lands_in_the_archive():
    loop = _loop(evaluate=lambda m: FitnessScore(70, 100, 1.0), propose=_mut())
    res = loop.step("misterdev/config.py", "rust")
    assert res.archived
    assert loop.archive.elite("rust").id == res.candidate_id
