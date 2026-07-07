import tempfile
from pathlib import Path

from misterdev.core.evolution import Candidate, EvolutionArchive, MutationPrior


def _archive():
    return EvolutionArchive(Path(tempfile.mkdtemp()) / "archive.json")


def _elite(arc, cid, niche, note, resolved=60):
    arc.consider(
        Candidate(
            id=cid, niche=niche, resolved=resolved, total=100, cost=1.0, note=note
        )
    )


def test_cold_start_has_no_prior():
    assert MutationPrior(_archive()).weights() == []
    assert MutationPrior(_archive()).favored_kinds() == []


def test_weights_are_elite_share_ranked():
    arc = _archive()
    _elite(arc, "a", "rust", "tag: prompt")
    _elite(arc, "b", "go", "prompt tweak")  # 'prompt' leading word -> same kind
    _elite(arc, "c", "python", "gate-tuning: loosen")
    weights = MutationPrior(arc).weights()
    top = weights[0]
    assert top.kind == "prompt" and top.elites == 2
    assert abs(top.weight - 2 / 3) < 1e-9


def test_favored_requires_evidence_bar():
    arc = _archive()
    _elite(arc, "a", "rust", "tag: prompt")
    _elite(arc, "b", "go", "tag: prompt")
    _elite(arc, "c", "python", "tag: oneoff")  # only 1 -> below the bar
    favored = MutationPrior(arc).favored_kinds()
    assert "prompt" in favored
    assert "oneoff" not in favored  # single anecdote must not dominate the prior


def test_untagged_notes_bucket_as_unknown_or_leading_word():
    arc = _archive()
    _elite(arc, "a", "rust", "")  # empty -> unknown
    weights = {w.kind: w.elites for w in MutationPrior(arc).weights()}
    assert weights.get("unknown") == 1
