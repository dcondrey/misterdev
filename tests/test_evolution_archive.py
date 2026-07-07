import tempfile
from pathlib import Path

from misterdev.core.evolution import Candidate, EvolutionArchive


def _archive(noise_band=0.0):
    return EvolutionArchive(
        Path(tempfile.mkdtemp()) / "archive.json", noise_band=noise_band
    )


def _cand(cid, niche, resolved, total=100, cost=1.0, regressions=0):
    return Candidate(
        id=cid,
        niche=niche,
        resolved=resolved,
        total=total,
        cost=cost,
        regressions=regressions,
    )


def test_first_candidate_takes_an_empty_niche():
    arc = _archive()
    assert arc.consider(_cand("a", "rust", 40))
    assert arc.elite("rust").id == "a"


def test_better_candidate_replaces_the_niche_elite():
    arc = _archive(noise_band=0.05)
    arc.consider(_cand("a", "rust", 40))
    assert arc.consider(_cand("b", "rust", 60))  # +20 beats the band
    assert arc.elite("rust").id == "b"


def test_within_band_candidate_does_not_replace_elite():
    arc = _archive(noise_band=0.05)
    arc.consider(_cand("a", "rust", 40))
    assert not arc.consider(_cand("b", "rust", 43))  # +3 is noise
    assert arc.elite("rust").id == "a"


def test_regressing_candidate_never_becomes_elite():
    arc = _archive()
    assert not arc.consider(_cand("a", "rust", 90, regressions=1))
    assert arc.elite("rust") is None


def test_niches_are_independent_stepping_stones():
    # A candidate that is globally worse but best in its own niche is KEPT — the
    # never-forget property. 'rust' elite outperforms the 'cost' elite globally,
    # yet both survive.
    arc = _archive()
    arc.consider(_cand("r", "rust", 80, cost=5.0))
    arc.consider(_cand("c", "cost", 50, cost=0.2))
    niches = {c.niche for c in arc.elites()}
    assert niches == {"rust", "cost"}


def test_champion_is_the_global_best_across_niches():
    arc = _archive()
    arc.consider(_cand("r", "rust", 80))
    arc.consider(_cand("c", "cost", 50))
    assert arc.champion().id == "r"


def test_champion_breaks_ties_on_cost():
    arc = _archive()
    arc.consider(_cand("a", "n1", 50, cost=2.0))
    arc.consider(_cand("b", "n2", 50, cost=1.0))  # same rate, cheaper
    assert arc.champion().id == "b"


def test_persists_across_instances():
    path = Path(tempfile.mkdtemp()) / "archive.json"
    EvolutionArchive(path).consider(_cand("a", "rust", 55))
    reopened = EvolutionArchive(path)
    assert reopened.elite("rust").id == "a"
    assert reopened.elite("rust").run == 1


def test_corrupt_archive_degrades_to_empty():
    path = Path(tempfile.mkdtemp()) / "archive.json"
    path.write_text("not json", encoding="utf-8")
    arc = EvolutionArchive(path)
    assert arc.elites() == []
    # And a fresh consider still works over the corrupt file.
    assert arc.consider(_cand("a", "rust", 40))
