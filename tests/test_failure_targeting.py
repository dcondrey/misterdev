from dataclasses import dataclass

from misterdev.core.learning.targeting import stream_blame, top_stream_target


@dataclass
class _Rec:
    language: str
    error: str
    category: str
    fp: str
    run: int
    resolved: bool = False


def test_empty_stream_has_no_target():
    assert stream_blame([]) == []
    assert top_stream_target([]) is None


def test_ranks_by_count_when_flat():
    recs = [
        _Rec("rust", "e1", "wrong_type", "a", 1),
        _Rec("rust", "e2", "wrong_type", "b", 1),
        _Rec("python", "e3", "name_error", "c", 1),
    ]
    top = top_stream_target(recs, by_category=True)
    assert top.niche == "rust/wrong_type"
    assert top.failures == 2
    assert top.source == "real-build failures"


def test_recurrence_outweighs_raw_count():
    # python has more raw failures, but each is a distinct one-off (fp count 1).
    # rust has fewer, but they are the SAME recurring failure (fp count 3), which
    # is the standing weakness worth targeting.
    recs = [
        _Rec("rust", "same", "wrong_type", "R", 1),
        _Rec("rust", "same", "wrong_type", "R", 2),
        _Rec("rust", "same", "wrong_type", "R", 3),
        _Rec("python", "a", "x", "p1", 3),
        _Rec("python", "b", "y", "p2", 3),
        _Rec("python", "c", "z", "p3", 3),
        _Rec("python", "d", "w", "p4", 3),
    ]
    top = top_stream_target(recs, by_category=False)
    assert top.niche == "rust"


def test_recency_decays_stale_niches():
    # rust failures are old (run 1); python is current (run 10). Same counts, but
    # recency should surface the current failure.
    recs = [
        _Rec("rust", "old1", "t", "r1", 1),
        _Rec("rust", "old2", "t", "r2", 1),
        _Rec("python", "new1", "t", "p1", 10),
        _Rec("python", "new2", "t", "p2", 10),
    ]
    top = top_stream_target(recs, by_category=False)
    assert top.niche == "python"


def test_examples_are_populated_for_proposer():
    recs = [_Rec("rust", "E0308 mismatched types", "wrong_type", "a", 1)]
    (blame,) = stream_blame(recs)
    assert blame.examples and "E0308" in blame.examples[0]
