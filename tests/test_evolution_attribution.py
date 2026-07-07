from dataclasses import dataclass

from misterdev.core.evolution import Blame, attribute, top_target


@dataclass
class _Result:
    resolved: bool
    language: str = "rust"
    output: str = ""
    name: str = "ex"


def test_passing_suite_produces_no_blame():
    res = [_Result(True, "rust"), _Result(True, "go")]
    assert attribute(res) == []
    assert top_target(res) is None


def test_ranks_niches_by_failure_count():
    res = [
        _Result(False, "rust"),
        _Result(False, "rust"),
        _Result(False, "go"),
        _Result(True, "python"),
    ]
    blames = attribute(res)
    assert [(b.niche, b.failures) for b in blames] == [("rust", 2), ("go", 1)]
    # python passed -> not blamed at all
    assert all(b.niche != "python" for b in blames)


def test_failure_rate_and_totals_track_passes():
    res = [_Result(False, "rust"), _Result(True, "rust")]
    (blame,) = attribute(res)
    assert blame.failures == 1 and blame.total == 2
    assert blame.failure_rate == 0.5


def test_by_category_subbins_failures():
    res = [
        _Result(False, "rust", "error[E0308]: mismatched types"),
        _Result(False, "rust", "error[E0308]: mismatched types"),
        _Result(False, "rust", "error[E0425]: cannot find value `x`"),
    ]
    blames = attribute(res, by_category=True)
    niches = {b.niche: b.failures for b in blames}
    assert niches == {"rust/wrong_type": 2, "rust/missing_symbol": 1}


def test_examples_are_captured_and_bounded():
    res = [_Result(False, "rust", f"error line {i}") for i in range(10)]
    (blame,) = attribute(res)
    assert 1 <= len(blame.examples) <= 3  # _MAX_EXAMPLES
    assert "error line" in blame.examples[0]


def test_missing_language_falls_back_to_unknown():
    @dataclass
    class _Bare:
        resolved: bool

    blames = attribute([_Bare(False)])
    assert blames[0].niche == "unknown"


def test_top_target_is_the_worst_niche():
    res = [_Result(False, "rust"), _Result(False, "go"), _Result(False, "go")]
    assert top_target(res, by_category=False).niche == "go"


def test_blame_rate_zero_total_safe():
    assert Blame(niche="x", failures=0, total=0).failure_rate == 0.0
