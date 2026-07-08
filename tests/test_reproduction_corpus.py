from dataclasses import dataclass

from misterdev.core.learning.reproduction import Case, ReproductionCorpus


@dataclass
class _R:
    name: str
    language: str
    resolved: bool
    error: str = ""


def test_update_and_failing_selection(tmp_path):
    corpus = ReproductionCorpus(tmp_path / "repro.json")
    corpus.update(
        [
            _R("a", "rust", False, "error[E0308]: mismatched types"),
            _R("b", "rust", True),
            _R("c", "python", False, "NameError"),
        ]
    )
    failing_ids = {c.id for c in corpus.failing()}
    assert failing_ids == {"a", "c"}
    # niche filtering: language-level and category-level.
    assert {c.id for c in corpus.failing(niche="rust")} == {"a"}
    assert corpus.failing(niche="python")[0].id == "c"


def test_streaks_track_consecutive_outcomes(tmp_path):
    corpus = ReproductionCorpus(tmp_path / "repro.json")
    corpus.update([_R("a", "rust", False, "boom")])
    corpus.update([_R("a", "rust", False, "boom")])
    corpus.update([_R("a", "rust", True)])
    case = next(iter(corpus._load().values()))
    assert case.pass_streak == 1 and case.fail_streak == 0 and case.runs == 3


def test_persistently_failing_ranked_first(tmp_path):
    corpus = ReproductionCorpus(tmp_path / "repro.json")
    # 'a' fails three times, 'b' fails once.
    for _ in range(3):
        corpus.update([_R("a", "rust", False, "boom"), _R("b", "rust", True)])
    corpus.update([_R("a", "rust", False, "boom"), _R("b", "rust", False, "bang")])
    ranked = corpus.failing(niche="rust")
    assert ranked[0].id == "a"  # higher fail_streak first


def test_guard_sample_is_deterministic_and_excludes(tmp_path):
    corpus = ReproductionCorpus(tmp_path / "repro.json")
    corpus.update([_R(f"p{i}", "rust", True) for i in range(10)])
    s1 = [c.id for c in corpus.guard_sample(3)]
    s2 = [c.id for c in corpus.guard_sample(3)]
    assert s1 == s2 and len(s1) == 3  # deterministic
    # excluded ids never appear.
    s3 = [c.id for c in corpus.guard_sample(5, exclude={"p0", "p1"})]
    assert "p0" not in s3 and "p1" not in s3


def test_guard_sample_bounds(tmp_path):
    corpus = ReproductionCorpus(tmp_path / "repro.json")
    corpus.update([_R("p0", "rust", True), _R("p1", "rust", True)])
    assert corpus.guard_sample(0) == []
    assert len(corpus.guard_sample(10)) == 2  # asking for more than exist -> all
    corpus2 = ReproductionCorpus(tmp_path / "empty.json")
    assert corpus2.guard_sample(3) == []


def test_niche_category_derived_from_error(tmp_path):
    corpus = ReproductionCorpus(tmp_path / "repro.json")
    corpus.update([_R("a", "rust", False, "error[E0308]: mismatched types")])
    case = corpus.failing()[0]
    assert case.category  # classify_error produced a category
    assert case.niche.startswith("rust/")


def test_recovery_clears_category(tmp_path):
    corpus = ReproductionCorpus(tmp_path / "repro.json")
    corpus.update([_R("a", "rust", False, "error[E0308]: mismatched types")])
    corpus.update([_R("a", "rust", True)])
    case = corpus._load()["a"]
    assert case.resolved and case.category == "" and case.niche == "rust"


def test_persists_and_reloads(tmp_path):
    path = tmp_path / "repro.json"
    ReproductionCorpus(path).update([_R("a", "rust", False, "boom")])
    reloaded = ReproductionCorpus(path)
    assert reloaded.known_case_ids() == {"a"}
    assert reloaded.stats() == {"total": 1, "failing": 1}


def test_missing_and_corrupt_degrade_to_empty(tmp_path):
    assert ReproductionCorpus(tmp_path / "nope.json").failing() == []
    bad = tmp_path / "bad.json"
    bad.write_text("not json")
    assert ReproductionCorpus(bad).failing() == []


def test_case_niche_property():
    assert (
        Case(id="x", language="rust", category="wrong_type").niche == "rust/wrong_type"
    )
    assert Case(id="x", language="go").niche == "go"
