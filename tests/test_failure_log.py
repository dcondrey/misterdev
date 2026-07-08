from dataclasses import dataclass, field
from typing import List

from misterdev.core.evolution import attribute, top_target
from misterdev.core.learning.failure_log import (
    FailureLog,
    FailureRecord,
    fingerprint,
    language_of,
)


@dataclass
class _Exec:
    logs: str = ""
    message: str = ""


@dataclass
class _Task:
    id: str
    execution_history: List[_Exec] = field(default_factory=list)
    files_to_modify: List[str] = field(default_factory=list)
    files_to_create: List[str] = field(default_factory=list)
    category: str = "fix"
    status: str = "failed"


def _failed(id, error, files=("src/lib.rs",), category="fix"):
    return _Task(
        id=id,
        execution_history=[_Exec(logs=error)],
        files_to_modify=list(files),
        category=category,
    )


def test_language_inferred_from_extension():
    assert language_of(["src/main.rs"]) == "rust"
    assert language_of(["a.py", "b.rs"]) == "python"  # first recognized wins
    assert language_of(["README"]) == "unknown"
    assert language_of([]) == "unknown"


def test_fingerprint_collapses_volatile_detail():
    a = fingerprint("error[E0308]: mismatched types at line 42 in /tmp/x9f/main.rs")
    b = fingerprint("error[E0308]: mismatched types at line 991 in /tmp/aa/main.rs")
    assert a == b and a != ""


def test_fingerprint_distinguishes_real_differences():
    assert fingerprint("mismatched types") != fingerprint("missing semicolon")
    assert fingerprint("") == ""


def test_fingerprint_preserves_error_codes():
    # Distinct compiler error codes are the most discriminating part of an error
    # and must NOT be collapsed by digit-stripping.
    a = fingerprint("error[E0433]: failed to resolve at line 5")
    b = fingerprint("error[E0499]: cannot borrow twice at line 5")
    assert a != b


def test_fingerprint_consistent_across_decimal_magnitude():
    # The same failure with a large vs small embedded count must fingerprint the
    # same (a long decimal is not an address).
    a = fingerprint("failed to allocate 268435456 bytes")
    b = fingerprint("failed to allocate 4096 bytes")
    assert a == b


def test_from_task_extracts_signal():
    rec = FailureRecord.from_task(_failed("T-1", "E0308 mismatched types"), run=1)
    assert rec is not None
    assert rec.name == "T-1"
    assert rec.language == "rust"
    assert rec.resolved is False
    assert rec.fp != ""
    assert rec.category  # classify_error returned something


def test_from_task_skips_contentless_failure():
    # A task with no recoverable error detail teaches nothing -> not logged.
    empty = _Task(id="T-empty", execution_history=[])
    assert FailureRecord.from_task(empty, run=1) is None


def test_record_and_load_roundtrip(tmp_path):
    log = FailureLog(tmp_path / "failures.jsonl")
    n = log.record_failures([_failed("T-1", "boom rust"), _failed("T-2", "bang rust")])
    assert n == 2
    loaded = log.load()
    assert {r.name for r in loaded} == {"T-1", "T-2"}
    assert all(not r.resolved for r in loaded)


def test_next_run_increments_across_batches(tmp_path):
    log = FailureLog(tmp_path / "failures.jsonl")
    log.record_failures([_failed("T-1", "boom")])
    log.record_failures([_failed("T-2", "bang")])
    runs = sorted({r.run for r in log.load()})
    assert runs == [1, 2]


def test_records_feed_attribution_directly(tmp_path):
    log = FailureLog(tmp_path / "failures.jsonl")
    log.record_failures(
        [
            _failed("T-1", "rust boom", files=["a.rs"]),
            _failed("T-2", "rust bang", files=["b.rs"]),
            _failed("T-3", "py oops", files=["c.py"]),
        ]
    )
    records = log.load()
    blames = attribute(records)
    # rust concentrates the blame; a FailureRecord duck-types as a benchmark result.
    assert blames[0].niche == "rust"
    assert blames[0].failures == 2
    assert top_target(records, by_category=False).niche == "rust"


def test_recurrence_counts_by_fingerprint(tmp_path):
    log = FailureLog(tmp_path / "failures.jsonl")
    log.record_failures([_failed("T-1", "mismatched types at line 1")])
    log.record_failures([_failed("T-2", "mismatched types at line 88")])
    log.record_failures([_failed("T-3", "totally different failure")])
    counts = log.recurrence()
    # The two line-varying failures collapse to one recurring fingerprint (count 2).
    assert max(counts.values()) == 2
    assert len(counts) == 2


def test_recency_weight_decays_with_age():
    rec = FailureRecord(name="x", language="rust", error="e", category="c", run=1)
    assert rec.recency_weight(current_run=1) == 1.0
    assert rec.recency_weight(current_run=6, half_life=5.0) == 0.5
    assert rec.recency_weight(current_run=11, half_life=5.0) == 0.25


def test_log_bounded_and_survives_corrupt_line(tmp_path):
    path = tmp_path / "failures.jsonl"
    path.write_text('not json\n{"name": "old", "run": 1, "error": "x"}\n')
    log = FailureLog(path)
    # The corrupt line is skipped, the valid one survives.
    assert [r.name for r in log.load()] == ["old"]
    # A fresh batch appends without crashing.
    assert log.record_failures([_failed("T-new", "boom")]) == 1
    assert "T-new" in {r.name for r in log.load()}


def test_missing_file_loads_empty(tmp_path):
    log = FailureLog(tmp_path / "nope.jsonl")
    assert log.load() == []
    assert log.next_run() == 1
    assert log.recurrence() == {}
