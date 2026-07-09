"""Offline validation of the native (Swift, C#) harness's pure logic.

Swift and C# had no empirical validation at all before this harness, so these
tests pin the two decisions that don't need a toolchain: discovery finds the
real fixtures, and the grader maps canned ``swift test`` / ``dotnet test`` output
to pass/fail. The actual swift/dotnet runs are never invoked here — the
subprocess-driven parts (grade, default_run_one) are exercised only through the
pure functions and injected fakes.
"""

from pathlib import Path

import pytest

from evaluation.native import (
    NativeExercise,
    SuiteReport,
    discover_exercises,
    grade_output,
    run_suite,
)
from evaluation.native.harness import RunResult

FIXTURES = str(
    Path(__file__).resolve().parent.parent / "evaluation" / "native" / "exercises"
)


def test_discover_finds_both_fixtures():
    exercises = discover_exercises(FIXTURES)
    by_lang = {(e.language, e.name) for e in exercises}
    assert ("swift", "two-fer") in by_lang
    assert ("csharp", "two-fer") in by_lang


def test_discover_loads_files_and_command():
    swift = next(
        e for e in discover_exercises(FIXTURES, ["swift"]) if e.name == "two-fer"
    )
    assert swift.solution_files == ["Sources/TwoFer/TwoFer.swift"]
    assert swift.test_files == ["Tests/TwoFerTests/TwoFerTests.swift"]
    assert swift.test_command == "swift test"
    assert "One for" in swift.instructions

    csharp = next(
        e for e in discover_exercises(FIXTURES, ["csharp"]) if e.name == "two-fer"
    )
    assert csharp.solution_files == ["TwoFer.cs"]
    assert csharp.test_command == "dotnet test"


def test_discover_language_filter_and_limit():
    assert all(e.language == "swift" for e in discover_exercises(FIXTURES, ["swift"]))
    assert discover_exercises(FIXTURES, limit=0) == []
    assert discover_exercises(FIXTURES, limit=-1) == []
    assert len(discover_exercises(FIXTURES, limit=1)) == 1


def test_discover_only_filters_by_slug():
    assert [e.name for e in discover_exercises(FIXTURES, only=["two-fer"])] == [
        "two-fer",
        "two-fer",
    ]
    assert discover_exercises(FIXTURES, only=["does-not-exist"]) == []


def test_default_test_command_by_language():
    assert NativeExercise("x", "swift", ["a"], ["b"]).test_command == "swift test"
    assert NativeExercise("x", "csharp", ["a"], ["b"]).test_command == "dotnet test"


def test_unsupported_language_raises():
    with pytest.raises(ValueError):
        NativeExercise("x", "haskell", ["a"], ["b"])


def test_grade_output_swift_pass():
    out = (
        "Test Suite 'All tests' passed at 2026-07-08.\n"
        "\t Executed 3 tests, with 0 failures (0 unexpected) in 0.001 seconds\n"
    )
    assert grade_output(0, out).resolved is True


def test_grade_output_swift_fail_nonzero_exit():
    out = (
        "Test Suite 'TwoFerTests' failed at 2026-07-08.\n"
        "\t Executed 3 tests, with 1 failure (0 unexpected) in 0.002 seconds\n"
    )
    res = grade_output(1, out)
    assert res.resolved is False
    assert "exited 1" in res.error


def test_grade_output_dotnet_pass():
    out = "Passed!  - Failed:     0, Passed:     3, Skipped:     0, Total:     3\n"
    assert grade_output(0, out).resolved is True


def test_grade_output_dotnet_fail():
    out = "Failed!  - Failed:     1, Passed:     2, Skipped:     0, Total:     3\n"
    res = grade_output(1, out)
    assert res.resolved is False


def test_grade_output_zero_exit_but_reported_failure():
    # A runner that exits 0 while still printing a failure banner must not be
    # scored as resolved — the exit code alone would wrongly pass it.
    res = grade_output(0, "Failed!  - Failed: 1, Passed: 2, Total: 3\n")
    assert res.resolved is False
    assert "reported failures" in res.error


def test_grade_output_build_failure():
    res = grade_output(1, "error: build failed\ncompilation terminated\n")
    assert res.resolved is False


def test_run_suite_uses_injected_run_one():
    # The suite drives an injected run_one (no misterdev, no toolchain) so the
    # discovery -> run -> aggregate path is exercised offline.
    seen = []

    def fake_run_one(exercise, dest):
        seen.append((exercise.language, exercise.name))
        return RunResult(exercise.name, exercise.language, resolved=True, cost=0.01)

    report = run_suite(FIXTURES, "/tmp/native-scratch", run_one=fake_run_one)
    assert report.total == 2
    assert report.resolved == 2
    assert report.resolved_rate == 1.0
    assert report.cost == pytest.approx(0.02)
    assert set(seen) == {("csharp", "two-fer"), ("swift", "two-fer")}


def test_suite_report_aggregates_and_serializes():
    report = SuiteReport(
        results=[
            RunResult("two-fer", "swift", True, cost=0.02),
            RunResult("two-fer", "csharp", False, cost=0.03, error="tests exited 1"),
        ]
    )
    assert report.cost == pytest.approx(0.05)
    d = report.to_dict()
    assert d["total"] == 2
    assert d["resolved"] == 1
    assert d["results"][1]["error"] == "tests exited 1"
    assert report.by_language() == {"swift": [1, 1], "csharp": [0, 1]}
