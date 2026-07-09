"""FailureView parses real pytest/jest/cargo output into exact assertions.

The snippets below are verbatim captures from the three runners (beer-song
pytest, complex-numbers jest, bowling cargo) so the parsers are validated
against ground truth, not assumed formats.
"""

from misterdev.core.execution.failure_view import (
    extract_failures,
    render_failure_view,
)

PYTEST = """\
=================================== FAILURES ===================================
_________________________ BeerSongTest.test_all_verses _________________________
E       AssertionError: None != ['99 bottles of beer on the wall, 99 bott[12735 chars]ll.']
=========================== short test summary info ============================
FAILED beer_song_test.py::BeerSongTest::test_all_verses - AssertionError: None...
FAILED beer_song_test.py::BeerSongTest::test_verse_with_1_bottle - AssertionError: None != ['1 bottle of beer']
8 failed in 0.02s
"""

JEST = """\
  ● Complex numbers › Real part of a purely real number

    expect(received).toEqual(expected) // deep equality

    Expected: 1
    Received: 2

      at Object.toEqual (complex-numbers.spec.js:8:20)

Tests:       1 failed, 30 skipped, 31 total
"""

CARGO = """\
test all_strikes_is_a_perfect_score_of_300 ... FAILED
thread 'all_strikes_is_a_perfect_score_of_300' (110780082) panicked at tests/bowling.rs:275:5:
assertion `left == right` failed
  left: None
 right: Some(300)
test result: FAILED. 8 passed; 23 failed; 0 ignored; 0 measured; 0 filtered out
"""


def test_pytest_extraction():
    fs = extract_failures(PYTEST, language="python")
    names = {f.test for f in fs}
    assert "test_all_verses" in names
    assert "test_verse_with_1_bottle" in names
    one = next(f for f in fs if f.test == "test_verse_with_1_bottle")
    assert one.actual == "None"
    assert "1 bottle of beer" in one.expected


def test_jest_extraction():
    fs = extract_failures(JEST, language="javascript")
    assert len(fs) == 1
    f = fs[0]
    assert f.test == "Real part of a purely real number"
    assert f.expected == "1" and f.actual == "2"
    assert f.location == "complex-numbers.spec.js:8:20"


def test_cargo_extraction():
    fs = extract_failures(CARGO, language="rust")
    assert len(fs) == 1
    f = fs[0]
    assert f.test == "all_strikes_is_a_perfect_score_of_300"
    assert f.actual == "None" and f.expected == "Some(300)"
    assert f.location == "tests/bowling.rs:275:5"


def test_runner_autodetected_without_language():
    # No language hint: the runner is recognized from the output signature.
    assert extract_failures(CARGO)[0].test.startswith("all_strikes")
    assert extract_failures(JEST)[0].expected == "1"
    assert extract_failures(PYTEST)


def test_unrecognized_output_is_empty():
    # Unknown output yields nothing so the caller keeps its existing fallback.
    assert extract_failures("some random build log\nwith no test failures") == []
    assert render_failure_view([]) == ""


def test_render_is_exact_and_bounded():
    fs = extract_failures(CARGO, language="rust")
    view = render_failure_view(fs)
    assert "expected: Some(300)" in view
    assert "actual:   None" in view
    assert "tests/bowling.rs:275:5" in view


XCTEST = """\
Test Case '-[DemoTests.DemoTests testAdd]' started.
/tmp/fv_swift/Tests/DemoTests/DemoTests.swift:4: error: -[DemoTests.DemoTests testAdd] : XCTAssertEqual failed: ("2") is not equal to ("3")
Test Case '-[DemoTests.DemoTests testAdd]' failed (0.918 seconds).
Executed 2 tests, with 1 failure (0 unexpected) in 0.918 seconds
"""

DOTNET = """\
[xUnit.net 00:00:00.09]     UnitTest1.AddIsWrong [FAIL]
  Failed UnitTest1.AddIsWrong [4 ms]
  Error Message:
   Assert.Equal() Failure: Values differ
Expected: 3
Actual:   2
  Stack Trace:
     at UnitTest1.AddIsWrong() in /tmp/fv_dotnet/UnitTest1.cs:line 5
Failed!  - Failed:     1, Passed:     1, Skipped:     0, Total:     2
"""

VITEST = """\
 FAIL  sum.test.js > adds numbers
AssertionError: expected 2 to be 3 // Object.is equality
 ❯ sum.test.js:2:44
      Tests  1 failed | 1 passed (2)
"""


def test_xctest_extraction():
    fs = extract_failures(XCTEST, language="swift")
    assert len(fs) == 1
    f = fs[0]
    assert f.test == "testAdd"
    assert f.actual == "2" and f.expected == "3"
    assert f.location == "/tmp/fv_swift/Tests/DemoTests/DemoTests.swift:4"


def test_dotnet_extraction():
    fs = extract_failures(DOTNET, language="csharp")
    assert len(fs) == 1
    f = fs[0]
    assert f.test == "AddIsWrong"
    assert f.expected == "3" and f.actual == "2"
    assert f.location == "/tmp/fv_dotnet/UnitTest1.cs:line 5"


def test_vitest_extraction_autodetected():
    # vitest output must not be mis-parsed by the jest parser; detection catches it.
    fs = extract_failures(VITEST, language="javascript")
    assert len(fs) == 1
    f = fs[0]
    assert f.test == "adds numbers"
    assert f.actual == "2" and f.expected == "3"
    assert f.location == "sum.test.js:2:44"


def test_render_caps_and_counts_overflow():
    fs = extract_failures(PYTEST, language="python")
    # Force overflow with a tiny cap; the tail count must be reported.
    view = render_failure_view(fs * 3, max_failures=2)
    assert "more failing the same way" in view
