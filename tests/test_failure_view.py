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


def test_render_caps_and_counts_overflow():
    fs = extract_failures(PYTEST, language="python")
    # Force overflow with a tiny cap; the tail count must be reported.
    view = render_failure_view(fs * 3, max_failures=2)
    assert "more failing the same way" in view
