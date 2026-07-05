import tempfile
from pathlib import Path

from misterdev.core.verification.validator import (
    CodeValidator,
    CertaintyScorer,
    StallDetector,
    ValidationResult,
    run_validation,
    run_health_check,
    _run_cmd,
    _tokenize,
    _parse_test_counts,
)


def test_run_health_check_absent_commands_count_as_pass():
    # An absent build/test/lint command is "not applicable", not "failing".
    with tempfile.TemporaryDirectory() as td:
        h = run_health_check(Path(td), None, None, None)
        assert h.builds is True
        assert h.tests_pass is True
        assert h.lint_clean is True


def test_parse_test_counts_sums_multiple_cargo_crates():
    out = "test result: ok. 3 passed; 0 failed\ntest result: ok. 4 passed; 1 failed\n"
    assert _parse_test_counts(out) == (8, 1)


def test_parse_test_counts_sums_multiple_pytest_blocks():
    out = "3 passed\n=== 5 passed, 1 failed in 0.1s ===\n"
    assert _parse_test_counts(out) == (9, 1)


def test_run_validation_all_pass():
    with tempfile.TemporaryDirectory() as td:
        r = run_validation(Path(td), "true", "true", "true")
        assert r.build_ok and r.tests_ok and r.lint_ok
        assert r.issues == []


def test_run_validation_records_each_failure():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        rb = run_validation(td, "false", "true", "true")
        assert not rb.build_ok and "Build failed during validation" in rb.issues
        rt = run_validation(td, "true", "false", "true")
        assert not rt.tests_ok and "Tests failed during validation" in rt.issues
        rl = run_validation(td, "true", "true", "false")
        assert not rl.lint_ok and any("Lint" in i for i in rl.issues)


def test_run_validation_skips_when_no_commands():
    with tempfile.TemporaryDirectory() as td:
        r = run_validation(Path(td), None, None, None)
        assert r.build_ok and r.tests_ok and r.lint_ok and r.issues == []
        # An absent gate must report SKIP, not OK — reporting OK would hide an
        # untested project behind a green-looking summary.
        assert not (r.build_ran or r.tests_ran or r.lint_ran)
        assert r.summary() == "build=SKIP | tests=SKIP | lint=SKIP"
        # Nothing actually ran, so nothing was verified -> not passed.
        assert r.passed is False


def test_passed_true_for_build_only_project():
    # A build-only project (no test/lint command) still passes on a green build;
    # an absent gate is non-blocking (run_validation sets *_ok True, *_ran False
    # for it), only an all-absent result fails.
    r = ValidationResult()
    r.build_ok = True
    r.tests_ok = True
    r.tests_ran = False
    r.lint_ok = True
    r.lint_ran = False
    assert r.passed is True


def test_validation_result_summary_status():
    r = ValidationResult()
    r.build_ok = True
    r.tests_ok = False
    r.lint_ran = False
    s = r.summary()
    assert "build=OK" in s and "tests=FAIL" in s and "lint=SKIP" in s
    assert r.passed is False


def test_run_cmd_timeout_and_env_prefix():
    with tempfile.TemporaryDirectory() as td:
        ok, out = _run_cmd("sleep 5", Path(td), timeout=1)
        assert not ok and "timed out" in out
        ok2, _ = _run_cmd("true", Path(td), env_activate="true")
        assert ok2  # env_activate prefix is chained with &&


def test_validate_code_unsupported_language_uses_brace_fallback():
    # Java has no tree-sitter gate here -> falls back to brace balancing.
    ok, err = CodeValidator.validate_code("class A { void m() { }", language="java")
    assert not ok and "delimiter" in err
    ok2, _ = CodeValidator.validate_code("class A { void m() {} }", language="java")
    assert ok2


def test_parse_test_counts_swift_xctest():
    out = "Test Suite 'All tests' passed\nExecuted 42 tests, with 0 failures (0 unexpected)"
    assert _parse_test_counts(out) == (42, 0)


def test_parse_test_counts_swift_with_failures():
    out = "Executed 10 tests, with 3 failures (0 unexpected) in 0.5 seconds"
    assert _parse_test_counts(out) == (10, 3)


def test_parse_test_counts_ctest():
    out = "100% tests passed, 0 tests failed out of 17"
    assert _parse_test_counts(out) == (17, 0)


def test_parse_test_counts_ctest_failures():
    out = "82% tests passed, 3 tests failed out of 17"
    assert _parse_test_counts(out) == (17, 3)


def test_parse_test_counts_dotnet_vstest():
    out = "Failed:     0, Passed:    24, Skipped:     1, Total:    25, Duration: 2 s"
    assert _parse_test_counts(out) == (25, 0)


def test_parse_test_counts_dotnet_vstest_failures():
    out = "Failed:     2, Passed:    23, Skipped:     0, Total:    25"
    assert _parse_test_counts(out) == (25, 2)


def test_parse_test_counts_dotnet_alt_format():
    out = "Total tests: 25. Passed: 23. Failed: 2. Skipped: 0."
    assert _parse_test_counts(out) == (25, 2)


def test_parse_test_counts_node_test_runner():
    out = "ℹ tests 169\nℹ suites 20\nℹ pass 163\nℹ fail 6\nℹ cancelled 0"
    assert _parse_test_counts(out) == (169, 6)


def test_parse_test_counts_node_tap_format():
    out = "# tests 169\n# pass 169\n# fail 0"
    assert _parse_test_counts(out) == (169, 0)


def test_code_validator_valid_python():
    valid, err = CodeValidator.validate_code("x = 1 + 2\ndef f(): pass")
    assert valid and err is None


def test_code_validator_invalid_python():
    valid, err = CodeValidator.validate_code("def f(")
    assert not valid and "Syntax error" in err


def test_code_validator_balanced_braces():
    valid, err = CodeValidator.validate_code(
        "fn main() { let x = [1, 2]; }", language="rust"
    )
    assert valid and err is None


def test_code_validator_unbalanced():
    valid, err = CodeValidator.validate_code(
        "fn main() { let x = [1, 2; }", language="rust"
    )
    assert not valid


def test_code_validator_rust_brace_in_string_passes():
    # Regression: brace-counting falsely rejected a brace inside a string; the
    # tree-sitter gate understands string literals.
    valid, err = CodeValidator.validate_code('fn f() { let s = "}"; }', language="rust")
    assert valid and err is None


def test_code_validator_rust_real_syntax_error():
    valid, err = CodeValidator.validate_code("fn f( { let", language="rust")
    assert not valid and "syntax error" in err


def test_code_validator_tsx_jsx_passes():
    valid, err = CodeValidator.validate_code(
        "const v = <div>{x}</div>;", language="typescript"
    )
    assert valid and err is None


def test_code_validator_csharp_syntax_error():
    valid, err = CodeValidator.validate_code("class A { void M( {", language="csharp")
    assert not valid


def test_certainty_high():
    score = CertaintyScorer.compute_score(
        "I have verified that this is correct. Tests pass successfully. The solution is complete."
    )
    assert score > 0.7


def test_certainty_low():
    score = CertaintyScorer.compute_score(
        "Maybe this could work, not sure, possibly wrong."
    )
    assert score < 0.3


def test_certainty_code_boost():
    score_no_code = CertaintyScorer.compute_score("Here is a solution.")
    score_code = CertaintyScorer.compute_score(
        "Here is a solution.\n```python\nx = 1\n```"
    )
    assert score_code > score_no_code


def test_stall_detector_no_stall():
    sd = StallDetector()
    r1 = sd.push_edit({"a.py": "def foo(): return 1"})
    assert r1 < 0.5


def test_stall_detector_identical():
    sd = StallDetector()
    sd.push_edit({"a.py": "def foo(): return 1"})
    r2 = sd.push_edit({"a.py": "def foo(): return 1"})
    assert r2 > 0.7


def test_stall_detector_similar():
    sd = StallDetector()
    sd.push_edit({"a.py": "def foo(): return 1"})
    r2 = sd.push_edit({"a.py": "def foo(): return 2"})
    assert r2 < 0.8  # similar but not identical


def test_tokenize():
    tokens = _tokenize("hello world_foo bar123 x")
    assert tokens == {"hello", "world_foo", "bar123", "x"}


def test_tokenize_empty():
    assert _tokenize("") == set()
    assert _tokenize("   ") == set()


def test_run_health_check_honors_per_command_timeouts(monkeypatch):
    # build_timeout/test_timeout must override the shared default per command,
    # so a slow compiler isn't falsely reported as a build failure.
    import misterdev.core.verification.validator as v

    calls = []

    def fake_run_cmd(cmd, cwd, env_activate=None, timeout=180):
        calls.append((cmd, timeout))
        return True, "ok"

    monkeypatch.setattr(v, "_run_cmd", fake_run_cmd)
    v.run_health_check(
        "/tmp",
        build_command="cargo build",
        test_command="cargo test",
        lint_command="cargo clippy",
        build_timeout=600,
        test_timeout=300,
    )
    by_cmd = dict(calls)
    assert by_cmd["cargo build"] == 600
    assert by_cmd["cargo test"] == 300
    assert by_cmd["cargo clippy"] == 300  # lint reuses test_timeout


def test_run_health_check_explicit_lint_timeout(monkeypatch):
    import misterdev.core.verification.validator as v

    calls = {}

    def fake_run_cmd(cmd, cwd, env_activate=None, timeout=180):
        calls[cmd] = timeout
        return True, "ok"

    monkeypatch.setattr(v, "_run_cmd", fake_run_cmd)
    v.run_health_check(
        "/tmp",
        build_command="b",
        test_command="t",
        lint_command="lint",
        test_timeout=300,
        lint_timeout=240,
    )
    assert calls["lint"] == 240  # explicit lint_timeout, not test_timeout
    assert calls["t"] == 300
