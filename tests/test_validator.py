from my_project_orchestrator.core.validator import (
    CodeValidator,
    CertaintyScorer,
    StallDetector,
    _tokenize,
    _parse_test_counts,
)


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
    import my_project_orchestrator.core.validator as v

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
    import my_project_orchestrator.core.validator as v

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
