from my_project_orchestrator.core.error_classifier import (
    classify_error, classify_and_guide, format_classified_error, ErrorCategory,
)


def test_classify_rust_syntax():
    error = 'error: expected `;`\n --> src/core/posting.rs:42:5'
    assert classify_error(error) == ErrorCategory.SYNTAX


def test_classify_rust_missing_import():
    error = "error[E0432]: unresolved import `crate::core::posting`"
    assert classify_error(error) == ErrorCategory.MISSING_IMPORT


def test_classify_rust_missing_export():
    error = "error[E0603]: struct `PostingShard` is private"
    assert classify_error(error) == ErrorCategory.MISSING_EXPORT


def test_classify_rust_wrong_type():
    error = "error[E0308]: mismatched types\nexpected `usize`, found `i32`"
    assert classify_error(error) == ErrorCategory.WRONG_TYPE


def test_classify_rust_missing_symbol():
    error = "error[E0425]: cannot find function `overlap_scan` in this scope"
    assert classify_error(error) == ErrorCategory.MISSING_SYMBOL


def test_classify_rust_trait_bound():
    error = "error[E0277]: the trait bound `Config: Clone` is not satisfied"
    assert classify_error(error) == ErrorCategory.WRONG_TYPE


def test_classify_test_failure():
    error = "thread 'test_posting' panicked at 'assertion failed: left == right'\nleft: 42\nright: 0"
    assert classify_error(error) == ErrorCategory.TEST_ASSERTION


def test_classify_python_syntax():
    error = "SyntaxError: invalid syntax (main.py, line 10)"
    assert classify_error(error) == ErrorCategory.SYNTAX


def test_classify_python_import():
    error = "ModuleNotFoundError: No module named 'nonexistent'"
    assert classify_error(error) == ErrorCategory.MISSING_IMPORT


def test_classify_unknown():
    error = "something completely unexpected happened"
    assert classify_error(error) == ErrorCategory.UNKNOWN


def test_classify_and_guide_returns_tuple():
    category, guidance = classify_and_guide("error[E0308]: mismatched types")
    assert category == ErrorCategory.WRONG_TYPE
    assert "TYPE MISMATCH" in guidance


def test_format_classified_error():
    output = format_classified_error("error[E0432]: unresolved import `foo`")
    assert "MISSING_IMPORT" in output
    assert "Fix Strategy" in output
    assert "unresolved import" in output


def test_format_truncates_long_output():
    long_error = "\n".join(f"error line {i}" for i in range(200))
    output = format_classified_error(long_error, max_lines=10)
    assert "190 more lines" in output


def test_multiple_indicators_picks_strongest():
    error = (
        "mismatched types\n"
        "expected type `String`\n"
        "incompatible type found\n"
        "cannot find module `foo`"
    )
    assert classify_error(error) == ErrorCategory.WRONG_TYPE


def test_rust_error_code_e0308():
    error = "error[E0308]: mismatched types\n  --> src/lib.rs:42:5"
    assert classify_error(error) == ErrorCategory.WRONG_TYPE


def test_rust_error_code_e0432():
    error = "error[E0432]: unresolved import `crate::core::posting`"
    assert classify_error(error) == ErrorCategory.MISSING_IMPORT


def test_rust_error_code_e0425():
    error = "error[E0425]: cannot find function `overlap_scan` in this scope"
    assert classify_error(error) == ErrorCategory.MISSING_SYMBOL


def test_rust_error_code_e0603():
    error = "error[E0603]: struct `PostingShard` is private"
    assert classify_error(error) == ErrorCategory.MISSING_EXPORT


def test_rust_error_code_e0277():
    error = "error[E0277]: the trait bound `Config: Clone` is not satisfied"
    assert classify_error(error) == ErrorCategory.WRONG_TYPE


def test_rust_error_code_takes_priority():
    # E0432 should win even though "cannot find" is also present
    error = "error[E0432]: unresolved import\ncannot find function"
    assert classify_error(error) == ErrorCategory.MISSING_IMPORT
