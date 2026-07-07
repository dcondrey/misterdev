from misterdev.core.execution.error_classifier import format_classified_error
from misterdev.core.execution.error_log_compressor import compress_error_log

RUSTC = """\
   Compiling acronym v0.1.0 (/tmp/acronym)
error[E0308]: mismatched types
  --> src/lib.rs:12:5
   |
12 |     phrase
   |     ^^^^^^ expected `String`, found `&str`
   |
   = note: expected struct `String`
              found reference `&str`

error[E0425]: cannot find value `words` in this scope
  --> src/lib.rs:8:17
   |
8  |     let acronym = words.iter().collect();
   |                   ^^^^^ not found in this scope

For more information about this error, try `rustc --explain E0308`.
error: aborting due to 2 previous errors
"""

PYTEST = """\
============================= test session starts ==============================
collected 3 items
tests/test_x.py::test_encode FAILED
    def test_encode():
>       assert encode("test") == "expected"
E       AssertionError: assert 'wrong' == 'expected'
E         - expected
E         + wrong
tests/test_x.py:14: AssertionError
=========================== 1 failed, 2 passed in 0.1s =========================
"""


def test_rustc_keeps_signal_drops_noise():
    out = compress_error_log(RUSTC, language="rust")
    # signal kept
    assert "E0308" in out and "E0425" in out
    assert "src/lib.rs:12:5" in out and "src/lib.rs:8:17" in out
    assert "mismatched types" in out
    # noise dropped
    assert "^^^^^^" not in out
    assert "= note" not in out
    assert "--explain" not in out
    assert "Compiling" not in out
    assert "12 |" not in out  # source echo
    assert "aborting due to" not in out  # summary line


def test_rustc_compresses_hard():
    out = compress_error_log(RUSTC, language="rust")
    assert len(out) < len(RUSTC) * 0.30  # >70% smaller
    assert out.count("\n") + 1 == 2  # exactly the two real errors


def test_rustc_dedup_identical_errors():
    dup = RUSTC + RUSTC  # same two errors twice
    out = compress_error_log(dup, language="rust")
    assert out.count("E0308") == 1 and out.count("E0425") == 1


def test_generic_pytest_keeps_assertion_drops_banner():
    out = compress_error_log(PYTEST, language="python")
    assert "AssertionError" in out  # the error is kept
    assert "test session starts" not in out  # banner noise dropped
    assert "collected 3 items" not in out  # collection noise dropped
    assert len(out) < len(PYTEST)


def test_empty_and_blank_return_empty():
    assert compress_error_log("") == ""
    assert compress_error_log("   \n  \n") == ""


def test_garbage_never_raises_and_shrinks_or_equals():
    junk = "\n".join(f"line {i} noise noise" for i in range(200))
    out = compress_error_log(junk)
    assert isinstance(out, str)
    assert len(out) <= len(junk)  # never grows a prompt


def test_format_classified_error_uses_compressor():
    formatted = format_classified_error(RUSTC)
    assert "E0308" in formatted and "src/lib.rs:12:5" in formatted
    assert "^^^^^^" not in formatted  # compressor ran, not raw dump
    assert len(formatted) < len(RUSTC)
