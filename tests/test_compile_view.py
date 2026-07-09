"""CompileView parses real rustc output into exact diagnostics."""

from misterdev.core.execution.compile_view import (
    extract_compile_errors,
    render_compile_view,
)

# Verbatim `cargo build` output for a type error + a name error.
RUSTC = """\
   Compiling cap v0.1.0 (/private/tmp/rustc_cap)
error[E0308]: mismatched types
 --> src/lib.rs:2:18
  |
2 |     let x: u32 = "hello";
  |            ---   ^^^^^^^ expected `u32`, found `&str`
  |            |
  |            expected due to this

error[E0425]: cannot find function `missing_fn` in this scope
 --> src/lib.rs:3:5
  |
3 |     missing_fn(x)
  |     ^^^^^^^^^^ not found in this scope

error: could not compile `cap` (lib) due to 2 previous errors
"""


def test_rustc_extraction():
    errs = extract_compile_errors(RUSTC, language="rust")
    # The two real diagnostics; the trailing "could not compile" summary carries
    # no code/location and is not a fixable diagnostic on its own.
    coded = [e for e in errs if e.code]
    assert {e.code for e in coded} == {"E0308", "E0425"}
    mismatch = next(e for e in coded if e.code == "E0308")
    assert mismatch.message == "mismatched types"
    assert mismatch.location == "src/lib.rs:2:18"
    assert "expected `u32`, found `&str`" in mismatch.detail
    name = next(e for e in coded if e.code == "E0425")
    assert name.location == "src/lib.rs:3:5"


def test_rustc_autodetected_without_language():
    errs = extract_compile_errors(RUSTC)
    assert any(e.code == "E0308" for e in errs)


def test_render_is_exact():
    errs = extract_compile_errors(RUSTC, language="rust")
    view = render_compile_view(errs)
    assert "[E0308] mismatched types (src/lib.rs:2:18)" in view
    assert "expected `u32`, found `&str`" in view


def test_unrecognized_output_is_empty():
    assert extract_compile_errors("just a log line, no compiler errors") == []
    assert render_compile_view([]) == ""
