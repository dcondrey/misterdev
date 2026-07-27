"""T2.1b — real go / swift / csharp compile_view adapters (were stubs).

Each parses verbatim compiler output into CompileError records (message, code,
location) and detects its own output for the no-language-hint fallback, at the same
altitude as the rust/tsc adapters. Cross-language detection must not bleed: a rust or
tsc diagnostic is never claimed by these parsers.
"""

from misterdev.core.execution.compile_view import (
    extract_compile_errors,
    get_adapter,
)

# --- verbatim compiler output -------------------------------------------------

GO = (
    "# example/app\n"
    "./main.go:10:6: undefined: helper\n"
    "./util.go:4:2: cannot use n (variable of type int) as string value in argument\n"
)
SWIFT = (
    "/src/App.swift:12:15: error: cannot find 'foo' in scope\n"
    "/src/App.swift:20:9: error: value of type 'Widget' has no member 'render'\n"
)
CSHARP = (
    "Program.cs(12,20): error CS0103: The name 'foo' does not exist in the current context\n"
    "src/Svc.cs(30,13): error CS1002: ; expected [/repo/App.csproj]\n"
)


def test_go_extracted():
    errs = extract_compile_errors(GO, language="go")
    locs = {e.location for e in errs}
    assert "./main.go:10:6" in locs
    assert any("undefined: helper" in e.message for e in errs)
    assert "./util.go:4:2" in locs


def test_go_autodetected():
    assert any(e.location == "./main.go:10:6" for e in extract_compile_errors(GO))


def test_swift_extracted():
    errs = extract_compile_errors(SWIFT, language="swift")
    by_loc = {e.location: e for e in errs}
    assert "/src/App.swift:12:15" in by_loc
    assert "cannot find 'foo' in scope" in by_loc["/src/App.swift:12:15"].message
    assert "/src/App.swift:20:9" in by_loc


def test_swift_autodetected():
    assert any(
        e.location == "/src/App.swift:12:15" for e in extract_compile_errors(SWIFT)
    )


def test_csharp_extracted():
    errs = extract_compile_errors(CSHARP, language="csharp")
    by_code = {e.code: e for e in errs}
    assert set(by_code) == {"CS0103", "CS1002"}
    assert by_code["CS0103"].location == "Program.cs:12:20"
    # the trailing MSBuild [project] tag is stripped from the message
    assert by_code["CS1002"].message == "; expected"
    assert by_code["CS1002"].location == "src/Svc.cs:30:13"


def test_csharp_autodetected():
    assert any(e.code == "CS0103" for e in extract_compile_errors(CSHARP))


def test_no_cross_language_detection_bleed():
    # rust and tsc outputs must not be claimed by go/swift/csharp adapters.
    rust = "error[E0308]: mismatched types\n --> src/lib.rs:2:18"
    tsc = "src/index.ts(12,7): error TS2322: Type 'string' is not assignable to type 'number'."
    for lang in ("go", "swift", "csharp"):
        ad = get_adapter(lang)
        assert ad.detect(rust) is False
        assert ad.detect(tsc) is False
