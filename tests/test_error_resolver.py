import tempfile
from pathlib import Path

from my_project_orchestrator.core.execution.error_resolver import ErrorResolver, ErrorLocation


def test_resolve_python_traceback_location():
    r = ErrorResolver(Path("."))
    out = 'File "src/app.py", line 42\n    raise ValueError'
    locs = r.resolve_errors(out)
    assert any(loc.file == "src/app.py" and loc.line == 42 for loc in locs)


def test_resolve_colon_location_and_rust_arrow():
    r = ErrorResolver(Path("."))
    out = "src/lib.rs:10: error: bad\n --> src/main.rs:7:3"
    files = {(loc.file, loc.line) for loc in r.resolve_errors(out)}
    assert ("src/lib.rs", 10) in files
    assert ("src/main.rs", 7) in files


def test_resolve_dedups_repeated_locations():
    r = ErrorResolver(Path("."))
    out = "a.py:5: error one\na.py:5: error two"
    locs = r.resolve_errors(out)
    assert len([loc for loc in locs if loc.file == "a.py" and loc.line == 5]) == 1


def test_resolve_no_locations_returns_empty():
    assert (
        ErrorResolver(Path(".")).resolve_errors("a vague failure, no file refs") == []
    )


def test_read_snippet_marks_error_line():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "m.py").write_text("\n".join(f"line{i}" for i in range(1, 21)))
        r = ErrorResolver(td)
        locs = r.resolve_errors("m.py:10: error: boom")
        loc = next(loc for loc in locs if loc.file == "m.py")
        assert "line10" in loc.snippet
        assert ">" in loc.snippet  # the error line is marked


def test_format_for_llm_includes_locations_and_caps():
    r = ErrorResolver(Path("."))
    locs = [ErrorLocation(f"f{i}.py", i, snippet=f"code{i}") for i in range(15)]
    out = r.format_for_llm(locs)
    assert "## Error Attribution" in out
    assert "f0.py:0" in out
    assert "code0" in out
    # capped at 10 entries
    assert "f12.py" not in out


def test_format_for_llm_empty():
    assert ErrorResolver(Path(".")).format_for_llm([]) == ""
