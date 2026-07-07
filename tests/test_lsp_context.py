import misterdev.core.context.lsp as lsp
from misterdev.core.context.lsp import (
    collect_and_format_lsp_context,
    format_lsp_context,
)


def test_none_and_empty_render_to_empty_string():
    assert format_lsp_context(None) == ""
    assert format_lsp_context([]) == ""


def test_diagnostics_render_as_injectable_block():
    diags = [
        {
            "file": "src/lib.rs",
            "line": 12,
            "message": "cannot find value `foo` in this scope",
        },
        {"file": "src/lib.rs", "line": 40, "message": "mismatched types"},
    ]
    out = format_lsp_context(diags)
    assert out.startswith("## Language-server diagnostics")
    assert "src/lib.rs:12: cannot find value `foo` in this scope" in out
    assert "src/lib.rs:40: mismatched types" in out


def test_cap_bounds_the_block_and_notes_remainder():
    diags = [{"file": "f.py", "line": i, "message": "err"} for i in range(30)]
    out = format_lsp_context(diags, cap=5)
    body = out.splitlines()
    # header + 5 capped lines + a remainder note
    assert len(body) == 1 + 5 + 1
    assert "and 25 more" in out


def test_collect_and_format_empty_for_unsupported_language():
    # No server is spun up for an unknown language; returns "" (no opinion).
    assert collect_and_format_lsp_context("/tmp", "cobol", ["x.cob"]) == ""


def test_collect_and_format_renders_when_diagnostics_present(monkeypatch):
    monkeypatch.setattr(
        lsp,
        "collect_diagnostics",
        lambda root, lang, files, timeout: [
            {"file": "src/a.rs", "line": 3, "message": "unresolved import"}
        ],
    )
    out = collect_and_format_lsp_context("/proj", "rust", ["src/a.rs"])
    assert out.startswith("## Language-server diagnostics")
    assert "src/a.rs:3: unresolved import" in out
