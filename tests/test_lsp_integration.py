"""Live LSP integration harness — drives a REAL language server (jedi, offline).

``multilspy`` is in the dev dependency group (same rationale as the MCP SDK), so
this exercises the real server path in CI instead of skipping. It skips *loudly*
only when the server genuinely can't start in the environment; when it does
start, a planted error MUST surface — a silent empty result is a failure, not a
skip. Server startup is a few seconds, bounded by ``collect_diagnostics``.
"""

import tempfile
from pathlib import Path

import pytest

pytest.importorskip(
    "multilspy", reason="install the misterdev[lsp] extra to run the LSP harness"
)

from misterdev.core.context.lsp import (  # noqa: E402
    collect_and_format_lsp_context,
    collect_diagnostics,
)

_TIMEOUT = 45.0


def _fixture(src: str) -> Path:
    d = Path(tempfile.mkdtemp())
    (d / "m.py").write_text(src, encoding="utf-8")
    return d


def _diags_or_skip(root: Path):
    """Run the real server; skip (loud) only if it didn't respond — ``None`` from
    ``collect_diagnostics`` means unavailable/timed-out, a list means it ran."""
    diags = collect_diagnostics(root, "python", ["m.py"], timeout=_TIMEOUT)
    if diags is None:
        pytest.skip("LSP server did not start/respond in this environment")
    return diags


def test_planted_syntax_error_surfaces_via_real_server():
    diags = _diags_or_skip(_fixture("def f(:\n    pass\n"))
    assert diags, "server ran but reported nothing for a definite syntax error"
    first = diags[0]
    assert first["line"] == 1
    assert "syntax" in first["message"].lower()
    assert first["file"].endswith("m.py")


def test_indentation_error_reports_correct_line():
    diags = _diags_or_skip(_fixture("def f():\nreturn 1\n"))
    assert any(d["line"] == 2 and "indent" in d["message"].lower() for d in diags), (
        diags
    )


def test_real_diagnostics_render_into_injectable_context():
    root = _fixture("x = = 1\n")
    diags = _diags_or_skip(root)
    assert diags  # a real error was captured
    ctx = collect_and_format_lsp_context(root, "python", ["m.py"], timeout=_TIMEOUT)
    assert ctx.startswith("## Language-server diagnostics")
    assert "m.py:1" in ctx


def test_clean_file_yields_no_error_diagnostics():
    root = _fixture("def f() -> int:\n    return 1\n")
    diags = _diags_or_skip(root)
    assert diags == []  # server ran, valid file -> no errors (no false positives)
    ctx = collect_and_format_lsp_context(root, "python", ["m.py"], timeout=_TIMEOUT)
    assert ctx == ""
