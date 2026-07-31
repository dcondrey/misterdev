"""Unit tests for InteractiveMixin."""

import pytest
from unittest.mock import MagicMock, patch

from misterdev.core.execution.interactive_mixin import InteractiveMixin


class _Orch(InteractiveMixin):
    pass


def _make_task(id_="t1", title="Do a thing"):
    t = MagicMock()
    t.id = id_
    t.title = title
    return t


# ---------------------------------------------------------------------------
# _interactive_prompt
# ---------------------------------------------------------------------------


def test_interactive_prompt_proceed():
    orch = _Orch()
    with patch(
        "misterdev.core.execution.interactive_mixin.Prompt.ask", return_value="y"
    ):
        result = orch._interactive_prompt(_make_task())
    assert result == "proceed"


def test_interactive_prompt_skip():
    orch = _Orch()
    with patch(
        "misterdev.core.execution.interactive_mixin.Prompt.ask", return_value="s"
    ):
        result = orch._interactive_prompt(_make_task())
    assert result == "skip"


def test_interactive_prompt_quit_n():
    orch = _Orch()
    with patch(
        "misterdev.core.execution.interactive_mixin.Prompt.ask", return_value="n"
    ):
        result = orch._interactive_prompt(_make_task())
    assert result == "quit"


def test_interactive_prompt_quit_q():
    orch = _Orch()
    with patch(
        "misterdev.core.execution.interactive_mixin.Prompt.ask", return_value="q"
    ):
        result = orch._interactive_prompt(_make_task())
    assert result == "quit"


def test_interactive_prompt_strategy_shown(capsys):
    orch = _Orch()
    with patch(
        "misterdev.core.execution.interactive_mixin.Prompt.ask", return_value="y"
    ):
        with patch(
            "misterdev.core.execution.interactive_mixin._console.print"
        ) as mock_print:
            orch._interactive_prompt(
                _make_task(title="Refactor auth"), strategy="parallel"
            )
    printed = mock_print.call_args[0][0]
    assert "Refactor auth" in printed
    assert "PARALLEL" in printed


# ---------------------------------------------------------------------------
# _staging_hint
# ---------------------------------------------------------------------------


def _make_symbol(file_path, name="Foo"):
    s = MagicMock()
    s.file_path = file_path
    s.name = name
    return s


def test_staging_hint_no_topography():
    orch = _Orch()
    proj = MagicMock()
    proj.topography = None
    assert orch._staging_hint(proj) == ""


def test_staging_hint_multi_file_goal_empty():
    orch = _Orch()
    graph = MagicMock()
    graph.symbols = {
        "A": _make_symbol("src/a.py"),
        "B": _make_symbol("src/b.py"),
    }
    proj = MagicMock()
    proj.topography.graph = graph
    with patch("misterdev.core.planning.verifier_decomposition.synthesize_stages"):
        result = orch._staging_hint(proj)
    assert result == ""


def test_staging_hint_test_files_excluded():
    orch = _Orch()
    graph = MagicMock()
    graph.symbols = {
        "A": _make_symbol("src/foo.py"),
        "T": _make_symbol("tests/test_foo.py"),
    }
    proj = MagicMock()
    proj.topography.graph = graph
    # Only one non-test file → single-file path; synthesize_stages returns <2 stages → empty
    with patch(
        "misterdev.core.planning.verifier_decomposition.synthesize_stages",
        return_value=["stage1"],
    ):
        result = orch._staging_hint(proj)
    assert result == ""


def test_staging_hint_single_file_with_stages():
    orch = _Orch()
    graph = MagicMock()
    graph.symbols = {
        "A": _make_symbol("src/engine.py", "Engine"),
        "B": _make_symbol("src/engine.py", "Engine.run"),
    }
    proj = MagicMock()
    proj.topography.graph = graph
    with (
        patch(
            "misterdev.core.planning.verifier_decomposition.synthesize_stages",
            return_value=["construct", "mutate"],
        ),
        patch(
            "misterdev.core.planning.verifier_decomposition.render_stages",
            return_value="1. construct\n2. mutate",
        ),
    ):
        result = orch._staging_hint(proj)
    assert "Suggested staging" in result
    assert "1. construct" in result


def test_staging_hint_exception_returns_empty():
    orch = _Orch()
    proj = MagicMock()
    proj.topography.graph = MagicMock(side_effect=RuntimeError("boom"))
    result = orch._staging_hint(proj)
    assert result == ""
