"""Relevance-selection tests for the Kotlin best-practice rule set."""

from misterdev.core.context.guidance._rules import render_rules, select_rules
from misterdev.core.context.guidance.kotlin import KOTLIN_RULES


def test_core_rules_present_and_untriggered():
    core = [r for r in KOTLIN_RULES if r.core]
    assert len(core) >= 2
    assert all(r.triggers == () for r in core)


def test_empty_context_yields_only_core():
    selected = select_rules(KOTLIN_RULES, "")
    assert selected
    assert all(r.core for r in selected)


def test_coroutine_context_pulls_coroutines_excludes_compose():
    ctx = "launch a suspend function on Dispatchers.IO with a Flow"
    selected = select_rules(KOTLIN_RULES, ctx)
    text = " ".join(r.text.lower() for r in selected)
    # coroutines rule present
    assert "coroutinescope" in text or "globalscope" in text
    # compose rule excluded
    assert "recomposition" not in text


def test_selected_selection_shorter_than_full_render():
    generic = select_rules(KOTLIN_RULES, "add a helper function")
    selected_render = render_rules("Kotlin", generic)
    full_render = render_rules("Kotlin", KOTLIN_RULES)
    assert len(selected_render) < len(full_render)
