"""Standalone tests for the CSS best-practice rule set."""

from misterdev.core.context.guidance._rules import render_rules, select_rules
from misterdev.core.context.guidance.css import CSS_RULES


def test_core_rules_present_and_untriggered():
    core = [r for r in CSS_RULES if r.core]
    assert len(core) >= 2
    assert all(r.triggers == () for r in core)


def test_empty_context_selects_only_core():
    selected = select_rules(CSS_RULES, "")
    assert selected
    assert all(r.core for r in selected)


def test_layout_context_pulls_layout_excludes_print():
    selected = select_rules(CSS_RULES, "build a responsive card grid with flexbox")
    text = render_rules("CSS", selected)
    assert "subgrid" in text or "grid-template" in text
    assert "break-inside" not in text


def test_selected_render_shorter_than_full():
    generic = render_rules("CSS", select_rules(CSS_RULES, "rename a variable"))
    full = render_rules("CSS", CSS_RULES)
    assert len(generic) < len(full)
