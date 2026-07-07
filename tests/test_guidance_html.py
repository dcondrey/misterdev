"""Relevance-selection tests for the HTML best-practice rule set."""

from misterdev.core.context.guidance._rules import render_rules, select_rules
from misterdev.core.context.guidance.html import HTML_RULES


def test_core_rules_present_and_untriggered():
    core = [r for r in HTML_RULES if r.core]
    assert len(core) >= 2
    assert all(r.triggers == () for r in core)


def test_empty_context_selects_only_core():
    selected = select_rules(HTML_RULES, "")
    assert selected
    assert all(r.core for r in selected)


def test_forms_context_pulls_forms_rule_and_excludes_security():
    selected = select_rules(HTML_RULES, "add a login form with an email input")
    rendered = render_rules("HTML", selected).lower()
    assert "fieldset" in rendered
    assert "autocomplete" in rendered
    assert "content-security-policy" not in rendered
    assert "csp" not in rendered


def test_selected_subset_shorter_than_full_set():
    selected = select_rules(HTML_RULES, "restructure the navigation landmarks")
    partial = render_rules("HTML", selected)
    full = render_rules("HTML", HTML_RULES)
    assert len(partial) < len(full)
