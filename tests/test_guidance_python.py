"""Tests for the Python best-practice rule set and its relevance selection."""

from misterdev.core.context.guidance._rules import render_rules, select_rules
from misterdev.core.context.guidance.python import PYTHON_RULES


def test_core_rules_present_and_untriggered():
    core = [r for r in PYTHON_RULES if r.core]
    assert len(core) >= 2
    assert all(r.triggers == () for r in core)


def test_empty_context_yields_only_core():
    selected = select_rules(PYTHON_RULES, "")
    assert selected
    assert all(r.core for r in selected)


def test_security_context_pulls_security_excludes_performance():
    selected = select_rules(
        PYTHON_RULES, "compare a password hash and unpickle the token"
    )
    text = render_rules("Python", selected).lower()
    assert "compare_digest" in text or "safe_load" in text
    assert "cprofile" not in text
    assert "numpy" not in text


def test_generic_selection_shorter_than_full_render():
    selected = select_rules(PYTHON_RULES, "refactor a small utility function")
    generic = render_rules("Python", selected)
    full = render_rules("Python", PYTHON_RULES)
    assert len(generic) < len(full)
