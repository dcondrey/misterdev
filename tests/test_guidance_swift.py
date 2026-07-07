from misterdev.core.context.guidance._rules import render_rules, select_rules
from misterdev.core.context.guidance.swift import SWIFT_RULES


def test_core_rules_are_always_on_with_empty_triggers():
    core = [r for r in SWIFT_RULES if r.core]
    assert len(core) >= 2
    assert all(r.triggers == () for r in core)


def test_empty_context_selects_only_core():
    selected = select_rules(SWIFT_RULES, "")
    assert selected
    assert all(r.core for r in selected)


def test_concurrency_context_pulls_concurrency_excludes_swiftui():
    selected = select_rules(SWIFT_RULES, "an actor with async await and Sendable state")
    text = render_rules("Swift", selected).lower()
    assert "mainactor" in text or "taskgroup" in text
    assert "@observable" not in text
    assert "foreach" not in text


def test_selection_shorter_than_full_render():
    selected = select_rules(SWIFT_RULES, "refactor a generic protocol conformance")
    assert len(render_rules("Swift", selected)) < len(
        render_rules("Swift", SWIFT_RULES)
    )
