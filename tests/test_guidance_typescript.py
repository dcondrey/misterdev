from misterdev.core.context.guidance._rules import render_rules, select_rules
from misterdev.core.context.guidance.typescript import TYPESCRIPT_RULES


def test_rules_have_a_core_baseline():
    core = [r for r in TYPESCRIPT_RULES if r.core]
    assert len(core) >= 2
    # core rules carry no triggers — they are unconditional.
    assert all(r.triggers == () for r in core)


def test_core_always_selected_with_empty_context():
    selected = select_rules(TYPESCRIPT_RULES, "")
    assert selected  # never empty
    assert all(r.core for r in selected)


def test_validation_context_pulls_validation_rule_only():
    selected = select_rules(
        TYPESCRIPT_RULES, "parse an untrusted API response with zod at the boundary"
    )
    text = render_rules("TypeScript", selected).lower()
    assert "zod" in text or "valibot" in text  # validation rule matched
    # testing rule NOT pulled in
    assert "vitest" not in text and "expect-type" not in text


def test_selection_is_smaller_than_the_whole_ruleset():
    everything = render_rules("TypeScript", TYPESCRIPT_RULES)
    typical = render_rules(
        "TypeScript", select_rules(TYPESCRIPT_RULES, "rename an internal helper")
    )
    assert len(typical) < len(everything)
