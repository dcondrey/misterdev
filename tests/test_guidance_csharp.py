from misterdev.core.context.guidance._rules import render_rules, select_rules
from misterdev.core.context.guidance.csharp import CSHARP_RULES


def test_core_rules_present_and_untriggered():
    core = [r for r in CSHARP_RULES if r.core]
    assert len(core) >= 2
    assert all(r.triggers == () for r in core)


def test_empty_context_selects_only_core():
    selected = select_rules(CSHARP_RULES, "")
    assert selected
    assert all(r.core for r in selected)


def test_async_context_pulls_async_excludes_tooling():
    selected = select_rules(CSHARP_RULES, "await a Task with a CancellationToken")
    rendered = render_rules("C#", selected).lower()
    assert "configureawait" in rendered or "valuetask" in rendered
    assert "benchmarkdotnet" not in rendered
    assert "editorconfig" not in rendered


def test_generic_selection_shorter_than_full():
    selected = render_rules("C#", select_rules(CSHARP_RULES, "add a helper method"))
    full = render_rules("C#", CSHARP_RULES)
    assert len(selected) < len(full)
