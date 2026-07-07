from misterdev.core.context.guidance._rules import render_rules, select_rules
from misterdev.core.context.guidance.elixir import ELIXIR_RULES


def test_core_rules_present_and_untriggered():
    core = [r for r in ELIXIR_RULES if r.core]
    assert len(core) >= 2
    assert all(r.triggers == () for r in core)


def test_empty_context_selects_only_core():
    selected = select_rules(ELIXIR_RULES, "")
    assert selected
    assert all(r.core for r in selected)


def test_otp_context_pulls_otp_excludes_phoenix():
    selected = select_rules(
        ELIXIR_RULES, "a GenServer supervised by a supervision tree"
    )
    text = render_rules("Elixir", selected).lower()
    assert "let it crash" in text or "supervision" in text
    assert "liveview" not in text


def test_generic_selection_shorter_than_full():
    selected = render_rules(
        "Elixir", select_rules(ELIXIR_RULES, "rename a helper function")
    )
    full = render_rules("Elixir", ELIXIR_RULES)
    assert len(selected) < len(full)
