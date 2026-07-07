from misterdev.core.context.guidance import get_language_guidance
from misterdev.core.context.guidance._rules import render_rules, select_rules
from misterdev.core.context.guidance.rust import RUST_RULES


def test_rules_have_a_core_baseline():
    core = [r for r in RUST_RULES if r.core]
    assert len(core) >= 3
    # core rules carry no triggers — they are unconditional.
    assert all(r.triggers == () for r in core)


def test_core_always_selected_with_empty_context():
    selected = select_rules(RUST_RULES, "")
    assert selected  # never empty
    assert all(r.core for r in selected)
    lower = render_rules("Rust", selected).lower()
    assert "unwrap" in lower  # a core anchor


def test_async_context_pulls_in_concurrency_rule_only():
    selected = select_rules(RUST_RULES, "add an async tokio handler that spawns tasks")
    text = render_rules("Rust", selected).lower()
    assert "await" in text and "joinset" in text  # concurrency rule matched
    assert "zeroize" not in text  # crypto rule NOT pulled in


def test_crypto_context_pulls_in_crypto_rule_only():
    selected = select_rules(
        RUST_RULES, "constant-time verify of a signature over a secret key"
    )
    text = render_rules("Rust", selected).lower()
    assert "zeroize" in text and "constanttimeeq" in text
    assert "rayon" not in text  # concurrency rule NOT pulled in


def test_selection_is_smaller_than_the_whole_ruleset():
    everything = render_rules("Rust", RUST_RULES)
    typical = render_rules(
        "Rust", select_rules(RUST_RULES, "refactor a helper function")
    )
    assert len(typical) < len(everything) / 2  # relevance cut, not the whole block


def test_get_language_guidance_threads_context():
    text = get_language_guidance("rust", "fuzz the cbor parser").lower()
    assert "cargo-fuzz" in text  # testing rule matched via "fuzz"/"parser"
