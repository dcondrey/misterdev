from misterdev.core.context.guidance import (
    get_language_guidance,
    guidance_for_files,
)


def test_language_lookup_normalizes_and_aliases():
    # Every language is rule-based: a rendered, relevance-selected titled block.
    assert get_language_guidance("Python").startswith("Python — ")
    assert get_language_guidance("py").startswith("Python — ")
    assert get_language_guidance("rust").startswith("Rust — ")
    assert get_language_guidance("rs").startswith("Rust — ")
    assert get_language_guidance("C#").startswith("C# — ")


def test_unknown_language_returns_empty_not_none():
    assert get_language_guidance("cobol") == ""
    assert get_language_guidance("") == ""


def test_files_prefer_extension_over_fallback_language():
    # A .rs file wins even when the project's declared language is python.
    assert guidance_for_files(["src/lib.rs"], "python").startswith("Rust — ")


def test_jsx_and_tsx_route_to_react():
    assert guidance_for_files(["ui/App.jsx"]).startswith("React — ")
    assert guidance_for_files(["ui/App.tsx"]).startswith("React — ")


def test_css_extension_resolves():
    text = guidance_for_files(["styles/main.scss"])
    assert text.startswith("CSS — ")
    assert text == get_language_guidance("css")


def test_no_extension_match_falls_back_to_language():
    assert guidance_for_files(["notes.txt"], "python").startswith("Python — ")
    assert guidance_for_files(["notes.txt"], "cobol") == ""
    assert guidance_for_files([], "rust").startswith("Rust — ")


def test_context_selects_relevant_rules_for_rule_languages():
    # The .rs path threads context through to rule selection.
    crypto = guidance_for_files(["src/sign.rs"], context="constant-time key comparison")
    assert "zeroize" in crypto.lower()
    assert "rayon" not in crypto.lower()
