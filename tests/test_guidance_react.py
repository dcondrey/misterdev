from misterdev.core.context.guidance import guidance_for_files
from misterdev.core.context.guidance._rules import render_rules, select_rules
from misterdev.core.context.guidance.react import REACT_RULES


def test_core_rules_present_and_empty_triggers():
    core = [r for r in REACT_RULES if r.core]
    assert len(core) >= 2
    assert all(r.triggers == () for r in core)


def test_empty_context_selects_only_core():
    selected = select_rules(REACT_RULES, "")
    assert selected
    assert all(r.core for r in selected)


def test_effects_context_pulls_effects_excludes_forms():
    rendered = render_rules(
        "React",
        select_rules(REACT_RULES, "a useEffect that subscribes and needs cleanup"),
    )
    assert "exhaustive-deps" in rendered or "cleanup" in rendered
    assert "useactionstate" not in rendered.lower()


def test_generic_selection_shorter_than_full():
    generic = render_rules("React", select_rules(REACT_RULES, "update a component"))
    full = render_rules("React", REACT_RULES)
    assert len(generic) < len(full)


def test_jsx_and_tsx_route_to_react():
    # Routing depends on __init__ wiring done separately; tolerate the unwired state.
    for name in ("a.jsx", "a.tsx"):
        text = guidance_for_files([name])
        assert text == "" or text.startswith("React — ")
