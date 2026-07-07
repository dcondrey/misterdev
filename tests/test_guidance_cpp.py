"""Standalone tests for the C++ guidance rule set.

Depends only on the cpp module + _rules (not the package __init__), so it runs
in isolation: `.venv/bin/python -m pytest tests/test_guidance_cpp.py -q`.
"""

from misterdev.core.context.guidance._rules import render_rules, select_rules
from misterdev.core.context.guidance.cpp import CPP_RULES


def test_core_rules_present_and_untriggered():
    core = [r for r in CPP_RULES if r.core]
    assert len(core) >= 2
    assert all(r.triggers == () for r in core)


def test_empty_context_selects_only_core():
    selected = select_rules(CPP_RULES, "")
    assert selected
    assert all(r.core for r in selected)


def test_safety_context_pulls_ub_rule_excludes_tooling():
    ctx = "fuzz the parser under asan for use-after-free"
    text = render_rules("C++", select_rules(CPP_RULES, ctx))
    # UB/safety rule is pulled.
    assert "sanitizer" in text or "ubsan" in text.lower()
    assert "ASan/UBSan/TSan/MSan" in text
    # Tooling rule is NOT pulled by this context.
    assert "clang-tidy" not in text
    assert "include-what-you-use" not in text


def test_generic_task_shorter_than_full_render():
    generic = render_rules("C++", select_rules(CPP_RULES, "refactor a helper function"))
    full = render_rules("C++", CPP_RULES)
    assert len(generic) < len(full)
