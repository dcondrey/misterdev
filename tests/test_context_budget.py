from my_project_orchestrator.core.economics.context_budget import (
    ContextBudget,
    estimate_tokens,
    _get_encoder,
)


def test_estimate_tokens():
    assert estimate_tokens("") == 1
    assert estimate_tokens("hello world") > 0
    # Monotonic in length regardless of backend (tiktoken or char heuristic).
    assert estimate_tokens("def f(): return 1\n" * 50) > estimate_tokens("def f()")


def test_estimate_tokens_matches_tiktoken_when_available():
    enc = _get_encoder()
    if enc is None:
        return  # tiktoken not installed; heuristic path covered above
    text = "def tokenize(source: str) -> list[str]: return source.split()"
    assert estimate_tokens(text) == len(enc.encode(text))


def test_no_truncation_under_budget():
    budget = ContextBudget(max_tokens=100000)
    budget.set("code", "short content", priority=1)
    budget.set("notes", "also short", priority=3)
    result = budget.allocate()
    assert result["code"] == "short content"
    assert result["notes"] == "also short"


def test_truncation_over_budget():
    budget = ContextBudget(max_tokens=200, reserved_tokens=50)
    # ~150 tokens available
    big = "\n".join(f"line {i}: some content here" for i in range(200))
    budget.set("big_section", big, priority=3)
    budget.set("essential", "keep this", priority=1)
    result = budget.allocate()
    assert result["essential"] == "keep this"
    assert len(result["big_section"]) < len(big)
    assert "omitted" in result["big_section"]


def test_priority_order():
    budget = ContextBudget(max_tokens=300, reserved_tokens=50)
    medium = "\n".join(f"line {i}" for i in range(100))
    budget.set("low_priority", medium, priority=3)
    budget.set("high_priority", medium, priority=1)
    result = budget.allocate()
    # Low priority should be truncated more
    assert len(result["low_priority"]) <= len(result["high_priority"])


def test_summary():
    budget = ContextBudget(max_tokens=10000)
    budget.set("code", "x" * 350, priority=1)
    budget.set("notes", "y" * 70, priority=3)
    summary = budget.summary()
    assert "budget=" in summary
    assert "code=" in summary
    assert "OK" in summary


def test_min_lines_respected():
    budget = ContextBudget(max_tokens=100, reserved_tokens=50)
    # Very tight budget
    small = "\n".join(f"line {i}" for i in range(5))
    budget.set("small", small, priority=3, min_lines=5)
    result = budget.allocate()
    # Should keep all 5 lines since min_lines=5
    assert "line 4" in result["small"]
