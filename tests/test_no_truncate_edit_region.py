"""T2.3 — the edit region is never truncated by the context budget.

The edit region (code_context) holds the exact verbatim lines a SEARCH/REPLACE edit
must match. Under budget pressure the allocator would trim it (priority 1 is trimmed
last, but still trimmed), leaving the model to edit blind against a tail it can no
longer see. A section registered `truncatable=False` must be kept verbatim even when
it alone overflows the budget; ordinary sections still truncate.
"""

from misterdev.core.economics.context_budget import ContextBudget

# An edit region that ALONE overflows the tiny budget below.
EDIT = "\n".join(
    f"{i:04d}: exact source line that must survive verbatim" for i in range(80)
)
FILLER = "\n".join(f"filler note {i}" for i in range(40))


def test_non_truncatable_edit_region_kept_verbatim_under_pressure():
    budget = ContextBudget(max_tokens=200, reserved_tokens=50)  # available=150
    budget.set("code_context", EDIT, priority=1, truncatable=False)
    budget.set("scratchpad", FILLER, priority=3)
    out = budget.allocate()
    assert out["code_context"] == EDIT
    assert "omitted" not in out["code_context"]


def test_ordinary_section_still_truncates():
    budget = ContextBudget(max_tokens=200, reserved_tokens=50)
    budget.set("code_context", EDIT, priority=1)  # truncatable by default
    budget.set("scratchpad", FILLER, priority=3)
    out = budget.allocate()
    assert "omitted" in out["code_context"]


def test_no_pressure_returns_everything_verbatim():
    budget = ContextBudget(max_tokens=100000)
    budget.set("code_context", EDIT, priority=1, truncatable=False)
    out = budget.allocate()
    assert out["code_context"] == EDIT
