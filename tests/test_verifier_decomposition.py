"""Tests for the pure verifier-decomposition ordering logic.

Covers the D3 staging heuristic: construction -> mutation -> query, contiguous
renumbering, dedup, empty handling, and the rendered prompt block.
"""

from types import SimpleNamespace

from misterdev.core.planning.verifier_decomposition import (
    StagedCheck,
    render_stages,
    synthesize_stages,
)


def _sym(name, kind=""):
    return SimpleNamespace(name=name, kind=kind)


def test_bowling_api_orders_constructor_mutator_query():
    # Deliberately out of order to prove ordering is imposed, not passthrough.
    contract = [
        _sym("score", kind="method"),
        _sym("roll", kind="method"),
        _sym("new", kind="constructor"),
    ]

    stages = synthesize_stages(contract)

    assert [c.symbol for c in stages] == ["new", "roll", "score"]
    assert [c.stage for c in stages] == [1, 2, 3]


def test_stages_group_by_role():
    contract = [
        _sym("Game", kind="class"),
        _sym("init"),
        _sym("add_frame"),
        _sym("set_pins"),
        _sym("total_score"),
        _sym("is_complete"),
    ]

    stages = synthesize_stages(contract)
    by_symbol = {c.symbol: c.stage for c in stages}

    assert by_symbol["Game"] == 1
    assert by_symbol["init"] == 1
    assert by_symbol["add_frame"] == 2
    assert by_symbol["set_pins"] == 2
    assert by_symbol["total_score"] == 3
    assert by_symbol["is_complete"] == 3


def test_is_prefix_is_a_query_not_a_mutator():
    # "insert" is a mutator prefix; "is_ready" must not be captured by it.
    stages = synthesize_stages([_sym("is_ready")])
    assert stages[0].stage == 1  # renumbered: only bucket present
    assert stages[0].symbol == "is_ready"


def test_single_function_yields_one_stage():
    stages = synthesize_stages([_sym("solve")])

    assert len(stages) == 1
    assert stages[0].stage == 1
    assert stages[0].symbol == "solve"


def test_mutator_only_contract_starts_at_stage_one():
    # No constructor/query: the single surviving bucket renumbers to stage 1.
    stages = synthesize_stages([_sym("push"), _sym("pop")])

    assert {c.stage for c in stages} == {1}
    assert [c.symbol for c in stages] == ["push", "pop"]


def test_dict_symbols_are_supported():
    contract = [
        {"name": "new", "kind": "constructor"},
        {"name": "roll", "kind": "method"},
        {"name": "score", "kind": "method"},
    ]

    stages = synthesize_stages(contract)

    assert [c.symbol for c in stages] == ["new", "roll", "score"]
    assert [c.stage for c in stages] == [1, 2, 3]


def test_duplicate_symbols_are_deduped():
    stages = synthesize_stages([_sym("roll"), _sym("roll")])

    assert len(stages) == 1
    assert stages[0].symbol == "roll"


def test_blank_names_skipped():
    stages = synthesize_stages([_sym(""), _sym("  "), _sym("new")])

    assert [c.symbol for c in stages] == ["new"]


def test_empty_input_yields_empty_list():
    assert synthesize_stages([]) == []
    assert synthesize_stages(None) == []


def test_description_and_rationale_populated():
    stages = synthesize_stages([_sym("new"), _sym("roll"), _sym("score")])

    for c in stages:
        assert isinstance(c, StagedCheck)
        assert c.description == f"implement + smoke-check {c.symbol}"
        assert c.rationale  # non-empty


def test_render_includes_stage_numbers_and_symbols():
    stages = synthesize_stages([_sym("new"), _sym("roll"), _sym("score")])

    text = render_stages(stages)

    assert "Stage 1:" in text
    assert "Stage 2:" in text
    assert "Stage 3:" in text
    assert "new" in text
    assert "roll" in text
    assert "score" in text


def test_render_groups_same_stage_symbols_on_one_line():
    stages = synthesize_stages([_sym("add"), _sym("set")])

    text = render_stages(stages)
    lines = [ln for ln in text.splitlines() if ln.strip()]

    assert len(lines) == 1
    assert "add, set" in lines[0]


def test_render_empty_plan_is_empty_string():
    assert render_stages([]) == ""


def test_instructions_arg_does_not_change_ordering():
    contract = [_sym("new"), _sym("roll"), _sym("score")]

    without = synthesize_stages(contract)
    with_instr = synthesize_stages(contract, instructions="focus on scoring")

    assert [(c.stage, c.symbol) for c in without] == [
        (c.stage, c.symbol) for c in with_instr
    ]


def _edge_sym(name, incoming=(), outgoing=(), kind=""):
    return SimpleNamespace(
        name=name,
        kind=kind,
        incoming_calls=set(incoming),
        outgoing_calls=set(outgoing),
    )


def test_call_graph_orders_dependency_before_its_callers():
    # `helper` is called by both `render` and `report`; by name shape it would
    # land late (a query-ish free helper), but the edges make it foundational.
    contract = [
        _edge_sym("render", outgoing=("helper",)),
        _edge_sym("report", outgoing=("helper",)),
        _edge_sym("helper", incoming=("render", "report")),
    ]

    stages = synthesize_stages(contract)
    by_symbol = {c.symbol: c.stage for c in stages}

    assert by_symbol["helper"] < by_symbol["render"]
    assert by_symbol["helper"] < by_symbol["report"]


def test_call_graph_ordering_is_deterministic_and_ties_keep_contract_order():
    contract = [
        _edge_sym("caller_b", outgoing=("base",)),
        _edge_sym("caller_a", outgoing=("base",)),
        _edge_sym("base", incoming=("caller_a", "caller_b")),
    ]

    first = [(c.stage, c.symbol) for c in synthesize_stages(contract)]
    second = [(c.stage, c.symbol) for c in synthesize_stages(contract)]

    assert first == second
    # base is foundational (stage 1); the two callers tie and keep input order.
    assert first[0] == (1, "base")
    assert [name for _, name in first[1:]] == ["caller_b", "caller_a"]


def test_malformed_edges_do_not_raise_and_fall_back_to_name_heuristic():
    # A non-iterable and a bare-string edge payload must be ignored, not crash;
    # with no usable graph the name-verb heuristic orders the plan.
    contract = [
        SimpleNamespace(name="new", kind="constructor", incoming_calls=None),
        SimpleNamespace(name="roll", kind="method", outgoing_calls=42),
        SimpleNamespace(name="score", kind="method", incoming_calls="oops"),
    ]

    stages = synthesize_stages(contract)

    assert [c.symbol for c in stages] == ["new", "roll", "score"]
    assert [c.stage for c in stages] == [1, 2, 3]


def test_no_edges_preserves_name_heuristic_order():
    # Regression: symbols without any edge info keep the exact name-verb staging.
    contract = [
        _sym("score", kind="method"),
        _sym("roll", kind="method"),
        _sym("new", kind="constructor"),
    ]

    stages = synthesize_stages(contract)

    assert [c.symbol for c in stages] == ["new", "roll", "score"]
    assert [c.stage for c in stages] == [1, 2, 3]
