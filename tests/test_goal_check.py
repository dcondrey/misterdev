import time

from my_project_orchestrator.core.goal_check import (
    GAP,
    SATISFIED,
    SKIP,
    GoalVerdict,
    build_evidence,
    run_goal_check,
    _extract_json_object,
    _parse_verdict,
)


# --- SKIP semantics ---------------------------------------------------------


def test_skip_when_no_goal_or_criteria():
    res = run_goal_check(
        "", "", "some diff", judge_call=lambda p: '{"satisfied": true}'
    )
    assert res.status == SKIP
    assert "no goal" in res.reason


def test_skip_when_no_judge_or_client():
    res = run_goal_check("ship feature X", "criteria", "diff", llm_client=None)
    assert res.status == SKIP
    assert "no LLM judge" in res.reason


def test_skip_on_unparseable_verdict():
    res = run_goal_check(
        "goal",
        "crit",
        "diff",
        judge_call=lambda p: "I think it is mostly fine but not sure.",
    )
    assert res.status == SKIP
    assert "unparseable" in res.reason


def test_skip_on_empty_verdict():
    res = run_goal_check("goal", "crit", "diff", judge_call=lambda p: "")
    assert res.status == SKIP


def test_skip_on_non_boolean_satisfied():
    res = run_goal_check(
        "goal", "crit", "diff", judge_call=lambda p: '{"satisfied": "yes"}'
    )
    assert res.status == SKIP
    assert "non-boolean" in res.reason


# --- SATISFIED / GAP --------------------------------------------------------


def test_satisfied_records_no_gaps():
    res = run_goal_check(
        "build a CLI",
        "accepts --help",
        "added argparse with --help",
        judge_call=lambda p: '{"satisfied": true, "gaps": []}',
    )
    assert res.status == SATISFIED
    assert res.satisfied
    assert res.gaps == []


def test_gap_records_gaps():
    res = run_goal_check(
        "build a CLI",
        "accepts --help and --version",
        "added --help only",
        judge_call=lambda p: '{"satisfied": false, "gaps": ["--version is missing"]}',
    )
    assert res.status == GAP
    assert res.has_gap
    assert res.gaps == ["--version is missing"]


def test_unsatisfied_with_no_gaps_gets_generic_gap():
    res = run_goal_check(
        "goal", "crit", "diff", judge_call=lambda p: '{"satisfied": false, "gaps": []}'
    )
    assert res.status == GAP
    assert len(res.gaps) == 1
    assert "no specific gap" in res.gaps[0]


def test_gaps_as_string_is_normalized():
    res = run_goal_check(
        "goal",
        "crit",
        "diff",
        judge_call=lambda p: '{"satisfied": false, "gaps": "the API is missing"}',
    )
    assert res.status == GAP
    assert res.gaps == ["the API is missing"]


def test_goal_or_criteria_alone_is_enough():
    # Goal only (no criteria) still runs the judge.
    res = run_goal_check(
        "implement login", "", "diff", judge_call=lambda p: '{"satisfied": true}'
    )
    assert res.status == SATISFIED
    # Criteria only (no goal) also runs.
    res2 = run_goal_check(
        "", "must hash passwords", "diff", judge_call=lambda p: '{"satisfied": true}'
    )
    assert res2.status == SATISFIED


# --- evidence is fed to the judge -------------------------------------------


def test_prompt_contains_goal_criteria_and_evidence():
    seen = {}

    def _call(prompt):
        seen["prompt"] = prompt
        return '{"satisfied": true}'

    run_goal_check("GOALTEXT", "CRITTEXT", "DIFFTEXT", judge_call=_call)
    assert "GOALTEXT" in seen["prompt"]
    assert "CRITTEXT" in seen["prompt"]
    assert "DIFFTEXT" in seen["prompt"]


def test_evidence_is_truncated():
    seen = {}

    def _call(prompt):
        seen["prompt"] = prompt
        return '{"satisfied": true}'

    huge = "x" * 50000
    run_goal_check("g", "c", huge, judge_call=_call)
    assert "x" * 16000 in seen["prompt"]
    assert "x" * 20000 not in seen["prompt"]


# --- never blocks / errors --------------------------------------------------


def test_hanging_judge_returns_within_timeout():
    def _slow(prompt):
        time.sleep(3600)
        return '{"satisfied": true}'

    start = time.monotonic()
    res = run_goal_check("g", "c", "d", judge_call=_slow, timeout=0.3)
    assert time.monotonic() - start < 10
    assert res.status == SKIP
    assert "timed out" in res.reason


def test_judge_error_is_skip_not_crash():
    def _boom(prompt):
        raise RuntimeError("model unreachable")

    res = run_goal_check("g", "c", "d", judge_call=_boom)
    assert res.status == SKIP
    assert "error" in res.reason


def test_no_client_default_call_is_none():
    res = run_goal_check("g", "c", "d", llm_client=None)
    assert res.status == SKIP


def test_client_without_generate_code_is_skip():
    class Bare:
        pass

    res = run_goal_check("g", "c", "d", llm_client=Bare())
    assert res.status == SKIP


def test_default_call_uses_client_generate_code():
    class FakeClient:
        def __init__(self):
            self.seen = {}

        def generate_code(self, prompt, system=""):
            self.seen["prompt"] = prompt
            self.seen["system"] = system
            return '{"satisfied": false, "gaps": ["x missing"]}'

    client = FakeClient()
    res = run_goal_check("g", "c", "d", llm_client=client)
    assert res.status == GAP
    assert res.gaps == ["x missing"]
    assert "g" in client.seen["prompt"]


# --- JSON extraction / parsing ----------------------------------------------


def test_extract_json_object_tolerates_prose_and_fences():
    text = 'Here is my verdict:\n```json\n{"satisfied": true, "gaps": []}\n```\ndone'
    obj = _extract_json_object(text)
    assert obj == {"satisfied": True, "gaps": []}


def test_extract_json_object_handles_braces_in_strings():
    text = '{"satisfied": false, "gaps": ["the }{ token broke parsing"]}'
    obj = _extract_json_object(text)
    assert obj["satisfied"] is False
    assert obj["gaps"] == ["the }{ token broke parsing"]


def test_extract_json_object_none_when_absent():
    assert _extract_json_object("no json here") is None
    assert _extract_json_object("") is None


def test_parse_verdict_satisfied():
    v = _parse_verdict('{"satisfied": true}')
    assert v.status == SATISFIED


def test_parse_verdict_gap():
    v = _parse_verdict('{"satisfied": false, "gaps": ["a", "b"]}')
    assert v.status == GAP
    assert v.gaps == ["a", "b"]


# --- evidence builder / result object ---------------------------------------


def test_build_evidence_combines_summary_and_diff():
    ev = build_evidence(diff="diff body", summary="summary body")
    assert "summary body" in ev
    assert "diff body" in ev
    assert ev.index("Summary") < ev.index("Diff")


def test_build_evidence_empty_when_both_empty():
    assert build_evidence("", "") == ""


def test_goal_verdict_repr_and_flags():
    v = GoalVerdict(SATISFIED)
    assert v.satisfied and not v.skipped and not v.has_gap
    assert "satisfied" in repr(v)
