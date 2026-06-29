from my_project_orchestrator.core.execution.outcomes import SKIP, GREEN, RED, GateOutcome


def test_green_is_passed_not_skipped():
    o = GateOutcome(GREEN)
    assert o.passed and not o.skipped


def test_skip_is_skipped_not_passed():
    o = GateOutcome(SKIP, reason="no config")
    assert o.skipped and not o.passed
    assert o.reason == "no config"


def test_red_is_neither():
    o = GateOutcome(RED, reason="failed")
    assert not o.passed and not o.skipped


def test_constants_are_distinct_strings():
    assert {SKIP, GREEN, RED} == {"skip", "green", "red"}


def test_gate_results_share_the_base_and_constants():
    # Each gate's result re-exports the same constants and subclasses the base,
    # so passed/skipped behave identically across them.
    from my_project_orchestrator.core.verification.vision_verify import VisionResult
    from my_project_orchestrator.core.execution.runtime import SmokeResult
    from my_project_orchestrator.core.verification.mutation_gate import MutationResult
    from my_project_orchestrator.core.verification.web_verify import WebResult

    for cls in (VisionResult, SmokeResult, MutationResult, WebResult):
        assert issubclass(cls, GateOutcome)
        assert cls(GREEN).passed
        assert cls(SKIP).skipped
        assert not cls(RED).passed
