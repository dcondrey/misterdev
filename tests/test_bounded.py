import time

from my_project_orchestrator.core.execution.bounded import run_bounded


def test_returns_work_result_when_fast():
    assert run_bounded(lambda: "ok", 1.0, "DEFAULT", "x") == "ok"


def test_returns_default_on_timeout():
    def slow():
        time.sleep(5)
        return "late"

    assert run_bounded(slow, 0.2, "DEFAULT", "x") == "DEFAULT"


def test_returns_default_when_work_raises():
    def boom():
        raise RuntimeError("nope")

    assert run_bounded(boom, 1.0, "DEFAULT", "x") == "DEFAULT"


def test_returns_control_within_timeout():
    def slow():
        time.sleep(10)

    start = time.monotonic()
    run_bounded(slow, 0.2, None, "x")
    assert time.monotonic() - start < 3.0


def test_default_can_be_any_object():
    sentinel = object()
    assert run_bounded(lambda: (_ for _ in ()).throw(ValueError()), 1.0, sentinel, "x") is sentinel
