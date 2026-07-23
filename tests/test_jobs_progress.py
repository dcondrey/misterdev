"""A2 — a job exposes task-level progress the running build reports.

A target may accept a ``report(**fields)`` callback and push done/total/phase/message;
``status`` surfaces them. Legacy zero-arg targets keep working unchanged.
"""

from misterdev.core.execution.jobs import JobRegistry


def _wait(reg, rid, timeout=3.0):
    job = reg.get(rid)
    if job is not None and job._thread is not None:
        job._thread.join(timeout)
    return reg.status(rid)


def test_status_reports_progress_from_target(tmp_path):
    r = JobRegistry()

    def target(report):
        report(done=1, total=3, phase="wave-1", message="task A")
        return "done"

    rid = r.start("build", str(tmp_path), target)
    st = _wait(r, rid)
    assert st["status"] == "succeeded"
    assert st["tasks_done"] == 1 and st["tasks_total"] == 3
    assert st["phase"] == "wave-1" and st["message"] == "task A"


def test_partial_progress_updates_only_given_fields(tmp_path):
    r = JobRegistry()

    def target(report):
        report(total=5)
        report(done=2)  # total must stay 5
        return "x"

    st = _wait(r, r.start("build", str(tmp_path), target))
    assert st["tasks_total"] == 5 and st["tasks_done"] == 2


def test_update_progress_unknown_id_returns_false():
    assert JobRegistry().update_progress("nope", done=1) is False


def test_zero_arg_target_backward_compatible(tmp_path):
    r = JobRegistry()
    rid = r.start("build", str(tmp_path), lambda: "ok")
    assert _wait(r, rid)["status"] == "succeeded"
