"""A1 — async jobs persist across an MCP server restart.

A JobRegistry with a store_path writes job state to disk, so a client can still poll
a run_id after the server restarts. A job persisted as "running" (its process died
mid-run) loads back as "interrupted", not as if it were still progressing.
"""

import json

from misterdev.core.execution.jobs import JobRegistry


def _wait_finished(reg, rid, timeout=3.0):
    job = reg.get(rid)
    if job is not None and job._thread is not None:
        job._thread.join(timeout)
    return reg.status(rid)


def test_finished_job_survives_restart(tmp_path):
    store = tmp_path / "jobs.json"
    r1 = JobRegistry(store_path=str(store))
    rid = r1.start("build", str(tmp_path), lambda: "the report")
    assert _wait_finished(r1, rid)["status"] == "succeeded"

    r2 = JobRegistry(store_path=str(store))  # simulates a server restart
    st = r2.status(rid)
    assert st is not None
    assert st["status"] == "succeeded"
    assert st["result"] == "the report"


def test_running_job_loads_as_interrupted(tmp_path):
    store = tmp_path / "jobs.json"
    store.write_text(
        json.dumps(
            [
                {
                    "run_id": "abc",
                    "kind": "build",
                    "project_path": str(tmp_path),
                    "status": "running",
                    "result": None,
                    "error": None,
                    "started_at": "t0",
                    "ended_at": None,
                    "stop_requested": False,
                }
            ]
        )
    )
    r = JobRegistry(store_path=str(store))
    assert r.status("abc")["status"] == "interrupted"
    # An interrupted (not running) job no longer blocks a new job for that path.
    rid = r.start("build", str(tmp_path), lambda: "ok")
    assert rid


def test_no_store_path_is_ephemeral(tmp_path):
    r = JobRegistry()  # backward compatible: no persistence
    rid = r.start("build", str(tmp_path), lambda: "x")
    assert _wait_finished(r, rid)["status"] == "succeeded"


def test_corrupt_store_starts_empty(tmp_path):
    store = tmp_path / "jobs.json"
    store.write_text("{not valid json")
    r = JobRegistry(store_path=str(store))
    assert r.list_jobs() == []
