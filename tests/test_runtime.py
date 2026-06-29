import os
import signal
import sys
import time

from my_project_orchestrator.core.execution.runtime import (
    GREEN,
    RED,
    SKIP,
    SmokeResult,
    run_smoke_gate,
)


def _py(code: str) -> str:
    # Run an inline python snippet via the test interpreter so the smoke gate
    # exercises a real process without depending on a container or shell tools.
    return f"{sys.executable} -c {code!r}"


# --- SKIP semantics ---------------------------------------------------------


def test_skip_when_no_config(tmp_path):
    assert run_smoke_gate(tmp_path, None).status == SKIP
    assert run_smoke_gate(tmp_path, {}).status == SKIP


def test_skip_when_no_launch(tmp_path):
    res = run_smoke_gate(tmp_path, {"expect": "hi"})
    assert res.status == SKIP
    assert res.skipped and not res.passed


# --- GREEN ------------------------------------------------------------------


def test_green_when_ready_and_expect_present(tmp_path):
    # Process prints READY, then echoes a probe line, then exits.
    code = (
        "import sys;"
        "print('READY', flush=True);"
        "line=sys.stdin.readline();"
        "print('PONG:'+line.strip(), flush=True)"
    )
    res = run_smoke_gate(
        tmp_path,
        {
            "launch": _py(code),
            "ready": "READY",
            "probe": "ping",
            "expect": "PONG:ping",
            "timeout": 10,
        },
    )
    assert res.status == GREEN
    assert res.passed
    assert "PONG:ping" in res.evidence


def test_green_on_stdout_substring_without_ready(tmp_path):
    code = "print('SERVICE-UP', flush=True)"
    res = run_smoke_gate(
        tmp_path,
        {"launch": _py(code), "expect": "SERVICE-UP", "timeout": 10},
    )
    assert res.status == GREEN


# --- RED --------------------------------------------------------------------


def test_red_when_expect_absent(tmp_path):
    code = "print('something else', flush=True)"
    res = run_smoke_gate(
        tmp_path,
        {"launch": _py(code), "expect": "NEVER_PRINTED", "timeout": 10},
    )
    assert res.status == RED
    assert not res.passed
    assert "NEVER_PRINTED" in res.reason


def test_red_when_process_exits_nonzero_no_expect(tmp_path):
    code = "import sys; sys.exit(3)"
    res = run_smoke_gate(tmp_path, {"launch": _py(code), "timeout": 10})
    assert res.status == RED
    assert res.exit_code == 3


def test_red_when_ready_never_appears(tmp_path):
    # Process exits cleanly but never prints the readiness signal.
    code = "print('nope', flush=True)"
    res = run_smoke_gate(
        tmp_path,
        {"launch": _py(code), "ready": "READY", "expect": "x", "timeout": 5},
    )
    assert res.status == RED


# --- never blocks (hard timeout) -------------------------------------------


def test_hanging_launch_returns_within_timeout(tmp_path):
    # A process that blocks forever waiting on stdin and never prints the
    # expectation must be abandoned by the hard timeout, not block the caller.
    code = "import time; time.sleep(3600)"
    start = time.monotonic()
    res = run_smoke_gate(
        tmp_path,
        {"launch": _py(code), "expect": "WILL_NEVER_APPEAR", "timeout": 1},
    )
    elapsed = time.monotonic() - start
    # join margin is timeout + 5; well under the 3600s the process would block.
    assert elapsed < 20
    # Either RED (inner timeout reached, expect absent) or SKIP (outer abandon);
    # both are non-blocking and non-passing, which is what matters.
    assert res.status in (RED, SKIP)


def test_hanging_launch_inner_deadline_fires_and_kills_process(tmp_path):
    # A launch that records its PID then sleeps far past the timeout. The gate
    # must (a) return via the INNER deadline (a real RED, not the outer
    # abandon) and (b) terminate the process — the old blocking readline left it
    # orphaned because the inner deadline never fired.
    code = (
        "import os, time;"
        "open('pid', 'w').write(str(os.getpid()));"
        "print('STARTED', flush=True);"
        "time.sleep(120)"
    )
    res = run_smoke_gate(
        tmp_path, {"launch": _py(code), "expect": "NEVER", "timeout": 2}
    )
    assert res.status == RED  # inner deadline fired before the outer abandon

    pid = int((tmp_path / "pid").read_text())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)  # still alive
            time.sleep(0.1)
        except OSError:
            return  # process gone -> no leak, as required
    os.kill(pid, signal.SIGKILL)  # cleanup so a failing run doesn't leak
    raise AssertionError("launched process was leaked (not terminated by the gate)")
    assert not res.passed


def test_outer_join_skips_when_worker_wedged(monkeypatch, tmp_path):
    # Force the inner work to outlast the outer join to prove the daemon-thread
    # abandonment path returns SKIP without blocking.
    import my_project_orchestrator.core.execution.runtime as runtime

    def _wedge(*a, **k):
        time.sleep(30)

    monkeypatch.setattr(runtime, "_smoke", _wedge)
    start = time.monotonic()
    res = run_smoke_gate(tmp_path, {"launch": "whatever", "timeout": 0.1})
    assert time.monotonic() - start < 10
    assert res.status == SKIP
    assert "timed out" in res.reason


# --- error handling ---------------------------------------------------------


def test_launch_failure_is_skip_not_crash(monkeypatch, tmp_path):
    import my_project_orchestrator.core.execution.runtime as runtime

    def _boom(*a, **k):
        raise OSError("cannot spawn")

    monkeypatch.setattr(runtime, "_smoke", _boom)
    res = run_smoke_gate(tmp_path, {"launch": "x", "expect": "y", "timeout": 5})
    assert res.status == SKIP
    assert "error" in res.reason


def test_smoke_result_repr_and_flags():
    r = SmokeResult(GREEN, evidence="e", exit_code=0)
    assert r.passed and not r.skipped
    assert "green" in repr(r)


# --- gatekeeper integration -------------------------------------------------


def test_gatekeeper_skips_smoke_when_off(tmp_path):
    from my_project_orchestrator.core.verification.gatekeeper import GateKeeper

    (tmp_path / "a.py").write_text("x = 1\n")
    # runtime_smoke off -> gate not run even with a (would-fail) spec present.
    keeper = GateKeeper(
        tmp_path,
        runtime_smoke=False,
        runtime_config={"smoke": {"launch": "false", "expect": "z"}},
    )
    success, issues, _ = keeper.run_gates({})
    assert not any("G4.6" in i for i in issues)


def test_gatekeeper_red_smoke_blocks_build(tmp_path):
    from my_project_orchestrator.core.verification.gatekeeper import GateKeeper

    (tmp_path / "a.py").write_text("x = 1\n")
    keeper = GateKeeper(
        tmp_path,
        runtime_smoke=True,
        runtime_config={
            "smoke": {
                "launch": _py("print('x', flush=True)"),
                "expect": "ABSENT_TOKEN",
                "timeout": 5,
            }
        },
    )
    success, issues, _ = keeper.run_gates({})
    assert not success
    assert any("G4.6" in i for i in issues)


def test_gatekeeper_green_smoke_passes(tmp_path):
    from my_project_orchestrator.core.verification.gatekeeper import GateKeeper

    (tmp_path / "a.py").write_text("x = 1\n")
    keeper = GateKeeper(
        tmp_path,
        runtime_smoke=True,
        runtime_config={
            "smoke": {
                "launch": _py("print('UP', flush=True)"),
                "expect": "UP",
                "timeout": 5,
            }
        },
    )
    success, issues, _ = keeper.run_gates({})
    assert not any("G4.6" in i for i in issues)
