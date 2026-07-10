"""ToolRunner: sandboxed execution of untrusted, model-authored tools.

The security invariant is the point: untrusted code never runs on the host, is
bounded, and degrades OFF when no sandbox exists. The container call is injected,
so these run without Docker.
"""

from pathlib import Path

from misterdev.core.evolution.tool_runner import (
    ToolRunner,
    ToolRunResult,
    _default_engine_factory,
)


class _FakeEngine:
    """Records the command it was asked to run and returns a canned result."""

    def __init__(self, result):
        self.result = result
        self.commands = []

    def run(self, command, timeout):
        self.commands.append((command, timeout))
        return self.result


def _runner_returning(result):
    engine = _FakeEngine(result)
    return ToolRunner(engine_factory=lambda work_dir: engine), engine


def test_no_sandbox_skips_and_never_runs_on_host():
    # The load-bearing safety property: no engine -> the code is NOT executed.
    runner = ToolRunner(engine_factory=lambda work_dir: None)
    r = runner.run("import os; os.system('rm -rf /')")
    assert r.status == "skip"
    assert not r.ran
    assert "no container sandbox" in r.stderr


def test_empty_source_is_rejected():
    runner, engine = _runner_returning((True, "x"))
    r = runner.run("   \n  ")
    assert r.status == "rejected"
    assert engine.commands == []  # never handed to the sandbox


def test_oversized_source_is_rejected():
    runner, engine = _runner_returning((True, "x"))
    r = runner.run("#" * (64 * 1024 + 1))
    assert r.status == "rejected"
    assert engine.commands == []


def test_successful_run_is_ok():
    runner, engine = _runner_returning((True, "42\n"))
    r = runner.run("print(40 + 2)")
    assert r.status == "ok" and r.ok and r.ran
    assert r.stdout == "42\n"
    assert r.exit_code == 0
    # stdin is piped from the mounted file, cwd is the isolated mount.
    assert engine.commands[0][0] == "python /work/tool.py < /work/stdin"


def test_nonzero_exit_is_error():
    runner, _ = _runner_returning((False, "Traceback: boom"))
    r = runner.run("raise SystemExit(1)")
    assert r.status == "error" and r.ran and not r.ok
    assert "boom" in r.stdout


def test_timeout_is_classified():
    runner, _ = _runner_returning((False, "Container command timed out after 20s: ..."))
    r = runner.run("while True: pass")
    assert r.status == "timeout" and r.ran
    assert r.exit_code is None


def test_output_is_truncated():
    runner, _ = _runner_returning((True, "A" * (16 * 1024 + 500)))
    r = runner.run("print('A'*100000)")
    assert r.status == "ok"
    assert "chars elided" in r.stdout
    assert len(r.stdout) < 16 * 1024 + 200


def test_stdin_is_passed_through(tmp_path):
    # The model supplies stdin; assert it reaches the run (via the redirect).
    runner, engine = _runner_returning((True, "echoed"))
    runner.run("import sys; print(sys.stdin.read())", stdin="hello-input")
    assert engine.commands[0][0].endswith("< /work/stdin")


def test_default_factory_applies_full_hardening(tmp_path, monkeypatch):
    # The security posture must actually be set on the real ContainerEngine:
    # no network, all caps dropped, no-new-privileges, resource caps.
    import misterdev.core.execution.container as container

    monkeypatch.setattr(container, "detect_engine", lambda preferred=None: "docker")
    factory = _default_engine_factory(
        image="python:3.12-slim", memory="256m", cpus="1", pids_limit=128
    )
    engine = factory(Path(tmp_path))
    assert engine is not None
    assert engine.network == "none"
    assert engine.cap_drop == ["ALL"]
    assert engine.security_opt == ["no-new-privileges"]
    assert engine.memory == "256m" and engine.cpus == "1" and engine.pids_limit == 128
    # And the argv it would run carries every hardening flag.
    argv = engine.wrap_command("python /work/tool.py", timeout=20)
    joined = " ".join(argv)
    assert "--network none" in joined
    assert "--cap-drop ALL" in joined
    assert "--security-opt no-new-privileges" in joined
    assert "--rm" in joined


def test_default_factory_returns_none_without_engine(tmp_path, monkeypatch):
    import misterdev.core.execution.container as container

    monkeypatch.setattr(container, "detect_engine", lambda preferred=None: None)
    factory = _default_engine_factory("img", "256m", "1", 128)
    assert factory(Path(tmp_path)) is None


def test_result_dataclass_semantics():
    assert ToolRunResult("ok", "", "").ran
    assert ToolRunResult("error", "", "").ran
    assert ToolRunResult("timeout", "", "").ran
    assert not ToolRunResult("skip", "", "").ran
    assert not ToolRunResult("rejected", "", "").ran
