import os
import subprocess
from pathlib import Path

import pytest

import my_project_orchestrator.core.container as container
from my_project_orchestrator.core.container import (
    ContainerEngine,
    detect_engine,
    image_for_language,
)
from my_project_orchestrator.core.gatekeeper import GateKeeper
from my_project_orchestrator.core.validator import _run_cmd
from my_project_orchestrator.environments.container_env import (
    ContainerEnvironmentManager,
)


# --- engine detection -------------------------------------------------------


def test_detect_engine_prefers_rootless_first(monkeypatch):
    seen = []

    def fake_probe(engine):
        seen.append(engine)
        return engine == "podman"

    monkeypatch.setattr(container, "_probe_engine", fake_probe)
    assert detect_engine() == "podman"
    assert seen[0] == "podman"  # rootless preferred over docker


def test_detect_engine_honors_preferred(monkeypatch):
    monkeypatch.setattr(container, "_probe_engine", lambda e: e == "docker")
    assert detect_engine(preferred="docker") == "docker"


def test_detect_engine_colima_maps_to_docker(monkeypatch):
    monkeypatch.setattr(container, "_probe_engine", lambda e: e == "colima")
    assert detect_engine() == "docker"


def test_detect_engine_none_when_no_engine(monkeypatch):
    monkeypatch.setattr(container, "_probe_engine", lambda e: False)
    assert detect_engine() is None


def test_probe_engine_never_raises_on_missing_binary(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("no such binary")

    monkeypatch.setattr(subprocess, "run", boom)
    assert container._probe_engine("podman") is False


def test_probe_engine_unavailable_on_timeout(monkeypatch):
    def slow(*a, **k):
        raise subprocess.TimeoutExpired(cmd="docker info", timeout=8)

    monkeypatch.setattr(subprocess, "run", slow)
    assert container._probe_engine("docker") is False


def test_image_for_language_maps_known_and_falls_back():
    assert "rust" in image_for_language("rust")
    assert "python" in image_for_language("python")
    assert "node" in image_for_language("typescript")
    # unknown language still yields a runnable generic image
    assert image_for_language("brainfuck") == container._DEFAULT_IMAGE
    assert image_for_language("") == container._DEFAULT_IMAGE
    assert image_for_language(None) == container._DEFAULT_IMAGE


# --- ContainerEngine command wrapping ---------------------------------------


def test_wrap_command_mounts_repo_and_sets_user(monkeypatch):
    monkeypatch.setattr(container.os, "getuid", lambda: 501, raising=False)
    monkeypatch.setattr(container.os, "getgid", lambda: 20, raising=False)
    eng = ContainerEngine("docker", "python:3.12-slim", Path("/repo"))
    argv = eng.wrap_command("pytest -q", timeout=60)
    assert argv[:3] == ["docker", "run", "--rm"]
    assert "-v" in argv and any(":/workspace" in a for a in argv)
    assert "--user" in argv and "501:20" in argv
    # user command passed verbatim to sh -c so shell features survive
    assert argv[-4:-1] == ["python:3.12-slim", "sh", "-c"]
    assert argv[-1] == "pytest -q"


def test_wrap_command_no_network_when_egress_none(monkeypatch):
    monkeypatch.setattr(container.os, "getuid", lambda: 501, raising=False)
    monkeypatch.setattr(container.os, "getgid", lambda: 20, raising=False)
    eng = ContainerEngine("docker", "img", Path("/repo"), network="none")
    argv = eng.wrap_command("pytest -q", timeout=60)
    assert "--network" in argv
    assert argv[argv.index("--network") + 1] == "none"


def test_wrap_command_omits_network_flag_by_default(monkeypatch):
    monkeypatch.setattr(container.os, "getuid", lambda: 501, raising=False)
    monkeypatch.setattr(container.os, "getgid", lambda: 20, raising=False)
    # default (network=None) and explicit "default" both leave the engine default
    for net in (None, "default"):
        eng = ContainerEngine("docker", "img", Path("/repo"), network=net)
        assert "--network" not in eng.wrap_command("pytest -q", timeout=60)


def test_wrap_command_omits_resource_flags_by_default(monkeypatch):
    monkeypatch.setattr(container.os, "getuid", lambda: 501, raising=False)
    monkeypatch.setattr(container.os, "getgid", lambda: 20, raising=False)
    # Unconfigured: argv must be byte-identical to the pre-hardening output.
    eng = ContainerEngine("docker", "img", Path("/repo"))
    argv = eng.wrap_command("pytest -q", timeout=60)
    assert "--memory" not in argv
    assert "--cpus" not in argv
    assert "--pids-limit" not in argv


def test_wrap_command_emits_resource_limits_when_set(monkeypatch):
    monkeypatch.setattr(container.os, "getuid", lambda: 501, raising=False)
    monkeypatch.setattr(container.os, "getgid", lambda: 20, raising=False)
    eng = ContainerEngine(
        "docker", "img", Path("/repo"), memory="512m", cpus="1.5", pids_limit=2048
    )
    argv = eng.wrap_command("pytest -q", timeout=60)
    assert argv[argv.index("--memory") + 1] == "512m"
    assert argv[argv.index("--cpus") + 1] == "1.5"
    assert argv[argv.index("--pids-limit") + 1] == "2048"
    # Limits precede the image/command, not after.
    assert argv.index("--memory") < argv.index("img")


def test_env_manager_passes_resource_limits_to_engine(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "my_project_orchestrator.environments.container_env.detect_engine",
        lambda preferred=None: "docker",
    )
    mgr = ContainerEnvironmentManager(
        {"type": "docker", "memory": "256m", "cpus": "2", "pids_limit": 512},
        tmp_path,
        language="python",
    )
    mgr.setup()
    eng = mgr.engine()
    assert eng.memory == "256m"
    assert eng.cpus == "2"
    assert eng.pids_limit == 512


def test_env_manager_passes_network_to_engine(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "my_project_orchestrator.environments.container_env.detect_engine",
        lambda preferred=None: "docker",
    )
    mgr = ContainerEnvironmentManager(
        {"type": "docker"}, tmp_path, language="python", network="none"
    )
    mgr.setup()
    assert mgr.engine().network == "none"


def test_project_wires_governance_network(tmp_path, monkeypatch):
    from my_project_orchestrator.core import project as project_mod

    monkeypatch.setattr(
        "my_project_orchestrator.environments.container_env.detect_engine",
        lambda preferred=None: "docker",
    )
    monkeypatch.setattr(project_mod.Project, "_init_llm_client", lambda self: None)

    on = project_mod.Project(
        tmp_path,
        {"environment": {"type": "docker"}, "governance": {"network": "none"}},
    )
    assert on.env_manager.network == "none"

    off = project_mod.Project(tmp_path, {"environment": {"type": "docker"}})
    assert off.env_manager.network is None


def test_engine_run_returns_ok_and_output(monkeypatch):
    class P:
        returncode = 0
        stdout = "all good"
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: P())
    eng = ContainerEngine("docker", "img", Path("/repo"))
    ok, out = eng.run("true", timeout=10)
    assert ok and "all good" in out


def test_engine_run_timeout_is_failure_not_raise(monkeypatch):
    def slow(*a, **k):
        raise subprocess.TimeoutExpired(cmd="docker run", timeout=10)

    monkeypatch.setattr(subprocess, "run", slow)
    eng = ContainerEngine("docker", "img", Path("/repo"))
    ok, out = eng.run("sleep 99", timeout=10)
    assert not ok and "timed out" in out


# --- _run_cmd runner seam ---------------------------------------------------


def test_run_cmd_uses_runner_when_supplied():
    captured = {}

    def runner(cmd, timeout):
        captured["cmd"] = cmd
        captured["timeout"] = timeout
        return True, "from-container"

    ok, out = _run_cmd("pytest -q", Path("."), "source .venv/bin/activate", 99, runner)
    assert ok and out == "from-container"
    # the host activation prefix is NOT applied when running via the runner
    assert captured["cmd"] == "pytest -q"
    assert captured["timeout"] == 99


def test_run_cmd_local_fallback_when_no_runner(tmp_path):
    ok, out = _run_cmd("echo hi", tmp_path, None, 30, None)
    assert ok and "hi" in out


# --- gatekeeper routing through container -----------------------------------


def _make_dir(tmp_path):
    (tmp_path / "x.py").write_text("x = 1\n")
    return tmp_path


def test_gatekeeper_routes_gates_through_container(tmp_path):
    calls = []

    class FakeEngine:
        def is_available(self):
            return True

        def run(self, cmd, timeout):
            calls.append(cmd)
            return True, "ok"

    keeper = GateKeeper(_make_dir(tmp_path), container=FakeEngine())
    success, issues, _health = keeper.run_gates(
        {
            "build_command": "cargo build",
            "test_command": "cargo test",
            "lint_command": "cargo clippy",
        }
    )
    assert success and not issues
    # all three gate commands executed in the container, none locally
    assert "cargo build" in calls
    assert "cargo test" in calls
    assert "cargo clippy" in calls


def test_gatekeeper_falls_back_local_when_engine_unavailable(tmp_path, monkeypatch):
    import my_project_orchestrator.core.gatekeeper as gk

    local_calls = []

    def fake_run_cmd(cmd, cwd, env_activate=None, timeout=180, runner=None):
        local_calls.append((cmd, runner))
        return True, "ok"

    monkeypatch.setattr(gk, "_run_cmd", fake_run_cmd)

    class DownEngine:
        def is_available(self):
            return False

        def run(self, cmd, timeout):  # pragma: no cover - must never be called
            raise AssertionError("container used despite being unavailable")

    keeper = gk.GateKeeper(_make_dir(tmp_path), container=DownEngine())
    keeper.run_gates({"build_command": "make", "test_command": "make test"})
    # engine unavailable -> container dropped, runner is None (local execution)
    assert all(runner is None for _cmd, runner in local_calls)


def test_gatekeeper_identical_result_local_vs_container(tmp_path):
    # The same gate set yields the same pass/fail whether routed through the
    # container or run locally, proving the substrate is transparent.
    class FailEngine:
        def is_available(self):
            return True

        def run(self, cmd, timeout):
            return (cmd != "false"), "out"

    cmds = {"build_command": "true", "test_command": "false"}
    via_container = GateKeeper(_make_dir(tmp_path), container=FailEngine()).run_gates(
        cmds
    )
    via_local = GateKeeper(_make_dir(tmp_path)).run_gates(cmds)
    assert via_container[0] == via_local[0] is False  # tests fail either way


# --- ContainerEnvironmentManager --------------------------------------------


def test_env_manager_setup_unavailable_returns_false(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "my_project_orchestrator.environments.container_env.detect_engine",
        lambda preferred=None: None,
    )
    mgr = ContainerEnvironmentManager({"type": "docker"}, tmp_path, language="python")
    assert mgr.setup() is False
    assert mgr.engine() is None
    assert mgr.activate_command() == ""  # empty prefix -> local fallback unmodified


def test_env_manager_setup_available_builds_engine(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "my_project_orchestrator.environments.container_env.detect_engine",
        lambda preferred=None: "podman",
    )
    mgr = ContainerEnvironmentManager(
        {"type": "container", "image": "rust:slim"}, tmp_path, language="rust"
    )
    assert mgr.setup() is True
    eng = mgr.engine()
    assert eng is not None and eng.is_available()
    assert eng.engine == "podman" and eng.image == "rust:slim"


def test_env_manager_auto_image_from_language(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "my_project_orchestrator.environments.container_env.detect_engine",
        lambda preferred=None: "docker",
    )
    mgr = ContainerEnvironmentManager({"type": "docker"}, tmp_path, language="rust")
    mgr.setup()
    assert "rust" in mgr.engine().image


# --- live integration (opt-in) ----------------------------------------------


def test_container_integration_runs_or_skips(tmp_path):
    # Opportunistic live integration: run real gate commands inside a throwaway
    # alpine container via a detected OCI engine. Gated behind
    # RUN_CONTAINER_INTEGRATION so the normal suite never needs a daemon; it also
    # skips cleanly when no engine is present.
    if not os.environ.get("RUN_CONTAINER_INTEGRATION"):
        pytest.skip("set RUN_CONTAINER_INTEGRATION=1 to exercise a real container")
    engine = detect_engine()
    if engine is None:
        pytest.skip("no OCI engine available")

    (tmp_path / "x.py").write_text("x = 1\n")
    eng = ContainerEngine(engine, "alpine", tmp_path)
    keeper = GateKeeper(tmp_path, container=eng)
    # `uname -s` reports the CONTAINER kernel (Linux), not the macOS host, which
    # proves the gate command actually executed inside the container.
    success, issues, _health = keeper.run_gates(
        {"build_command": "true", "test_command": "uname -s"}
    )
    assert success, f"containerized gates failed: {issues}"
    ok, out = eng.run("uname -s", timeout=120)
    assert ok and "Linux" in out, f"expected Linux kernel from container, got: {out!r}"

    # Egress is blocked when network is "none": an outbound fetch must fail.
    no_net = ContainerEngine(engine, "alpine", tmp_path, network="none")
    blocked_ok, _blocked_out = no_net.run(
        "wget -q -T3 -O- http://example.com", timeout=120
    )
    assert blocked_ok is False, "expected egress to be blocked with network=none"
