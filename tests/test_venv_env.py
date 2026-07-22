"""M2 — venv setup subprocess is timeout-bounded and captures output.

A hung `pip install` / `python -m venv` with no timeout blocks the whole run
indefinitely. setup() must bound each command with a timeout and fail cleanly
(returning False) on a timeout instead of hanging.
"""

import subprocess

import misterdev.environments.venv_env as ve
from misterdev.environments.venv_env import VenvEnvironmentManager


def _mgr(tmp_path):
    return VenvEnvironmentManager({"root_dir": "vv"}, tmp_path)


def test_setup_passes_a_timeout(tmp_path, monkeypatch):
    seen = {}

    def fake_run(cmd, **kw):
        seen.update(kw)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(ve.subprocess, "run", fake_run)
    assert _mgr(tmp_path).setup() is True
    assert "timeout" in seen and seen["timeout"] > 0


def test_setup_returns_false_on_timeout(tmp_path, monkeypatch):
    def fake_run(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, kw.get("timeout", 1))

    monkeypatch.setattr(ve.subprocess, "run", fake_run)
    assert _mgr(tmp_path).setup() is False


def test_setup_returns_false_on_command_error(tmp_path, monkeypatch):
    def fake_run(cmd, **kw):
        raise subprocess.CalledProcessError(1, cmd, output="", stderr="boom")

    monkeypatch.setattr(ve.subprocess, "run", fake_run)
    assert _mgr(tmp_path).setup() is False
