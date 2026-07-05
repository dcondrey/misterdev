import subprocess
import tempfile
from pathlib import Path

from misterdev.core.gitcmd import run_git


def _init_repo():
    d = Path(tempfile.mkdtemp())
    for args in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "t@t.t"],
        ["git", "config", "user.name", "t"],
    ):
        subprocess.run(args, cwd=d, check=True, capture_output=True)
    return d


def test_run_git_returns_completed_process():
    d = _init_repo()
    proc = run_git("git rev-parse --is-inside-work-tree", d)
    assert proc is not None
    assert proc.returncode == 0
    assert proc.stdout.strip() == "true"


def test_run_git_nonzero_returncode_outside_repo():
    d = Path(tempfile.mkdtemp())
    proc = run_git("git rev-parse --is-inside-work-tree", d)
    # Ran, but git reports failure -> non-None process with non-zero return code.
    assert proc is not None
    assert proc.returncode != 0


def test_run_git_missing_binary_returns_none(monkeypatch):
    def _boom(*a, **k):
        raise FileNotFoundError("git not found")

    monkeypatch.setattr(subprocess, "run", _boom)
    assert run_git("git status", Path(".")) is None


def test_run_git_timeout_returns_none(monkeypatch):
    def _timeout(*a, **k):
        raise subprocess.TimeoutExpired(cmd="git", timeout=1)

    monkeypatch.setattr(subprocess, "run", _timeout)
    assert run_git("git log", Path("."), timeout=1) is None
