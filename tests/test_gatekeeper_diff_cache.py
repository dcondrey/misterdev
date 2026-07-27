"""H3 — the Gatekeeper git-diff is computed once per run_gates, not once per gate.

G5/G6/G9 each called `_iter_diff_added_lines` (3+ git subprocesses + untracked-file
reads), but the tree is identical across gates within one run_gates. Memoize it and
invalidate at the start of each run_gates so a later run (a changed tree) recomputes.
"""

import misterdev.core.verification.gatekeeper as gk_mod
from misterdev.core.verification.gatekeeper import GateKeeper


class _R:
    def __init__(self, stdout, returncode=0):
        self.stdout = stdout
        self.returncode = returncode


def _gk(tmp_path):
    gk = GateKeeper.__new__(GateKeeper)
    gk.project_path = tmp_path
    return gk


def test_diff_is_memoized_within_a_run(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_run_git(cmd, path):
        calls["n"] += 1
        return _R("true" if "rev-parse" in cmd else "")

    monkeypatch.setattr(gk_mod, "run_git", fake_run_git)
    gk = _gk(tmp_path)

    gk._iter_diff_added_lines()
    after_first = calls["n"]
    assert after_first > 0
    gk._iter_diff_added_lines()
    assert calls["n"] == after_first  # 2nd call served from cache; no new subprocesses


def test_invalidation_forces_recompute(tmp_path, monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(
        gk_mod,
        "run_git",
        lambda cmd, path: (
            calls.__setitem__("n", calls["n"] + 1),
            _R("true" if "rev-parse" in cmd else ""),
        )[1],
    )
    gk = _gk(tmp_path)
    gk._iter_diff_added_lines()
    after_first = calls["n"]
    gk._diff_cache_valid = False  # what run_gates does at its start
    gk._iter_diff_added_lines()
    assert calls["n"] > after_first  # a fresh run recomputes
