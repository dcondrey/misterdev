"""T3.2 — on a detected stall, reset to a fresh candidate (clean task base).

`_reset_to_task_base` returns the working tree to the task's clean base so the next
attempt re-derives from scratch, but — unlike `_abort_task` — stays ON the task branch
and in the retry loop (it must NOT check out the base or delete the branch). This drives
the helper directly with a duck fixture.
"""

from types import SimpleNamespace

from misterdev.task_executors.markdown_plan_executor.git_mixin import GitMixin


class _Ex(GitMixin):
    def __init__(self):
        self.git_cmds = []
        self.reverted = None
        self.orphans_cleaned = False

    def _git(self, project, cmd):
        self.git_cmds.append(cmd)
        return (True, "")

    def _revert_files(self, project, snapshot):
        self.reverted = snapshot

    def _clean_task_orphans(self, project, untracked_before):
        self.orphans_cleaned = True


def _project():
    return SimpleNamespace(topography=None)


def test_git_mode_resets_hard_without_leaving_branch():
    ex = _Ex()
    ex._reset_to_task_base(_project(), "task-branch", "main", None, set())
    assert any("reset --hard" in c for c in ex.git_cmds)
    # Must NOT end the task: no checkout of base, no branch delete.
    assert not any("checkout" in c for c in ex.git_cmds)
    assert not any("branch -D" in c for c in ex.git_cmds)
    assert ex.orphans_cleaned


def test_snapshot_mode_reverts_files():
    ex = _Ex()
    snap = {"a.py": "original"}
    ex._reset_to_task_base(_project(), None, None, snap, set())
    assert ex.reverted == snap
    assert ex.orphans_cleaned
