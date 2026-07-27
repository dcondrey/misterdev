"""M3 — a worktree is torn down if a prep step raises (no leak).

`_prepare_task_worktree` creates the worktree, then primes/healthchecks it. If any
prep step raises, the wave cleanup (which only iterates fully-prepared tasks) would
never remove that worktree. The helper must tear it down and surface the error.
"""

from types import SimpleNamespace

from misterdev.core.execution.parallel import ParallelExecutionMixin


class _FakeGit:
    def __init__(self):
        self.removed = []
        self.deleted = []

    def worktree_add(self, project, path, branch, new_branch=True):
        return True, ""

    def worktree_remove(self, project, path):
        self.removed.append(path)

    def branch_delete(self, project, branch):
        self.deleted.append(branch)


def _task():
    return SimpleNamespace(id="T1")


def test_prep_raise_tears_down_worktree(tmp_path):
    class _Ex(ParallelExecutionMixin):
        def _prime_worktree_by_clone(self, *a, **k):
            raise RuntimeError("clone blew up")

    git = _FakeGit()
    prep, err = _Ex()._prepare_task_worktree(
        None, git, _task(), tmp_path, True, None, None, 1, None
    )
    assert prep is None
    assert isinstance(err, RuntimeError)
    assert git.removed and git.deleted  # torn down, not leaked


def test_successful_prep_returns_tuple_no_teardown(tmp_path):
    class _Ex(ParallelExecutionMixin):
        def _prime_worktree_by_clone(self, *a, **k):
            return True  # primed; skips healthcheck

    git = _FakeGit()
    prep, err = _Ex()._prepare_task_worktree(
        None, git, _task(), tmp_path, True, None, None, 1, None
    )
    assert err is None
    task, wt_path, branch = prep
    assert task.id == "T1" and branch.startswith("task/T1-")
    assert not git.removed and not git.deleted


def test_worktree_add_failure_returns_error_no_teardown(tmp_path):
    class _FailGit(_FakeGit):
        def worktree_add(self, project, path, branch, new_branch=True):
            return False, "add failed"

    git = _FailGit()
    prep, err = ParallelExecutionMixin()._prepare_task_worktree(
        None, git, _task(), tmp_path, True, None, None, 1, None
    )
    assert prep is None and isinstance(err, RuntimeError)
    assert not git.removed  # nothing to tear down; it was never created
