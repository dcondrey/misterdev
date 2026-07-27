"""Copy-on-write cloning of a base checkout's dependency dirs into a worktree."""

import os
import tempfile
from pathlib import Path

import pytest

from misterdev.core.execution.dep_clone import (
    clone_dependencies,
    clone_supported,
    find_node_modules_dirs,
)


def _make_base(root: Path):
    """A minimal monorepo: a root node_modules, a workspace package, and a server
    whose node_modules links that package with a RELATIVE symlink (as pnpm does)."""
    (root / "packages" / "shared").mkdir(parents=True)
    (root / "packages" / "shared" / "index.js").write_text("export const x = 1\n")
    (root / "node_modules").mkdir()
    (root / "node_modules" / ".marker").write_text("root\n")
    scope = root / "apps" / "server" / "node_modules" / "@scope"
    scope.mkdir(parents=True)
    (root / "apps" / "server" / "node_modules" / "realdep").mkdir()
    (root / "apps" / "server" / "node_modules" / "realdep" / "index.js").write_text("1")
    # Relative workspace symlink, exactly like pnpm's.
    os.symlink("../../../../packages/shared", scope / "shared")
    # Nested node_modules inside a package's node_modules must NOT be found
    # separately (it rides along with the parent clone).
    (root / "node_modules" / "dep" / "node_modules").mkdir(parents=True)


def test_find_node_modules_dirs_roots_and_workspaces(tmp_path):
    _make_base(tmp_path)
    found = {str(p) for p in find_node_modules_dirs(tmp_path)}
    assert found == {"node_modules", os.path.join("apps", "server", "node_modules")}
    # The nested node_modules inside node_modules/dep is pruned.
    assert os.path.join("node_modules", "dep", "node_modules") not in found


def test_find_skips_git_and_orchestrator(tmp_path):
    (tmp_path / ".git" / "node_modules").mkdir(parents=True)
    (tmp_path / ".orchestrator" / "node_modules").mkdir(parents=True)
    (tmp_path / "node_modules").mkdir()
    assert [str(p) for p in find_node_modules_dirs(tmp_path)] == ["node_modules"]


def test_clone_supported_returns_bool(tmp_path):
    assert isinstance(clone_supported(tmp_path), bool)


def test_clone_dependencies_no_node_modules_returns_false(tmp_path):
    (tmp_path / "src").mkdir()
    assert clone_dependencies(tmp_path, tmp_path / "wt") == (False, [])


@pytest.mark.skipif(
    not clone_supported(tempfile.gettempdir()),
    reason="filesystem does not support CoW/hardlink cloning",
)
def test_clone_preserves_relative_workspace_symlink_into_worktree(tmp_path):
    """The cloned node_modules' relative workspace symlink resolves to the
    WORKTREE's own package, not the base — the property that makes a cloned
    worktree usable with no reinstall."""
    base = tmp_path / "base"
    base.mkdir()
    _make_base(base)
    # A worktree has the SOURCE checked out (packages/shared) but no node_modules.
    wt = tmp_path / "wt"
    (wt / "packages" / "shared").mkdir(parents=True)
    (wt / "packages" / "shared" / "index.js").write_text("export const x = 1\n")
    (wt / "apps" / "server").mkdir(parents=True)

    ok, dirs = clone_dependencies(base, wt)
    assert ok
    assert set(dirs) == {"node_modules", os.path.join("apps", "server", "node_modules")}

    # Root clone carried its regular files.
    assert (wt / "node_modules" / ".marker").read_text() == "root\n"
    assert (wt / "apps" / "server" / "node_modules" / "realdep" / "index.js").exists()

    # The relative workspace symlink was preserved AND resolves inside the worktree.
    link = wt / "apps" / "server" / "node_modules" / "@scope" / "shared"
    assert link.is_symlink()
    resolved = os.path.realpath(link)
    assert resolved.startswith(os.path.realpath(wt))
    assert not resolved.startswith(os.path.realpath(base))
    assert resolved == os.path.realpath(wt / "packages" / "shared")


@pytest.mark.skipif(
    not clone_supported(tempfile.gettempdir()),
    reason="filesystem does not support CoW/hardlink cloning",
)
def test_prime_worktree_by_clone_gates_on_sanity_probe(tmp_path):
    """_prime_worktree_by_clone returns True only when the clone passes the probe,
    and False (→ install fallback) when the probe fails or there is no probe."""
    import misterdev.agent as agent_mod
    from unittest.mock import MagicMock

    base = tmp_path / "base"
    base.mkdir()
    _make_base(base)
    wt = tmp_path / "wt"
    wt.mkdir()

    project = MagicMock()
    project.path = base
    orch = agent_mod.ProjectOrchestrator()

    class _Tool:
        def __init__(self, ok):
            self.ok = ok

        def execute(self, project, command, cwd=None, timeout=None):
            return self.ok, "" if self.ok else "error TS2307: cannot find module"

    # Probe passes -> primed by clone.
    assert orch._prime_worktree_by_clone(
        project, MagicMock(id="T"), wt, "tsc --version", 60, _Tool(True)
    )
    # Probe fails -> decline, fall back to install.
    assert not orch._prime_worktree_by_clone(
        project, MagicMock(id="T"), tmp_path / "wt2", "tsc --version", 60, _Tool(False)
    )
    # No probe command -> cannot verify, decline.
    assert not orch._prime_worktree_by_clone(
        project, MagicMock(id="T"), tmp_path / "wt3", None, 60, _Tool(True)
    )
