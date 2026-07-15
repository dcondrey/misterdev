"""Regression tests for the devplan correctness fixes.

Covers: atomic writes, LLM-edit path validation, secret-scan false-positive
reduction, new error-classifier categories, validation SKIP status, formatter
path handling, per-task change attribution, opt-in file-overlap dependencies,
ephemeral cleanup, and report persistence.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from misterdev.utils.file_utils import atomic_write
from misterdev.task_executors.markdown_plan_executor import (
    MarkdownPlanExecutor,
    _detect_language,
    _LANG_MAP,
)
from misterdev.core.verification.gatekeeper import GateKeeper
from misterdev.core.execution.error_classifier import classify_error, ErrorCategory
from misterdev.core.verification.validator import ValidationResult, CodeValidator
from misterdev.core.context.change_tracker import ChangeTracker
from misterdev.core.planning.sovereign import EphemeralCodeManager


class _FakeProject:
    def __init__(self, path, config=None):
        self.path = Path(path)
        self.config = config or {}


class _FakeTask:
    def __init__(self, modify=None, create=None):
        self.files_to_modify = modify or []
        self.files_to_create = create or []


# --- atomic_write -----------------------------------------------------------


def test_atomic_write_creates_and_overwrites():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "sub" / "state.json"
        atomic_write(p, "first")
        assert p.read_text() == "first"
        atomic_write(p, "second")
        assert p.read_text() == "second"
        # No stray temp files left behind.
        assert list(p.parent.glob("*.tmp")) == []


# --- _detect_language / _LANG_MAP -------------------------------------------


def test_detect_language_falls_back_to_text():
    assert _detect_language("notes.md") == "text"
    assert _detect_language("data.csv") == "text"
    assert _detect_language("script.sh") == "shell"
    assert _LANG_MAP[".py"] == "python"


# --- LLM edit path validation -----------------------------------------------


def test_apply_edits_is_atomic_restores_on_partial_failure():
    # A write that fails mid-batch must leave NO partial changes: pre-existing
    # files are restored to their prior content and newly-created files removed.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "existing.py").write_text("original\n")
        # "blocker" is a FILE, so writing to "blocker/x.py" fails when write_file
        # tries to mkdir the parent — a real mid-batch write failure.
        (root / "blocker").write_text("i am a file\n")
        proj = _FakeProject(td)
        ex = MarkdownPlanExecutor()
        edits = {
            "existing.py": "clobbered\n",  # written first, must be rolled back
            "new.py": "created\n",  # written second, must be removed
            "blocker/x.py": "boom\n",  # third write raises
        }
        with pytest.raises(OSError):
            ex._apply_edits(proj, edits)
        assert (root / "existing.py").read_text() == "original\n"
        assert not (root / "new.py").exists()


def test_apply_edits_writes_all_on_success():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        proj = _FakeProject(td)
        MarkdownPlanExecutor()._apply_edits(proj, {"a.py": "aa\n", "sub/b.py": "bb\n"})
        assert (root / "a.py").read_text() == "aa\n"
        assert (root / "sub" / "b.py").read_text() == "bb\n"


def test_validate_edit_paths_rejects_escape_and_empty():
    with tempfile.TemporaryDirectory() as td:
        proj = _FakeProject(td)
        task = _FakeTask(modify=["src/in_scope.py"])
        ex = MarkdownPlanExecutor()
        edits = {
            "src/in_scope.py": "x = 1\n",
            "../escape.py": "evil\n",
            "/etc/passwd": "evil\n",
            "src/empty.py": "   \n",
        }
        valid = ex._validate_edit_paths(proj, task, edits)
        assert "src/in_scope.py" in valid
        assert "../escape.py" not in valid
        assert "/etc/passwd" not in valid
        assert "src/empty.py" not in valid


# --- secret scan false-positive reduction -----------------------------------


def test_secret_assignment_heuristic():
    # Real assigned literal -> flagged.
    assert GateKeeper._is_secret_assignment('api_key = "abcdef123456"')
    # Ordinary source constructs -> not flagged.
    assert not GateKeeper._is_secret_assignment("token: String,")
    assert not GateKeeper._is_secret_assignment("let token = get_token()")
    assert not GateKeeper._is_secret_assignment('api_key = os.environ["API_KEY"]')


def test_secret_scan_ignores_code_keeps_high_signal():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "engine.rs").write_text("pub struct S { token: String, secret: u64 }\n")
        (root / "leak.py").write_text('KEY = "sk-abc123def"\n')
        gk = GateKeeper(root)
        found = gk._scan_secrets()
        assert any("leak.py" in f for f in found)
        assert not any("engine.rs" in f for f in found)


# --- error classifier new categories ----------------------------------------


def test_classify_manifest_error():
    assert (
        classify_error("error: failed to parse manifest at Cargo.toml")
        == ErrorCategory.MANIFEST
    )


def test_classify_file_not_found():
    assert (
        classify_error("OSError: No such file or directory: 'x'")
        == ErrorCategory.FILE_NOT_FOUND
    )


# --- validation SKIP status -------------------------------------------------


def test_validation_summary_skip_for_unrun_gates():
    v = ValidationResult()
    v.build_ran, v.build_ok = True, True
    v.tests_ran = False  # no test command configured
    v.lint_ran, v.lint_ok = True, True
    s = v.summary()
    assert "tests=SKIP" in s
    assert "build=OK" in s


def test_shell_skips_delimiter_check():
    ok, err = CodeValidator.validate_code('x=$(echo "hi")\n', language="shell")
    assert ok and err is None


# --- formatter path handling ------------------------------------------------


def test_formatter_runs_project_wide_without_placeholder():
    from unittest.mock import patch
    from misterdev.tools.formatter import FormatterTool

    tool = FormatterTool.__new__(FormatterTool)
    tool.config = {"command": "ruff format ."}
    with patch(
        "misterdev.tools.command.CommandTool.execute",
        autospec=True,
        return_value=(True, ""),
    ) as m:
        tool.execute(_FakeProject("."), file_path="ignored.py")
    # No {path} placeholder -> command runs as-is, not per file.
    assert m.call_args.kwargs.get("command") == "ruff format ."


def test_formatter_substitutes_path_when_placeholder_present():
    from unittest.mock import patch
    from misterdev.tools.formatter import FormatterTool

    tool = FormatterTool.__new__(FormatterTool)
    tool.config = {"command": "rustfmt {path}"}
    with patch(
        "misterdev.tools.command.CommandTool.execute",
        autospec=True,
        return_value=(True, ""),
    ) as m:
        tool.execute(_FakeProject("."), file_path="src/lib.rs")
    assert m.call_args.kwargs.get("command") == "rustfmt src/lib.rs"


# --- per-task change attribution --------------------------------------------


def _git(path, *args):
    subprocess.run(["git", *args], cwd=path, capture_output=True, text=True)


def test_change_tracker_attributes_per_task_commit():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        _git(td, "init")
        _git(td, "config", "user.email", "t@t.t")
        _git(td, "config", "user.name", "t")
        (td / "f.txt").write_text("a\n")
        _git(td, "add", "-A")
        _git(td, "commit", "-m", "init")
        # Task A commit
        (td / "f.txt").write_text("a\nb\n")
        _git(td, "add", "-A")
        _git(td, "commit", "-m", "task(T-A): add b")
        # A later unrelated commit so HEAD is no longer T-A.
        (td / "g.txt").write_text("z\n")
        _git(td, "add", "-A")
        _git(td, "commit", "-m", "task(T-B): add g")

        ct = ChangeTracker(td)
        change = ct.record_task_changes("T-A", ["f.txt"])
        # T-A's own commit is found despite T-B being HEAD.
        assert change.additions >= 1


# --- opt-in file-overlap dependency detection -------------------------------


def _write_task(devplan: Path, tid: str, modify):
    devplan.mkdir(parents=True, exist_ok=True)
    body = (
        "---\n"
        "status: pending\n"
        f"title: {tid}\n"
        "files_to_modify:\n" + "".join(f"- {m}\n" for m in modify) + "---\nbody\n"
    )
    (devplan / f"{tid}.md").write_text(body)


def _make_tm(root: Path, auto: bool):
    from misterdev.core.task import TaskManager

    class _P:
        path = root
        config = {
            "devplan_dir": "devplan",
            "orchestrator": {"auto_detect_dependencies": auto},
        }

    return TaskManager(_P())


def test_file_overlap_dependencies_opt_in():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        dp = root / "devplan"
        _write_task(dp, "001-a", ["src/mod.rs"])
        _write_task(dp, "002-b", ["src/mod.rs"])

        tm_off = _make_tm(root, auto=False)
        tm_off.discover_tasks()
        assert tm_off.tasks["002-b"].dependencies == []

        tm_on = _make_tm(root, auto=True)
        tm_on.discover_tasks()
        assert "001-a" in tm_on.tasks["002-b"].dependencies


# --- ephemeral cleanup ------------------------------------------------------


def test_ephemeral_context_manager_cleans_up():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        with EphemeralCodeManager(root) as em:
            ephemeral_dir = em.ephemeral_dir
            assert ephemeral_dir.exists()
        assert not ephemeral_dir.exists()


# --- commit only the task's files, never `git add -A` (data safety) ----------


def test_commit_task_does_not_sweep_unrelated_untracked():
    from misterdev.task_executors.markdown_plan_executor import (
        MarkdownPlanExecutor,
    )

    class _P:
        env_manager = None

    class _T:
        id = "T-1"
        title = "do thing"
        description = "d"

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        _git(td, "init")
        _git(td, "config", "user.email", "t@t.t")
        _git(td, "config", "user.name", "t")
        (td / "seed.txt").write_text("x")
        _git(td, "add", "-A")
        _git(td, "commit", "-m", "init")
        (td / "task_file.txt").write_text("the task's edit")
        (td / "user_unrelated.txt").write_text(
            "uncommitted user work"
        )  # untracked, NOT the task's

        proj = _P()
        proj.path = td
        MarkdownPlanExecutor()._commit_task(proj, None, None, _T(), ["task_file.txt"])

        committed = subprocess.run(
            ["git", "show", "--name-only", "--format=", "HEAD"],
            cwd=td,
            capture_output=True,
            text=True,
        ).stdout
        assert "task_file.txt" in committed
        assert "user_unrelated.txt" not in committed
        # The unrelated file is untouched, still untracked.
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=td, capture_output=True, text=True
        ).stdout
        assert "?? user_unrelated.txt" in status


def test_commit_task_empty_file_list_does_not_sweep():
    """With no files, commit must be empty -- never fall back to git add -A,
    which would sweep unrelated untracked work (the rideshare data-loss path)."""
    from misterdev.task_executors.markdown_plan_executor import (
        MarkdownPlanExecutor,
    )

    class _P:
        env_manager = None

    class _T:
        id = "T-2"
        title = "no edits"
        description = "d"

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        _git(td, "init")
        _git(td, "config", "user.email", "t@t.t")
        _git(td, "config", "user.name", "t")
        (td / "seed.txt").write_text("x")
        _git(td, "add", "-A")
        _git(td, "commit", "-m", "init")
        (td / "user_untracked.txt").write_text("uncommitted user work")

        proj = _P()
        proj.path = td
        MarkdownPlanExecutor()._commit_task(proj, None, None, _T(), [])

        committed = subprocess.run(
            ["git", "show", "--name-only", "--format=", "HEAD"],
            cwd=td,
            capture_output=True,
            text=True,
        ).stdout
        assert "user_untracked.txt" not in committed  # not swept
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=td, capture_output=True, text=True
        ).stdout
        assert "?? user_untracked.txt" in status  # still safely untracked


# --- probe generation f-string bug (found via live build) --------------------


def test_generate_probes_does_not_crash_on_json_braces():
    from misterdev.core.planning.sovereign import ProbeGenerator

    class _LLM:
        def generate_code(self, prompt, system_prompt=""):
            # The literal JSON example must survive the f-string as real braces.
            assert '{"name"' in prompt
            return '[{"name": "p", "purpose": "x", "script": "print(1)"}]'

    probes = ProbeGenerator(_LLM()).generate_probes("a spec", "a summary")
    assert probes and probes[0]["name"] == "p"


# --- git worktree parallel execution (019) ----------------------------------


def test_execute_parallel_worktrees_merges_back():
    from unittest.mock import MagicMock
    import misterdev.agent as agent_mod

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        _git(td, "init")
        _git(td, "config", "user.email", "t@t.t")
        _git(td, "config", "user.name", "t")
        (td / "base.txt").write_text("base\n")
        _git(td, "add", "-A")
        _git(td, "commit", "-m", "init")

        project = MagicMock()
        project.path = td
        project.config = {
            "orchestrator": {"parallel_mode": "worktree", "max_workers": 2}
        }

        class FakeExec:
            def execute(self, task, proj, use_git_branch=True):
                # Runs inside the task's worktree; write + commit there.
                f = Path(proj.path) / f"{task.id}.txt"
                f.write_text(task.id)
                _git(proj.path, "add", "-A")
                _git(proj.path, "commit", "-m", f"task({task.id}): work")
                r = MagicMock()
                r.status = "completed"
                return r

        tasks = [_mock_task("T-A"), _mock_task("T-B")]
        orch = agent_mod.ProjectOrchestrator()
        results = orch._execute_parallel_worktrees(tasks, FakeExec(), project)

        assert {t.id for t, r, e in results} == {"T-A", "T-B"}
        assert all(r is not None and r.status == "completed" for _, r, _ in results)
        # Both worktrees' files merged into the base tree.
        assert (td / "T-A.txt").exists() and (td / "T-B.txt").exists()
        # Worktrees cleaned up.
        wt_root = td / ".orchestrator" / "worktrees"
        assert not any(wt_root.iterdir()) if wt_root.exists() else True


def test_execute_parallel_worktrees_tolerates_stale_task_branch():
    """Regression: a leftover ``task/<id>`` branch from a prior failed run must not
    collide with this run's worktree creation, and no throwaway branch is left
    behind (whether the task succeeds or fails)."""
    from unittest.mock import MagicMock
    import misterdev.agent as agent_mod

    def _branches(path):
        out = subprocess.run(
            ["git", "branch", "--format=%(refname:short)"],
            cwd=path,
            capture_output=True,
            text=True,
        ).stdout
        return [b.strip() for b in out.splitlines() if b.strip()]

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        _git(td, "init")
        _git(td, "config", "user.email", "t@t.t")
        _git(td, "config", "user.name", "t")
        (td / "base.txt").write_text("base\n")
        _git(td, "add", "-A")
        _git(td, "commit", "-m", "init")
        # Two stale leftover branches from an imagined prior run — the exact names
        # the old fixed-name scheme would try to re-create and fail on.
        _git(td, "branch", "task/T-A")
        _git(td, "branch", "task/T-B")

        project = MagicMock()
        project.path = td
        project.config = {
            "orchestrator": {"parallel_mode": "worktree", "max_workers": 2}
        }

        class FakeExec:
            def execute(self, task, proj, use_git_branch=True):
                if task.id == "T-B":  # this one fails: it must still clean up
                    raise RuntimeError("boom")
                f = Path(proj.path) / f"{task.id}.txt"
                f.write_text(task.id)
                _git(proj.path, "add", "-A")
                _git(proj.path, "commit", "-m", f"task({task.id}): work")
                r = MagicMock()
                r.status = "completed"
                return r

        orch = agent_mod.ProjectOrchestrator()
        results = orch._execute_parallel_worktrees(
            [_mock_task("T-A"), _mock_task("T-B")], FakeExec(), project
        )

        by_id = {t.id: (r, e) for t, r, e in results}
        # The stale branch did not block T-A; its work merged despite the collision.
        assert by_id["T-A"][0] is not None and by_id["T-A"][0].status == "completed"
        assert (td / "T-A.txt").exists()
        assert by_id["T-B"][1] is not None  # T-B surfaced its error, didn't wedge
        # No run-created throwaway branches remain (only the two inert stale names).
        assert sorted(_branches(td)) == ["main", "task/T-A", "task/T-B"] or sorted(
            _branches(td)
        ) == ["master", "task/T-A", "task/T-B"]


def test_worktree_setup_command_detection(tmp_path):
    """Priming command: explicit config wins ('' disables); else auto-detect from
    the lockfile so a gate never pays a full install inside its own timeout."""
    from unittest.mock import MagicMock
    import misterdev.agent as agent_mod

    orch = agent_mod.ProjectOrchestrator()

    def proj(sub, files=(), cfg=None):
        d = tmp_path / sub
        d.mkdir()
        for f in files:
            (d / f).write_text("")
        p = MagicMock()
        p.path = d
        p.config = cfg or {}
        return p

    explicit = proj("a", cfg={"orchestrator": {"worktree_setup_command": "make deps"}})
    assert orch._worktree_setup_command(explicit) == "make deps"
    disabled = proj("b", cfg={"orchestrator": {"worktree_setup_command": ""}})
    assert orch._worktree_setup_command(disabled) is None
    assert "pnpm install" in orch._worktree_setup_command(proj("c", ["pnpm-lock.yaml"]))
    assert orch._worktree_setup_command(proj("d", ["package-lock.json"])) == "npm ci"
    assert "yarn install" in orch._worktree_setup_command(proj("e", ["yarn.lock"]))
    assert orch._worktree_setup_command(proj("f")) is None  # nothing to install


def test_execute_parallel_worktrees_primes_deps_before_gate():
    """The setup command runs INSIDE each worktree before the task body, so the
    gate finds dependencies already present instead of installing on the hot path."""
    from unittest.mock import MagicMock
    import misterdev.agent as agent_mod

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        _git(td, "init")
        _git(td, "config", "user.email", "t@t.t")
        _git(td, "config", "user.name", "t")
        (td / "base.txt").write_text("base\n")
        _git(td, "add", "-A")
        _git(td, "commit", "-m", "init")

        project = MagicMock()
        project.path = td
        project.config = {
            "orchestrator": {
                "parallel_mode": "worktree",
                "max_workers": 2,
                # a marker-writing stand-in for `pnpm install`
                "worktree_setup_command": "touch .primed",
            }
        }

        seen = {}

        class FakeExec:
            def execute(self, task, proj, use_git_branch=True):
                # The prime step must have already run in this worktree.
                seen[task.id] = (Path(proj.path) / ".primed").exists()
                r = MagicMock()
                r.status = "completed"
                return r

        orch = agent_mod.ProjectOrchestrator()
        orch._execute_parallel_worktrees(
            [_mock_task("T-A"), _mock_task("T-B")], FakeExec(), project
        )
        assert seen == {"T-A": True, "T-B": True}


def _worktree_repo(td: Path):
    """A minimal committed git repo usable as a parallel-worktree base."""
    _git(td, "init")
    _git(td, "config", "user.email", "t@t.t")
    _git(td, "config", "user.name", "t")
    (td / "base.txt").write_text("base\n")
    _git(td, "add", "-A")
    _git(td, "commit", "-m", "init")


def test_worktree_healthcheck_probes_after_prime():
    """The health probe runs INSIDE each primed worktree; a passing probe lets the
    task proceed and leaves its marker behind."""
    from unittest.mock import MagicMock
    import misterdev.agent as agent_mod

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        _worktree_repo(td)
        project = MagicMock()
        project.path = td
        project.config = {
            "orchestrator": {
                "parallel_mode": "worktree",
                "max_workers": 2,
                "worktree_setup_command": "touch .primed",
                # passes only if priming ran first, and records that it ran
                "worktree_healthcheck_command": "test -f .primed && touch .probed",
            }
        }

        seen = {}

        class FakeExec:
            def execute(self, task, proj, use_git_branch=True):
                seen[task.id] = (Path(proj.path) / ".probed").exists()
                r = MagicMock()
                r.status = "completed"
                return r

        agent_mod.ProjectOrchestrator()._execute_parallel_worktrees(
            [_mock_task("T-A")], FakeExec(), project
        )
        assert seen == {"T-A": True}  # probe ran (and passed) before the task body


def test_worktree_healthcheck_reprimes_once_on_failure():
    """A failing probe re-primes the deps exactly once and re-probes; a probe that
    then passes lets the task run without flagging the environment."""
    from unittest.mock import MagicMock
    import misterdev.agent as agent_mod

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        _worktree_repo(td)
        project = MagicMock()
        project.path = td
        project.config = {
            "orchestrator": {
                "parallel_mode": "worktree",
                "max_workers": 1,
                # each prime appends one line; the probe passes only after TWO primes
                "worktree_setup_command": "echo x >> .primes",
                "worktree_healthcheck_command": "test $(wc -l < .primes) -ge 2",
            }
        }

        primes = {}

        class FakeExec:
            def execute(self, task, proj, use_git_branch=True):
                primes[task.id] = (Path(proj.path) / ".primes").read_text().count("x")
                r = MagicMock()
                r.status = "completed"
                return r

        agent_mod.ProjectOrchestrator()._execute_parallel_worktrees(
            [_mock_task("T-A")], FakeExec(), project
        )
        # Primed exactly twice: the initial prime plus one re-prime after the
        # probe failed. A third prime would mean the "once" bound was violated.
        assert primes == {"T-A": 2}


def test_worktree_healthcheck_flags_unhealthy_environment(caplog):
    """A probe that stays red after the single re-prime logs the worktree as an
    ENVIRONMENT fault (not the code), and does not re-prime more than once."""
    import logging
    from unittest.mock import MagicMock
    import misterdev.agent as agent_mod

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        _worktree_repo(td)
        project = MagicMock()
        project.path = td
        project.config = {
            "orchestrator": {
                "parallel_mode": "worktree",
                "max_workers": 1,
                "worktree_setup_command": "echo x >> .primes",
                "worktree_healthcheck_command": "false",  # never resolves
            }
        }

        class FakeExec:
            def execute(self, task, proj, use_git_branch=True):
                r = MagicMock()
                r.status = "completed"
                return r

        with caplog.at_level(logging.WARNING, logger="project_orchestrator"):
            agent_mod.ProjectOrchestrator()._execute_parallel_worktrees(
                [_mock_task("T-A")], FakeExec(), project
            )
        # Exactly one re-prime attempt (the "once" bound) and one unhealthy report.
        assert caplog.text.count("re-priming deps once") == 1
        assert caplog.text.count("ENVIRONMENT unhealthy for T-A") == 1


def test_post_merge_healthcheck_reverts_broken_base_keeps_clean():
    """A merged change that fails the base-branch gate is rolled back (base tree
    restored, task returned unfinished); a clean merge is kept."""
    import subprocess
    from unittest.mock import MagicMock
    import misterdev.agent as agent_mod

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        _worktree_repo(td)
        project = MagicMock()
        project.path = td
        project.config = {
            "orchestrator": {"parallel_mode": "worktree", "max_workers": 1},
            # top-level gate (no targets): the base is broken iff `.broken` exists
            "typecheck_command": "test ! -f .broken",
        }

        class FakeExec:
            def execute(self, task, proj, use_git_branch=True):
                wt = Path(proj.path)
                if task.id == "T-break":
                    (wt / ".broken").write_text("boom\n")
                else:
                    (wt / f"ok-{task.id}.txt").write_text("ok\n")
                subprocess.run(["git", "add", "-A"], cwd=wt, check=True)
                subprocess.run(
                    ["git", "commit", "-m", f"task {task.id}"],
                    cwd=wt,
                    check=True,
                    capture_output=True,
                )
                r = MagicMock()
                r.status = "completed"
                return r

            def _run_command(self, project, command, timeout=120, cwd=None):
                p = subprocess.run(
                    command,
                    shell=True,
                    cwd=cwd or project.path,
                    capture_output=True,
                    text=True,
                )
                return p.returncode == 0, (p.stdout or "") + (p.stderr or "")

        results = agent_mod.ProjectOrchestrator()._execute_parallel_worktrees(
            [_mock_task("T-break"), _mock_task("T-ok")], FakeExec(), project
        )

        # Base branch is NOT left broken: the breaking merge was rolled back...
        assert not (td / ".broken").exists()
        # ...and the clean merge was kept.
        assert (td / "ok-T-ok.txt").exists()

        by_id = {t.id: (res, err) for t, res, err in results}
        # The broken task is returned unfinished (no completed result, error set).
        assert by_id["T-break"][0] is None and by_id["T-break"][1] is not None
        # The clean task stays completed.
        assert getattr(by_id["T-ok"][0], "status", None) == "completed"


def test_post_merge_healthcheck_disabled_keeps_broken_merge():
    """With orchestrator.post_merge_healthcheck false, a merge is never gated, so a
    base-breaking change is kept (the opt-out path)."""
    import subprocess
    from unittest.mock import MagicMock
    import misterdev.agent as agent_mod

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        _worktree_repo(td)
        project = MagicMock()
        project.path = td
        project.config = {
            "orchestrator": {
                "parallel_mode": "worktree",
                "max_workers": 1,
                "post_merge_healthcheck": False,
            },
            "typecheck_command": "test ! -f .broken",
        }

        class FakeExec:
            def execute(self, task, proj, use_git_branch=True):
                (Path(proj.path) / ".broken").write_text("boom\n")
                subprocess.run(["git", "add", "-A"], cwd=proj.path, check=True)
                subprocess.run(
                    ["git", "commit", "-m", "break"],
                    cwd=proj.path,
                    check=True,
                    capture_output=True,
                )
                r = MagicMock()
                r.status = "completed"
                return r

            def _run_command(self, *a, **k):  # gate must not be consulted
                raise AssertionError("post-merge gate ran while disabled")

        agent_mod.ProjectOrchestrator()._execute_parallel_worktrees(
            [_mock_task("T-break")], FakeExec(), project
        )
        assert (td / ".broken").exists()  # merge kept, no gate, no revert


def test_worktree_healthcheck_command_resolution(tmp_path: Path):
    """Explicit config wins and "" disables; otherwise a node project auto-detects
    a toolchain probe (tsc when TS is used, else a dependency resolve) and a
    non-node project has no probe."""
    from misterdev.agent_helpers import worktree_healthcheck_command

    assert worktree_healthcheck_command({}, tmp_path) is None  # non-node: no probe
    cfg = {"orchestrator": {"worktree_healthcheck_command": "make check"}}
    assert worktree_healthcheck_command(cfg, tmp_path) == "make check"
    off = {"orchestrator": {"worktree_healthcheck_command": ""}}
    (tmp_path / "package.json").write_text('{"dependencies":{"hono":"^4"}}')
    assert worktree_healthcheck_command(off, tmp_path) is None  # "" disables
    # package.json with a dependency but no tsconfig -> resolve the first dep.
    assert (
        worktree_healthcheck_command({}, tmp_path)
        == "node -e \"require.resolve('hono')\""
    )
    # tsconfig.json present -> prefer the TypeScript toolchain probe.
    (tmp_path / "tsconfig.json").write_text("{}")
    assert (
        worktree_healthcheck_command({}, tmp_path) == "npx --no-install tsc --version"
    )


# --- streaming with early abort (028) ---------------------------------------


def test_code_gen_abort_check():
    from misterdev.llm.client import code_gen_abort_check

    assert code_gen_abort_check("I'll help you write this function...")
    assert code_gen_abort_check("x" * 2500)  # long, no code fence
    assert not code_gen_abort_check("```python\ncode\n```")
    assert not code_gen_abort_check("short")


def test_generate_stream_aborts_early():
    from misterdev.llm.client import BaseLLMClient, code_gen_abort_check

    class _Streamer(BaseLLMClient):
        def __init__(self, parts):
            super().__init__({"build": {}})
            self.model = "m"
            self._parts = parts
            self.emitted = 0

        def _call(self, p, s):
            raise NotImplementedError

        def _call_stream(self, p, s):
            for part in self._parts:
                self.emitted += 1
                yield part

    # Conversational opener trips the abort on the first chunk.
    bad = _Streamer(["I'll help you ", "with that. ", "more text"])
    r = bad.generate_stream("p", abort_check=code_gen_abort_check)
    assert r.finish_reason == "aborted"
    assert bad.emitted == 1

    good = _Streamer(["```python\n", "print(1)\n", "```"])
    r2 = good.generate_stream("p", abort_check=code_gen_abort_check)
    assert r2.finish_reason == "stop"
    assert r2.content == "```python\nprint(1)\n```"


# --- regression bisect (029) -------------------------------------------------


def test_bisect_first_failing_pure():
    from misterdev.task_executors.markdown_plan_executor import (
        _bisect_first_failing,
    )

    # passes, passes, FAILS, fails  -> first failing index is 2
    states = [True, True, False, False]
    assert _bisect_first_failing(len(states), lambda i: states[i]) == 2
    allpass = [True, True, True]
    # nothing fails -> returns last index (caller re-checks)
    assert _bisect_first_failing(len(allpass), lambda i: allpass[i]) == 2


def test_bisect_regression_end_to_end_git():
    from misterdev.task_executors.markdown_plan_executor import (
        MarkdownPlanExecutor,
    )

    class _P:
        env_manager = None

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        _git(td, "init")
        _git(td, "config", "user.email", "t@t.t")
        _git(td, "config", "user.name", "t")
        (td / "f.txt").write_text("clean\n")
        _git(td, "add", "-A")
        _git(td, "commit", "-m", "init")
        # T-A: still clean
        (td / "f.txt").write_text("clean\nA\n")
        _git(td, "add", "-A")
        _git(td, "commit", "-m", "task(T-A): a")
        # T-B: introduces BUG (the regression)
        (td / "f.txt").write_text("clean\nA\nBUG\n")
        _git(td, "add", "-A")
        _git(td, "commit", "-m", "task(T-B): b")
        # T-C: more clean work
        (td / "f.txt").write_text("clean\nA\nBUG\nC\n")
        _git(td, "add", "-A")
        _git(td, "commit", "-m", "task(T-C): c")

        proj = _P()
        proj.path = td
        ex = MarkdownPlanExecutor()
        commits = [
            (tid, ex.find_task_commit(proj, tid)) for tid in ("T-A", "T-B", "T-C")
        ]
        assert all(sha for _, sha in commits)
        # "test" fails when BUG is present.
        culprit = ex.bisect_regression(proj, commits, "! grep -q BUG f.txt", timeout=30)
        assert culprit == "T-B"


# --- prompt caching cost accounting (026) -----------------------------------


def test_cache_read_is_cheaper_than_fresh_input():
    from misterdev.llm.client import AnthropicLLMClient

    c = AnthropicLLMClient.__new__(AnthropicLLMClient)
    c.model = "claude-opus-4-8"  # 15/75 per 1M
    fresh = c._estimate_cost(1_000_000, 0, cache_creation=0, cache_read=0)
    cached = c._estimate_cost(0, 0, cache_creation=0, cache_read=1_000_000)
    assert fresh == 15.0
    assert abs(cached - 1.5) < 1e-9  # cache read = 10% of input
    creation = c._estimate_cost(0, 0, cache_creation=1_000_000, cache_read=0)
    assert abs(creation - 18.75) < 1e-9  # cache write = 125% of input


# --- per-task cost attribution (020) ----------------------------------------


def test_cost_attributed_per_task():
    from misterdev.llm.client import BaseLLMClient, LLMResponse, LLMUsage

    class _C(BaseLLMClient):
        def __init__(self):
            super().__init__({"build": {"budget": 100.0}})
            self.model = "m"

        def _call(self, p, s):
            return LLMResponse(
                content="x",
                model="m",
                usage=LLMUsage(total_tokens=10, estimated_cost=0.5),
            )

    c = _C()
    with c.track_task("T-1"):
        c.generate("p")
        c.generate("p")
    with c.track_task("T-2"):
        c.generate("p")
    c.generate("p")  # outside any task -> overhead
    assert c.cost_by_task["T-1"] == 1.0
    assert c.cost_by_task["T-2"] == 0.5
    assert c.cost_by_task["overhead"] == 0.5


# --- model routing (021) -----------------------------------------------------


def test_with_model_restores_original():
    from misterdev.llm.client import BaseLLMClient

    class _C(BaseLLMClient):
        def __init__(self):
            super().__init__({"build": {}})
            self.model = "default"

        def _call(self, p, s):
            raise NotImplementedError

    c = _C()
    with c.with_model("cheap"):
        assert c.model == "cheap"
    assert c.model == "default"


def test_resolve_model_by_complexity_and_strategy():
    ex = MarkdownPlanExecutor()
    cfg = {
        "llm": {
            "models": {"simple": "haiku", "complex": "opus", "default": "sonnet"},
            "routing": {"small": "simple", "large": "complex", "surgical": "simple"},
        }
    }
    proj = _FakeProject(".", cfg)

    class _T:
        complexity = "large"

    assert ex._resolve_model(proj, _T(), "iterative") == "opus"

    class _T2:
        complexity = "small"

    assert ex._resolve_model(proj, _T2(), "iterative") == "haiku"

    # Falls back to strategy when complexity has no route.
    class _T3:
        complexity = "medium"

    assert ex._resolve_model(proj, _T3(), "surgical") == "haiku"

    # No routing config -> None (keep client default).
    assert (
        MarkdownPlanExecutor()._resolve_model(_FakeProject(".", {}), _T(), "x") is None
    )


# --- provider failover (030) -------------------------------------------------


def _make_failover(primary, fallbacks):
    from misterdev.llm.client import FailoverLLMClient, BaseLLMClient

    fc = FailoverLLMClient.__new__(FailoverLLMClient)
    BaseLLMClient.__init__(fc, {"build": {}})
    fc.primary = primary
    fc.failover_clients = fallbacks
    return fc


class _StubClient:
    def __init__(self, behavior, model="m"):
        self.behavior = behavior  # "ok" or an LLMCallError to raise
        self.model = model

    def _call(self, prompt, system_prompt):
        from misterdev.llm.client import LLMResponse, LLMUsage

        if self.behavior == "ok":
            return LLMResponse(content="ok", model=self.model, usage=LLMUsage())
        raise self.behavior


def test_failover_advances_on_retryable_error():
    from misterdev.llm.client import LLMCallError

    fc = _make_failover(
        _StubClient(LLMCallError("503 overloaded", retryable=True), model="primary"),
        [_StubClient("ok", model="backup")],
    )
    resp = fc._call("p", "s")
    assert resp.content == "ok"
    assert resp.model == "backup"


def test_failover_stops_on_non_retryable():
    from misterdev.llm.client import LLMCallError

    fc = _make_failover(
        _StubClient(LLMCallError("400 bad request", retryable=False), model="primary"),
        [_StubClient("ok", model="backup")],
    )
    with pytest.raises(LLMCallError):
        fc._call("p", "s")  # non-retryable must not fall through to backup


def test_failover_factory_wraps_when_configured(monkeypatch):
    from misterdev.llm import client as cl

    monkeypatch.setattr(cl, "_create_single_client", lambda cfg: _StubClient("ok"))
    wrapped = cl.create_llm_client(
        {"llm": {"provider": "openrouter", "failover": [{"provider": "anthropic"}]}}
    )
    assert isinstance(wrapped, cl.FailoverLLMClient)
    plain = cl.create_llm_client({"llm": {"provider": "openrouter"}})
    assert not isinstance(plain, cl.FailoverLLMClient)


# --- incremental rerun (022) -------------------------------------------------


def test_progress_needs_rerun_on_hash_change():
    from misterdev.core.execution.progress import ProgressTracker

    with tempfile.TemporaryDirectory() as td:
        pt = ProgressTracker(Path(td))
        assert pt.needs_rerun("T-1", "h1")  # never completed
        pt.mark_completed("T-1", "h1")
        assert not pt.needs_rerun("T-1", "h1")  # same hash -> skip
        assert pt.needs_rerun("T-1", "h2")  # changed -> rerun
        # Hash persists across reloads.
        pt2 = ProgressTracker(Path(td))
        assert not pt2.needs_rerun("T-1", "h1")


def test_compute_task_hash_content_based_and_mtime_stable():
    import os
    from types import SimpleNamespace

    from misterdev.core.execution.progress import compute_task_hash

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "a.py").write_text("print(1)\n")

        def task(desc="do it"):
            return SimpleNamespace(
                id="T1",
                title="t",
                description=desc,
                acceptance_criteria="",
                files_to_create=[],
                files_to_modify=["a.py"],
            )

        h1 = compute_task_hash(task(), root)
        # Stable across an mtime bump with identical content (the resume fix).
        os.utime(root / "a.py", None)
        assert compute_task_hash(task(), root) == h1
        # Changes when the committed file CONTENT changes.
        (root / "a.py").write_text("print(2)\n")
        assert compute_task_hash(task(), root) != h1
        # Changes when the task SPEC changes (an intentional edit re-runs it).
        (root / "a.py").write_text("print(1)\n")
        assert compute_task_hash(task(desc="do it differently"), root) != h1


# --- tree-sitter rust contract extraction (013) -----------------------------


def test_rust_contracts_tree_sitter_multiline():
    from misterdev.core.context.contracts import _extract_public_symbols

    src = (
        "pub struct Engine<T: Clone> {\n"
        "    pub name: String,\n"
        "    secret: u64,\n"
        "}\n\n"
        "pub fn overlap_scan(\n"
        "    query: &[u16],\n"
        "    limit: usize,\n"
        ") -> Vec<u32> {\n"
        "    Vec::new()\n"
        "}\n\n"
        "impl Engine<T> {\n"
        "    pub fn new(name: String) -> Self { todo!() }\n"
        "    fn private_helper(&self) {}\n"
        "}\n"
    )
    syms = _extract_public_symbols(src, "rust")
    by_name = {s["name"]: s for s in syms}
    # Multi-line fn signature captured whole.
    assert "overlap_scan" in by_name
    assert "limit: usize" in by_name["overlap_scan"]["signature"]
    # Generic struct with pub field, private field excluded.
    assert "<T: Clone>" in by_name["Engine"]["signature"]
    assert "pub name" in by_name["Engine"]["signature"]
    assert "secret" not in by_name["Engine"]["signature"]
    # Impl: pub method qualified, private method excluded.
    assert "Engine::new" in by_name
    assert not any("private_helper" in n for n in by_name)


# --- typescript topography (010, TS only) -----------------------------------


def test_topography_typescript_symbols():
    from misterdev.core.context.topography import SymbolGraph, _get_ts_parsers

    if "typescript" not in _get_ts_parsers():
        pytest.skip("typescript grammar not installed")
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "app.ts").write_text(
            "export function greet(name: string): string { return name; }\n"
            "export class Service {\n"
            "  start(): void {}\n"
            "}\n"
            "export interface Repo { find(id: number): string; }\n"
            "export type ID = number;\n"
        )
        g = SymbolGraph(Path(td))
        g.build()
        names = {k.split(":", 1)[1] for k in g.symbols}
    assert {"greet", "Service", "Service.start", "Repo", "ID"} <= names


# --- depth-limited scan (012) ------------------------------------------------


def test_walk_limited_prunes_and_bounds_depth():
    from misterdev.analyzers.project_analyzer import _walk_limited

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "a.py").write_text("x")
        (root / "node_modules").mkdir()
        (root / "node_modules" / "junk.js").write_text("x")
        deep = root / "d1" / "d2" / "d3" / "d4"
        deep.mkdir(parents=True)
        (deep / "deep.py").write_text("x")
        found = {p.name for p in _walk_limited(root, max_depth=2)}
        assert "a.py" in found
        assert "junk.js" not in found  # node_modules pruned
        assert "deep.py" not in found  # beyond max_depth


# --- preflight validation (023) ---------------------------------------------


def test_preflight_flags_dangling_dependency():
    from misterdev.core.verification.preflight import PreflightValidator
    from misterdev.core.models import Task

    good = Task(
        id="001-a",
        description="d",
        type="markdown_planner",
        project_ref=".",
        dependencies=[],
    )
    bad = Task(
        id="002-b",
        description="d",
        type="markdown_planner",
        project_ref=".",
        dependencies=["999-missing"],
    )
    with tempfile.TemporaryDirectory() as td:
        issues = PreflightValidator().validate([good, bad], Path(td))
    assert PreflightValidator.has_errors(issues)
    assert any("999-missing" in i.message and i.severity == "error" for i in issues)


def test_preflight_clean_plan_has_no_errors():
    from misterdev.core.verification.preflight import PreflightValidator
    from misterdev.core.models import Task

    a = Task(
        id="001-a",
        description="d",
        type="markdown_planner",
        project_ref=".",
        dependencies=[],
    )
    b = Task(
        id="002-b",
        description="d",
        type="markdown_planner",
        project_ref=".",
        dependencies=["001-a"],
    )
    with tempfile.TemporaryDirectory() as td:
        issues = PreflightValidator().validate([a, b], Path(td))
    assert not PreflightValidator.has_errors(issues)


# --- run_project orchestration (001 / 016) ----------------------------------


def _mock_task(tid, deps=None):
    from unittest.mock import MagicMock

    t = MagicMock()
    t.id = tid
    t.dependencies = deps or []
    t.title = tid
    t.description = "d"
    t.complexity = "small"
    t.category = "fix"
    t.files_to_modify = []
    t.files_to_create = []
    t.context_files = []
    t.processor_data = {}
    t.execution_history = []
    return t


def _patched_run(tmp, tasks, dry_run=False):
    from unittest.mock import patch, MagicMock
    import misterdev.agent as agent_mod

    project = MagicMock()
    project.name = "p"
    project.path = Path(tmp)
    project.env_manager = None
    project.config = {
        "language": "python",
        "orchestrator": {"max_consecutive_failures": 3},
    }
    project.task_manager.get_pending_tasks.return_value = tasks

    completed_result = MagicMock()
    completed_result.status = "completed"

    orch = agent_mod.ProjectOrchestrator()
    with (
        patch.object(
            agent_mod.ProjectOrchestrator, "_get_or_register", return_value=project
        ),
        patch("misterdev.agent.topological_sort", side_effect=lambda x: x),
        patch("misterdev.agent.Scratchpad"),
        patch("misterdev.agent.ContractRegistry"),
        patch("misterdev.agent.ChangeTracker"),
        patch("misterdev.agent.StrategyOptimizer") as MockStrat,
        patch("misterdev.agent.ProgressTracker") as MockProg,
        patch("misterdev.agent.MarkdownPlanExecutor") as MockExec,
    ):
        MockProg.return_value.completed = []
        MockProg.return_value.is_done.return_value = False
        MockStrat.return_value.select_best_strategy.return_value = "iterative"
        MockExec.return_value.execute.return_value = completed_result
        orch.run_project(tmp, dry_run=dry_run)
    return MockExec.return_value


def test_run_project_dry_run_does_not_execute():
    with tempfile.TemporaryDirectory() as td:
        ex = _patched_run(td, [_mock_task("001-a")], dry_run=True)
        ex.execute.assert_not_called()


def test_run_project_executes_in_dependency_order():
    with tempfile.TemporaryDirectory() as td:
        a = _mock_task("001-a")
        b = _mock_task("002-b", deps=["001-a"])
        ex = _patched_run(td, [b, a])  # deliberately out of order
        executed = [c.args[0].id for c in ex.execute.call_args_list]
        assert executed == ["001-a", "002-b"]


# --- budget kill-switch / graceful checkpointing ----------------------------


def _fresh_report():
    from datetime import datetime, timezone
    from misterdev.core.reporting.report import BuildReport
    from misterdev.core.planning.assessment import ProjectAssessment
    from misterdev.core.modes import BuildMode

    return BuildReport(
        BuildMode.COMPLETE, "p", ProjectAssessment(), datetime.now(timezone.utc)
    )


class _BudgetClient:
    """Minimal stand-in for BaseLLMClient exposing the budget hooks agent uses."""

    def __init__(self, budget_remaining=100.0, max_cost_per_task=None):
        self.budget_remaining = budget_remaining
        self._max_cost_per_task = max_cost_per_task
        self.cost_by_task = {}

    def task_cost(self, task_id):
        return self.cost_by_task.get(task_id, 0.0)

    def task_cost_exceeded(self, task_id):
        if self._max_cost_per_task is None or task_id is None:
            return False
        return self.task_cost(task_id) >= self._max_cost_per_task


def _budget_project(tmp, max_cost_per_task=None, budget_remaining=100.0):
    from unittest.mock import MagicMock

    project = MagicMock()
    project.path = Path(tmp)
    project.config = {
        "language": "python",
        "orchestrator": {
            "max_consecutive_failures": 3,
            **(
                {"max_cost_per_task": max_cost_per_task}
                if max_cost_per_task is not None
                else {}
            ),
        },
    }
    project.llm_client = _BudgetClient(
        budget_remaining=budget_remaining, max_cost_per_task=max_cost_per_task
    )
    return project


def test_per_task_cost_cap_reverts_and_defers_not_failure():
    from unittest.mock import patch, MagicMock
    import misterdev.agent as agent_mod
    from misterdev.core.modes import BuildFlags

    with tempfile.TemporaryDirectory() as td:
        project = _budget_project(td, max_cost_per_task=0.05)
        task = _mock_task("T-cap")

        reverted = []

        class _Exec:
            def __init__(self, *a, **k):
                pass

            def execute(self, t, proj):
                # Pretend the task burned past its cap during execution.
                project.llm_client.cost_by_task[t.id] = 0.10
                r = MagicMock()
                r.status = "failed"
                return r

            def find_task_commit(self, proj, tid):
                return "deadbeef"

            def revert_task_commit(self, proj, sha):
                reverted.append(sha)
                return True

            def _run_command(self, *a, **k):
                return True, ""

        with (
            patch.object(agent_mod, "MarkdownPlanExecutor", _Exec),
            patch("misterdev.agent.Scratchpad"),
            patch("misterdev.agent.RealTimeAligner"),
            patch("misterdev.agent.ContractRegistry"),
            patch("misterdev.agent.ChangeTracker"),
            patch("misterdev.agent.ProgressTracker") as MockProg,
            patch("misterdev.agent.StrategyOptimizer") as MockStrat,
        ):
            MockProg.return_value.completed = []
            MockProg.return_value.needs_rerun.return_value = True
            MockStrat.return_value.select_best_strategy.return_value = "iterative"
            orch = agent_mod.ProjectOrchestrator()
            report = _fresh_report()
            flags = BuildFlags(no_rollback=True)
            orch._execute_tasks([task], project, flags, report)

        assert reverted == ["deadbeef"]
        assert task in report.deferred_tasks
        assert task not in report.failed_tasks
        assert any("per-task cost cap" in d for d in report.key_decisions)


class _SkipExec:
    """Executor stand-in that records every task it is asked to run, so a test can
    assert an already-satisfied task never reaches execution (no worktree)."""

    executed: list

    def __init__(self, *a, **k):
        pass

    def execute(self, t, proj, *a, **k):
        from unittest.mock import MagicMock

        type(self).executed.append(t.id)
        r = MagicMock()
        r.status = "completed"
        return r

    def _run_command(self, *a, **k):
        return True, ""

    def find_task_commit(self, *a, **k):
        return None


def _run_execute_tasks(project, tasks):
    from unittest.mock import patch
    import misterdev.agent as agent_mod
    from misterdev.core.modes import BuildFlags

    _SkipExec.executed = []
    # Patch the peripheral collaborators (as the other _execute_tasks tests do),
    # but keep a REAL ProgressTracker so it loads the on-disk ledger this test
    # pre-populated — that ledger is what the skip decision reads.
    with (
        patch.object(agent_mod, "MarkdownPlanExecutor", _SkipExec),
        patch("misterdev.agent.Scratchpad"),
        patch("misterdev.agent.RealTimeAligner"),
        patch("misterdev.agent.ContractRegistry"),
        patch("misterdev.agent.ChangeTracker"),
        patch("misterdev.agent.StrategyOptimizer") as MockStrat,
    ):
        MockStrat.return_value.select_best_strategy.return_value = "iterative"
        orch = agent_mod.ProjectOrchestrator()
        report = _fresh_report()
        orch._execute_tasks(tasks, project, BuildFlags(no_rollback=True), report)
    return report, list(_SkipExec.executed)


def test_satisfied_task_skipped_before_worktree():
    """A ready task whose content hash matches its recorded completion is marked
    done WITHOUT ever reaching the executor (so no worktree is spawned)."""
    from misterdev.core.execution.progress import ProgressTracker, compute_task_hash

    with tempfile.TemporaryDirectory() as td:
        project = _budget_project(td)
        task = _mock_task("T-done")
        task.acceptance_criteria = ""  # a real str -> a deterministic hash
        # Record a prior completion at the task's CURRENT content hash.
        ProgressTracker(Path(td)).mark_completed(
            task.id, compute_task_hash(task, Path(td))
        )

        report, executed = _run_execute_tasks(project, [task])

        assert executed == []  # never dispatched -> no worktree
        assert task in report.completed_tasks


def test_changed_or_absent_hash_still_runs():
    """A recorded completion at a STALE hash, and a task with no recorded
    completion at all, both re-run — the skip keys on the content hash, not the id
    alone."""
    from misterdev.core.execution.progress import ProgressTracker

    with tempfile.TemporaryDirectory() as td:
        project = _budget_project(td)
        stale = _mock_task("T-stale")
        stale.acceptance_criteria = ""
        absent = _mock_task("T-absent")
        absent.acceptance_criteria = ""
        # 'stale' is recorded completed but at a hash that no longer matches;
        # 'absent' has no record. Neither may be skipped.
        ProgressTracker(Path(td)).mark_completed("T-stale", "0000stalehash000")

        _, executed = _run_execute_tasks(project, [stale, absent])

        assert set(executed) == {"T-stale", "T-absent"}


def test_skip_satisfied_tasks_flag_off_forces_rerun():
    """With orchestrator.skip_satisfied_tasks false, even a hash-matched completion
    re-runs (the skip is opt-out)."""
    from misterdev.core.execution.progress import ProgressTracker, compute_task_hash

    with tempfile.TemporaryDirectory() as td:
        project = _budget_project(td)
        project.config["orchestrator"]["skip_satisfied_tasks"] = False
        task = _mock_task("T-done")
        task.acceptance_criteria = ""
        ProgressTracker(Path(td)).mark_completed(
            task.id, compute_task_hash(task, Path(td))
        )

        _, executed = _run_execute_tasks(project, [task])

        assert executed == ["T-done"]  # flag off -> re-run despite the match


def test_wave_infra_count_counts_only_unrecovered_infra_failures():
    from unittest.mock import MagicMock
    import misterdev.agent as agent_mod

    def _res(status, logs=""):
        r = MagicMock()
        r.status = status
        r.logs = logs
        r.message = ""
        return r

    results = [
        (_mock_task("A"), _res("failed", "Command timed out after 120s"), None),
        (_mock_task("B"), _res("completed", "Command timed out after 120s"), None),
        (_mock_task("C"), _res("failed", "AssertionError: 1 != 2"), None),
        (_mock_task("D"), None, RuntimeError("waiting for the lock on the store")),
    ]
    # A (infra, failed) and D (infra, in the raised error) count; B completed
    # (self-healed) and C (a real code error) do not.
    assert agent_mod.ProjectOrchestrator._wave_infra_count(results) == 2


def test_apply_wave_tuning_scales_config_from_base():
    from unittest.mock import MagicMock
    import misterdev.agent as agent_mod
    from misterdev.core.execution.adaptive import WaveTuning

    project = MagicMock()
    project.config = {"orchestrator": {}, "build": {}}
    base = {"workers": 8, "setup": 600, "build": 120, "test": 180}
    agent_mod.ProjectOrchestrator()._apply_wave_tuning(
        project, WaveTuning(2, 2.0), base
    )
    assert project.config["orchestrator"]["max_workers"] == 2
    assert project.config["orchestrator"]["worktree_setup_timeout"] == 1200
    assert project.config["build"]["build_timeout"] == 240
    assert project.config["build"]["test_timeout"] == 360


def test_adaptive_backoff_applies_to_next_wave():
    """An infra failure in wave 1 backs off concurrency for wave 2: the wave-2
    task sees a halved max_workers and a doubled timeout in the live config."""
    from unittest.mock import patch, MagicMock
    import misterdev.agent as agent_mod
    from misterdev.core.modes import BuildFlags
    from misterdev.config import get_setting

    with tempfile.TemporaryDirectory() as td:
        project = _budget_project(td)
        project.config["orchestrator"].update(
            {
                "max_workers": 4,
                "adaptive_infra_threshold": 0,  # any infra fault triggers backoff
            }
        )
        # A fails on infra; B completes (wave 1). C depends on B (wave 2).
        a = _mock_task("A")
        b = _mock_task("B")
        c = _mock_task("C", deps=["B"])

        seen_workers = {}

        class _Exec:
            def __init__(self, *a, **k):
                pass

            def execute(self, t, proj):
                seen_workers[t.id] = get_setting(
                    proj.config, "orchestrator", "max_workers"
                )
                r = MagicMock()
                if t.id == "A":
                    r.status = "failed"
                    r.logs = "Command timed out after 120s"
                else:
                    r.status = "completed"
                    r.logs = ""
                r.message = ""
                return r

            def _run_command(self, *a, **k):
                return True, ""

        with (
            patch.object(agent_mod, "MarkdownPlanExecutor", _Exec),
            patch("misterdev.agent.Scratchpad"),
            patch("misterdev.agent.RealTimeAligner"),
            patch("misterdev.agent.ContractRegistry"),
            patch("misterdev.agent.ChangeTracker"),
            patch("misterdev.agent.StrategyOptimizer") as MockStrat,
        ):
            MockStrat.return_value.select_best_strategy.return_value = "iterative"
            orch = agent_mod.ProjectOrchestrator()
            orch._execute_tasks(
                [a, b, c], project, BuildFlags(no_rollback=True), _fresh_report()
            )

        assert seen_workers["A"] == 4  # wave 1: full concurrency
        assert seen_workers["C"] == 2  # wave 2: halved after the infra fault
        # Config is restored to the configured base after the run.
        assert get_setting(project.config, "orchestrator", "max_workers") == 4


def test_budget_exhausted_before_wave_defers_gracefully():
    from unittest.mock import patch, MagicMock
    import misterdev.agent as agent_mod
    from misterdev.core.modes import BuildFlags

    with tempfile.TemporaryDirectory() as td:
        project = _budget_project(td, budget_remaining=0.0)
        tasks = [_mock_task("T-1"), _mock_task("T-2")]

        executed = []

        class _Exec:
            def __init__(self, *a, **k):
                pass

            def execute(self, t, proj):
                executed.append(t.id)
                r = MagicMock()
                r.status = "completed"
                return r

            def _run_command(self, *a, **k):
                return True, ""

        with patch.object(agent_mod, "MarkdownPlanExecutor", _Exec):
            orch = agent_mod.ProjectOrchestrator()
            report = _fresh_report()
            flags = BuildFlags(no_rollback=True)
            orch._execute_tasks(tasks, project, flags, report)

        assert executed == []  # no wave launched
        assert {t.id for t in report.deferred_tasks} == {"T-1", "T-2"}
        assert report.failed_tasks == []
        assert any("budget exhausted" in d for d in report.key_decisions)


# --- report persistence -----------------------------------------------------


def test_report_save_writes_markdown_and_json():
    from datetime import datetime, timezone
    from misterdev.core.reporting.report import BuildReport
    from misterdev.core.planning.assessment import ProjectAssessment
    from misterdev.core.modes import BuildMode

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        report = BuildReport(
            BuildMode.COMPLETE, "proj", ProjectAssessment(), datetime.now(timezone.utc)
        )
        report.finalize()
        path = report.save(root)
        assert path is not None and path.exists()
        json_path = path.with_suffix(".json")
        data = json.loads(json_path.read_text())
        assert data["project"] == "proj"


# --- health-check command detection (tests=none blindness) ------------------
def test_detect_test_command_pytest_uv():
    from misterdev.analyzers.project_analyzer import detect_test_command

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "tests").mkdir()
        (root / "uv.lock").write_text("", encoding="utf-8")
        assert detect_test_command(root) == "uv run pytest -q"


def test_detect_test_command_pytest_no_uv():
    from misterdev.analyzers.project_analyzer import detect_test_command

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "pyproject.toml").write_text(
            "[tool.pytest.ini_options]\n", encoding="utf-8"
        )
        assert detect_test_command(root) == "pytest -q"


def test_detect_test_command_npm_and_cargo_and_none():
    from misterdev.analyzers.project_analyzer import detect_test_command

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "package.json").write_text(
            '{"scripts": {"test": "jest"}}', encoding="utf-8"
        )
        assert detect_test_command(root) == "npm test"
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "Cargo.toml").write_text("[package]\n", encoding="utf-8")
        assert detect_test_command(root) == "cargo test"
    with tempfile.TemporaryDirectory() as td:
        assert detect_test_command(Path(td)) is None


def test_detect_test_command_node_test_runner():
    # The rideshare bug: package.json with no `test` script but a *.test.js suite
    # must resolve to `node --test`, not be left ungated.
    from misterdev.analyzers.project_analyzer import detect_test_command

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "package.json").write_text('{"type": "module"}', encoding="utf-8")
        (root / "tests" / "unit").mkdir(parents=True)
        (root / "tests" / "unit" / "crypto.test.js").write_text("", encoding="utf-8")
        assert detect_test_command(root) == "node --test"


def test_detect_test_command_rust_with_tests_dir_uses_cargo():
    # Regression: Rust uses tests/ for integration tests; a bare tests/ dir must
    # NOT shadow cargo with pytest.
    from misterdev.analyzers.project_analyzer import detect_test_command

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "Cargo.toml").write_text("[package]\n", encoding="utf-8")
        (root / "tests").mkdir()
        (root / "tests" / "integration.rs").write_text("", encoding="utf-8")
        assert detect_test_command(root) == "cargo test"


def test_detect_test_command_bare_tests_dir_is_not_pytest():
    # A tests/ dir with no Python signal and no python test files -> not pytest.
    from misterdev.analyzers.project_analyzer import detect_test_command

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "tests").mkdir()
        (root / "tests" / "data.txt").write_text("", encoding="utf-8")
        assert detect_test_command(root) is None


def test_detect_test_command_pytest_from_py_test_files():
    from misterdev.analyzers.project_analyzer import detect_test_command

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "tests").mkdir()
        (root / "tests" / "test_foo.py").write_text(
            "def test_x():\n    pass\n", encoding="utf-8"
        )
        assert detect_test_command(root) == "pytest -q"


def test_detect_test_command_go():
    from misterdev.analyzers.project_analyzer import detect_test_command

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "go.mod").write_text("module x\n", encoding="utf-8")
        assert detect_test_command(root) == "go test ./..."


def test_has_test_files():
    from misterdev.analyzers.project_analyzer import has_test_files

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        assert has_test_files(root) is False
        (root / "tests").mkdir()
        (root / "tests" / "x.test.js").write_text("", encoding="utf-8")
        assert has_test_files(root) is True


def test_warn_if_no_test_gate_records_when_tests_exist_without_command():
    import types
    from misterdev.agent import _warn_if_no_test_gate
    from misterdev.core.reporting.report import BuildReport
    from misterdev.core.planning.assessment import (
        ProjectAssessment,
        HealthCheck,
        ProjectStructure,
        TechnicalDebt,
        RiskAssessment,
    )
    from misterdev.core.modes import BuildMode
    from datetime import datetime, timezone

    def _assessment(test_command):
        return ProjectAssessment(
            structure=ProjectStructure(
                project_type="web-app",
                languages=["javascript"],
                test_command=test_command,
            ),
            health=HealthCheck(builds=True),
            tech_debt=TechnicalDebt(score=10),
            risk=RiskAssessment(level="low"),
        )

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "tests").mkdir()
        (root / "tests" / "a.test.js").write_text("", encoding="utf-8")
        proj = types.SimpleNamespace(path=root)
        # No test command + tests exist -> recorded as a degraded subsystem.
        rep = BuildReport(
            BuildMode.SMART,
            "x",
            _assessment(None),
            datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        _warn_if_no_test_gate(_assessment(None), proj, rep)
        assert any("No test gate" in d for d in rep.degraded_subsystems)
        # Command present -> no warning.
        rep2 = BuildReport(
            BuildMode.SMART,
            "x",
            _assessment("node --test"),
            datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        _warn_if_no_test_gate(_assessment("node --test"), proj, rep2)
        assert not any("No test gate" in d for d in rep2.degraded_subsystems)


def test_analyze_project_fills_test_command_when_llm_returns_null():
    """The real regression: LLM left test_command null -> tests=none. The
    deterministic fallback must populate the assessment so the suite runs."""
    from unittest.mock import patch
    from misterdev.analyzers import project_analyzer
    from misterdev.core.planning.assessment import HealthCheck

    class _EmptyLLM:
        def generate_code(self, prompt, system_prompt=""):
            return "{}"

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "tests").mkdir()
        (root / "uv.lock").write_text("", encoding="utf-8")
        with patch.object(
            project_analyzer, "run_health_check", return_value=HealthCheck()
        ):
            assessment = project_analyzer.analyze_project(
                root, _EmptyLLM(), parallel=False
            )
        assert assessment.structure.test_command == "uv run pytest -q"


# --- model preflight health check -------------------------------------------
def test_health_check_ok_and_failure_preserve_budget():
    from misterdev.llm.client import BaseLLMClient, LLMResponse, LLMUsage

    class _OK(BaseLLMClient):
        model = "good/model"

        def _call(self, prompt, system_prompt):
            return LLMResponse(content="OK", usage=LLMUsage(estimated_cost=0.5))

    client = _OK({"build": {"budget": 10.0}})
    client.cumulative_usage.estimated_cost = 0.0
    ok, detail = client.health_check()
    assert ok and detail == "good/model"
    assert client.cumulative_usage.estimated_cost == 0.0  # probe cost rolled back

    class _Dead(BaseLLMClient):
        model = "dead/model"

        def _call(self, prompt, system_prompt):
            raise RuntimeError("404 No endpoints found")

    bad = _Dead({"build": {"budget": 10.0}})
    ok, detail = bad.health_check()
    assert not ok and "dead/model" in detail and "404" in detail


# --- integration gate: real git + real pytest (would have caught self-build) -
def _git(root, *args):
    import subprocess

    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True)


def _make_repo_with_passing_suite(root):
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@t.t")
    _git(root, "config", "user.name", "t")
    pkg = root / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    tdir = root / "tests"
    tdir.mkdir()
    (tdir / "test_basic.py").write_text(
        "from pkg import VALUE\n\ndef test_value():\n    assert VALUE == 1\n",
        encoding="utf-8",
    )
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "baseline")


def _commit_task(root, task_id, mutate):
    """Apply a mutation and commit it with the task(<id>): convention."""
    mutate(root)
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", f"task({task_id}): change")


def test_integration_gate_reverts_regressing_task():
    from types import SimpleNamespace
    from misterdev.agent import ProjectOrchestrator
    from misterdev.task_executors.markdown_plan_executor import (
        MarkdownPlanExecutor,
    )

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _make_repo_with_passing_suite(root)

        # A task that breaks the suite the way the self-build did: a module
        # referencing a name that no longer exists -> ImportError at collection.
        def break_import(r):
            (r / "pkg" / "__init__.py").write_text(
                "from pkg.missing import GONE\nVALUE = 1\n", encoding="utf-8"
            )

        _commit_task(root, "T-bad", break_import)

        project = SimpleNamespace(path=root, env_manager=None)
        executor = MarkdownPlanExecutor()
        orch = ProjectOrchestrator()
        task = SimpleNamespace(id="T-bad")
        branch_before = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
        ).stdout.strip()

        # Sanity: suite is red at HEAD before the gate runs.
        ok_before, _ = executor._run_command(
            project, f"{sys.executable} -m pytest -q", timeout=120
        )
        assert not ok_before

        reverted = orch._integration_gate(
            project, executor, f"{sys.executable} -m pytest -q", [task], timeout=120
        )
        assert reverted == ["T-bad"]
        ok_after, _ = executor._run_command(
            project, f"{sys.executable} -m pytest -q", timeout=120
        )
        assert ok_after  # tree restored to green

        # HEAD must remain attached to the branch (not detached by bisect), or
        # subsequent waves would branch from / merge into a detached head.
        attached = subprocess.run(
            ["git", "symbolic-ref", "-q", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
        )
        assert attached.returncode == 0, "HEAD left detached after gate"
        branch_after = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
        ).stdout.strip()
        assert branch_after == branch_before


def test_integration_gate_noop_when_suite_stays_green():
    from types import SimpleNamespace
    from misterdev.agent import ProjectOrchestrator
    from misterdev.task_executors.markdown_plan_executor import (
        MarkdownPlanExecutor,
    )

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _make_repo_with_passing_suite(root)

        def harmless(r):
            (r / "pkg" / "extra.py").write_text("X = 2\n", encoding="utf-8")

        _commit_task(root, "T-ok", harmless)

        project = SimpleNamespace(path=root, env_manager=None)
        executor = MarkdownPlanExecutor()
        orch = ProjectOrchestrator()
        task = SimpleNamespace(id="T-ok")
        reverted = orch._integration_gate(
            project, executor, f"{sys.executable} -m pytest -q", [task], timeout=120
        )
        assert reverted == []


# --- interactive planner: advisor + goal selection --------------------------
def test_recommend_work_parses_and_normalizes():
    from misterdev.core.planning.advisor import recommend_work
    from misterdev.core.planning.assessment import ProjectAssessment

    class _LLM:
        def generate_code(self, prompt, system_prompt=""):
            return (
                'Here you go: [{"title": "Fix the import gate", '
                '"rationale": "suite is red", "work_type": "debug"}, '
                '{"title": "Polish docs", "rationale": "thin", '
                '"work_type": "bogus"}]'
            )

    recs = recommend_work(ProjectAssessment(), _LLM())
    assert [r.title for r in recs] == ["Fix the import gate", "Polish docs"]
    assert recs[0].work_type == "debug"
    assert recs[1].work_type == "complete"  # invalid type normalized


def test_recommend_work_bad_json_returns_empty():
    from misterdev.core.planning.advisor import recommend_work
    from misterdev.core.planning.assessment import ProjectAssessment

    class _LLM:
        def generate_code(self, prompt, system_prompt=""):
            return "no json here"

    assert recommend_work(ProjectAssessment(), _LLM()) == []


def test_choose_goal_number_text_and_quit():
    from unittest.mock import patch
    from misterdev.agent import ProjectOrchestrator
    from misterdev.core.planning.advisor import Recommendation
    from misterdev.core.modes import BuildMode

    orch = ProjectOrchestrator()
    recs = [
        Recommendation("Fix imports", "red suite", "debug"),
        Recommendation("Add feature X", "users want it", "feature"),
    ]

    with patch("misterdev.agent.Prompt.ask", return_value="1"):
        goal, mode = orch._choose_goal(recs)
    assert goal == "Fix imports" and mode == BuildMode.DEBUG

    with patch("misterdev.agent.Prompt.ask", return_value="make it faster"):
        goal, mode = orch._choose_goal(recs)
    assert goal == "make it faster"

    with patch("misterdev.agent.Prompt.ask", return_value="q"):
        goal, mode = orch._choose_goal(recs)
    assert goal is None


def test_interactive_plan_cancels_when_no_goal():
    from unittest.mock import patch, MagicMock
    from misterdev.agent import ProjectOrchestrator
    from misterdev.core.planning.assessment import ProjectAssessment

    orch = ProjectOrchestrator()
    project = MagicMock()
    project.name = "p"
    project.path = Path("/tmp/x")
    project.env_manager = None
    project.config = {}
    project.llm_client.health_check.return_value = (True, "model")

    with (
        patch.object(orch, "_get_or_register", return_value=project),
        patch(
            "misterdev.agent.analyze_project",
            return_value=ProjectAssessment(),
        ),
        patch("misterdev.agent.recommend_work", return_value=[]),
        patch.object(orch, "_choose_goal", return_value=(None, None)),
        patch.object(orch, "_run_pipeline") as mock_pipeline,
    ):
        result = orch.interactive_plan("/tmp/x")
    assert "Cancelled" in result
    mock_pipeline.assert_not_called()


def test_interactive_plan_runs_pipeline_with_confirm():
    from unittest.mock import patch, MagicMock
    from misterdev.agent import ProjectOrchestrator
    from misterdev.core.planning.assessment import ProjectAssessment
    from misterdev.core.modes import BuildMode

    orch = ProjectOrchestrator()
    project = MagicMock()
    project.name = "p"
    project.path = Path("/tmp/x")
    project.env_manager = None
    project.config = {}
    project.llm_client.health_check.return_value = (True, "model")

    with (
        patch.object(orch, "_get_or_register", return_value=project),
        patch(
            "misterdev.agent.analyze_project",
            return_value=ProjectAssessment(),
        ),
        patch("misterdev.agent.recommend_work", return_value=[]),
        patch.object(
            orch, "_choose_goal", return_value=("Fix imports", BuildMode.DEBUG)
        ),
        patch.object(orch, "_run_pipeline", return_value="REPORT") as mock_pipeline,
    ):
        result = orch.interactive_plan("/tmp/x")
    assert result == "REPORT"
    assert mock_pipeline.call_args.kwargs.get("confirm_plan") is True


# --- health-check test-count parsing (tests=none display bug) ----------------
def test_parse_test_counts_pytest_and_cargo():
    from misterdev.core.verification.validator import _parse_test_counts

    assert _parse_test_counts("332 passed in 34.18s") == (332, 0)
    assert _parse_test_counts("3 failed, 317 passed in 53s") == (320, 3)
    assert _parse_test_counts("5 passed, 2 skipped") == (5, 0)
    assert _parse_test_counts("test result: ok. 42 passed; 0 failed") == (42, 0)
    assert _parse_test_counts("no recognizable summary") == (0, 0)


def test_run_health_check_populates_test_count():
    from misterdev.core.verification.validator import run_health_check

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "test_x.py").write_text(
            "def test_a():\n    assert True\n", encoding="utf-8"
        )
        health = run_health_check(root, None, f"{sys.executable} -m pytest -q", None)
        assert health.test_count == 1 and health.test_failures == 0
        assert health.tests_pass


def test_health_ground_truth_string():
    from misterdev.analyzers.project_analyzer import _health_ground_truth
    from misterdev.core.planning.assessment import HealthCheck

    h = HealthCheck(builds=True, tests_pass=True, test_count=332, test_failures=0)
    g = _health_ground_truth(h)
    assert "build passes" in g and "332/332 tests passing" in g


# --- stale cross-build progress no longer causes spurious skips --------------
def test_task_hash_reflects_content_for_idless_llm_tasks():
    from types import SimpleNamespace
    from misterdev.core.execution.progress import compute_task_hash

    a = SimpleNamespace(
        id="T-001",
        title="Add scan tests",
        description="cover scan",
        files_to_create=["tests/test_scan.py"],
        files_to_modify=[],
    )
    b = SimpleNamespace(
        id="T-001",
        title="Add build tests",
        description="cover build",
        files_to_create=["tests/test_build.py"],
        files_to_modify=[],
    )
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        # Same reused id, different content -> different hash.
        assert compute_task_hash(a, root) != compute_task_hash(b, root)


def test_needs_rerun_skips_stale_idmatch_without_hash():
    from types import SimpleNamespace
    from misterdev.core.execution.progress import ProgressTracker, compute_task_hash

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        pt = ProgressTracker(root)
        # Simulate a prior build that completed T-001 WITHOUT recording a hash
        # (the old stale-progress state from the broken self-build).
        pt.mark_completed("T-001")
        new_task = SimpleNamespace(
            id="T-001",
            title="fresh",
            description="new plan",
            files_to_create=["x.py"],
            files_to_modify=[],
        )
        # A freshly decomposed T-001 must NOT be skipped against stale progress.
        assert pt.needs_rerun("T-001", compute_task_hash(new_task, root))
        # After completing it WITH its hash, an identical resume is skipped.
        h = compute_task_hash(new_task, root)
        pt.mark_completed("T-001", h)
        assert not pt.needs_rerun("T-001", h)


# --- ephemeral probe name with path chars must not crash (live build #3) -----
def test_ephemeral_script_name_with_slash_does_not_crash():
    from misterdev.core.planning.sovereign import EphemeralCodeManager

    with tempfile.TemporaryDirectory() as td:
        with EphemeralCodeManager(Path(td)) as mgr:
            # LLM named a probe "CLI Runner / Invocation Mechanism Probe" -> the
            # '/' previously turned the filename into a missing subdir and raised.
            ok, out = mgr.run_ephemeral_script(
                "print('hi')", name="probe_CLI Runner / Invocation Mechanism Probe"
            )
            assert ok and "hi" in out


# --- LLM-identifier sanitization (root of the whole crash class) -------------
def test_safe_ref_slug_neutralizes_path_and_ref_chars():
    from misterdev.utils.file_utils import safe_ref_slug

    assert (
        safe_ref_slug("CLI Runner / Invocation Probe") == "CLI_Runner_Invocation_Probe"
    )
    assert safe_ref_slug("T 001") == "T_001"
    assert safe_ref_slug("feat:auth~bug") == "feat_auth_bug"
    assert safe_ref_slug("T-001") == "T-001"  # already-clean ids untouched
    assert safe_ref_slug("...", fallback="x") == "x"
    assert safe_ref_slug("../escape") == "escape"


def test_decompose_sanitizes_task_ids_and_deps():
    from misterdev.core.planning.decomposer import decompose_spec
    from misterdev.core.planning.assessment import ProjectAssessment
    from misterdev.core.modes import BuildMode

    class _LLM:
        def generate_code(self, prompt, system_prompt=""):
            return (
                '[{"id": "T 1", "title": "a", "dependencies": []},'
                ' {"id": "T/2", "title": "b", "dependencies": ["T 1"]}]'
            )

    tasks = decompose_spec("spec", ProjectAssessment(), BuildMode.SMART, _LLM(), ".")
    ids = [t.id for t in tasks]
    assert ids == ["T_1", "T_2"]  # branch-safe
    # Dependency ref sanitized with the SAME function so it still resolves.
    assert tasks[1].dependencies == ["T_1"]


# --- offline executor end-to-end (first real coverage of execute()) ----------
def _fake_project(repo: Path, monkeypatch, edit_response: str, build_command="true"):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    from misterdev.config import DEFAULT_CONFIG
    from misterdev.core.execution.project import Project
    from tests.test_llm_client import FakeLLMClient
    from misterdev.llm.client import LLMResponse, LLMUsage

    cfg = json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy
    cfg["name"] = "fixture"
    cfg["build_command"] = build_command  # per-task build check
    cfg.pop("test_command", None)
    project = Project(repo, cfg)
    fake = FakeLLMClient(
        responses=[LLMResponse(content=edit_response, usage=LLMUsage())] * 4
    )
    project.llm_client = fake
    return project


def test_executor_rejects_task_when_build_gate_stays_red(monkeypatch):
    """End-to-end against a real repo with a REAL failing build gate: an edit
    that never satisfies the build command must NOT be reported completed, and
    the base branch must be left clean (no bad code committed)."""
    from types import SimpleNamespace
    from misterdev.task_executors.markdown_plan_executor import MarkdownPlanExecutor

    with tempfile.TemporaryDirectory() as td:
        repo = Path(td)
        _git(repo, "init")
        _git(repo, "config", "user.email", "t@t.t")
        _git(repo, "config", "user.name", "t")
        (repo / "seed.py").write_text("X = 1\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-m", "init")
        head_before = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True
        ).stdout.strip()

        # The edit is syntactically valid (passes the pre-apply syntax gate) but
        # wrong: the build command asserts answer()==42, and this returns 0, so
        # the build gate is RED on every attempt.
        edit = "```python:feature.py\ndef answer():\n    return 0\n```\n"
        build_cmd = (
            'python -c "import feature, sys; '
            'sys.exit(0 if feature.answer() == 42 else 1)"'
        )
        project = _fake_project(repo, monkeypatch, edit, build_command=build_cmd)
        task = SimpleNamespace(
            id="T-red",
            title="implement answer",
            description="implement answer",
            acceptance_criteria="",
            files_to_modify=[],
            files_to_create=["feature.py"],
            context_files=[],
            dependencies=[],
            complexity="small",
            category="feature",
            processor_data={"strategy": "surgical"},
            execution_history=[],
        )

        result = MarkdownPlanExecutor().execute(task, project)

        assert result.status != "completed"
        # No bad code committed: HEAD is unchanged and feature.py is untracked.
        tracked = subprocess.run(
            ["git", "ls-files"], cwd=repo, capture_output=True, text=True
        ).stdout
        assert "feature.py" not in tracked
        head_after = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True
        ).stdout.strip()
        assert head_after == head_before
        # Revert cleans the orphan file the failed task created (not left behind).
        assert not (repo / "feature.py").exists()


def test_completed_status_persists_to_committed_markdown(monkeypatch):
    """A completed task's status:completed is committed into its source .md and
    survives the merge — otherwise a finished devplan still reads 'pending' and a
    rerun redoes done work."""
    from misterdev.core.models import Task
    from misterdev.task_executors.markdown_plan_executor import MarkdownPlanExecutor

    with tempfile.TemporaryDirectory() as td:
        repo = Path(td)
        _git(repo, "init")
        _git(repo, "config", "user.email", "t@t.t")
        _git(repo, "config", "user.name", "t")
        (repo / "devplan").mkdir()
        md = repo / "devplan" / "010-x.md"
        md.write_text("---\nstatus: pending\n---\ndo the thing\n")
        (repo / "seed.py").write_text("X = 1\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-m", "init")

        edit = "```python:feature.py\nY = 2\n```\n"
        project = _fake_project(repo, monkeypatch, edit)  # build_command "true"
        task = Task(
            id="010-x",
            description="do the thing",
            project_ref=str(repo),
            source_ref=str(md),
            files_to_create=["feature.py"],
            processor_data={"strategy": "surgical"},
        )
        project.task_manager.tasks[task.id] = task

        result = MarkdownPlanExecutor().execute(task, project)
        assert result.status == "completed"
        committed = subprocess.run(
            ["git", "show", "HEAD:devplan/010-x.md"],
            cwd=repo,
            capture_output=True,
            text=True,
        ).stdout
        assert "status: completed" in committed
        # No stray uncommitted status write left in the tree.
        assert (
            "devplan/010-x.md"
            not in subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo,
                capture_output=True,
                text=True,
            ).stdout
        )


def test_abort_preserves_preexisting_untracked_files(monkeypatch):
    """Revert cleanup removes only orphans the task created, never a user's
    pre-existing untracked work."""
    from types import SimpleNamespace
    from misterdev.task_executors.markdown_plan_executor import MarkdownPlanExecutor

    with tempfile.TemporaryDirectory() as td:
        repo = Path(td)
        _git(repo, "init")
        _git(repo, "config", "user.email", "t@t.t")
        _git(repo, "config", "user.name", "t")
        (repo / "seed.py").write_text("X = 1\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-m", "init")
        # Pre-existing untracked user work — must survive a task revert.
        (repo / "user_notes.txt").write_text("keep me\n")

        edit = "```python:feature.py\ndef answer():\n    return 0\n```\n"
        build_cmd = 'python -c "import feature, sys; sys.exit(1)"'
        project = _fake_project(repo, monkeypatch, edit, build_command=build_cmd)
        task = SimpleNamespace(
            id="T-orphan",
            title="x",
            description="x",
            acceptance_criteria="",
            files_to_modify=[],
            files_to_create=["feature.py"],
            context_files=[],
            dependencies=[],
            complexity="small",
            category="feature",
            processor_data={"strategy": "surgical"},
            execution_history=[],
        )
        MarkdownPlanExecutor().execute(task, project)
        assert not (repo / "feature.py").exists()  # orphan cleaned
        assert (repo / "user_notes.txt").read_text() == "keep me\n"  # preserved


def test_executor_execute_commits_real_and_out_of_scope_files(monkeypatch):
    """End-to-end: a real LLM-edit response is applied, BOTH the declared file
    and an out-of-scope-but-in-root file are committed, and the commit is
    findable by the integration gate. No live API, no git detach, no orphans."""
    from types import SimpleNamespace
    from misterdev.task_executors.markdown_plan_executor import (
        MarkdownPlanExecutor,
    )

    with tempfile.TemporaryDirectory() as td:
        repo = Path(td)
        _git(repo, "init")
        _git(repo, "config", "user.email", "t@t.t")
        _git(repo, "config", "user.name", "t")
        (repo / "seed.py").write_text("X = 1\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-m", "init")
        branch_before = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
        ).stdout.strip()

        edit = (
            "Here are the edits:\n"
            "```python:tests/test_new.py\n"
            "def test_ok():\n    assert True\n"
            "```\n\n"
            "```python:helper_extra.py\n"
            "Y = 2\n"
            "```\n"
        )
        project = _fake_project(repo, monkeypatch, edit)
        task = SimpleNamespace(
            id="T-x",
            title="add a test",
            description="add a test",
            acceptance_criteria="",
            files_to_modify=[],
            files_to_create=["tests/test_new.py"],
            context_files=[],
            dependencies=[],
            complexity="small",
            category="test",
            processor_data={"strategy": "surgical"},
            execution_history=[],
        )

        result = MarkdownPlanExecutor().execute(task, project)

        assert result.status == "completed"
        # Both files written and committed (out-of-scope one not orphaned).
        # HEAD is the --no-ff merge commit, so assert via the tracked set.
        assert (repo / "tests/test_new.py").exists()
        assert (repo / "helper_extra.py").exists()
        tracked = subprocess.run(
            ["git", "ls-files"], cwd=repo, capture_output=True, text=True
        ).stdout
        assert "tests/test_new.py" in tracked
        assert "helper_extra.py" in tracked
        # The integration gate must be able to find this task's commit.
        sha = MarkdownPlanExecutor().find_task_commit(project, "T-x")
        assert sha
        # HEAD back on the base branch (merged), attached, clean tree.
        attached = subprocess.run(
            ["git", "symbolic-ref", "-q", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
        )
        assert attached.returncode == 0
        assert (
            subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=repo,
                capture_output=True,
                text=True,
            ).stdout.strip()
            == branch_before
        )
        assert (
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo,
                capture_output=True,
                text=True,
            ).stdout.strip()
            == ""
        )


# --- working-tree safety: dirty guard + formatter-spillover cleanup ----------
def test_working_tree_dirty_detects_changes_and_ignores_gitignored():
    from misterdev.agent import ProjectOrchestrator
    from types import SimpleNamespace

    orch = ProjectOrchestrator()
    with tempfile.TemporaryDirectory() as td:
        repo = Path(td)
        _git(repo, "init")
        _git(repo, "config", "user.email", "t@t.t")
        _git(repo, "config", "user.name", "t")
        (repo / ".gitignore").write_text(".orchestrator/\n")
        (repo / "seed.py").write_text("X = 1\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-m", "init")
        project = SimpleNamespace(path=repo)
        assert orch._working_tree_dirty(project) == ""  # clean

        # gitignored runtime cache must NOT count as dirty
        (repo / ".orchestrator").mkdir()
        (repo / ".orchestrator" / "progress.json").write_text("{}")
        assert orch._working_tree_dirty(project) == ""

        # a real uncommitted change is reported
        (repo / "seed.py").write_text("X = 2\n")
        assert "seed.py" in orch._working_tree_dirty(project)


def test_build_aborts_on_dirty_tree():
    from unittest.mock import MagicMock, patch
    from misterdev.agent import ProjectOrchestrator

    orch = ProjectOrchestrator()
    project = MagicMock()
    project.path = Path("/tmp/x")
    with (
        patch.object(orch, "_get_or_register", return_value=project),
        patch.object(orch, "_working_tree_dirty", return_value="3 file(s), e.g. a.py"),
    ):
        result = orch.build("/tmp/x", "do a thing")
    assert "Error" in result and "uncommitted" in result
    # The model health check must NOT have run -- we aborted before it.
    project.llm_client.health_check.assert_not_called()


def test_commit_task_discards_formatter_spillover():
    """A project-wide formatter dirties files outside the task; those must not
    be carried across the branch switch and left as a permanently dirty tree."""
    from misterdev.task_executors.markdown_plan_executor import (
        MarkdownPlanExecutor,
    )

    class _P:
        env_manager = None

    class _T:
        id = "T-9"
        title = "scoped task"
        description = "d"

    with tempfile.TemporaryDirectory() as td:
        repo = Path(td)
        _git(repo, "init")
        _git(repo, "config", "user.email", "t@t.t")
        _git(repo, "config", "user.name", "t")
        (repo / "other.py").write_text("X=1\n")
        (repo / "task_file.py").write_text("OLD=1\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-m", "init")
        base = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
        ).stdout.strip()

        ex = MarkdownPlanExecutor()
        proj = _P()
        proj.path = repo
        # Simulate a task branch where the task edited task_file.py AND a
        # project-wide formatter reformatted the unrelated other.py.
        ex._git(proj, "git checkout -b task/T-9")
        (repo / "task_file.py").write_text("NEW = 1\n")
        (repo / "other.py").write_text("X = 1  # reformatted spillover\n")

        ex._commit_task(proj, "task/T-9", base, _T(), ["task_file.py"])

        # Back on base, merged, and CLEAN -- spillover to other.py was dropped.
        assert (
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo,
                capture_output=True,
                text=True,
            ).stdout.strip()
            == ""
        )
        assert (repo / "task_file.py").read_text() == "NEW = 1\n"  # task change kept
        assert (repo / "other.py").read_text() == "X=1\n"  # spillover reverted


# --- FULL PIPELINE offline e2e: probes -> decompose -> execute -> gate -------
class _ScriptedLLM:
    """Routes generate_code by prompt content so the whole SMART _run_pipeline
    runs offline. Bakes in hostile inputs (probe name with '/', task id with a
    space and '/') to prove sanitization holds end-to-end."""

    def __init__(self):
        self.cumulative_usage = __import__(
            "misterdev.llm.client", fromlist=["LLMUsage"]
        ).LLMUsage()
        self._budget = 100.0
        self.calls = []

    def generate(self, prompt, system_prompt=""):
        from misterdev.llm.client import LLMResponse

        return LLMResponse(
            content=self.generate_code(prompt, system_prompt), finish_reason="stop"
        )

    def generate_code(self, prompt, system_prompt=""):
        self.calls.append(prompt[:60])
        p = prompt
        if "REFLECTIVE" in p or "Probe" in p or "assumptions" in p:
            return (
                '[{"name": "Bad / Name Probe", "purpose": "x", "script": "print(1)"}]'
            )
        if "comprehensive project spec" in p:
            return "Add one passing test."  # short -> AB-MCTS skipped
        if "JSON array of task objects" in p:
            return (
                '[{"id": "T 1/x", "title": "add test", "description": "add a '
                'passing test", "acceptance_criteria": "", "files_to_create": '
                '["tests/test_added.py"], "files_to_modify": [], "context_files": '
                '[], "dependencies": [], "complexity": "small", "category": "test"}]'
            )
        if "execution strategy" in p or "surgical, iterative" in p:
            return "iterative"
        if "Files to Edit" in p or "markdown code blocks" in p:
            return (
                "```python:tests/test_added.py\ndef test_added():\n    assert True\n```"
            )
        return "[]"

    def health_check(self):
        # Offline preflight: report the model as reachable so build() proceeds
        # without a network round-trip.
        return True, "ok"

    # context-manager / routing shims used by the executor
    def track_task(self, task_id):
        from contextlib import nullcontext

        return nullcontext()

    def with_model(self, model):
        from contextlib import nullcontext

        return nullcontext()

    def generate_stream(self, prompt, system_prompt="", abort_check=None):
        from misterdev.llm.client import LLMResponse

        return LLMResponse(content=self.generate_code(prompt, system_prompt))


def test_full_pipeline_offline_smart_build(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    from misterdev.config import DEFAULT_CONFIG
    from misterdev.core.execution.project import Project
    from misterdev.core.planning.assessment import (
        ProjectAssessment,
        HealthCheck,
    )
    from misterdev.core.modes import BuildMode, BuildFlags
    from misterdev.core.reporting.report import BuildReport
    from misterdev.task_executors.markdown_plan_executor import (
        MarkdownPlanExecutor,
    )
    from misterdev.agent import ProjectOrchestrator

    with tempfile.TemporaryDirectory() as td:
        repo = Path(td)
        _git(repo, "init")
        _git(repo, "config", "user.email", "t@t.t")
        _git(repo, "config", "user.name", "t")
        (repo / ".gitignore").write_text(".orchestrator/\n__pycache__/\n*.pyc\n")
        (repo / "tests").mkdir()
        (repo / "tests" / "test_seed.py").write_text(
            "def test_seed():\n    assert True\n"
        )
        (repo / "seed.py").write_text("X = 1\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-m", "init")

        cfg = json.loads(json.dumps(DEFAULT_CONFIG))
        cfg["name"] = "fixture"
        cfg["build_command"] = "true"
        project = Project(repo, cfg)
        project.llm_client = _ScriptedLLM()

        assessment = ProjectAssessment()
        assessment.structure.build_command = "true"
        assessment.structure.test_command = f"{sys.executable} -m pytest -q"
        assessment.health = HealthCheck(builds=True, tests_pass=True, test_count=1)

        from datetime import datetime, timezone

        report = BuildReport(
            BuildMode.SMART, "fixture", assessment, datetime.now(timezone.utc)
        )
        flags = BuildFlags(budget=100.0)

        orch = ProjectOrchestrator()
        result = orch._run_pipeline(
            project,
            "add a passing test",
            BuildMode.SMART,
            flags,
            assessment,
            None,
            report,
        )

        # Pipeline finished with a real report (not crash/cancel) despite the
        # hostile probe name and task id.
        assert "Error" not in result and "Cancelled" not in result
        # The hostile id "T 1/x" was sanitized to a branch-safe slug and the
        # task completed + committed its file.
        assert report.completed_tasks, "no task completed"
        done_id = report.completed_tasks[0].id
        assert "/" not in done_id and " " not in done_id
        assert (repo / "tests" / "test_added.py").exists()
        assert MarkdownPlanExecutor().find_task_commit(project, done_id)
        # Full suite still green (integration gate did not revert), tree clean.
        assert (
            subprocess.run(
                [sys.executable, "-m", "pytest", "-q"],
                cwd=repo,
                capture_output=True,
                text=True,
            ).returncode
            == 0
        )
        assert (
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo,
                capture_output=True,
                text=True,
            ).stdout.strip()
            == ""
        )


def test_build_pipeline_offline_converges_and_writes_report(monkeypatch):
    # End-to-end through the real assessment + pipeline, fully offline (scripted
    # LLM, dynamic selection / free-model harvest / semantic retrieval off so no
    # network). Asserts the build converges with gates green and persists a
    # report file under .orchestrator/reports.
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    from misterdev.config import DEFAULT_CONFIG
    from misterdev.core.execution.project import Project
    from misterdev.core.modes import BuildMode, BuildFlags
    from misterdev.core.reporting.report import BuildReport
    from misterdev.agent import ProjectOrchestrator
    from datetime import datetime, timezone

    with tempfile.TemporaryDirectory() as td:
        repo = Path(td)
        _git(repo, "init")
        _git(repo, "config", "user.email", "t@t.t")
        _git(repo, "config", "user.name", "t")
        (repo / ".gitignore").write_text(".orchestrator/\n__pycache__/\n*.pyc\n")
        (repo / "tests").mkdir()
        (repo / "tests" / "test_seed.py").write_text(
            "def test_seed():\n    assert True\n"
        )
        (repo / "seed.py").write_text("X = 1\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-m", "init")

        cfg = json.loads(json.dumps(DEFAULT_CONFIG))
        cfg["name"] = "throwaway"
        cfg["build_command"] = "true"
        cfg["test_command"] = f"{sys.executable} -m pytest -q"
        cfg["llm"]["use_free_models"] = False
        cfg["llm"]["dynamic_selection"] = False
        cfg["llm"]["semantic_retrieval"] = False
        project = Project(repo, cfg)
        project.llm_client = _ScriptedLLM()

        orch = ProjectOrchestrator()
        assessment = orch._analyze(project, None)
        report = BuildReport(
            BuildMode.SMART, "throwaway", assessment, datetime.now(timezone.utc)
        )
        result = orch._run_pipeline(
            project,
            "add a passing test",
            BuildMode.SMART,
            BuildFlags(budget=100.0),
            assessment,
            None,
            report,
        )

        assert "Error" not in result and "Cancelled" not in result
        assert "PASSED" in result  # gates green
        reports = list((repo / ".orchestrator" / "reports").glob("report_*.md"))
        assert reports, "no report file was persisted"
        assert "Build Report" in reports[0].read_text()


# --- metacognition: LLM returning objects must not crash the audit -----------
def test_save_lessons_handles_dict_rules_without_crashing():
    from misterdev.core.planning.metacognition import SessionAuditor

    class _LLM:
        def generate_code(self, prompt, system_prompt=""):
            return "[]"

    with tempfile.TemporaryDirectory() as td:
        auditor = SessionAuditor(Path(td), _LLM())
        # The LLM returned objects, not strings -- previously crashed set() dedup.
        auditor._save_lessons(["use ruff", {"rule": "close db"}, "use ruff"])
        saved = json.loads(auditor.lessons_file.read_text())
        texts = [le["text"] for le in saved["lessons"]]
        assert "use ruff" in texts  # plain string kept
        assert any("close db" in t for t in texts)  # object coerced, not lost
        assert texts.count("use ruff") == 1  # deduped


# --- consolidated JSON-array extraction (was duplicated 4x) ------------------
def test_extract_json_array_handles_prose_fences_and_garbage():
    from misterdev.llm.responses import extract_json_array

    assert extract_json_array("Here: [1, 2, 3] done") == [1, 2, 3]
    assert extract_json_array('```json\n["a","b"]\n```') == ["a", "b"]
    assert extract_json_array("no array here") == []
    assert extract_json_array("[broken", default=None) == []
    assert extract_json_array("", default=["x"]) == ["x"]
    # An object, not an array -> default (rejects non-arrays cleanly)
    assert extract_json_array('{"k": 1}') == []


def test_lazy_topography_not_built_at_registration(monkeypatch):
    """Project construction must NOT eagerly build the symbol graph -- every
    CLI command registers all known projects, so eager scanning is wasted."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    from misterdev.config import DEFAULT_CONFIG
    from misterdev.core.execution.project import Project

    with tempfile.TemporaryDirectory() as td:
        repo = Path(td)
        (repo / "mod.py").write_text("def f():\n    return 1\n")
        cfg = json.loads(json.dumps(DEFAULT_CONFIG))
        cfg["name"] = "fixture"
        project = Project(repo, cfg)
        # Graph not built yet (lazy).
        assert project.topography._initialized is False
        # First explicit use builds it (idempotent).
        project.topography.initialize()
        assert project.topography._initialized is True


# --- convergence loop: outer execute+gate iteration --------------------------
class _CountingLLM:
    """Offline LLM for convergence tests. decompose_spec is monkeypatched so
    generate_code is only hit by best-effort phases (which are also stubbed)."""

    def __init__(self, budget_remaining=100.0):
        self.budget_remaining = budget_remaining
        self._budget = budget_remaining
        self.cumulative_usage = __import__(
            "misterdev.llm.client", fromlist=["LLMUsage"]
        ).LLMUsage()
        self.cost_by_task = {}

    def generate_code(self, prompt, system_prompt=""):
        return "spec"


class _ScriptedGate:
    """A GateKeeper stand-in returning a scripted (success, issues, health)
    per run_gates call so a test can drive fail-then-pass sequences."""

    sequence: list = []
    _calls = 0

    def __init__(self, *a, **k):
        pass

    def run_gates(self, commands):
        from misterdev.core.planning.assessment import HealthCheck

        idx = min(_ScriptedGate._calls, len(_ScriptedGate.sequence) - 1)
        success, issues = _ScriptedGate.sequence[idx]
        _ScriptedGate._calls += 1
        health = HealthCheck(
            builds=success,
            tests_pass=success,
            lint_clean=success,
            test_output="" if success else "boom",
        )
        return success, list(issues), health


def _run_convergence_pipeline(gate_sequence, max_iterations, budget=100.0):
    """Drive _run_pipeline in DEBUG mode (no LLM spec/probe calls) with a
    scripted gate and a counting _execute_tasks. Returns (report, exec_calls,
    decompose_calls)."""
    from unittest.mock import patch
    import misterdev.agent as agent_mod
    from misterdev.config import DEFAULT_CONFIG
    from misterdev.core.execution.project import Project
    from misterdev.core.planning.assessment import ProjectAssessment, HealthCheck
    from misterdev.core.modes import BuildMode, BuildFlags
    from misterdev.core.reporting.report import BuildReport
    from datetime import datetime, timezone

    _ScriptedGate.sequence = list(gate_sequence)
    _ScriptedGate._calls = 0

    counters = {"exec": 0, "decompose": 0}

    with tempfile.TemporaryDirectory() as td:
        repo = Path(td)
        cfg = json.loads(json.dumps(DEFAULT_CONFIG))
        cfg["name"] = "fixture"
        cfg["orchestrator"]["max_build_iterations"] = max_iterations
        project = Project(repo, cfg)
        project.llm_client = _CountingLLM(budget_remaining=budget)

        assessment = ProjectAssessment()
        assessment.structure.build_command = "true"
        assessment.structure.test_command = "true"
        assessment.health = HealthCheck(builds=True, tests_pass=True)

        report = BuildReport(
            BuildMode.DEBUG, "fixture", assessment, datetime.now(timezone.utc)
        )
        flags = BuildFlags(budget=budget)
        orch = agent_mod.ProjectOrchestrator()

        def fake_exec(tasks, project, flags, report):
            counters["exec"] += 1

        def fake_decompose(spec, assessment, mode, client, path, **kwargs):
            counters["decompose"] += 1
            return [_mock_task(f"FIX-{counters['decompose']}")]

        with (
            patch.object(orch, "_execute_tasks", side_effect=fake_exec),
            patch.object(agent_mod, "decompose_spec", side_effect=fake_decompose),
            patch.object(agent_mod, "topological_sort", side_effect=lambda x: x),
            patch.object(agent_mod, "GateKeeper", _ScriptedGate),
            patch.object(
                agent_mod.ProjectOrchestrator,
                "_maybe_rollback_regression",
                return_value=None,
            ),
            patch.object(agent_mod, "SessionAuditor") as MockAuditor,
        ):
            MockAuditor.return_value.get_lessons_context.return_value = ""
            MockAuditor.return_value.audit_session.return_value = None
            orch._run_pipeline(
                project, "fix things", BuildMode.DEBUG, flags, assessment, None, report
            )

    return report, counters["exec"], counters["decompose"]


def _run_convergence_pipeline_with_cfg(gate_sequence, orchestrator_cfg):
    """Like _run_convergence_pipeline but sets the orchestrator config block
    verbatim, so the absent-key default (single pass) can be exercised."""
    from unittest.mock import patch
    import misterdev.agent as agent_mod
    from misterdev.config import DEFAULT_CONFIG
    from misterdev.core.execution.project import Project
    from misterdev.core.planning.assessment import ProjectAssessment, HealthCheck
    from misterdev.core.modes import BuildMode, BuildFlags
    from misterdev.core.reporting.report import BuildReport
    from datetime import datetime, timezone

    _ScriptedGate.sequence = list(gate_sequence)
    _ScriptedGate._calls = 0
    counters = {"exec": 0, "decompose": 0}

    with tempfile.TemporaryDirectory() as td:
        repo = Path(td)
        cfg = json.loads(json.dumps(DEFAULT_CONFIG))
        cfg["name"] = "fixture"
        cfg["orchestrator"] = dict(orchestrator_cfg)
        project = Project(repo, cfg)
        project.llm_client = _CountingLLM()

        assessment = ProjectAssessment()
        assessment.structure.build_command = "true"
        assessment.structure.test_command = "true"
        assessment.health = HealthCheck(builds=True, tests_pass=True)

        report = BuildReport(
            BuildMode.DEBUG, "fixture", assessment, datetime.now(timezone.utc)
        )
        flags = BuildFlags(budget=100.0)
        orch = agent_mod.ProjectOrchestrator()

        def fake_exec(tasks, project, flags, report):
            counters["exec"] += 1

        def fake_decompose(spec, assessment, mode, client, path, **kwargs):
            counters["decompose"] += 1
            return [_mock_task(f"FIX-{counters['decompose']}")]

        with (
            patch.object(orch, "_execute_tasks", side_effect=fake_exec),
            patch.object(agent_mod, "decompose_spec", side_effect=fake_decompose),
            patch.object(agent_mod, "topological_sort", side_effect=lambda x: x),
            patch.object(agent_mod, "GateKeeper", _ScriptedGate),
            patch.object(
                agent_mod.ProjectOrchestrator,
                "_maybe_rollback_regression",
                return_value=None,
            ),
            patch.object(agent_mod, "SessionAuditor") as MockAuditor,
        ):
            MockAuditor.return_value.get_lessons_context.return_value = ""
            MockAuditor.return_value.audit_session.return_value = None
            orch._run_pipeline(
                project, "fix things", BuildMode.DEBUG, flags, assessment, None, report
            )

    return report, counters["exec"], counters["decompose"]


def test_convergence_single_pass_when_cap_is_one():
    # Gate fails, but cap=1 -> exactly one execute + one gate, no re-decompose.
    # decompose_calls counts the baseline Phase-3 decompose (1); a fix
    # re-decompose would push it to 2.
    report, exec_calls, decompose_calls = _run_convergence_pipeline(
        [(False, ["build broke"])], max_iterations=1
    )
    assert exec_calls == 1
    assert decompose_calls == 1  # baseline only; no fix re-decompose
    assert report.validation_passed is False
    assert not any("Convergence iteration" in d for d in report.key_decisions)


def test_convergence_passes_on_first_gate():
    report, exec_calls, decompose_calls = _run_convergence_pipeline(
        [(True, [])], max_iterations=3
    )
    assert exec_calls == 1
    assert decompose_calls == 1  # baseline Phase-3 decompose only
    assert report.validation_passed is True


def test_convergence_fail_then_pass_runs_second_iteration():
    # First gate red, second green: a second execute + re-decompose runs, build
    # converges, and the final report reflects the LAST (passing) gate.
    report, exec_calls, decompose_calls = _run_convergence_pipeline(
        [(False, ["tests fail"]), (True, [])], max_iterations=3
    )
    assert exec_calls == 2
    assert decompose_calls == 2  # baseline + one fix spec for iteration 2
    assert report.validation_passed is True
    assert any("Convergence iteration 2" in d for d in report.key_decisions)


def test_convergence_stops_at_iteration_cap():
    # Gate stays red with DIFFERENT issues each time so the no-progress guard
    # doesn't trip; the cap must bound the loop at max_iterations executes.
    report, exec_calls, decompose_calls = _run_convergence_pipeline(
        [
            (False, ["a"]),
            (False, ["b"]),
            (False, ["c"]),
            (False, ["d"]),
        ],
        max_iterations=3,
    )
    assert exec_calls == 3  # bounded by cap
    assert report.validation_passed is False


def test_convergence_stops_on_no_progress_identical_failures():
    # Identical issues on iterations 1 and 2 -> halt after the second gate even
    # though the cap allows more.
    report, exec_calls, decompose_calls = _run_convergence_pipeline(
        [(False, ["same"]), (False, ["same"]), (False, ["same"])],
        max_iterations=5,
    )
    assert exec_calls == 2
    assert any("identical gate failures" in d for d in report.key_decisions)


def test_convergence_stops_on_budget_exhaustion():
    # Cap allows more iterations, but budget is zero -> no second iteration.
    report, exec_calls, decompose_calls = _run_convergence_pipeline(
        [(False, ["x"]), (True, [])], max_iterations=5, budget=0.0
    )
    assert exec_calls == 1
    assert any(
        "budget exhausted before next iteration" in d for d in report.key_decisions
    )


def test_combine_commands():
    from misterdev.agent import _combine_commands

    assert (
        _combine_commands("pytest", "pytest tests/golden")
        == "(pytest) && (pytest tests/golden)"
    )
    assert _combine_commands("pytest", None) == "(pytest)"
    assert _combine_commands(None, "golden") == "(golden)"
    assert _combine_commands(None, None) is None
    assert _combine_commands("", None) is None


def test_check_golden_config_warns_on_half_configuration(caplog):
    import logging
    from misterdev.agent import _check_golden_config

    with caplog.at_level(logging.WARNING):
        _check_golden_config({"orchestrator": {"golden_command": "pytest g"}})
    assert any("not protected from edits" in r.message.lower() for r in caplog.records)

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        _check_golden_config({"orchestrator": {"golden_paths": ["tests/golden/"]}})
    assert any("never run as a gate" in r.message.lower() for r in caplog.records)

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        # Both set (consistent) or both empty -> no warning.
        _check_golden_config(
            {"orchestrator": {"golden_paths": ["g/"], "golden_command": "pytest g"}}
        )
        _check_golden_config({"orchestrator": {}})
    assert not caplog.records


def test_apply_budget_ceiling_takes_tighter_cap():
    from misterdev.agent import _apply_budget_ceiling

    class _C:
        pass

    c = _C()
    c._budget = 20.0
    _apply_budget_ceiling(c, 100.0)  # config cap tighter than CLI default
    assert c._budget == 20.0

    c._budget = 100.0
    _apply_budget_ceiling(c, 5.0)  # CLI flag tighter than config
    assert c._budget == 5.0

    c._budget = object()  # non-numeric (test double) -> use flag
    _apply_budget_ceiling(c, 7.0)
    assert c._budget == 7.0


def test_warn_if_baseline_broken_records_only_on_failure():
    from datetime import datetime, timezone
    from misterdev.agent import _warn_if_baseline_broken
    from misterdev.core.planning.assessment import ProjectAssessment, HealthCheck
    from misterdev.core.modes import BuildMode
    from misterdev.core.reporting.report import BuildReport

    def _report(builds):
        a = ProjectAssessment()
        a.health = HealthCheck(builds=builds, build_output="error: mismatched types")
        r = BuildReport(BuildMode.COMPLETE, "p", a, datetime.now(timezone.utc))
        _warn_if_baseline_broken(a, r)
        return r

    failed = _report(False)
    assert any("baseline build was failing" in d for d in failed.key_decisions)

    ok = _report(True)
    assert not any("baseline build" in d for d in ok.key_decisions)


def test_enable_ab_mcts_default_off():
    from misterdev.config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["orchestrator"]["enable_ab_mcts"] is False


def test_warn_if_no_test_gate_silent_for_multi_target():
    import types
    from misterdev.agent import _warn_if_no_test_gate
    from misterdev.core.reporting.report import BuildReport
    from misterdev.core.planning.assessment import (
        ProjectAssessment,
        HealthCheck,
        ProjectStructure,
        TechnicalDebt,
        RiskAssessment,
    )
    from misterdev.core.modes import BuildMode
    from datetime import datetime, timezone

    def _assess():
        return ProjectAssessment(
            structure=ProjectStructure(project_type="monorepo", test_command=None),
            health=HealthCheck(builds=True),
            tech_debt=TechnicalDebt(score=5),
            risk=RiskAssessment(level="low"),
        )

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "tests").mkdir()
        (root / "tests" / "a.test.js").write_text("", encoding="utf-8")
        # No top-level test command, but a target declares one -> no warning.
        proj = types.SimpleNamespace(
            path=root,
            config={
                "targets": [{"name": "web", "path": "web", "build_command": "tsc"}]
            },
        )
        rep = BuildReport(
            BuildMode.SMART, "x", _assess(), datetime(2026, 1, 1, tzinfo=timezone.utc)
        )
        _warn_if_no_test_gate(_assess(), proj, rep)
        assert not any("No test gate" in d for d in rep.degraded_subsystems)
