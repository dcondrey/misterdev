"""Regression tests for the devplan correctness fixes.

Covers: atomic writes, LLM-edit path validation, secret-scan false-positive
reduction, new error-classifier categories, validation SKIP status, formatter
path handling, per-task change attribution, opt-in file-overlap dependencies,
ephemeral cleanup, and report persistence.
"""

import json
import subprocess
import tempfile
from pathlib import Path

import pytest

from my_project_orchestrator.utils.file_utils import atomic_write
from my_project_orchestrator.task_executors.markdown_plan_executor import (
    MarkdownPlanExecutor,
    _detect_language,
    _LANG_MAP,
)
from my_project_orchestrator.core.gatekeeper import SOTAGateKeeper
from my_project_orchestrator.core.error_classifier import classify_error, ErrorCategory
from my_project_orchestrator.core.validator import ValidationResult, SOTAValidator
from my_project_orchestrator.core.change_tracker import ChangeTracker
from my_project_orchestrator.core.sovereign import EphemeralCodeManager


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
    assert SOTAGateKeeper._is_secret_assignment('api_key = "abcdef123456"')
    # Ordinary source constructs -> not flagged.
    assert not SOTAGateKeeper._is_secret_assignment("token: String,")
    assert not SOTAGateKeeper._is_secret_assignment("let token = get_token()")
    assert not SOTAGateKeeper._is_secret_assignment('api_key = os.environ["API_KEY"]')


def test_secret_scan_ignores_code_keeps_high_signal():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "engine.rs").write_text("pub struct S { token: String, secret: u64 }\n")
        (root / "leak.py").write_text('KEY = "sk-abc123def"\n')
        gk = SOTAGateKeeper(root)
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
    ok, err = SOTAValidator.validate_code('x=$(echo "hi")\n', language="shell")
    assert ok and err is None


# --- formatter path handling ------------------------------------------------


def test_formatter_runs_project_wide_without_placeholder():
    from unittest.mock import patch
    from my_project_orchestrator.tools.formatter import FormatterTool

    tool = FormatterTool.__new__(FormatterTool)
    tool.config = {"command": "ruff format ."}
    with patch(
        "my_project_orchestrator.tools.command.CommandTool.execute",
        autospec=True,
        return_value=(True, ""),
    ) as m:
        tool.execute(_FakeProject("."), file_path="ignored.py")
    # No {path} placeholder -> command runs as-is, not per file.
    assert m.call_args.kwargs.get("command") == "ruff format ."


def test_formatter_substitutes_path_when_placeholder_present():
    from unittest.mock import patch
    from my_project_orchestrator.tools.formatter import FormatterTool

    tool = FormatterTool.__new__(FormatterTool)
    tool.config = {"command": "rustfmt {path}"}
    with patch(
        "my_project_orchestrator.tools.command.CommandTool.execute",
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
    from my_project_orchestrator.core.task import TaskManager

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
    from my_project_orchestrator.task_executors.markdown_plan_executor import (
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
    from my_project_orchestrator.task_executors.markdown_plan_executor import (
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
    from my_project_orchestrator.core.sovereign import ProbeGenerator

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
    import my_project_orchestrator.agent as agent_mod

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


# --- streaming with early abort (028) ---------------------------------------


def test_code_gen_abort_check():
    from my_project_orchestrator.llm.client import code_gen_abort_check

    assert code_gen_abort_check("I'll help you write this function...")
    assert code_gen_abort_check("x" * 2500)  # long, no code fence
    assert not code_gen_abort_check("```python\ncode\n```")
    assert not code_gen_abort_check("short")


def test_generate_stream_aborts_early():
    from my_project_orchestrator.llm.client import BaseLLMClient, code_gen_abort_check

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
    from my_project_orchestrator.task_executors.markdown_plan_executor import (
        _bisect_first_failing,
    )

    # passes, passes, FAILS, fails  -> first failing index is 2
    states = [True, True, False, False]
    assert _bisect_first_failing(len(states), lambda i: states[i]) == 2
    allpass = [True, True, True]
    # nothing fails -> returns last index (caller re-checks)
    assert _bisect_first_failing(len(allpass), lambda i: allpass[i]) == 2


def test_bisect_regression_end_to_end_git():
    from my_project_orchestrator.task_executors.markdown_plan_executor import (
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
    from my_project_orchestrator.llm.client import AnthropicLLMClient

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
    from my_project_orchestrator.llm.client import BaseLLMClient, LLMResponse, LLMUsage

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
    from my_project_orchestrator.llm.client import BaseLLMClient

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
    from my_project_orchestrator.llm.client import FailoverLLMClient, BaseLLMClient

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
        from my_project_orchestrator.llm.client import LLMResponse, LLMUsage

        if self.behavior == "ok":
            return LLMResponse(content="ok", model=self.model, usage=LLMUsage())
        raise self.behavior


def test_failover_advances_on_retryable_error():
    from my_project_orchestrator.llm.client import LLMCallError

    fc = _make_failover(
        _StubClient(LLMCallError("503 overloaded", retryable=True), model="primary"),
        [_StubClient("ok", model="backup")],
    )
    resp = fc._call("p", "s")
    assert resp.content == "ok"
    assert resp.model == "backup"


def test_failover_stops_on_non_retryable():
    from my_project_orchestrator.llm.client import LLMCallError

    fc = _make_failover(
        _StubClient(LLMCallError("400 bad request", retryable=False), model="primary"),
        [_StubClient("ok", model="backup")],
    )
    with pytest.raises(LLMCallError):
        fc._call("p", "s")  # non-retryable must not fall through to backup


def test_failover_factory_wraps_when_configured(monkeypatch):
    from my_project_orchestrator.llm import client as cl

    monkeypatch.setattr(cl, "_create_single_client", lambda cfg: _StubClient("ok"))
    wrapped = cl.create_llm_client(
        {"llm": {"provider": "openrouter", "failover": [{"provider": "anthropic"}]}}
    )
    assert isinstance(wrapped, cl.FailoverLLMClient)
    plain = cl.create_llm_client({"llm": {"provider": "openrouter"}})
    assert not isinstance(plain, cl.FailoverLLMClient)


# --- incremental rerun (022) -------------------------------------------------


def test_progress_needs_rerun_on_hash_change():
    from my_project_orchestrator.core.progress import ProgressTracker

    with tempfile.TemporaryDirectory() as td:
        pt = ProgressTracker(Path(td))
        assert pt.needs_rerun("T-1", "h1")  # never completed
        pt.mark_completed("T-1", "h1")
        assert not pt.needs_rerun("T-1", "h1")  # same hash -> skip
        assert pt.needs_rerun("T-1", "h2")  # changed -> rerun
        # Hash persists across reloads.
        pt2 = ProgressTracker(Path(td))
        assert not pt2.needs_rerun("T-1", "h1")


def test_compute_task_hash_changes_with_spec():
    from my_project_orchestrator.core.progress import compute_task_hash

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        spec = root / "task.md"
        spec.write_text("v1")

        class _T:
            source_ref = str(spec)
            files_to_modify = []

        h1 = compute_task_hash(_T(), root)
        spec.write_text("v2 changed")
        h2 = compute_task_hash(_T(), root)
        assert h1 != h2


# --- tree-sitter rust contract extraction (013) -----------------------------


def test_rust_contracts_tree_sitter_multiline():
    from my_project_orchestrator.core.contracts import _extract_public_symbols

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
    from my_project_orchestrator.core.topography import SymbolGraph, _get_ts_parsers

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
    from my_project_orchestrator.analyzers.project_analyzer import _walk_limited

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
    from my_project_orchestrator.core.preflight import PreflightValidator
    from my_project_orchestrator.core.models import Task

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
    from my_project_orchestrator.core.preflight import PreflightValidator
    from my_project_orchestrator.core.models import Task

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
    import my_project_orchestrator.agent as agent_mod

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
        patch(
            "my_project_orchestrator.agent.topological_sort", side_effect=lambda x: x
        ),
        patch("my_project_orchestrator.agent.Scratchpad"),
        patch("my_project_orchestrator.agent.ContractRegistry"),
        patch("my_project_orchestrator.agent.ChangeTracker"),
        patch("my_project_orchestrator.agent.StrategyOptimizer") as MockStrat,
        patch("my_project_orchestrator.agent.ProgressTracker") as MockProg,
        patch("my_project_orchestrator.agent.MarkdownPlanExecutor") as MockExec,
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


# --- report persistence -----------------------------------------------------


def test_report_save_writes_markdown_and_json():
    from datetime import datetime, timezone
    from my_project_orchestrator.core.report import BuildReport
    from my_project_orchestrator.core.assessment import ProjectAssessment
    from my_project_orchestrator.core.modes import BuildMode

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
    from my_project_orchestrator.analyzers.project_analyzer import detect_test_command

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "tests").mkdir()
        (root / "uv.lock").write_text("", encoding="utf-8")
        assert detect_test_command(root) == "uv run pytest -q"


def test_detect_test_command_pytest_no_uv():
    from my_project_orchestrator.analyzers.project_analyzer import detect_test_command

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "pyproject.toml").write_text(
            "[tool.pytest.ini_options]\n", encoding="utf-8"
        )
        assert detect_test_command(root) == "pytest -q"


def test_detect_test_command_npm_and_cargo_and_none():
    from my_project_orchestrator.analyzers.project_analyzer import detect_test_command

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


def test_analyze_project_fills_test_command_when_llm_returns_null():
    """The real regression: LLM left test_command null -> tests=none. The
    deterministic fallback must populate the assessment so the suite runs."""
    from unittest.mock import patch
    from my_project_orchestrator.analyzers import project_analyzer
    from my_project_orchestrator.core.assessment import HealthCheck

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
    from my_project_orchestrator.llm.client import BaseLLMClient, LLMResponse, LLMUsage

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
    from my_project_orchestrator.agent import ProjectOrchestrator
    from my_project_orchestrator.task_executors.markdown_plan_executor import (
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
            project, "python -m pytest -q", timeout=120
        )
        assert not ok_before

        reverted = orch._integration_gate(
            project, executor, "python -m pytest -q", [task], timeout=120
        )
        assert reverted == ["T-bad"]
        ok_after, _ = executor._run_command(project, "python -m pytest -q", timeout=120)
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
    from my_project_orchestrator.agent import ProjectOrchestrator
    from my_project_orchestrator.task_executors.markdown_plan_executor import (
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
            project, executor, "python -m pytest -q", [task], timeout=120
        )
        assert reverted == []


# --- interactive planner: advisor + goal selection --------------------------
def test_recommend_work_parses_and_normalizes():
    from my_project_orchestrator.core.advisor import recommend_work, Recommendation
    from my_project_orchestrator.core.assessment import ProjectAssessment

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
    from my_project_orchestrator.core.advisor import recommend_work
    from my_project_orchestrator.core.assessment import ProjectAssessment

    class _LLM:
        def generate_code(self, prompt, system_prompt=""):
            return "no json here"

    assert recommend_work(ProjectAssessment(), _LLM()) == []


def test_choose_goal_number_text_and_quit():
    from unittest.mock import patch
    from my_project_orchestrator.agent import ProjectOrchestrator
    from my_project_orchestrator.core.advisor import Recommendation
    from my_project_orchestrator.core.modes import BuildMode

    orch = ProjectOrchestrator()
    recs = [
        Recommendation("Fix imports", "red suite", "debug"),
        Recommendation("Add feature X", "users want it", "feature"),
    ]

    with patch("my_project_orchestrator.agent.Prompt.ask", return_value="1"):
        goal, mode = orch._choose_goal(recs)
    assert goal == "Fix imports" and mode == BuildMode.DEBUG

    with patch(
        "my_project_orchestrator.agent.Prompt.ask", return_value="make it faster"
    ):
        goal, mode = orch._choose_goal(recs)
    assert goal == "make it faster"

    with patch("my_project_orchestrator.agent.Prompt.ask", return_value="q"):
        goal, mode = orch._choose_goal(recs)
    assert goal is None


def test_interactive_plan_cancels_when_no_goal():
    from unittest.mock import patch, MagicMock
    from my_project_orchestrator.agent import ProjectOrchestrator
    from my_project_orchestrator.core.assessment import ProjectAssessment

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
            "my_project_orchestrator.agent.analyze_project",
            return_value=ProjectAssessment(),
        ),
        patch("my_project_orchestrator.agent.recommend_work", return_value=[]),
        patch.object(orch, "_choose_goal", return_value=(None, None)),
        patch.object(orch, "_run_pipeline") as mock_pipeline,
    ):
        result = orch.interactive_plan("/tmp/x")
    assert "Cancelled" in result
    mock_pipeline.assert_not_called()


def test_interactive_plan_runs_pipeline_with_confirm():
    from unittest.mock import patch, MagicMock
    from my_project_orchestrator.agent import ProjectOrchestrator
    from my_project_orchestrator.core.assessment import ProjectAssessment
    from my_project_orchestrator.core.modes import BuildMode

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
            "my_project_orchestrator.agent.analyze_project",
            return_value=ProjectAssessment(),
        ),
        patch("my_project_orchestrator.agent.recommend_work", return_value=[]),
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
    from my_project_orchestrator.core.validator import _parse_test_counts

    assert _parse_test_counts("332 passed in 34.18s") == (332, 0)
    assert _parse_test_counts("3 failed, 317 passed in 53s") == (320, 3)
    assert _parse_test_counts("5 passed, 2 skipped") == (5, 0)
    assert _parse_test_counts("test result: ok. 42 passed; 0 failed") == (42, 0)
    assert _parse_test_counts("no recognizable summary") == (0, 0)


def test_run_health_check_populates_test_count():
    from my_project_orchestrator.core.validator import run_health_check

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "test_x.py").write_text(
            "def test_a():\n    assert True\n", encoding="utf-8"
        )
        health = run_health_check(root, None, "python -m pytest -q", None)
        assert health.test_count == 1 and health.test_failures == 0
        assert health.tests_pass


def test_health_ground_truth_string():
    from my_project_orchestrator.analyzers.project_analyzer import _health_ground_truth
    from my_project_orchestrator.core.assessment import HealthCheck

    h = HealthCheck(builds=True, tests_pass=True, test_count=332, test_failures=0)
    g = _health_ground_truth(h)
    assert "build passes" in g and "332/332 tests passing" in g


# --- stale cross-build progress no longer causes spurious skips --------------
def test_task_hash_reflects_content_for_idless_llm_tasks():
    from types import SimpleNamespace
    from my_project_orchestrator.core.progress import compute_task_hash

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
    from my_project_orchestrator.core.progress import ProgressTracker, compute_task_hash

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
    from my_project_orchestrator.core.sovereign import EphemeralCodeManager

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
    from my_project_orchestrator.utils.file_utils import safe_ref_slug

    assert (
        safe_ref_slug("CLI Runner / Invocation Probe") == "CLI_Runner_Invocation_Probe"
    )
    assert safe_ref_slug("T 001") == "T_001"
    assert safe_ref_slug("feat:auth~bug") == "feat_auth_bug"
    assert safe_ref_slug("T-001") == "T-001"  # already-clean ids untouched
    assert safe_ref_slug("...", fallback="x") == "x"
    assert safe_ref_slug("../escape") == "escape"


def test_decompose_sanitizes_task_ids_and_deps():
    from my_project_orchestrator.core.decomposer import decompose_spec
    from my_project_orchestrator.core.assessment import ProjectAssessment
    from my_project_orchestrator.core.modes import BuildMode

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
def _fake_project(repo: Path, monkeypatch, edit_response: str):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    from my_project_orchestrator.config import DEFAULT_CONFIG
    from my_project_orchestrator.core.project import Project
    from tests.test_llm_client import FakeLLMClient
    from my_project_orchestrator.llm.client import LLMResponse, LLMUsage

    cfg = json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy
    cfg["name"] = "fixture"
    cfg["build_command"] = "true"  # always-passing per-task build check
    cfg.pop("test_command", None)
    project = Project(repo, cfg)
    fake = FakeLLMClient(
        responses=[LLMResponse(content=edit_response, usage=LLMUsage())] * 4
    )
    project.llm_client = fake
    return project


def test_executor_execute_commits_real_and_out_of_scope_files(monkeypatch):
    """End-to-end: a real LLM-edit response is applied, BOTH the declared file
    and an out-of-scope-but-in-root file are committed, and the commit is
    findable by the integration gate. No live API, no git detach, no orphans."""
    from types import SimpleNamespace
    from my_project_orchestrator.task_executors.markdown_plan_executor import (
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
            processor_data={"sota_strategy": "surgical"},
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
    from my_project_orchestrator.agent import ProjectOrchestrator
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
    from my_project_orchestrator.agent import ProjectOrchestrator

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
    from my_project_orchestrator.task_executors.markdown_plan_executor import (
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
            cwd=repo, capture_output=True, text=True,
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
        assert subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo,
            capture_output=True, text=True,
        ).stdout.strip() == ""
        assert (repo / "task_file.py").read_text() == "NEW = 1\n"  # task change kept
        assert (repo / "other.py").read_text() == "X=1\n"  # spillover reverted
