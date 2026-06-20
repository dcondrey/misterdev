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
    MarkdownPlanExecutor, _detect_language, _LANG_MAP,
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
    assert classify_error("error: failed to parse manifest at Cargo.toml") == ErrorCategory.MANIFEST


def test_classify_file_not_found():
    assert classify_error("OSError: No such file or directory: 'x'") == ErrorCategory.FILE_NOT_FOUND


# --- validation SKIP status -------------------------------------------------

def test_validation_summary_skip_for_unrun_gates():
    v = ValidationResult()
    v.build_ran, v.build_ok = True, True
    v.tests_ran = False           # no test command configured
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
        autospec=True, return_value=(True, ""),
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
        autospec=True, return_value=(True, ""),
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
        "files_to_modify:\n" + "".join(f"- {m}\n" for m in modify) +
        "---\nbody\n"
    )
    (devplan / f"{tid}.md").write_text(body)


def _make_tm(root: Path, auto: bool):
    from my_project_orchestrator.core.task import TaskManager

    class _P:
        path = root
        config = {"devplan_dir": "devplan",
                  "orchestrator": {"auto_detect_dependencies": auto}}
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
    from my_project_orchestrator.task_executors.markdown_plan_executor import MarkdownPlanExecutor

    class _P:
        env_manager = None

    class _T:
        id = "T-1"
        title = "do thing"
        description = "d"

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        _git(td, "init"); _git(td, "config", "user.email", "t@t.t"); _git(td, "config", "user.name", "t")
        (td / "seed.txt").write_text("x")
        _git(td, "add", "-A"); _git(td, "commit", "-m", "init")
        (td / "task_file.txt").write_text("the task's edit")
        (td / "user_unrelated.txt").write_text("uncommitted user work")  # untracked, NOT the task's

        proj = _P(); proj.path = td
        MarkdownPlanExecutor()._commit_task(proj, None, None, _T(), ["task_file.txt"])

        committed = subprocess.run(["git", "show", "--name-only", "--format=", "HEAD"],
                                   cwd=td, capture_output=True, text=True).stdout
        assert "task_file.txt" in committed
        assert "user_unrelated.txt" not in committed
        # The unrelated file is untouched, still untracked.
        status = subprocess.run(["git", "status", "--porcelain"], cwd=td, capture_output=True, text=True).stdout
        assert "?? user_unrelated.txt" in status


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
        _git(td, "init"); _git(td, "config", "user.email", "t@t.t"); _git(td, "config", "user.name", "t")
        (td / "base.txt").write_text("base\n")
        _git(td, "add", "-A"); _git(td, "commit", "-m", "init")

        project = MagicMock()
        project.path = td
        project.config = {"orchestrator": {"parallel_mode": "worktree", "max_workers": 2}}

        class FakeExec:
            def execute(self, task, proj, use_git_branch=True):
                # Runs inside the task's worktree; write + commit there.
                f = Path(proj.path) / f"{task.id}.txt"
                f.write_text(task.id)
                _git(proj.path, "add", "-A")
                _git(proj.path, "commit", "-m", f"task({task.id}): work")
                r = MagicMock(); r.status = "completed"; return r

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
    assert code_gen_abort_check("x" * 2500)                       # long, no code fence
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
    from my_project_orchestrator.task_executors.markdown_plan_executor import _bisect_first_failing
    # passes, passes, FAILS, fails  -> first failing index is 2
    states = [True, True, False, False]
    assert _bisect_first_failing(len(states), lambda i: states[i]) == 2
    allpass = [True, True, True]
    # nothing fails -> returns last index (caller re-checks)
    assert _bisect_first_failing(len(allpass), lambda i: allpass[i]) == 2


def test_bisect_regression_end_to_end_git():
    from my_project_orchestrator.task_executors.markdown_plan_executor import MarkdownPlanExecutor

    class _P:
        env_manager = None

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        _git(td, "init")
        _git(td, "config", "user.email", "t@t.t")
        _git(td, "config", "user.name", "t")
        (td / "f.txt").write_text("clean\n")
        _git(td, "add", "-A"); _git(td, "commit", "-m", "init")
        # T-A: still clean
        (td / "f.txt").write_text("clean\nA\n")
        _git(td, "add", "-A"); _git(td, "commit", "-m", "task(T-A): a")
        # T-B: introduces BUG (the regression)
        (td / "f.txt").write_text("clean\nA\nBUG\n")
        _git(td, "add", "-A"); _git(td, "commit", "-m", "task(T-B): b")
        # T-C: more clean work
        (td / "f.txt").write_text("clean\nA\nBUG\nC\n")
        _git(td, "add", "-A"); _git(td, "commit", "-m", "task(T-C): c")

        proj = _P(); proj.path = td
        ex = MarkdownPlanExecutor()
        commits = [(tid, ex.find_task_commit(proj, tid)) for tid in ("T-A", "T-B", "T-C")]
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
    assert abs(cached - 1.5) < 1e-9          # cache read = 10% of input
    creation = c._estimate_cost(0, 0, cache_creation=1_000_000, cache_read=0)
    assert abs(creation - 18.75) < 1e-9      # cache write = 125% of input


# --- per-task cost attribution (020) ----------------------------------------

def test_cost_attributed_per_task():
    from my_project_orchestrator.llm.client import BaseLLMClient, LLMResponse, LLMUsage

    class _C(BaseLLMClient):
        def __init__(self):
            super().__init__({"build": {"budget": 100.0}})
            self.model = "m"

        def _call(self, p, s):
            return LLMResponse(content="x", model="m",
                               usage=LLMUsage(total_tokens=10, estimated_cost=0.5))

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
    cfg = {"llm": {
        "models": {"simple": "haiku", "complex": "opus", "default": "sonnet"},
        "routing": {"small": "simple", "large": "complex", "surgical": "simple"},
    }}
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
    assert MarkdownPlanExecutor()._resolve_model(_FakeProject(".", {}), _T(), "x") is None


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
    wrapped = cl.create_llm_client({"llm": {"provider": "openrouter",
                                            "failover": [{"provider": "anthropic"}]}})
    assert isinstance(wrapped, cl.FailoverLLMClient)
    plain = cl.create_llm_client({"llm": {"provider": "openrouter"}})
    assert not isinstance(plain, cl.FailoverLLMClient)


# --- incremental rerun (022) -------------------------------------------------

def test_progress_needs_rerun_on_hash_change():
    from my_project_orchestrator.core.progress import ProgressTracker
    with tempfile.TemporaryDirectory() as td:
        pt = ProgressTracker(Path(td))
        assert pt.needs_rerun("T-1", "h1")          # never completed
        pt.mark_completed("T-1", "h1")
        assert not pt.needs_rerun("T-1", "h1")      # same hash -> skip
        assert pt.needs_rerun("T-1", "h2")          # changed -> rerun
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
        assert "junk.js" not in found        # node_modules pruned
        assert "deep.py" not in found        # beyond max_depth


# --- preflight validation (023) ---------------------------------------------

def test_preflight_flags_dangling_dependency():
    from my_project_orchestrator.core.preflight import PreflightValidator
    from my_project_orchestrator.core.models import Task

    good = Task(id="001-a", description="d", type="markdown_planner", project_ref=".", dependencies=[])
    bad = Task(id="002-b", description="d", type="markdown_planner", project_ref=".", dependencies=["999-missing"])
    with tempfile.TemporaryDirectory() as td:
        issues = PreflightValidator().validate([good, bad], Path(td))
    assert PreflightValidator.has_errors(issues)
    assert any("999-missing" in i.message and i.severity == "error" for i in issues)


def test_preflight_clean_plan_has_no_errors():
    from my_project_orchestrator.core.preflight import PreflightValidator
    from my_project_orchestrator.core.models import Task

    a = Task(id="001-a", description="d", type="markdown_planner", project_ref=".", dependencies=[])
    b = Task(id="002-b", description="d", type="markdown_planner", project_ref=".", dependencies=["001-a"])
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
    project.config = {"language": "python", "orchestrator": {"max_consecutive_failures": 3}}
    project.task_manager.get_pending_tasks.return_value = tasks

    completed_result = MagicMock()
    completed_result.status = "completed"

    orch = agent_mod.ProjectOrchestrator()
    with (
        patch.object(agent_mod.ProjectOrchestrator, "_get_or_register", return_value=project),
        patch("my_project_orchestrator.agent.topological_sort", side_effect=lambda x: x),
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
        report = BuildReport(BuildMode.COMPLETE, "proj", ProjectAssessment(),
                             datetime.now(timezone.utc))
        report.finalize()
        path = report.save(root)
        assert path is not None and path.exists()
        json_path = path.with_suffix(".json")
        data = json.loads(json_path.read_text())
        assert data["project"] == "proj"
