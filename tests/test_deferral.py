"""Walk-away deferral: park tasks that need human input instead of failing."""

from pathlib import Path
from types import SimpleNamespace

from misterdev.core.execution.blocker import blocked_reason
from misterdev.core.execution.deferral import DeferralBook
from misterdev.core.models import Task
from misterdev.task_executors.markdown_plan_executor import MarkdownPlanExecutor
from misterdev.task_executors.markdown_plan_executor.helpers import _extract_needs_input


# --- blocker classifier: external-resource signals vs real code errors -------


def test_blocked_reason_flags_external_resources():
    for out in (
        "Error: Not logged in. Run `wrangler login`.",
        "HTTP 401 Unauthorized",
        "403 Forbidden: account lacks permission",
        "CLOUDFLARE_API_TOKEN is required",
        "ANTHROPIC_API_KEY not set",
        "getaddrinfo ENOTFOUND api.example.com",
        "permission denied",
    ):
        assert blocked_reason(out), out


def test_blocked_reason_ignores_real_code_errors():
    for out in (
        "AssertionError: expected 3 got 4",
        "SyntaxError: unexpected token",
        "TypeError: x is not a function",
        "2 failed, 5 passed in 0.3s",
        "error[E0308]: mismatched types",
    ):
        assert blocked_reason(out) is None, out


# --- NEEDS_INPUT model marker -------------------------------------------------


def test_extract_needs_input():
    assert _extract_needs_input("prose\nNEEDS_INPUT: Which DB should I use?\nx") == (
        "Which DB should I use?"
    )
    assert _extract_needs_input("NEEDS-INPUT - pick a name") == "pick a name"
    assert _extract_needs_input("no marker here") is None
    assert _extract_needs_input("") is None


# --- DeferralBook: QUESTIONS.md round-trip, answer persistence ----------------


def test_deferral_book_writes_and_reads_answers(tmp_path: Path):
    book = DeferralBook(tmp_path)
    book.write(
        [
            {
                "id": "T004",
                "title": "Env",
                "reason": "blocked: needs CF creds",
                "questions": ["Provide credentials or say skip."],
            },
            {
                "id": "T038",
                "title": "Security review",
                "reason": "judgment task",
                "questions": ["Is the salt rotation acceptable?"],
            },
        ]
    )
    assert book.md_path.exists()
    assert book.load_answers() == {}  # placeholders, nothing answered yet
    # User fills one answer.
    txt = book.md_path.read_text(encoding="utf-8").replace(
        "_(write your answer here)_", "skip the deploy checks", 1
    )
    book.md_path.write_text(txt, encoding="utf-8")
    ans = book.load_answers()
    assert ans.get("T004") == "skip the deploy checks" and "T038" not in ans
    # Re-writing the book preserves the typed answer for a still-parked task.
    book.write([{"id": "T004", "title": "Env", "reason": "x", "questions": ["q"]}])
    assert book.load_answers().get("T004") == "skip the deploy checks"


def test_deferral_book_empty_is_noop(tmp_path: Path):
    book = DeferralBook(tmp_path)
    book.write([])
    assert not book.md_path.exists()
    assert book.load_answers() == {}


# --- executor deferral decision + terminal seam ------------------------------


def _task():
    return Task(id="T1", description="do a thing", project_ref="p", title="Do a thing")


def test_deferral_reason_three_shapes():
    e = MarkdownPlanExecutor()
    # Blocked on an external resource.
    reason, q = e._deferral_reason(_task(), "Error: 401 Unauthorized", has_gate=True)
    assert reason.startswith("blocked") and "re-run" in q
    # Judgment/review task (no automated gate).
    reason, q = e._deferral_reason(_task(), "", has_gate=False)
    assert "no automated verification" in reason and "review" in q
    # Genuine inability with a gate.
    reason, q = e._deferral_reason(_task(), "boom: could not compile", has_gate=True)
    assert "could not complete" in reason and "How should I proceed" in q


def test_defer_task_returns_deferred_status_with_questions():
    e = MarkdownPlanExecutor()
    statuses = []
    project = SimpleNamespace(
        task_manager=SimpleNamespace(
            update_task_status=lambda tid, s: statuses.append(s)
        )
    )
    result = e._defer_task(project, _task(), "needs input", ["a question", ""])
    assert result.status == "deferred"
    assert result.questions == ["a question"]  # empties filtered
    assert statuses == ["deferred"]


# --- run_project: deferral is non-blocking and defers dependents -------------


def test_run_project_deferral_is_nonblocking(tmp_path: Path, monkeypatch):
    from misterdev.agent import ProjectOrchestrator
    from misterdev.core.models import ExecutionResult

    def mk(tid, deps=()):
        return Task(
            id=tid,
            description=tid,
            project_ref="p",
            title=tid,
            dependencies=list(deps),
        )

    # T003 depends on the (soon-parked) T001 and the completed T002.
    tasks = [mk("T001"), mk("T002"), mk("T003", ["T001", "T002"])]
    task_mgr = SimpleNamespace(
        discover_tasks=lambda: None,
        get_pending_tasks=lambda: tasks,
        update_task_status=lambda *a: None,
    )
    project = SimpleNamespace(
        path=tmp_path,
        name="proj",
        config={},
        env_manager=None,
        task_manager=task_mgr,
        llm_client=None,
    )
    orch = ProjectOrchestrator()
    monkeypatch.setattr(orch, "_get_or_register", lambda p: project)
    monkeypatch.setattr(orch, "_inject_task_context", lambda *a, **k: None)

    def fake_execute(self, task, project, **kw):
        # T001 parks on a missing credential; T002 completes normally.
        if task.id == "T001":
            return ExecutionResult(
                status="deferred",
                message="blocked: needs Cloudflare credentials",
                questions=["Provide CLOUDFLARE_API_TOKEN or say skip."],
            )
        return ExecutionResult(status="completed", message="ok")

    monkeypatch.setattr(MarkdownPlanExecutor, "execute", fake_execute)

    # Must not raise / abort: the run completes, parking T001 and (transitively) T003.
    orch.run_project(tmp_path, skip_preflight=True)

    md = (tmp_path / ".orchestrator" / "QUESTIONS.md").read_text(encoding="utf-8")
    assert "T001" in md  # the directly-parked task
    assert "T003" in md and "blocked by parked task" in md  # deferred dependent
    # T002 (independent) still completed despite T001 parking.
    progress = (tmp_path / ".orchestrator" / "progress.json").read_text(
        encoding="utf-8"
    )
    assert "T002" in progress


def test_run_project_parallel_path_runs_and_records_wave(tmp_path, monkeypatch):
    from misterdev.agent import ProjectOrchestrator
    from misterdev.core.models import ExecutionResult

    def mk(tid, deps=()):
        return Task(
            id=tid, description=tid, project_ref="p", title=tid, dependencies=list(deps)
        )

    tasks = [mk("T1"), mk("T2"), mk("T3")]  # independent wave
    task_mgr = SimpleNamespace(
        discover_tasks=lambda: None,
        get_pending_tasks=lambda: tasks,
        update_task_status=lambda *a: None,
    )
    # run_parallel on; no .git in tmp so _execute_parallel uses shared/thread mode.
    project = SimpleNamespace(
        path=tmp_path,
        name="p",
        config={"orchestrator": {"run_parallel": True, "max_workers": 2}},
        env_manager=None,
        task_manager=task_mgr,
        llm_client=None,
    )
    orch = ProjectOrchestrator()
    monkeypatch.setattr(orch, "_get_or_register", lambda p: project)
    monkeypatch.setattr(orch, "_inject_task_context", lambda *a, **k: None)
    seen = []
    real_parallel = orch._execute_parallel

    def spy(ready, executor, proj):
        seen.append(sorted(t.id for t in ready))
        return real_parallel(ready, executor, proj)

    monkeypatch.setattr(orch, "_execute_parallel", spy)
    monkeypatch.setattr(
        MarkdownPlanExecutor,
        "execute",
        lambda self, task, project, **k: ExecutionResult(
            status="completed", message="ok"
        ),
    )
    orch.run_project(tmp_path, skip_preflight=True, proceed=True)
    assert seen and seen[0] == ["T1", "T2", "T3"]  # the wave went through parallel
    progress = (tmp_path / ".orchestrator" / "progress.json").read_text(
        encoding="utf-8"
    )
    assert all(t in progress for t in ("T1", "T2", "T3"))  # all recorded completed
