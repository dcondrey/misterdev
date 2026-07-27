"""Requirements preflight: review the plan for user-supplied inputs up front."""

from pathlib import Path
from types import SimpleNamespace

from misterdev.core.models import Task
from misterdev.core.planning import requirements as R
from misterdev.task_executors.markdown_plan_executor import MarkdownPlanExecutor


def _t(tid, text="", accept="", deps=()):
    return Task(
        id=tid,
        description=text,
        project_ref="p",
        title=tid,
        acceptance_criteria=accept,
        dependencies=list(deps),
    )


# --- heuristic scan ----------------------------------------------------------


def test_scan_detects_env_vars_and_accounts():
    tasks = [
        _t("T1", "wire the worker", "set ADMIN_TOKEN and CLOUDFLARE_API_TOKEN"),
        _t("T2", "deploy to Cloudflare with wrangler deploy"),
        _t("T3", "npm publish the package"),
        _t("T4", "just typecheck", "pnpm typecheck = 0"),  # nothing to require
    ]
    reqs = {r["key"]: r for r in R.scan_requirements(tasks)}
    assert reqs["ADMIN_TOKEN"]["kind"] == "env"
    assert reqs["CLOUDFLARE_API_TOKEN"]["kind"] == "env"
    assert reqs["CLOUDFLARE_ACCOUNT"]["kind"] == "account"  # real deploy
    assert reqs["NPM_TOKEN"]["kind"] == "account"
    assert reqs["ADMIN_TOKEN"]["task_ids"] == ["T1"]


def test_scan_ignores_local_dry_run_and_lowercase():
    # A dry-run deploy needs no account; lowercase words are not env vars.
    tasks = [_t("T1", "run wrangler deploy --dry-run; the token is validated")]
    reqs = R.scan_requirements(tasks)
    assert reqs == []


# --- fan-out (blast radius) --------------------------------------------------


def test_fanout_counts_transitive_dependents():
    tasks = [_t("A"), _t("B", deps=["A"]), _t("C", deps=["B"]), _t("D", deps=["A"])]
    assert R.fanout(["A"], tasks) == 3  # B, C, D
    assert R.fanout(["C"], tasks) == 0  # leaf


# --- satisfaction + smart gate ----------------------------------------------


def test_check_satisfied_env(monkeypatch):
    monkeypatch.setenv("PRESENT_TOKEN", "x")
    monkeypatch.delenv("ABSENT_TOKEN", raising=False)
    assert R.check_satisfied({"kind": "env", "key": "PRESENT_TOKEN"})
    assert not R.check_satisfied({"kind": "env", "key": "ABSENT_TOKEN"})
    assert not R.check_satisfied({"kind": "account", "key": "CLOUDFLARE_ACCOUNT"})


def test_gating_only_foundational_missing_accounts():
    # An account need that gates 3+ downstream tasks gates the run; a leaf one and
    # an env-var need (advisory) never do.
    tasks = [
        _t("T1", "deploy to cloudflare"),  # foundational account
        _t("T2", deps=["T1"]),
        _t("T3", deps=["T2"]),
        _t("T4", deps=["T1"]),
    ]
    reqs = [
        {
            "key": "CLOUDFLARE_ACCOUNT",
            "kind": "account",
            "task_ids": ["T1"],
            "satisfied": False,
        },
        {"key": "SOME_TOKEN", "kind": "env", "task_ids": ["T1"], "satisfied": False},
    ]
    gating = R.gating_requirements(reqs, tasks, threshold=3)
    assert [g["key"] for g in gating] == ["CLOUDFLARE_ACCOUNT"]  # env is advisory


def test_gating_skips_answered_accounts():
    # A foundational account need the user has answered (a typed decision, e.g. "skip
    # the deploy") no longer stops the run — the user made the call.
    tasks = [
        _t("T1", "deploy to cloudflare"),
        _t("T2", deps=["T1"]),
        _t("T3", deps=["T2"]),
        _t("T4", deps=["T1"]),
    ]
    reqs = [
        {
            "key": "CLOUDFLARE_ACCOUNT",
            "kind": "account",
            "task_ids": ["T1"],
            "satisfied": False,
            "answered": True,
        },
    ]
    assert R.gating_requirements(reqs, tasks, threshold=3) == []
    # Same account need but leaf (no dependents) does not gate.
    leaf = [_t("L1", "deploy to cloudflare")]
    assert (
        R.gating_requirements(
            [
                {
                    "key": "CLOUDFLARE_ACCOUNT",
                    "kind": "account",
                    "task_ids": ["L1"],
                    "satisfied": False,
                }
            ],
            leaf,
        )
        == []
    )


def test_review_merges_llm_extras():
    tasks = [_t("T1", "deploy to cloudflare")]

    def fake_llm(prompt, system):
        return (
            '[{"key":"OAUTH_APP","kind":"account","summary":"an OAuth app",'
            '"how_to_provide":"register one","task_ids":["T1"]}]'
        )

    reqs = {r["key"]: r for r in R.review_requirements(tasks, llm=fake_llm)}
    assert "CLOUDFLARE_ACCOUNT" in reqs  # heuristic
    assert "OAUTH_APP" in reqs  # llm enrichment


# --- RequirementsBook --------------------------------------------------------


def test_requirements_book_writes_and_reads_decisions(tmp_path: Path):
    book = R.RequirementsBook(tmp_path)
    book.write(
        [
            {
                "key": "CLOUDFLARE_ACCOUNT",
                "kind": "account",
                "summary": "cf",
                "how_to_provide": "wrangler login",
                "task_ids": ["T1"],
                "satisfied": False,
            },
            {
                "key": "DB_CHOICE",
                "kind": "decision",
                "summary": "which db",
                "how_to_provide": "",
                "task_ids": ["T2"],
                "satisfied": False,
            },
        ]
    )
    assert book.md_path.exists()
    assert book.load_answers() == {}
    # Every unsatisfied requirement is answerable, not just decisions: the account
    # and the decision each carry an Answer line to fill.
    assert book.md_path.read_text(encoding="utf-8").count("- Answer:") == 2
    # Fill the account requirement (the first block) and the decision one.
    txt = book.md_path.read_text(encoding="utf-8")
    txt = txt.replace(
        "_(provide this, or write your decision here)_", "skip the deploy", 1
    ).replace("_(provide this, or write your decision here)_", "use D1", 1)
    book.md_path.write_text(txt, encoding="utf-8")
    answers = book.load_answers()
    assert answers.get("CLOUDFLARE_ACCOUNT") == "skip the deploy"
    assert answers.get("DB_CHOICE") == "use D1"
    # A re-write preserves both typed answers.
    book.write(
        [
            {
                "key": "CLOUDFLARE_ACCOUNT",
                "kind": "account",
                "summary": "cf",
                "task_ids": ["T1"],
                "satisfied": False,
            },
            {
                "key": "DB_CHOICE",
                "kind": "decision",
                "summary": "which db",
                "task_ids": ["T2"],
                "satisfied": False,
            },
        ]
    )
    assert book.load_answers().get("CLOUDFLARE_ACCOUNT") == "skip the deploy"


# --- run_project smart gate: stop vs proceed --------------------------------


def _run_with_gate(tmp_path, monkeypatch, proceed):
    from misterdev.agent import ProjectOrchestrator
    from misterdev.core.models import ExecutionResult

    # T1 needs a live Cloudflare deploy (account, missing) and gates T2/T3/T4.
    tasks = [
        _t("T1", "deploy to cloudflare"),
        _t("T2", deps=["T1"]),
        _t("T3", deps=["T2"]),
        _t("T4", deps=["T1"]),
    ]
    task_mgr = SimpleNamespace(
        discover_tasks=lambda: None,
        get_pending_tasks=lambda: tasks,
        update_task_status=lambda *a: None,
    )
    project = SimpleNamespace(
        path=tmp_path,
        name="p",
        config={},
        env_manager=None,
        task_manager=task_mgr,
        llm_client=None,
    )
    orch = ProjectOrchestrator()
    monkeypatch.setattr(orch, "_get_or_register", lambda p: project)
    monkeypatch.setattr(orch, "_inject_task_context", lambda *a, **k: None)
    calls = []
    monkeypatch.setattr(
        MarkdownPlanExecutor,
        "execute",
        lambda self, task, project, **k: (
            calls.append(task.id) or ExecutionResult(status="completed", message="ok")
        ),
    )
    orch.run_project(tmp_path, skip_preflight=True, proceed=proceed)
    return calls, tmp_path / ".orchestrator" / "REQUIREMENTS.md"


def test_run_project_stops_on_foundational_missing_input(tmp_path, monkeypatch):
    calls, req_md = _run_with_gate(tmp_path, monkeypatch, proceed=False)
    assert calls == []  # stopped before executing anything
    assert req_md.exists() and "CLOUDFLARE_ACCOUNT" in req_md.read_text()


def test_run_project_proceed_flag_overrides_gate(tmp_path, monkeypatch):
    calls, _ = _run_with_gate(tmp_path, monkeypatch, proceed=True)
    assert "T1" in calls  # --proceed runs despite the missing foundational input
