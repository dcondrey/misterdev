"""Built-in changed-region mutation: score whether a passing suite actually
verifies a fix. Pure generator + fake-runner scoring (no real test runs)."""

from misterdev.core.execution.outcomes import GREEN, RED
from misterdev.core.verification.changed_region_mutation import (
    changed_line_indices,
    generate_mutants,
    run_changed_region_mutation,
)


def test_changed_line_indices_diff():
    old = "a\nb\nc\n"
    new = "a\nB\nc\nd\n"  # b->B (replace at 1), +d (insert at 3)
    idx = changed_line_indices(old, new)
    assert 1 in idx and 3 in idx and 0 not in idx


def test_changed_line_indices_whole_file_replacement_marks_all():
    idx = changed_line_indices("", "one\ntwo\n")
    assert idx == {0, 1}


def test_generate_mutants_operators_and_scope():
    src = "def f(x):\n    return x == 5 and True\n"
    muts = generate_mutants(src, {1})  # only line index 1 changed
    descs = [d for _, d in muts]
    assert any("'==' -> '!='" in d for d in descs)  # comparison swap
    assert any("'and' -> 'or'" in d for d in descs)  # connective swap
    assert any("'True' -> 'False'" in d for d in descs)  # bool flip
    assert any("'5' -> '6'" in d for d in descs)  # numeric bump
    assert all(d.startswith("L2:") for d in descs)  # the def line (L1) is untouched


def test_generate_mutants_are_distinct_and_capped():
    # Several DIFFERENT operators on one line (one mutant per operator) -> capped.
    src = "x = (a == b) and (c >= d) and True\n"
    muts = generate_mutants(src, {0}, max_mutants=2)
    assert len(muts) == 2
    assert len({m for m, _ in muts}) == 2  # distinct sources


def test_strong_suite_kills_all_mutants(tmp_path):
    f = tmp_path / "m.py"
    new = "def gt(a, b):\n    return a >= b\n"
    f.write_text(new)

    def strong(cmd, timeout):
        # A suite that pins behavior: any deviation from the accepted fix fails.
        return f.read_text() == new, ""

    res = run_changed_region_mutation(
        tmp_path,
        "m.py",
        "def gt(a, b):\n    return a > b\n",
        new,
        "pytest",
        runner=strong,
        min_score=0.5,
    )
    assert res.score == 1.0 and res.status == GREEN
    assert f.read_text() == new  # restored to the accepted fix


def test_weak_suite_survives_all_and_reds_when_gated(tmp_path):
    f = tmp_path / "m.py"
    new = "def gt(a, b):\n    return a >= b\n"
    f.write_text(new)

    def weak(cmd, timeout):
        return True, ""  # the suite passes no matter what -> nothing killed

    res = run_changed_region_mutation(
        tmp_path, "m.py", "orig", new, "pytest", runner=weak, min_score=0.5
    )
    assert res.score == 0.0 and res.status == RED
    assert "does not verify the fix" in res.reason
    assert f.read_text() == new  # restored


def test_executor_seam_is_a_noop_when_disabled(tmp_path):
    # The per-task wiring must be inert (and never raise) unless explicitly enabled.
    from types import SimpleNamespace
    from misterdev.task_executors.markdown_plan_executor import MarkdownPlanExecutor

    ex = MarkdownPlanExecutor()
    proj = SimpleNamespace(path=tmp_path, config={})  # flag absent -> off
    assert (
        ex._changed_region_mutation_check(proj, {"a.py": "x"}, "pytest", None) is None
    )


def test_advisory_default_never_reds(tmp_path):
    f = tmp_path / "m.py"
    new = "def gt(a, b):\n    return a >= b\n"
    f.write_text(new)
    res = run_changed_region_mutation(
        tmp_path, "m.py", "orig", new, "pytest", runner=lambda c, t: (True, "")
    )  # min_score defaults to 0.0
    assert res.score == 0.0 and res.status == GREEN  # measurement, not a gate


def _enabled_executor(tmp_path):
    from types import SimpleNamespace
    from misterdev.task_executors.markdown_plan_executor import MarkdownPlanExecutor

    (tmp_path / "mod.py").write_text("def gt(a, b):\n    return a >= b\n")
    proj = SimpleNamespace(
        path=tmp_path, config={"orchestrator": {"changed_region_mutation": True}}
    )
    return MarkdownPlanExecutor(), proj


def test_seam_logs_score_when_enabled(tmp_path, caplog, monkeypatch):
    # Flag on + a real mutable change -> the score is surfaced (observable).
    import logging
    import misterdev.core.verification.changed_region_mutation as crm
    from misterdev.core.verification.mutation_gate import MutationResult
    from misterdev.core.execution.outcomes import GREEN

    ex, proj = _enabled_executor(tmp_path)
    monkeypatch.setattr(
        crm,
        "run_changed_region_mutation",
        lambda *a, **k: MutationResult(GREEN, 1.0, reason="2/2 killed"),
    )
    with caplog.at_level(logging.INFO):
        ex._changed_region_mutation_check(proj, {"mod.py": "orig"}, "pytest", None)
    assert any("Changed-region mutation [mod.py]" in r.message for r in caplog.records)


def test_seam_stays_observable_when_skipped(tmp_path, caplog, monkeypatch):
    # Regression: a no-op change (SKIP) must NOT silently return — the seam that
    # is enabled always logs, so "wired live" is verifiable in the run log.
    import logging
    import misterdev.core.verification.changed_region_mutation as crm
    from misterdev.core.verification.mutation_gate import MutationResult
    from misterdev.core.execution.outcomes import SKIP

    ex, proj = _enabled_executor(tmp_path)
    monkeypatch.setattr(
        crm,
        "run_changed_region_mutation",
        lambda *a, **k: MutationResult(SKIP, reason="no mutants"),
    )
    with caplog.at_level(logging.INFO):
        ex._changed_region_mutation_check(proj, {"mod.py": "orig"}, "pytest", None)
    assert any("no mutable source change" in r.message for r in caplog.records)


def test_seam_falls_back_to_project_test_command(tmp_path, caplog, monkeypatch):
    # Regression: a task that merges with NO per-task test gate must still be
    # scored against the project suite. Without this the seam silently no-ops on
    # every no-test completion path (the common case).
    import logging
    import misterdev.core.verification.changed_region_mutation as crm
    from misterdev.core.verification.mutation_gate import MutationResult
    from misterdev.core.execution.outcomes import GREEN

    ex, proj = _enabled_executor(tmp_path)
    proj.config["test_command"] = "python -m pytest -q"
    seen = {}
    monkeypatch.setattr(
        crm,
        "run_changed_region_mutation",
        lambda *a, **k: (
            seen.update(cmd=a[4]) or MutationResult(GREEN, 1.0, reason="2/2 killed")
        ),
    )
    with caplog.at_level(logging.INFO):
        ex._changed_region_mutation_check(proj, {"mod.py": "orig"}, None, None)
    assert seen.get("cmd") == "python -m pytest -q"  # used the project fallback
    assert any("Changed-region mutation [mod.py]" in r.message for r in caplog.records)
