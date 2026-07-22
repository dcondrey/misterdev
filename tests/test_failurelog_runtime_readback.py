"""T5.1 — the FailureLog is read back at runtime to inform later attempts.

`_failure_priors` loads THIS project's FailureLog and surfaces the prior failures of
the current task (by id) into the edit context, so a repeated attempt within the same
run does not rediscover an error it already hit. Best-effort: no log / no match -> "".
"""

import json
from types import SimpleNamespace

from misterdev.task_executors.markdown_plan_executor.context_mixin import ContextMixin


class _Ex(ContextMixin):
    pass


def _project(tmp_path):
    return SimpleNamespace(path=tmp_path)


def _write_failures(tmp_path, rows):
    d = tmp_path / ".orchestrator"
    d.mkdir(parents=True, exist_ok=True)
    (d / "failures.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )


def test_prior_failures_for_task_are_surfaced(tmp_path):
    _write_failures(
        tmp_path,
        [
            {
                "name": "T1",
                "language": "python",
                "error": "AssertionError: 1 != 2",
                "category": "test_assertion",
            },
            {
                "name": "T1",
                "language": "python",
                "error": "TypeError: bad arg",
                "category": "wrong_type",
            },
            {
                "name": "OTHER",
                "language": "python",
                "error": "unrelated",
                "category": "x",
            },
        ],
    )
    out = _Ex()._failure_priors(_project(tmp_path), SimpleNamespace(id="T1"))
    assert "Prior failures" in out
    assert "AssertionError: 1 != 2" in out
    assert "TypeError: bad arg" in out
    assert "unrelated" not in out  # another task's failure is not surfaced


def test_no_log_returns_empty(tmp_path):
    assert _Ex()._failure_priors(_project(tmp_path), SimpleNamespace(id="T1")) == ""


def test_no_match_returns_empty(tmp_path):
    _write_failures(
        tmp_path,
        [{"name": "OTHER", "language": "python", "error": "e", "category": "c"}],
    )
    assert _Ex()._failure_priors(_project(tmp_path), SimpleNamespace(id="T1")) == ""
