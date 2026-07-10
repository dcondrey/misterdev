"""Capture wiring (two-timescale P2c): invent_tool's sink collects run tools, and
_record_invented_tools folds them into the tool corpus with the task outcome.
"""

from misterdev.core.evolution.tool_invention import invent_tool
from misterdev.core.evolution.tool_runner import ToolRunResult
from misterdev.core.evolution.tool_corpus import ToolCorpus
from misterdev.task_executors.markdown_plan_executor import MarkdownPlanExecutor
from misterdev.core.models import Task


class _FakeRunner:
    def __init__(self, *results):
        self._results = list(results)

    def run(self, source, stdin=""):
        return self._results.pop(0) if self._results else ToolRunResult("ok", "", "")


def test_sink_collects_only_tools_that_ran():
    sink = []
    invent_tool(
        _FakeRunner(ToolRunResult("ok", "out", "")),
        lambda p: "```tool\nprint(1)\n```",
        max_rounds=1,
        sink=sink,
    )
    assert sink == ["print(1)"]


def test_sink_skips_unsandboxed_tool():
    sink = []
    invent_tool(
        _FakeRunner(ToolRunResult("skip", "", "no sandbox")),
        lambda p: "```tool\nprint(1)\n```",
        max_rounds=1,
        sink=sink,
    )
    assert sink == []  # nothing ran -> nothing captured


def test_complete_task_records_invented_tools_resolved(tmp_path):
    class _P:
        path = tmp_path
        config = {"language": "python"}

        class task_manager:
            @staticmethod
            def update_task_status(*a):
                pass

    task = Task(id="T-1", description="x", project_ref=str(tmp_path))
    task.processor_data["invented_tools"] = ["print('inverse')"]
    MarkdownPlanExecutor()._complete_task(_P(), task, "done", "")
    corpus = ToolCorpus(tmp_path / ".orchestrator" / "evolution" / "tool_corpus.json")
    recs = corpus.records()
    assert len(recs) == 1
    assert recs[0].outcomes == {"T-1": True}
    assert recs[0].niche == "python"


def test_fail_task_records_invented_tools_unresolved(tmp_path):
    class _P:
        path = tmp_path
        config = {"language": "python"}

        class task_manager:
            @staticmethod
            def update_task_status(*a):
                pass

    task = Task(id="T-2", description="x", project_ref=str(tmp_path))
    task.processor_data["invented_tools"] = ["print('bad')"]
    MarkdownPlanExecutor()._fail_task(_P(), task, "nope")
    corpus = ToolCorpus(tmp_path / ".orchestrator" / "evolution" / "tool_corpus.json")
    assert corpus.records()[0].outcomes == {"T-2": False}


def test_no_tools_is_a_noop(tmp_path):
    class _P:
        path = tmp_path
        config = {"language": "python"}

        class task_manager:
            @staticmethod
            def update_task_status(*a):
                pass

    task = Task(id="T-3", description="x", project_ref=str(tmp_path))
    MarkdownPlanExecutor()._complete_task(_P(), task, "done", "")
    # No corpus file is created when nothing was invented.
    assert not (tmp_path / ".orchestrator" / "evolution" / "tool_corpus.json").exists()
