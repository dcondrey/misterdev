"""The _runtime_tool executor seam: off by default (byte-identical no-op), and
when on it routes through invent_tool with a sandboxed ToolRunner. No Docker/LLM.
"""

from misterdev.task_executors.markdown_plan_executor import MarkdownPlanExecutor
from misterdev.core.models import Task


class _Client:
    def __init__(self, reply="NO_TOOL"):
        self.reply = reply
        self.prompts = []

    def generate_code(self, prompt, system=""):
        self.prompts.append(prompt)
        return self.reply


def _proj(cfg, client):
    class _P:
        config = cfg
        llm_client = client

    return _P()


def _task():
    t = Task(id="T-1", description="compute a modular inverse", project_ref="/x")
    return t


def test_off_by_default_is_noop():
    client = _Client()
    proj = _proj({"orchestrator": {}}, client)
    assert MarkdownPlanExecutor()._runtime_tool(proj, _task()) == ""
    assert client.prompts == []  # the model was never even asked


def test_on_but_no_sandbox_degrades_to_empty(monkeypatch):
    # Flag on, but no container engine -> ToolRunner returns skip -> "" (never
    # runs untrusted code on the host).
    import misterdev.core.execution.container as container

    monkeypatch.setattr(container, "detect_engine", lambda preferred=None: None)
    client = _Client(reply="```tool\nprint('inverse=9')\n```")
    proj = _proj(
        {"orchestrator": {"runtime_tooling": True, "runtime_tooling_rounds": 2}},
        client,
    )
    out = MarkdownPlanExecutor()._runtime_tool(proj, _task())
    assert out == ""  # sandbox unavailable -> nothing injected, nothing executed


def test_on_with_fake_sandbox_injects_tool_output(monkeypatch):
    # Flag on + a fake engine -> the authored tool runs and its output is injected.
    import misterdev.core.execution.container as container

    class _FakeEngine:
        def __init__(self, *a, **k):
            pass

        def run(self, command, timeout):
            return True, "inverse=9\n"

    monkeypatch.setattr(container, "detect_engine", lambda preferred=None: "docker")
    monkeypatch.setattr(container, "ContainerEngine", _FakeEngine)
    client = _Client(reply="I'll compute it.\n```tool\nprint('inverse=9')\n```")
    proj = _proj(
        {"orchestrator": {"runtime_tooling": True, "runtime_tooling_rounds": 1}},
        client,
    )
    out = MarkdownPlanExecutor()._runtime_tool(proj, _task())
    assert "inverse=9" in out
    assert "sandboxed" in out.lower()
