import tempfile
from pathlib import Path


from my_project_orchestrator.tools.command import CommandTool


class FakeProject:
    def __init__(self, path: Path):
        self.path = path


# Minimal tool config required by BaseTool
TOOL_CONFIG = {"name": "command", "type": "command"}


def make_tool():
    return CommandTool(TOOL_CONFIG)


def test_successful_command():
    with tempfile.TemporaryDirectory() as td:
        project = FakeProject(Path(td))
        tool = make_tool()
        success, output = tool.execute(project, "echo hello")
        assert success is True
        assert "hello" in output


def test_failed_command():
    with tempfile.TemporaryDirectory() as td:
        project = FakeProject(Path(td))
        tool = make_tool()
        success, output = tool.execute(project, "false")
        assert success is False


def test_stderr_captured():
    with tempfile.TemporaryDirectory() as td:
        project = FakeProject(Path(td))
        tool = make_tool()
        # Write to stderr explicitly
        success, output = tool.execute(project, "echo error_message >&2")
        # The command itself succeeds (exit 0), but stderr should appear in output
        assert "error_message" in output


def test_default_cwd_uses_project_path():
    with tempfile.TemporaryDirectory() as td:
        project = FakeProject(Path(td))
        tool = make_tool()
        # pwd should return the project path
        success, output = tool.execute(project, "pwd")
        assert success is True
        # Resolve both to handle symlinks (e.g. /var vs /private/var on macOS)
        assert Path(output.strip()).resolve() == Path(td).resolve()


def test_custom_cwd_overrides_default():
    with tempfile.TemporaryDirectory() as td:
        with tempfile.TemporaryDirectory() as custom_td:
            project = FakeProject(Path(td))
            tool = make_tool()
            success, output = tool.execute(project, "pwd", cwd=custom_td)
            assert success is True
            assert Path(output.strip()).resolve() == Path(custom_td).resolve()


def test_timeout_fires():
    with tempfile.TemporaryDirectory() as td:
        project = FakeProject(Path(td))
        tool = make_tool()
        success, output = tool.execute(project, "sleep 10", timeout=1)
        assert success is False
        assert "timed out" in output
        assert "1s" in output
