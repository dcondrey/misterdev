"""Built-in tools, registered on import so the registry resolves them.

Importing this package registers the built-ins with ``misterdev.plugins.TOOLS``;
third-party tools add themselves via the ``misterdev.tools`` entry-point group.
"""

from misterdev.plugins import TOOLS
from misterdev.tools.command import CommandTool
from misterdev.tools.file_io import FileIOTool
from misterdev.tools.formatter import FormatterTool
from misterdev.tools.git_tool import GitTool

TOOLS.register("command", CommandTool)
TOOLS.register("test_runner", CommandTool)  # alias: a test runner is a command
TOOLS.register("file_io", FileIOTool)
TOOLS.register("formatter", FormatterTool)
TOOLS.register("git", GitTool)

__all__ = ["CommandTool", "FileIOTool", "FormatterTool", "GitTool"]
