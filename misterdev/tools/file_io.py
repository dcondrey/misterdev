from typing import Any, Tuple, Optional
import os

from misterdev.tools.base_tool import BaseTool
from misterdev.utils.file_utils import read_file_capped, write_file
from misterdev.logging_setup import setup_logger

logger = setup_logger(__name__)


class FileIOTool(BaseTool):
    """Tool for explicit file I/O operations."""

    def execute(
        self, project: Any, action: str, path: str, content: Optional[str] = None
    ) -> Tuple[bool, Any]:
        """
        Executes a file I/O action.
        Actions: read, write, exists, delete
        """
        project_root = project.path.resolve()
        full_path = (project.path / path).resolve()
        if not full_path.is_relative_to(project_root):
            return False, f"Path traversal blocked: {path}"
        # An empty/"."/"./ " path resolves to the project root itself, which passes
        # the traversal guard above. Refuse to operate on the root so a model
        # emitting such a path cannot delete (rmtree) the whole project.
        if full_path == project_root:
            return False, f"Refusing to operate on the project root: {path!r}"
        # The orchestrator's rollback/bisect safety paths depend on version
        # history, so a stray delete of .git would break its own ability to
        # revert bad work. This tool edits project source, never git internals.
        git_dir = project_root / ".git"
        if full_path == git_dir or git_dir in full_path.parents:
            return False, f"Refusing to operate on the git directory: {path!r}"

        try:
            if action == "read":
                if not full_path.exists():
                    return False, f"File not found: {path}"
                return True, read_file_capped(full_path)

            elif action == "write":
                if content is None:
                    return False, "No content provided for write action."
                write_file(full_path, content)
                return True, f"Successfully wrote to {path}"

            elif action == "exists":
                return True, full_path.exists()

            elif action == "delete":
                if full_path.exists():
                    if full_path.is_file():
                        os.remove(full_path)
                    else:
                        import shutil

                        shutil.rmtree(full_path)
                    return True, f"Deleted {path}"
                return True, f"{path} did not exist."

            else:
                return False, f"Unsupported action: {action}"

        except Exception as e:
            error_msg = f"File I/O error: {e}"
            logger.error(error_msg)
            return False, error_msg
