from typing import Any, Tuple, Optional
from pathlib import Path
import os

from my_project_orchestrator.tools.base_tool import BaseTool
from my_project_orchestrator.utils.file_utils import read_file, write_file
from my_project_orchestrator.logging_setup import setup_logger

logger = setup_logger(__name__)

class FileIOTool(BaseTool):
    """Tool for explicit file I/O operations."""
    
    def execute(self, project: Any, action: str, path: str, content: Optional[str] = None) -> Tuple[bool, Any]:
        """
        Executes a file I/O action.
        Actions: read, write, exists, delete
        """
        full_path = (project.path / path).resolve()
        if not full_path.is_relative_to(project.path.resolve()):
            return False, f"Path traversal blocked: {path}"

        try:
            if action == "read":
                if not full_path.exists():
                    return False, f"File not found: {path}"
                return True, read_file(full_path)
            
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
