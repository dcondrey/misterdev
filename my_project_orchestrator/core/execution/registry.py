import json
from pathlib import Path
from typing import Dict, Optional, Any

from my_project_orchestrator.core.execution.project import Project
from my_project_orchestrator.config import ConfigManager
from my_project_orchestrator.logging_setup import setup_logger

logger = setup_logger(__name__)


class ProjectRegistry:
    """Manages discovery, instantiation, and persistence of projects."""

    def __init__(self, state_file: Optional[str | Path] = None):
        self.projects: Dict[str, Project] = {}
        self.config_manager = ConfigManager()

        if state_file:
            self.state_file = Path(state_file)
        else:
            self.state_file = Path.home() / ".project_orchestrator" / "registry.json"

        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self._load_state()

    def _load_state(self):
        """Loads registered project paths from the state file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    state = json.load(f)
                    for project_path in state.get("registered_paths", []):
                        try:
                            self.register_project(project_path, save=False)
                        except Exception as e:
                            logger.error(
                                f"Failed to reload project at {project_path}: {e}"
                            )
            except Exception as e:
                logger.error(f"Failed to load registry state: {e}")

    def _save_state(self):
        """Saves registered project paths to the state file."""
        state = {"registered_paths": list(self.projects.keys())}
        try:
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save registry state: {e}")

    def discover_projects(self, root_dir: str | Path):
        """Scans for project.yaml files and registers projects."""
        root_path = Path(root_dir)
        logger.info(f"Scanning for projects in {root_path}")

        found_any = False
        for yaml_path in root_path.rglob("project.yaml"):
            project_dir = yaml_path.parent
            try:
                self.register_project(project_dir)
                found_any = True
            except Exception as e:
                logger.error(f"Failed to register project at {project_dir}: {e}")

        if found_any:
            self._save_state()

    def register_project(self, project_path: str | Path, save: bool = True) -> Project:
        """Loads config and initializes a Project object."""
        path = Path(project_path).resolve()
        project_id = str(path)

        if project_id in self.projects:
            logger.debug(f"Project already registered: {project_id}")
            return self.projects[project_id]

        config = self.config_manager.load_project_config(path)
        project = Project(path, config)
        self.projects[project_id] = project
        logger.info(f"Registered project: {project.name} at {path}")

        if save:
            self._save_state()

        return project

    def get_project(self, project_path: str | Path) -> Optional[Project]:
        path = str(Path(project_path).resolve())
        return self.projects.get(path)

    def list_projects(self) -> Dict[str, Any]:
        """Returns a summary of all registered projects."""
        return {
            path: {"name": p.name, "description": p.description, "path": str(p.path)}
            for path, p in self.projects.items()
        }
