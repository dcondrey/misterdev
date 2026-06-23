from abc import ABC, abstractmethod
from pathlib import Path


class BaseEnvironmentManager(ABC):
    def __init__(self, config: dict, project_path: str | Path):
        self.config = config
        self.project_path = Path(project_path)

    @abstractmethod
    def setup(self) -> bool:
        """Sets up the environment."""
        pass

    @abstractmethod
    def activate_command(self) -> str:
        """Returns the command snippet to activate the environment."""
        pass
