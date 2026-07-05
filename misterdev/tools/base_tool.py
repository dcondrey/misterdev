from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    def __init__(self, config: dict):
        self.config = config
        self.name = config.get("name", "Unnamed Tool")
        self.type = config.get("type", "base")

    @abstractmethod
    def execute(self, project: Any, **kwargs) -> Any:
        """Executes the tool's action."""
        pass
