"""Compatibility re-export of the package-level configuration.

The canonical configuration lives in ``my_project_orchestrator.config``. This
module exposes the same ``DEFAULT_CONFIG`` and ``ConfigManager`` under the
``core`` namespace so callers can import from either location.
"""

from my_project_orchestrator.config import DEFAULT_CONFIG, ConfigManager

__all__ = ["DEFAULT_CONFIG", "ConfigManager"]
