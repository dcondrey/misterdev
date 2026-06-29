"""Shared logger for the contracts package.

Uses the original module path as the logger name so log output is
identical to the pre-split single-file module.
"""

from my_project_orchestrator.logging_setup import setup_logger

logger = setup_logger("my_project_orchestrator.core.context.contracts")
