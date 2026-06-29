"""Shared package logger.

Named for the package (not this submodule) so log output keeps the pre-split
logger name ``my_project_orchestrator.core.context.topography`` after the
god-module was broken into sections.
"""

from my_project_orchestrator.logging_setup import setup_logger

logger = setup_logger(__name__.rsplit(".", 1)[0])
