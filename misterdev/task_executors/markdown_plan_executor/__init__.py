"""Markdown plan executor - executes tasks via LLM with Try-Test-Fix loop.

This package preserves the original ``markdown_plan_executor`` module's import
path: the executor class and every module-level helper it historically exported
are re-exported here unchanged. The implementation is split into cohesive
mixins purely as code movement; behaviour is identical.
"""

from typing import Optional

from misterdev.core.context.scratchpad import Scratchpad
from misterdev.task_executors.base_executor import BaseTaskExecutor

from .helpers import (
    logger,
    _is_golden_path,
    _LANG_MAP,
    EDIT_FORMAT_INSTRUCTIONS,
    FULL_FILE_FALLBACK_INSTRUCTIONS,
    JUDGE_MIN_BUDGET_FRACTION,
    _relevant_line_ranges,
    _merge_ranges,
    _window_lines,
    _bisect_first_failing,
    _is_truncated,
    _detect_language,
    _is_test_file,
    _extract_acceptance_command,
    _test_metrics,
    _count_tautologies,
    _diagnose_tampering,
    _diagnose_py_tampering,
)
from .execute_mixin import ExecuteMixin
from .git_mixin import GitMixin
from .commands_mixin import CommandsMixin
from .context_mixin import ContextMixin
from .llm_mixin import LLMMixin
from .gates_mixin import GatesMixin
from .critic_spec_mixin import CriticSpecMixin
from .edits_mixin import EditsMixin
from .results_mixin import ResultsMixin

# Re-exports preserving the original module's public surface (see module docstring).
__all__ = [
    "MarkdownPlanExecutor",
    "logger",
    "_is_golden_path",
    "_LANG_MAP",
    "EDIT_FORMAT_INSTRUCTIONS",
    "FULL_FILE_FALLBACK_INSTRUCTIONS",
    "JUDGE_MIN_BUDGET_FRACTION",
    "_relevant_line_ranges",
    "_merge_ranges",
    "_window_lines",
    "_bisect_first_failing",
    "_is_truncated",
    "_detect_language",
    "_is_test_file",
    "_extract_acceptance_command",
    "_test_metrics",
    "_count_tautologies",
    "_diagnose_tampering",
    "_diagnose_py_tampering",
]


class MarkdownPlanExecutor(
    ExecuteMixin,
    GitMixin,
    CommandsMixin,
    ContextMixin,
    LLMMixin,
    GatesMixin,
    CriticSpecMixin,
    EditsMixin,
    ResultsMixin,
    BaseTaskExecutor,
):
    """Executes tasks with a Try-Test-Fix loop.

    Uses git branch-per-task for atomic execution: each task runs on a
    temporary branch. Success merges to the current branch. Failure
    deletes the branch, leaving the repo clean.
    """

    def __init__(self, scratchpad: Optional[Scratchpad] = None):
        self.scratchpad = scratchpad or Scratchpad()
