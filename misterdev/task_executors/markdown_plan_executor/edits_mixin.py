"""Edit path validation, test-tamper detection, and edit application."""

from pathlib import Path
from typing import Dict, Optional, Tuple

from misterdev.core.models import Task
from misterdev.core.execution.project import Project
from misterdev.llm.responses import (
    EditConflictError,
    LLMResponseParser,
    apply_search_replace,
)
from misterdev.config import get_setting
from misterdev.utils.file_utils import write_file

from .helpers import (
    logger,
    _is_golden_path,
    _is_test_file,
    _diagnose_py_tampering,
    _diagnose_tampering,
)


class EditsMixin:
    def _validate_edit_paths(
        self, project: Project, task: Task, edits: Dict[str, str]
    ) -> Dict[str, str]:
        """Reject hallucinated or out-of-scope edits before they touch disk.

        Drops edits that escape the project root (absolute paths, ``..``
        traversal) or that would create empty files, and warns when the LLM
        touches files outside the task's declared scope. This is what prevents
        a misrouted edit from clobbering files outside the project.
        """
        project_root = project.path.resolve()
        expected = set(task.files_to_modify + task.files_to_create)
        golden_paths = get_setting(project.config, "orchestrator", "golden_paths")
        valid: Dict[str, str] = {}
        for path, content in edits.items():
            if ".." in Path(path).parts or Path(path).is_absolute():
                logger.error(f"Rejected edit with unsafe path: {path}")
                continue
            if _is_golden_path(path, golden_paths):
                logger.error(f"Rejected edit to protected golden file: {path}")
                continue
            full = (project.path / path).resolve()
            try:
                inside = full.is_relative_to(project_root)
            except ValueError:
                inside = False
            if not inside:
                logger.error(f"Rejected edit to path outside project root: {path}")
                continue
            if not content.strip():
                logger.warning(f"Rejected empty-content edit: {path}")
                continue
            if expected and path not in expected:
                logger.warning(f"LLM modified file outside task scope: {path}")
            valid[path] = content
        return valid

    def _detect_test_tampering(
        self, project: Project, edits: Dict[str, str]
    ) -> Optional[str]:
        """Reject edits that weaken existing test files (deterministic gate).

        For each edited TEST file that already exists on disk, compares the
        current (pre-edit) content against the proposed content. An edit that
        reduces tests/assertions or adds skip markers is tampering: the test
        gate must not be satisfied by gutting the gate. New test files and
        purely additive edits pass. Set ``orchestrator.allow_test_edits`` to
        skip the check (escape hatch); it defaults to off (check enforced).
        Must be called BEFORE ``_apply_edits`` so the on-disk content still
        reflects the pre-edit state.
        """
        if get_setting(project.config, "orchestrator", "allow_test_edits"):
            return None
        reasons = []
        for file_path, content in edits.items():
            if not _is_test_file(file_path):
                continue
            full_path = project.path / file_path
            if not full_path.exists():
                continue
            try:
                before = full_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            # Python uses precise assertion-survival diagnosis (resistant to
            # rename/move/split/merge); other languages fall back to the
            # cross-language regex totals, which can't parse structure.
            if file_path.endswith((".py", ".pyi")):
                reason = _diagnose_py_tampering(before, content)
            else:
                reason = _diagnose_tampering(before, content)
            if reason:
                reasons.append(f"{file_path} ({reason})")
        return "; ".join(reasons) if reasons else None

    def _resolve_edits(
        self, project: Project, llm_response: str
    ) -> Tuple[Dict[str, str], Optional[str]]:
        """Turn the LLM response into ``{path: full_content}`` edits.

        Prefers surgical SEARCH/REPLACE hunks: each is applied against the
        current on-disk file so the model never has to reproduce a large file
        in full (the whole-file path truncates past the output-token limit).
        The resolved full content then flows through the same path/syntax/
        tamper gates as before. Returns ``(edits, error)``; a non-empty error
        means a hunk did not apply and the attempt should retry rather than
        write a partial file. Falls back to the whole-file parser when the
        response contains no SEARCH/REPLACE markers.
        """
        sr_edits = LLMResponseParser.parse_search_replace_blocks(llm_response)
        if not sr_edits:
            return LLMResponseParser.parse_file_edits(llm_response), None
        by_path: Dict[str, list] = {}
        for edit in sr_edits:
            by_path.setdefault(edit.path, []).append(edit)
        resolved: Dict[str, str] = {}
        for path, hunks in by_path.items():
            full_path = project.path / path
            try:
                original = (
                    full_path.read_text(encoding="utf-8") if full_path.exists() else ""
                )
            except (UnicodeDecodeError, OSError) as exc:
                return {}, f"{path}: could not read file to apply edit ({exc})"
            try:
                resolved[path] = apply_search_replace(original, hunks)
            except EditConflictError as exc:
                return {}, str(exc)
        return resolved, None

    def _apply_edits(self, project: Project, edits: Dict[str, str]):
        for file_path, content in edits.items():
            full_path = project.path / file_path
            write_file(full_path, content)

    def _run_formatters(self, project: Project, files):
        # Per-file formatters substitute {path}; project-wide formatters run
        # once. Pass each file only to formatters whose command templates use
        # {path}; otherwise invoke the formatter a single time.
        file_list = list(files)
        for tool_name, tool in project.tool_manager.tools.items():
            if getattr(tool, "type", None) != "formatter":
                continue
            template = (
                getattr(tool, "config", {}).get("command", "")
                if hasattr(tool, "config")
                else ""
            )
            if "{path}" in template:
                for file_path in file_list:
                    tool.execute(project, file_path=file_path)
            else:
                tool.execute(project)
