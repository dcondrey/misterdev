"""Code-context assembly and MCP awareness/gathering."""

from pathlib import Path
from typing import List, Optional

from my_project_orchestrator.core.integration.mcp_gather import gather_context
from my_project_orchestrator.core.models import Task
from my_project_orchestrator.core.execution.project import Project
from my_project_orchestrator.config import get_setting

from .helpers import logger, _relevant_line_ranges, _window_lines


class ContextMixin:
    def _mcp_gather(self, project: Project, task: Task) -> str:
        """Run the bounded agentic MCP gathering loop, or "" when off.

        Additive and behind ``orchestrator.mcp_tool_use`` (off by default): when
        the flag is off, no MCP manager is configured, or discovery found no
        tools, this returns "" so the task context — and the entire edit path —
        is byte-for-byte unchanged from the no-MCP build. When on, the model may
        request up to ``orchestrator.mcp_max_tool_rounds`` bounded MCP tool calls
        (each via the timeout-guarded, never-raising ``MCPManager.call_tool``)
        whose results are gathered into a context block. Never raises into the
        build; any failure degrades to whatever was gathered so far.
        """
        if not get_setting(project.config, "orchestrator", "mcp_tool_use"):
            return ""
        mcp = getattr(project, "mcp", None)
        if mcp is None:
            return ""
        max_rounds = get_setting(project.config, "orchestrator", "mcp_max_tool_rounds")

        def _ask(prompt: str) -> Optional[str]:
            return project.llm_client.generate_code(prompt, "")

        try:
            return gather_context(
                mcp,
                _ask,
                task_description=task.description,
                max_rounds=max_rounds,
            )
        except Exception as e:  # gathering is best-effort; never sink the build
            logger.warning(f"MCP tool-gathering skipped (error: {e}).")
            return ""

    def _mcp_awareness(self, project: Project) -> str:
        """Render the available-MCP-tools section, or "" when off / no tools.

        Additive and behind ``orchestrator.mcp_enabled``: when the flag is off,
        no MCP manager is configured, or discovery found nothing, this returns
        an empty string so the task context is byte-for-byte unchanged from the
        no-MCP build. It only informs the model the tools exist (awareness); it
        does not enable the model to call them — that agentic loop is a separate,
        out-of-scope phase that would drive ``project.mcp.call_tool``.
        """
        if not get_setting(project.config, "orchestrator", "mcp_enabled"):
            return ""
        mcp = getattr(project, "mcp", None)
        if mcp is None:
            return ""
        described = mcp.describe_tools()
        if not described:
            return ""
        return (
            "\n\n## Available MCP tools (informational)\n"
            "These external tools exist in this environment. You cannot invoke "
            "them directly in your edits; they are listed so you understand what "
            "capabilities are available.\n" + described
        )

    def _get_code_context(
        self,
        project: Project,
        target_files: List[str],
        context_files: List[str],
        max_lines: int = 500,
        task: Optional[Task] = None,
    ) -> str:
        context = ""
        topo = getattr(project, "topography", None)
        threshold = (
            get_setting(project.config, "orchestrator", "large_file_line_threshold")
            or 800
        )
        if target_files:
            context += "### Files to Modify/Create\n"
            for file_path in target_files:
                context += self._render_target_file(
                    project, topo, file_path, task, threshold
                )
        if context_files:
            context += "\n### Reference/Context Files (Read-Only)\n"
            for file_path in context_files:
                context += self._read_file_for_context(
                    project.path / file_path, file_path, max_lines
                )
        return context

    def _read_file_for_context(
        self, full_path: Path, rel_path: str, max_lines: Optional[int]
    ) -> str:
        if not full_path.exists():
            return f"\n# File: {rel_path} (Does not exist yet)\n"
        try:
            content = full_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            return f"\n# File: {rel_path} (binary or unreadable, skipped)\n"
        lines = content.splitlines()
        if max_lines is not None and len(lines) > max_lines:
            content = (
                "\n".join(lines[:max_lines])
                + f"\n... ({len(lines)} lines total, truncated)"
            )
        return f"\n# File: {rel_path}\n{content}\n"

    def _render_target_file(
        self,
        project: Project,
        topo,
        file_path: str,
        task: Optional[Task],
        threshold: int,
    ) -> str:
        """Render one target file for the edit prompt.

        Small files are sent in full. Large files are sent as a symbol outline
        plus verbatim windows of the task-relevant symbols (with elision markers
        for the rest), so the model navigates via the outline and edits the
        relevant regions exactly while context scales with the edit, not the
        file. SEARCH/REPLACE still applies against the full on-disk content, so
        windowing only narrows what the model reads, never what it can change.
        """
        full_path = project.path / file_path
        outline = topo.get_file_outline(file_path) if topo is not None else ""
        header = (
            f"\n# Outline of {file_path} (symbol: line range):\n{outline}\n"
            if outline
            else ""
        )
        if not full_path.exists():
            return header + f"\n# File: {file_path} (Does not exist yet)\n"
        try:
            content = full_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            return header + f"\n# File: {file_path} (binary or unreadable, skipped)\n"

        lines = content.split("\n")
        if topo is None or len(lines) <= threshold:
            return header + f"\n# File: {file_path}\n{content}\n"

        symbols = topo.get_file_symbols(file_path)
        keep = _relevant_line_ranges(symbols, task, len(lines))
        if keep is None:
            # No symbol matched the task: never strand the model, send it all.
            return header + f"\n# File: {file_path}\n{content}\n"
        body = _window_lines(lines, keep)
        return (
            header
            + f"\n# File: {file_path} (large file: windowed to relevant symbols; "
            "use the outline above to locate anything not shown)\n" + f"{body}\n"
        )
