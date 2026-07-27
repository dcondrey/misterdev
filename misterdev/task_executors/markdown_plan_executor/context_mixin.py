"""Code-context assembly and MCP awareness/gathering."""

from pathlib import Path
from typing import List, Optional

from misterdev.core.integration.mcp_gather import gather_context
from misterdev.core.models import Task
from misterdev.core.execution.project import Project
from misterdev.config import get_setting

from .helpers import logger, _relevant_line_ranges, _window_lines


class ContextMixin:
    def _runtime_tool(
        self, project: Project, task: Task, error_context: str = ""
    ) -> str:
        """Let the model author + run a sandboxed helper tool, or "" when off.

        Additive and behind ``orchestrator.runtime_tooling`` (off by default):
        when the flag is off, or no container sandbox is available, this returns
        "" so the edit path is byte-for-byte unchanged. When on, the model may
        author a small Python tool that runs in the hardened, network-less
        :class:`ToolRunner` (untrusted code never touches the host or repo); its
        output is prepended to the edit context. ``error_context`` (a prior
        failure) lets the model write a tool to diagnose it. Never raises into the
        build; any failure degrades to whatever was produced (usually nothing).
        """
        if not get_setting(project.config, "orchestrator", "runtime_tooling"):
            return ""
        from misterdev.core.evolution.tool_invention import invent_tool
        from misterdev.core.evolution.tool_runner import ToolRunner

        max_rounds = get_setting(
            project.config, "orchestrator", "runtime_tooling_rounds"
        )

        def _ask(prompt: str) -> Optional[str]:
            return project.llm_client.generate_code(prompt, "")

        # Captured tools are folded into the tool corpus with this task's outcome
        # at the terminal seam (see ResultsMixin._record_invented_tools).
        sink = task.processor_data.setdefault("invented_tools", [])
        # Seed with tools promoted from past runs (held-out-gated), so capability
        # compounds instead of being reinvented — the two-timescale payoff. Empty
        # until a promotion pass has admitted tools; best-effort.
        seeds: list = []
        try:
            from misterdev.core.evolution.tool_library import ToolLibrary

            lib = ToolLibrary(
                project.path / ".orchestrator" / "evolution" / "tool_library.json"
            )
            seeds = [t.source for t in lib.seed(limit=5)]
        except Exception as e:  # seeding is best-effort
            logger.debug(f"Tool-library seed skipped: {e}")
        try:
            from misterdev.core.execution.container import detect_engine

            if detect_engine() is None:
                logger.warning(
                    "runtime_tooling is enabled but no container sandbox is available "
                    "(Docker or Podman not found); tool invention disabled for this run."
                )
                return ""
            return invent_tool(
                ToolRunner(),
                _ask,
                task_description=task.description,
                error_context=error_context,
                max_rounds=max_rounds,
                sink=sink,
                seeds=seeds,
            )
        except Exception as e:  # invention is best-effort; never sink the build
            logger.warning(f"Runtime tool-invention skipped (error: {e}).")
            return ""

    def _mcp_gather(self, project: Project, task: Task, error_context: str = "") -> str:
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
        local_tools = self._gather_safe_tools(project)
        if mcp is None and not local_tools:
            return ""
        max_rounds = get_setting(project.config, "orchestrator", "mcp_max_tool_rounds")

        def _ask(prompt: str) -> Optional[str]:
            return project.llm_client.generate_code(prompt, "")

        # On-demand provisioning: when enabled, let the model FIND and mount a new
        # server mid-gather via the same trust ladder. Off by default (it is
        # model-driven local code execution); mcp is None-checked above. Refused
        # under network=none isolation — running unvetted registry code on the
        # host would defeat the sandbox the project explicitly chose.
        provide = None
        mcp_cfg = project.config.get("mcp") or {}
        isolated = bool(getattr(project, "_host_exec_isolated", lambda: False)())
        if mcp is not None and mcp_cfg.get("discover_on_demand", True) and not isolated:
            from misterdev.core.integration.mcp_registry import (
                RegistryCache,
                provide_capability,
            )

            trusted = mcp_cfg.get("trusted_namespaces") or None
            min_trust = float(mcp_cfg.get("min_trust", 0.5))
            cache_path = getattr(project, "_mcp_cache_path", lambda: None)()
            cache = RegistryCache(cache_path) if cache_path else None

            source = str(mcp_cfg.get("discover_source", "cgcone"))

            def _provide(
                capability: str, _t=trusted, _m=min_trust, _c=cache, _s=source
            ):
                kwargs = {"min_trust": _m, "cache": _c, "source": _s}
                if _t:
                    kwargs["trusted_namespaces"] = _t
                # Skip re-discovering anything already mounted, by launch identity
                # (locked snapshot — the manager is shared across parallel tasks).
                kwargs["known_identities"] = mcp.known_identities()
                return provide_capability(capability, **kwargs)

            provide = _provide
        elif mcp_cfg.get("discover_on_demand", True) and isolated:
            logger.info(
                "MCP on-demand provisioning disabled under host-execution isolation."
            )

        # Query-on-failure: on a retry, frame the gather around the actual gate
        # error so the model looks up what it needs to FIX the failure (a doc, an
        # API signature, the meaning of an error) — the moment a tool is most
        # useful — rather than only speculatively before the first attempt.
        description = task.description
        if error_context:
            description = (
                f"{task.description}\n\nThe previous attempt FAILED a gate with:\n"
                f"{error_context[:1500]}\n\nGather information that helps fix this "
                "specific failure before the next attempt."
            )

        # Audit sink: record provisions and tool calls to the build's append-only
        # trail. Installing/running internet code and calling external tools are
        # the security-relevant events here; the logger alone is not forensic.
        audit = getattr(project, "audit_trail", None)
        on_event = None
        if audit is not None:

            def on_event(kind: str, details: dict, _a=audit, _tid=task.id):
                _a.record(kind, task_id=_tid, **details)

        try:
            return gather_context(
                mcp,
                _ask,
                task_description=description,
                max_rounds=max_rounds,
                local_tools=local_tools,
                provide=provide,
                max_provisions=int(mcp_cfg.get("discover_on_demand_max", 2)),
                on_event=on_event,
            )
        except Exception as e:  # gathering is best-effort; never sink the build
            logger.warning(f"MCP tool-gathering skipped (error: {e}).")
            return ""

    def _gather_safe_tools(self, project: Project) -> dict:
        """Configured tools that opt into the gathering loop, as
        ``{name: (description, call)}``.

        A tool participates only if it is in ``project.config['tools']`` (operator
        opted in) AND its class sets ``gather_safe = True`` — so a mutating tool
        (command, file_io write/delete) is never exposed to this read-only
        context-gathering pass; a plugin ships a read-only tool by declaring the
        flag. Off by default: no built-in tool is gather-safe, so behaviour is
        unchanged unless a gather-safe tool is configured.
        """
        import misterdev.tools  # noqa: F401 - registers built-in tools
        from misterdev.plugins import TOOLS

        local: dict = {}
        for tc in project.config.get("tools") or []:
            tool_cls = TOOLS.get(tc.get("type"))
            if tool_cls is None or not getattr(tool_cls, "gather_safe", False):
                continue
            instance = tool_cls(tc)
            name = tc.get("name") or tc.get("type")
            desc = getattr(tool_cls, "gather_description", f"{tc.get('type')} tool")

            def _call(args, _instance=instance):
                ok, out = _instance.execute(project, **(args or {}))
                return str(out) if ok else None

            local[name] = (desc, _call)
        return local

    def _solved_task_priors(self, project: Project, task: Task) -> str:
        """Warm-start THIS task from its nearest previously-solved tasks, or "".

        The build already seeds the SPEC with priors for the overall goal; this
        closes the same cross-build learning loop at the EXECUTOR, keyed on the
        individual task description, so a recurring task shape starts from a proven
        approach instead of re-deriving it cold. Reads the project's append-only
        solved-task index and reuses the project's embedder for dense+lexical
        retrieval (lexical-only without one). Best-effort: a missing/unreadable
        index or no prior match degrades to "" and never breaks execution.
        """
        try:
            from misterdev.core.learning.warm_start import SolvedTaskIndex

            embedder = getattr(
                getattr(project, "semantic_ranker", None), "embedder", None
            )
            index = SolvedTaskIndex(
                project.path / ".orchestrator" / "solved_tasks.jsonl",
                embedder=embedder,
            )
            return index.context(task.description)
        except Exception as e:  # warm-start is best-effort; never sink the build
            logger.debug(f"Solved-task warm-start skipped: {e}")
            return ""

    def _failure_priors(self, project: Project, task: Task) -> str:
        """Runtime read-back of this project's FailureLog for THIS task, or "".

        The FailureLog was write-only at runtime (read only by the evolution loop);
        this closes the loop within a run: a task that already failed sees its own
        prior failures (by task id) so a later attempt does not rediscover the same
        error. Best-effort — a missing/unreadable log or no match degrades to "".
        """
        try:
            from misterdev.core.learning.failure_log import FailureLog

            records = FailureLog(
                project.path / ".orchestrator" / "failures.jsonl"
            ).load()
            mine = [r for r in records if r.name == task.id and r.error]
            if not mine:
                return ""
            lines = ["## Prior failures of this task (do NOT repeat these):"]
            for r in mine[::-1][:3]:  # most recent first, capped
                cat = f"[{r.category}] " if r.category else ""
                lines.append(f"- {cat}{r.error.strip()[:300]}")
            return "\n".join(lines)
        except Exception as e:  # read-back is best-effort; never sink the build
            logger.debug(f"Failure-prior read-back skipped: {e}")
            return ""

    def _localize_target_files(self, project: Project, task: Task) -> List[str]:
        """Find edit targets for a task that declares none, or [].

        Decomposition usually names a task's files, but a bare issue ("fix the X
        bug") can arrive with none — and the executor would then edit blind, with
        no target-file context. This ranks the tree-sitter symbol graph against the
        task description (lexical + call-graph, semantic when an embedder exists;
        see :mod:`misterdev.core.context.localizer`) and returns the owning files,
        best-first. A seed, not a cage: the model may still edit beyond them. Best-
        effort — no graph, no match, or any error degrades to [] (edit blind, the
        prior behavior)."""
        topo = getattr(project, "topography", None)
        if topo is None:
            return []
        try:
            ranked = topo.localize_files(
                task.description, ranker=getattr(project, "semantic_ranker", None)
            )
            return [file_path for file_path, _score in ranked]
        except Exception as e:  # localization is best-effort; never sink the build
            logger.debug(f"Target-file localization skipped: {e}")
            return []

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

    def _fully_shown_target_files(
        self, project: Project, target_files: List[str]
    ) -> set:
        """Target files small enough to be sent verbatim IN FULL by code_context.

        These (existing files at or under the large-file threshold) have their
        complete text in code_context, so topo can drop their own symbols to
        avoid duplicating the same code across two sections. A large (windowed)
        file is NOT included: its out-of-window symbols are still useful in topo.
        """
        threshold = (
            get_setting(project.config, "orchestrator", "large_file_line_threshold")
            or 800
        )
        shown = set()
        for file_path in target_files:
            full = project.path / file_path
            try:
                if full.exists() and full.stat().st_size:
                    line_count = full.read_text(encoding="utf-8").count("\n") + 1
                    if line_count <= threshold:
                        shown.add(file_path)
            except (UnicodeDecodeError, OSError):
                continue
        return shown

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
