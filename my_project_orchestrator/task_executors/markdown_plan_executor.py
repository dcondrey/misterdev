"""Markdown plan executor - executes tasks via LLM with Try-Test-Fix loop."""
import shlex
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from my_project_orchestrator.core.context_budget import ContextBudget
from my_project_orchestrator.core.models import Task, ExecutionResult
from my_project_orchestrator.core.project import Project
from my_project_orchestrator.core.scratchpad import Scratchpad
from my_project_orchestrator.llm.prompt_manager import PromptManager
from my_project_orchestrator.logging_setup import setup_logger
from my_project_orchestrator.task_executors.base_executor import BaseTaskExecutor
from my_project_orchestrator.llm.responses import LLMResponseParser
from my_project_orchestrator.llm.client import code_gen_abort_check
from my_project_orchestrator.core.validator import SOTAValidator, CertaintyScorer, StallDetector
from my_project_orchestrator.core.error_resolver import ErrorResolver
from my_project_orchestrator.core.error_classifier import format_classified_error, classify_error
from my_project_orchestrator.utils.file_utils import write_file

logger = setup_logger(__name__)

# Maps file extensions to language identifiers for syntax validation and
# contract extraction. Unknown extensions fall back to "text".
_LANG_MAP = {
    ".py": "python", ".pyi": "python",
    ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".rs": "rust", ".go": "go", ".java": "java",
    ".c": "c", ".h": "c", ".cpp": "cpp", ".hpp": "cpp", ".cc": "cpp",
    ".rb": "ruby", ".php": "php", ".sh": "shell", ".bash": "shell",
}


def _bisect_first_failing(n: int, passes_at) -> int:
    """Binary-search [0, n) for the first index where passes_at(i) is False.

    Assumes a monotonic pass->fail boundary (all-pass prefix, then failures).
    Returns n-1 if nothing fails; callers should re-check that index.
    """
    lo, hi = 0, n - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if passes_at(mid):
            lo = mid + 1
        else:
            hi = mid
    return lo


def _detect_language(file_path: str) -> str:
    """Detect a source language from a file path's extension.

    Returns "text" for extensions with no known language mapping so callers
    fall back to language-agnostic validation rather than guessing.
    """
    ext = Path(file_path).suffix.lower()
    return _LANG_MAP.get(ext, "text")


class MarkdownPlanExecutor(BaseTaskExecutor):
    """Executes tasks with a Try-Test-Fix loop.

    Uses git branch-per-task for atomic execution: each task runs on a
    temporary branch. Success merges to the current branch. Failure
    deletes the branch, leaving the repo clean.
    """

    def __init__(self, scratchpad: Optional[Scratchpad] = None):
        self.scratchpad = scratchpad or Scratchpad()
        self.stall_detector = StallDetector()

    def execute(self, task: Task, project: Project, use_git_branch: bool = True, _depth: int = 0) -> ExecutionResult:
        logger.info(f"Starting execution of task: {task.id}")
        project.task_manager.update_task_status(task.id, "in_progress")

        prompt_manager = PromptManager(project.config)
        processor_config = self._get_processor_config(project)

        # Read max_task_attempts from orchestrator config, falling back to processor config, then hardcoded default
        orch_cfg = project.config.get("orchestrator", {})
        default_attempts = orch_cfg.get("max_task_attempts", 3)
        max_retries = processor_config.get("max_retries_per_task", default_attempts)

        # Read context budget tokens from orchestrator config or task processor_data
        context_budget_tokens = task.processor_data.get(
            "context_budget_tokens",
            orch_cfg.get("context_budget_tokens", 100000),
        )

        files_key = processor_config.get("target_files_key", "files_to_modify")
        test_cmd_key = processor_config.get("test_command_key", "test_command")

        target_files = task.processor_data.get(files_key, [])
        if isinstance(target_files, str):
            target_files = [target_files]
        target_files = list(set(target_files + task.files_to_modify + task.files_to_create))

        test_command = task.processor_data.get(test_cmd_key)
        build_cfg = project.config.get("build", {})
        build_timeout = build_cfg.get("build_timeout", 120)
        test_timeout = build_cfg.get("test_timeout", 180)

        # Atomic execution: git branch per task (disabled in parallel mode to avoid races)
        can_branch = use_git_branch and self._is_git_repo(project)
        branch_name = f"task/{task.id}" if can_branch else None
        base_branch = None

        if branch_name:
            base_branch = self._get_current_branch(project)
            if not self._create_task_branch(project, branch_name):
                logger.warning("Failed to create task branch; falling back to file snapshots")
                branch_name = None

        # Fallback: file snapshots if git branching isn't available
        snapshot = None
        if not branch_name:
            snapshot = self._snapshot_files(project, target_files)

        error_logs = None
        prior_errors: List[str] = []
        resolver = ErrorResolver(project.path, project.topography.graph)

        for attempt in range(max_retries):
            logger.info(f"Attempt {attempt + 1}/{max_retries} for task {task.id}")

            code_context = self._get_code_context(project, target_files, task.context_files)
            topo_context = project.topography.get_context_for_task(task.description, target_files)
            scratchpad_context = self.scratchpad.format_context(
                files=target_files + task.context_files, tags=[task.category],
            )

            strategy = task.processor_data.get("sota_strategy", "iterative")
            consensus = task.processor_data.get("consensus_context", "None")
            interface_contracts = task.processor_data.get("interface_contracts", "")
            recent_changes = task.processor_data.get("recent_changes", "")

            # Budget-aware context allocation using configured token limit
            budget = ContextBudget(max_tokens=context_budget_tokens)
            budget.set("code_context", code_context, priority=1)
            budget.set("topo_context", topo_context, priority=3)
            budget.set("error_logs", error_logs or "", priority=1, min_lines=20)
            budget.set("scratchpad", scratchpad_context, priority=3)
            budget.set("interface_contracts", interface_contracts, priority=2)
            budget.set("recent_changes", recent_changes, priority=2)
            budget.set("consensus_context", consensus, priority=3)
            allocated = budget.allocate()

            full_code_context = allocated["code_context"]
            if allocated["topo_context"]:
                full_code_context += "\n\n" + allocated["topo_context"]
            if allocated["recent_changes"]:
                full_code_context += "\n\n" + allocated["recent_changes"]

            context_dict = {
                "project": project,
                "task": task,
                "code_context": full_code_context,
                "error_logs": allocated["error_logs"] or None,
                "task.description": task.description,
                "task.target_files": ", ".join(target_files) if target_files else "None explicitly specified",
                "scratchpad": allocated["scratchpad"],
                "acceptance_criteria": task.acceptance_criteria,
                "consensus_context": allocated["consensus_context"],
                "interface_contracts": allocated["interface_contracts"],
                "sota_strategy": strategy.upper(),
                "sota_invariants": (
                    f"SOTA Strategy: {strategy.upper()}. Output MUST be syntactically valid. "
                    "Provide certainty indicators."
                ),
            }

            system_prompt = prompt_manager.format_prompt("system", context_dict)
            if error_logs and attempt > 0:
                prompt = prompt_manager.format_prompt("error_correction_instruction", context_dict)
            else:
                prompt = prompt_manager.format_prompt("task_completion_instruction", context_dict)

            routed_model = self._resolve_model(project, task, strategy)
            aborted = False
            try:
                with project.llm_client.track_task(task.id):
                    if routed_model:
                        logger.info(f"[{task.id}] routing to {routed_model} ({task.complexity}/{strategy})")
                        with project.llm_client.with_model(routed_model):
                            llm_response, aborted = self._invoke_llm(project, prompt, system_prompt)
                    else:
                        llm_response, aborted = self._invoke_llm(project, prompt, system_prompt)
            except Exception as e:
                msg = f"LLM generation failed: {e}"
                logger.error(msg)
                self._abort_task(project, branch_name, base_branch, snapshot)
                return self._fail_task(project, task, msg)

            if aborted:
                logger.warning(f"LLM stream aborted for {task.id}; retrying with stricter instruction.")
                error_logs = "SOTA ERROR: response was not code. Output ONLY file edits as code blocks with file paths."
                continue

            certainty = CertaintyScorer.compute_score(llm_response)
            logger.info(f"LLM Certainty Score: {certainty:.2f}")

            edits = LLMResponseParser.parse_file_edits(llm_response)
            edits = self._validate_edit_paths(project, task, edits)
            if not edits:
                logger.warning("No file edits detected in LLM response.")
            else:
                stall_risk = self.stall_detector.push_edit(edits)
                if stall_risk > 0.7:
                    logger.warning(f"High stall risk detected ({stall_risk:.2f}).")
                    if attempt > 1:
                        error_logs = "SOTA ERROR: Stalling detected. Try a fundamentally different approach."
                        continue

                validation_failed = False
                for file_path, content in edits.items():
                    lang = _detect_language(file_path)
                    valid, error = SOTAValidator.validate_code(content, language=lang)
                    if not valid:
                        logger.error(f"SOTA Validation Failed for {file_path}: {error}")
                        error_logs = f"SOTA SYNTAX ERROR in {file_path}:\n{error}"
                        validation_failed = True
                        break

                if validation_failed:
                    continue

                self._apply_edits(project, edits)
                self._run_formatters(project, edits.keys())

            build_cmd = project.config.get("build_command")
            if build_cmd:
                success, output = self._run_command(project, build_cmd, timeout=build_timeout)
                if not success:
                    logger.warning(f"Build failed on attempt {attempt + 1}")
                    locations = resolver.resolve_errors(output)
                    attributed_error = resolver.format_for_llm(locations)
                    classified = format_classified_error(output)
                    error_logs = self._build_error_context(
                        prior_errors, attempt, output, classified, attributed_error
                    )
                    continue

            if test_command:
                success, output = self._run_command(project, test_command, timeout=test_timeout)
                if success:
                    logger.info("Tests passed successfully.")
                    self._record_success(task, target_files)
                    self._commit_task(project, branch_name, base_branch, task, target_files)
                    return self._complete_task(project, task, "Task completed and tests passed.", output)
                else:
                    logger.warning(f"Tests failed on attempt {attempt + 1}.")
                    locations = resolver.resolve_errors(output)
                    attributed_error = resolver.format_for_llm(locations)
                    classified = format_classified_error(output)
                    error_logs = self._build_error_context(
                        prior_errors, attempt, output, classified, attributed_error
                    )
            else:
                self._record_success(task, target_files)
                self._commit_task(project, branch_name, base_branch, task, target_files)
                return self._complete_task(project, task, "Task completed (no tests run).", llm_response)

        # Strategy escalation: if current strategy failed, try one more attempt
        # with "surgical". Guarded by _depth so escalation can never recurse more
        # than once, even if the strategy-selection logic changes later.
        current_strategy = task.processor_data.get("sota_strategy", "iterative")
        if current_strategy != "surgical" and _depth < 1:
            logger.info(f"Escalating strategy from {current_strategy} to surgical for final attempt")
            self._abort_task(project, branch_name, base_branch, snapshot)

            task.processor_data["sota_strategy"] = "surgical"
            task.processor_data["sota_invariants"] = (
                "ESCALATED: Previous strategy failed. Use SURGICAL approach: "
                "make the smallest possible change to fix the immediate error. "
                "Do not refactor or restructure. Minimal, targeted fix only."
            )
            # Recursive single attempt with surgical strategy
            return self.execute(task, project, use_git_branch=use_git_branch, _depth=_depth + 1)
        elif _depth >= 1:
            logger.warning(f"Strategy escalation blocked at depth {_depth}: already exhausted all strategies")

        logger.warning(f"Task {task.id} failed after all attempts including escalation. Reverting.")
        self._abort_task(project, branch_name, base_branch, snapshot)
        self.scratchpad.record(
            category="pitfall",
            discovery=(
                f"Failed after {max_retries} attempts + escalation: "
                f"{error_logs[:200] if error_logs else 'unknown'}"
            ),
            task_id=task.id,
            files=target_files,
        )
        return self._fail_task(
            project, task,
            f"Task failed after {max_retries} attempts + escalation.",
            error_logs,
        )

    # ----------------------------------------------------------------
    # Git branch-per-task operations
    # ----------------------------------------------------------------

    def _is_git_repo(self, project: Project) -> bool:
        return (project.path / ".git").exists()

    # ----------------------------------------------------------------
    # Regression bisection (post-build gate failure)
    # ----------------------------------------------------------------

    def find_task_commit(self, project: Project, task_id: str) -> Optional[str]:
        """SHA of the commit recording a task (message 'task(<id>):'), or None."""
        ok, out = self._git(
            project,
            f"git log --all -n 1 --format=%H --fixed-strings --grep={shlex.quote(f'task({task_id}):')}",
        )
        sha = out.strip().splitlines()[0] if ok and out.strip() else ""
        return sha or None

    def bisect_regression(self, project: Project, task_commits: List, test_command: str,
                          timeout: int = 180) -> Optional[str]:
        """Find the earliest task commit whose checkout fails the test command.

        task_commits is [(task_id, sha)] ordered oldest->newest. Returns the
        culprit task_id, or None if no checked-out commit actually fails (so a
        flaky/non-task regression isn't misattributed). Restores HEAD after.
        """
        if not task_commits:
            return None
        ok, head = self._git(project, "git rev-parse HEAD")
        restore = head.strip() if ok else None

        def passes_at(i: int) -> bool:
            self._git(project, f"git checkout {shlex.quote(task_commits[i][1])}")
            success, _ = self._run_command(project, test_command, timeout=timeout)
            return success

        try:
            idx = _bisect_first_failing(len(task_commits), passes_at)
            culprit = None if passes_at(idx) else task_commits[idx][0]
        finally:
            if restore:
                self._git(project, f"git checkout {shlex.quote(restore)}")
        return culprit

    def revert_task_commit(self, project: Project, sha: str) -> bool:
        """Revert a task's commit, leaving an explicit revert commit."""
        ok, _ = self._git(project, f"git revert --no-edit {shlex.quote(sha)}")
        return ok

    def _get_current_branch(self, project: Project) -> Optional[str]:
        ok, output = self._git(project, "git rev-parse --abbrev-ref HEAD")
        return output.strip() if ok else None

    def _create_task_branch(self, project: Project, branch_name: str) -> bool:
        ok, _ = self._git(project, f"git checkout -b {shlex.quote(branch_name)}")
        if ok:
            logger.info(f"Created task branch: {branch_name}")
        return ok

    def _commit_task(self, project: Project, branch_name: Optional[str], base_branch: Optional[str],
                     task: Task, files: Optional[List[str]] = None):
        """Commit the task's own changes and merge the task branch back to base.

        Stages only the task's target files, never `git add -A`: a blanket add
        sweeps unrelated untracked files (other uncommitted user work) into the
        commit, which then get destroyed if the task is later reverted.
        """
        msg = f"task({task.id}): {task.title or task.description[:50]}"
        if files:
            quoted = " ".join(shlex.quote(f) for f in files)
            self._git(project, f"git add -- {quoted}")
        else:
            self._git(project, "git add -A")
        self._git(project, f"git commit -m {shlex.quote(msg)} --allow-empty")

        if branch_name and base_branch:
            self._git(project, f"git checkout {shlex.quote(base_branch)}")
            ok, output = self._git(
                project,
                f"git merge --no-ff {shlex.quote(branch_name)} -m {shlex.quote(f'Merge {branch_name}')}",
            )
            if ok:
                self._git(project, f"git branch -d {shlex.quote(branch_name)}")
                logger.info(f"Merged and cleaned up branch: {branch_name}")
            else:
                logger.error(f"Merge failed for {branch_name}: {output}")

    def _abort_task(
        self,
        project: Project,
        branch_name: Optional[str],
        base_branch: Optional[str],
        snapshot: Optional[Dict],
    ):
        """Revert a failed task: delete branch or restore file snapshots."""
        if branch_name and base_branch:
            self._git(project, "git reset --hard")
            self._git(project, f"git checkout {shlex.quote(base_branch)}")
            self._git(project, f"git branch -D {shlex.quote(branch_name)}")
            logger.info(f"Aborted and deleted branch: {branch_name}")
        elif snapshot is not None:
            self._revert_files(project, snapshot)

    def _git(self, project: Project, command: str) -> tuple:
        try:
            proc = subprocess.run(
                command, shell=True, cwd=project.path,
                capture_output=True, text=True, timeout=30,
            )
            output = proc.stdout
            if proc.stderr:
                output += "\n" + proc.stderr
            return proc.returncode == 0, output.strip()
        except Exception as e:
            return False, str(e)

    # ----------------------------------------------------------------
    # Command execution and file operations
    # ----------------------------------------------------------------

    def _run_command(self, project: Project, command: str, timeout: int = 120) -> tuple:
        logger.info(f"Running: {command} (timeout={timeout}s)")
        cmd = command
        if project.env_manager:
            activation = project.env_manager.activate_command()
            cmd = f"{activation} && {command}"
        try:
            proc = subprocess.run(
                cmd, shell=True, cwd=project.path,
                capture_output=True, text=True, timeout=timeout,
            )
            output = proc.stdout
            if proc.stderr:
                output += "\n" + proc.stderr
            return proc.returncode == 0, output
        except subprocess.TimeoutExpired:
            return False, f"Command timed out after {timeout}s: {command}"
        except Exception as e:
            return False, str(e)

    def _snapshot_files(self, project: Project, files: List[str]) -> Dict[str, Optional[str]]:
        snapshot = {}
        for file_path in files:
            full_path = project.path / file_path
            if full_path.exists():
                try:
                    snapshot[file_path] = full_path.read_text(encoding="utf-8")
                except (UnicodeDecodeError, OSError):
                    snapshot[file_path] = None
            else:
                snapshot[file_path] = None
        return snapshot

    def _revert_files(self, project: Project, snapshot: Dict[str, Optional[str]]) -> None:
        for file_path, content in snapshot.items():
            full_path = project.path / file_path
            if content is None:
                if full_path.exists():
                    full_path.unlink()
            else:
                write_file(full_path, content)

    def _record_success(self, task: Task, files: List[str]) -> None:
        self.scratchpad.record(
            category="pattern",
            discovery=f"Task {task.id} completed successfully",
            task_id=task.id,
            files=files,
            tags=[task.category],
        )

    def _get_processor_config(self, project: Project) -> dict:
        processors = project.config.get("task_processors", [])
        for p in processors:
            if p.get("type") == "markdown_planner":
                return p.get("settings", {})
        return {}

    def _get_code_context(
        self,
        project: Project,
        target_files: List[str],
        context_files: List[str],
        max_lines: int = 500,
    ) -> str:
        context = ""
        if target_files:
            context += "### Files to Modify/Create\n"
            for file_path in target_files:
                context += self._read_file_for_context(project.path / file_path, file_path, max_lines)
        if context_files:
            context += "\n### Reference/Context Files (Read-Only)\n"
            for file_path in context_files:
                context += self._read_file_for_context(project.path / file_path, file_path, max_lines)
        return context

    def _read_file_for_context(self, full_path: Path, rel_path: str, max_lines: int) -> str:
        if not full_path.exists():
            return f"\n# File: {rel_path} (Does not exist yet)\n"
        try:
            content = full_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            return f"\n# File: {rel_path} (binary or unreadable, skipped)\n"
        lines = content.splitlines()
        if len(lines) > max_lines:
            content = "\n".join(lines[:max_lines]) + f"\n... ({len(lines)} lines total, truncated)"
        return f"\n# File: {rel_path}\n{content}\n"

    def _invoke_llm(self, project: Project, prompt: str, system_prompt: str):
        """Call the LLM, optionally streaming with early abort (config opt-in).

        Returns (text, aborted). Streaming is enabled via llm.streaming=true.
        """
        if project.config.get("llm", {}).get("streaming"):
            resp = project.llm_client.generate_stream(prompt, system_prompt, abort_check=code_gen_abort_check)
            return resp.content, resp.finish_reason == "aborted"
        return project.llm_client.generate_code(prompt, system_prompt), False

    def _resolve_model(self, project: Project, task: Task, strategy: str) -> Optional[str]:
        """Pick a model for this task from llm.routing/llm.models config.

        Routes by task complexity first, then strategy. Returns None when no
        routing is configured so the client keeps its default model.
        """
        llm_cfg = project.config.get("llm", {})
        routing = llm_cfg.get("routing", {})
        models = llm_cfg.get("models", {})
        if not routing or not models:
            return None
        tier = routing.get(task.complexity) or routing.get(strategy)
        if not tier:
            return None
        return models.get(tier) or models.get("default")

    def _build_error_context(self, prior_errors: List[str], attempt: int, output: str,
                             classified: str, attributed_error: str) -> str:
        """Combine the current error with a summary of prior failed attempts.

        Surfacing what already failed (and how it was classified) stops the LLM
        from re-submitting the same broken fix across retries.
        """
        prior_errors.append(f"Attempt {attempt + 1}: {classify_error(output)}")
        history = ""
        if len(prior_errors) > 1:
            past = "\n".join(f"- {e}" for e in prior_errors[:-1])
            history = (
                "### Previous Attempt Failures (a different approach is required)\n"
                f"{past}\n\n"
            )
        return f"{history}{classified}\n\n{attributed_error}"

    def _validate_edit_paths(self, project: Project, task: Task, edits: Dict[str, str]) -> Dict[str, str]:
        """Reject hallucinated or out-of-scope edits before they touch disk.

        Drops edits that escape the project root (absolute paths, ``..``
        traversal) or that would create empty files, and warns when the LLM
        touches files outside the task's declared scope. This is what prevents
        a misrouted edit from clobbering files outside the project.
        """
        project_root = project.path.resolve()
        expected = set(task.files_to_modify + task.files_to_create)
        valid: Dict[str, str] = {}
        for path, content in edits.items():
            if ".." in Path(path).parts or Path(path).is_absolute():
                logger.error(f"Rejected edit with unsafe path: {path}")
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
            template = getattr(tool, "config", {}).get("command", "") if hasattr(tool, "config") else ""
            if "{path}" in template:
                for file_path in file_list:
                    tool.execute(project, file_path=file_path)
            else:
                tool.execute(project)

    def _complete_task(self, project: Project, task: Task, msg: str, logs: str) -> ExecutionResult:
        project.task_manager.update_task_status(task.id, "completed")
        return ExecutionResult(status="completed", message=msg, logs=logs)

    def _fail_task(self, project: Project, task: Task, msg: str, logs: str = "") -> ExecutionResult:
        project.task_manager.update_task_status(task.id, "failed")
        return ExecutionResult(status="failed", message=msg, logs=logs)