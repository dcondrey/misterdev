"""The Try-Test-Fix execution loop (``execute``)."""

from typing import List, Optional

from misterdev.core.economics.context_budget import ContextBudget
from misterdev.core.models import Task, ExecutionResult
from misterdev.core.execution.project import Project
from misterdev.core.verification.validator import (
    CodeValidator,
    CertaintyScorer,
)
from misterdev.core.execution.error_resolver import ErrorResolver
from misterdev.core.execution.error_classifier import (
    format_classified_error,
)
from misterdev.core.execution.escalation import (
    choose_rung,
    should_count_failure,
)
from misterdev.llm.client import CACHE_BREAKPOINT, SYSTEM_CACHE_SPLIT
from misterdev.llm.prompt_manager import PromptManager
from misterdev.config import get_setting
from misterdev.core.context.guidance import guidance_for_files

from .helpers import (
    logger,
    _is_golden_path,
    _detect_language,
    _extract_needs_input,
    EDIT_FORMAT_INSTRUCTIONS,
    FULL_FILE_FALLBACK_INSTRUCTIONS,
    NEEDS_INPUT_INSTRUCTION,
)


class ExecuteMixin:
    def execute(
        self, task: Task, project: Project, use_git_branch: bool = True, _depth: int = 0
    ) -> ExecutionResult:
        logger.info(f"Starting execution of task: {task.id}")
        project.task_manager.update_task_status(task.id, "in_progress")

        prompt_manager = PromptManager(project.config)
        processor_config = self._get_processor_config(project)

        # Read max_task_attempts from orchestrator config, falling back to processor config, then hardcoded default
        default_attempts = get_setting(
            project.config, "orchestrator", "max_task_attempts"
        )
        max_retries = processor_config.get("max_retries_per_task", default_attempts)

        # Read context budget tokens from orchestrator config or task processor_data
        context_budget_tokens = task.processor_data.get(
            "context_budget_tokens",
            get_setting(project.config, "orchestrator", "context_budget_tokens"),
        )

        # Minimum LLM certainty required to accept a task as completed when no
        # test gate ran. Deterministic: compared against the heuristic certainty
        # score only, never another LLM call.
        certainty_threshold = get_setting(
            project.config, "orchestrator", "certainty_threshold"
        )

        # Per-task acceptance gate. Cheap deterministic command path is on by
        # default; the LLM-judge fallback is off by default so the default path
        # adds zero extra LLM calls. When no command is found and the judge is
        # off, acceptance is a no-op (behaviour identical to before this gate).
        verify_acceptance = get_setting(
            project.config, "orchestrator", "verify_acceptance"
        )
        llm_acceptance_judge = get_setting(
            project.config, "orchestrator", "llm_acceptance_judge"
        )

        files_key = processor_config.get("target_files_key", "files_to_modify")
        test_cmd_key = processor_config.get("test_command_key", "test_command")
        typecheck_cmd_key = processor_config.get(
            "typecheck_command_key", "typecheck_command"
        )

        target_files = task.processor_data.get(files_key, [])
        if isinstance(target_files, str):
            target_files = [target_files]
        target_files = list(
            set(target_files + task.files_to_modify + task.files_to_create)
        )

        # Golden suite: never task the model with these files and never read
        # them into its context. Combined with the edit-time rejection in
        # _validate_edit_paths, the model cannot see or alter them.
        golden_paths = get_setting(project.config, "orchestrator", "golden_paths")
        target_files = [f for f in target_files if not _is_golden_path(f, golden_paths)]
        context_files = [
            f for f in task.context_files if not _is_golden_path(f, golden_paths)
        ]

        # No declared targets (a bare issue): localize them from the symbol graph
        # so the task edits with real file context instead of blind. Runs BEFORE
        # routing so a routed target is selected from the localized files too;
        # golden-filtered; best-effort ([] leaves the prior edit-blind path).
        if not target_files:
            localized = [
                f
                for f in self._localize_target_files(project, task)
                if not _is_golden_path(f, golden_paths)
            ]
            if localized:
                logger.info(f"No target files declared; localized to: {localized}")
                target_files = localized

        # Multi-target routing: pick the sub-project that owns this task's files
        # and gate with ITS build/test/typecheck commands. No targets / no match
        # -> top-level commands, i.e. the single-target path is unchanged.
        from misterdev.core.planning.targets import (
            select_target,
            target_commands,
        )

        routed_target = select_target(project.config.get("targets") or [], target_files)
        target_cmds = target_commands(routed_target, project.config)
        # Routed gates run in the TARGET's directory (so `npm run typecheck`
        # resolves under clients/web, not the repo root). None -> project.path.
        task_cwd = None
        if routed_target is not None:
            tp = (routed_target.get("path") or "").strip("/")
            task_cwd = project.path / tp if tp else project.path
            logger.info(
                f"Task routed to target "
                f"'{routed_target.get('name') or routed_target.get('path')}' "
                f"(cwd={tp or '.'}): build={target_cmds['build_command']!r}, "
                f"test={target_cmds['test_command']!r}"
            )

        test_command = task.processor_data.get(test_cmd_key)
        if routed_target is not None and target_cmds["test_command"] is not None:
            test_command = target_cmds["test_command"]
        typecheck_command = (
            task.processor_data.get(typecheck_cmd_key)
            or target_cmds["typecheck_command"]
        )
        task_build_command = target_cmds["build_command"]
        build_timeout = get_setting(project.config, "build", "build_timeout")
        test_timeout = get_setting(project.config, "build", "test_timeout")
        # Flaky-test quarantine for the per-task gate (0 disables, matching the
        # integration gate). A red test that does not reproduce on re-run is a
        # flake and must not revert a correct edit.
        flaky_reruns = get_setting(project.config, "orchestrator", "flaky_reruns")
        # Walk-away mode: park (defer with a question) instead of failing when a
        # task can't be completed/verified. Whether ANY objective gate exists for
        # this task shapes the parked question (judgment task vs real inability).
        ask_when_stuck = get_setting(project.config, "orchestrator", "ask_when_stuck")
        has_gate = bool(task_build_command or typecheck_command or test_command)

        # Atomic execution: git branch per task (disabled in parallel mode to avoid races)
        can_branch = use_git_branch and self._is_git_repo(project)
        branch_name = f"task/{task.id}" if can_branch else None
        base_branch = None

        if branch_name:
            base_branch = self._get_current_branch(project)
            if not self._create_task_branch(project, branch_name):
                logger.warning(
                    "Failed to create task branch; falling back to file snapshots"
                )
                branch_name = None

        # Untracked files present BEFORE the task, so revert can clean only the
        # orphans this task creates without touching pre-existing untracked work.
        untracked_before = (
            self._untracked_files(project) if self._is_git_repo(project) else set()
        )

        # Fallback: file snapshots if git branching isn't available
        snapshot = None
        if not branch_name:
            snapshot = self._snapshot_files(project, target_files)

        # Seed the first attempt with the baseline test failures so a fix task
        # debugs against the REAL errors from the start, not blindly. Empty on a
        # green baseline -> error_logs stays None and attempt 0 uses the normal
        # task template (unchanged behavior).
        seed_output = getattr(project, "baseline_test_output", "") or ""
        error_logs = (
            f"The project's test suite is currently failing. Fix the failures "
            f"below (focus on this task's files):\n{seed_output[:4000]}"
            if seed_output.strip()
            else None
        )
        prior_errors: list = []
        # Accumulated failure reflections (Reflexion): each failed attempt adds a
        # short root-cause reflection that the NEXT attempt sees, so a retry fixes
        # the underlying problem rather than re-patching the symptom.
        reflections: List[str] = []
        # Count anchored-edit application failures so we can fall back to a
        # full-file rewrite when SEARCH/REPLACE keeps not matching (a stall that
        # otherwise makes no progress across attempts).
        apply_failures = 0
        # Build the symbol graph now (idempotent, lazy): it isn't constructed at
        # project registration anymore, and task execution is the first consumer.
        try:
            project.topography.initialize()
        except Exception as e:
            logger.warning(f"Topography init failed (continuing without graph): {e}")
        resolver = ErrorResolver(project.path, project.topography.graph)
        # Every in-root path the LLM actually wrote, across all attempts. Commit
        # exactly these (plus declared targets) so an out-of-scope-but-valid
        # edit isn't applied-then-orphaned by staging only the declared files.
        edited_files: set = set()
        # Original content of every file the task touches, captured the FIRST time
        # each is about to be edited (before any attempt mutates it). The optional
        # post-pass changed-region mutation check diffs this task-level baseline
        # against the final committed file — NOT the last attempt's pre-edit, which
        # would miss a fix that landed on an earlier attempt. A new file maps to "".
        task_pre_edit: dict = {}

        # Optional adversarial critic (independent second component): reviews each
        # candidate edit before it is applied and can force a regeneration with
        # concrete objections. Off by default. Bounded so it defers to the real
        # gates after a few rejections rather than starving the attempt loop.
        critic_enabled = self._critic_enabled_for(project, task)
        critic_max_rejections = get_setting(
            project.config, "orchestrator", "critic_max_rejections"
        )
        critic_rejections = 0

        # Spec-as-tests (opt-in): generate a failing test from the acceptance
        # criteria BEFORE implementation, written under .orchestrator/spec_tests/
        # (outside the project suite, so it never flips the integration-gate
        # baseline). The SOURCE is injected into the edit context as the concrete
        # reproduction target (directed / TDD); after the task's gates pass it is
        # run scoped and must now pass (red -> green), blocking under
        # spec_as_tests_block, advisory otherwise.
        spec_test_path, spec_test_source = self._maybe_generate_spec_test(
            project, task, validate_timeout=test_timeout
        )
        spec_test_block = get_setting(
            project.config, "orchestrator", "spec_as_tests_block"
        )

        # The most recent attempt awaiting an outcome. Recorded as a failure at
        # the top of the next iteration (we only loop again when an attempt
        # failed) and at loop exit; recorded as a success at the success seams.
        pending_attempt: Optional[dict] = None
        # Baseline full-suite failure count (from analysis), so a RED baseline
        # doesn't reject every task. The test gate then accepts an attempt that
        # leaves the suite no worse (failures <= baseline), letting a multi-failure
        # project be reduced incrementally instead of demanding one task fix the
        # whole suite. 0 (a green baseline) keeps the gate strictly green-only.
        baseline_failures = int(getattr(project, "baseline_test_failures", 0) or 0)
        if routed_target is not None:
            # Never apply the top-level (e.g. core) baseline to a DIFFERENT
            # target's gate. Use that target's own measured baseline if present,
            # else 0 (strict green-only) — far safer than inheriting core's count.
            per_target = getattr(project, "target_baselines", None) or {}
            key = routed_target.get("name") or routed_target.get("path")
            baseline_failures = int(per_target.get(key, 0) or 0)
        _selector = getattr(project, "model_selector", None)
        track_models = (
            _selector is not None
            and _selector.enabled
            and hasattr(project.llm_client, "task_cost")
        )

        # Whole-project structural map (every file + its symbols). Computed once
        # — it is identical across attempts — so the model edits with the entire
        # project's shape in view, not just the target files. ContextBudget
        # trims it first (priority 3) when space is tight.
        topo = getattr(project, "topography", None)
        project_outline = topo.get_project_outline() if topo is not None else ""

        # Cross-build warm-start: seed THIS task with how the nearest previously-
        # solved tasks were done, keyed on the task description. Computed once (it
        # is identical across attempts); "" when the index is empty or nothing
        # matches. Closes the learning loop at the executor, not just the planner.
        solved_priors = self._solved_task_priors(project, task)

        # Shared task-list preamble: the global conventions a numbered devplan
        # states once up front (canonical constants, locked dependency versions,
        # "never guess" rules) that every task must honor. The tasklist parser
        # attaches it to each task; inject it so the model sees those constraints
        # on this task, not only at planning time. "" when the plan had no preamble.
        shared_context = str(task.processor_data.get("shared_context", "") or "")

        # The user's answer to a question this task was parked on in a prior run
        # (loaded by run_project from QUESTIONS.md). A direct instruction — inject
        # it at top priority so the retry follows it. "" for a first-time task.
        user_answer = str(task.processor_data.get("user_answer", "") or "")

        # Optional, off-by-default agentic pre-edit gathering: when enabled and
        # an MCP manager with tools exists, the model may request bounded MCP
        # tool calls to gather information. The result is prepended to the task
        # context; off → "" and the path below is byte-identical to today.
        mcp_gathered = self._mcp_gather(project, task)
        # Optional, off-by-default runtime tool-invention (two-timescale P2): the
        # model may author a small helper tool that runs sandboxed, its output fed
        # into the edit context. Off → "" and the path below is byte-identical.
        runtime_tool_ctx = self._runtime_tool(project, task)

        # A no-usable-edit response (not code, an anchor miss, or no edit at all)
        # changed nothing on disk — it is a formatting failure, not a solve
        # attempt — so it must not consume the solve budget. Grant a bounded
        # number of EXTRA iterations for such no-output responses (the model still
        # escalates a tier each iteration); the cap keeps the empty-response spin
        # the escalation prompt already guards from ever becoming unbounded.
        attempt = -1
        attempt_cap = max_retries
        no_output_forgiven = 0
        forgiveness_cap = 2
        # Escalation ladder: count only NON-infra (real code) gate failures, and
        # climb widen_context -> stronger_model -> decompose as they accumulate.
        # Infra faults self-heal and must never advance the ladder. Off -> the
        # counter stays 0 and every attempt is the plain "normal" rung.
        escalation_on = get_setting(
            project.config, "orchestrator", "escalation_enabled"
        )
        code_failures = 0
        while True:
            attempt += 1
            if attempt >= attempt_cap:
                break
            logger.info(f"Attempt {attempt + 1}/{attempt_cap} for task {task.id}")
            # Pick this attempt's rung from the code-failure count so far. On the
            # decompose rung, stop retrying the whole task and request a split into
            # named sub-steps instead of burning the remaining attempts.
            rung = (
                self._escalation_rung(project, code_failures)
                if escalation_on
                else "normal"
            )
            if rung == "decompose":
                logger.info(
                    f"Task {task.id} escalated to DECOMPOSE after {code_failures} "
                    "code failure(s); requesting a split into sub-steps."
                )
                self._abort_task(
                    project, branch_name, base_branch, snapshot, untracked_before
                )
                return self._escalate_decompose(project, task, error_logs)
            widen_context = rung in ("widen_context", "stronger_model")
            force_stronger = rung == "stronger_model"
            # Diagnostic: sizes of the context that ACCUMULATES across attempts,
            # so growth (or a runaway component) is visible per retry.
            logger.debug(
                f"[attempt-ctx] task={task.id} #{attempt + 1} "
                f"error_logs={len(error_logs or '')}c "
                f"reflections={len(reflections)}x/{sum(len(r) for r in reflections)}c "
                f"prior_errors={len(prior_errors)}"
            )
            if pending_attempt is not None:
                self._ledger_record(project, task, pending_attempt, success=False)
                pending_attempt = None

            # Reflexion: a prior attempt failed (error_logs set). Before retrying,
            # reflect on the ROOT CAUSE and fold the running reflections into the
            # error context, so this attempt fixes the underlying problem rather
            # than re-patching the symptom. One central seam covers every gate.
            if attempt > 0 and error_logs:
                error_logs = self._apply_reflection(
                    project, task, error_logs, reflections
                )
                # Fold in the language server's semantic diagnostics for the
                # touched files — per-file, per-line errors the compiler's raw
                # stderr may not spell out. Gated behind lsp_diagnostics (off by
                # default) and bounded, so it only costs a server round-trip on a
                # retry when the user opted in; "" when the LSP has no opinion.
                if get_setting(project.config, "orchestrator", "lsp_diagnostics"):
                    from misterdev.core.context.lsp import (
                        collect_and_format_lsp_context,
                    )

                    lsp_ctx = collect_and_format_lsp_context(
                        project.path,
                        project.config.get("language") or "",
                        target_files,
                    )
                    if lsp_ctx:
                        error_logs = f"{error_logs}\n\n{lsp_ctx}"

                # Query-on-failure: on the FIRST failure, let the model consult
                # MCP tools about the actual error (look up a doc / API / the
                # error's meaning) and fold the result into the gather context for
                # every subsequent attempt. Bounded to once per task; "" when MCP
                # tool use is off, so the path is unchanged for non-MCP builds.
                if attempt == 1:
                    failure_ctx = self._mcp_gather(
                        project, task, error_context=error_logs
                    )
                    if failure_ctx:
                        mcp_gathered = f"{mcp_gathered}{failure_ctx}"

            code_context = self._get_code_context(
                project, target_files, context_files, task=task
            )
            topo_context = project.topography.get_context_for_task(
                task.description,
                target_files,
                ranker=getattr(project, "semantic_ranker", None),
                exclude_files=self._fully_shown_target_files(project, target_files),
            )
            # Exhaustive external references to the symbols being edited, so a
            # delete/rename/refactor updates every call site in one attempt
            # instead of chasing them one build-error at a time (the dominant
            # failure mode: missing_symbol/wrong_type across attempts).
            reference_sites = project.topography.reference_sites(target_files)
            scratchpad_context = self.scratchpad.format_context(
                files=target_files + context_files,
                tags=[task.category],
            )

            strategy = task.processor_data.get("strategy", "iterative")
            consensus = task.processor_data.get("consensus_context", "None")
            interface_contracts = task.processor_data.get("interface_contracts", "")
            recent_changes = task.processor_data.get("recent_changes", "")

            # Budget-aware context allocation using configured token limit
            budget = ContextBudget(max_tokens=context_budget_tokens)
            budget.set("code_context", code_context, priority=1)
            # Correctness-critical: the complete reference set must survive
            # truncation, or a rename/delete misses sites and the build fails.
            budget.set("reference_sites", reference_sites, priority=1, min_lines=0)
            # The self-authored reproduction test IS the task's concrete target;
            # it must survive truncation so the model always edits toward it.
            budget.set("spec_test", spec_test_source or "", priority=1, min_lines=0)
            # topo_context is the task-ranked (relevant-first) symbol-graph map
            # that cross-file tasks (delete-all-refs, wire call-sites) need. At
            # priority=3/min_lines=10 it collapsed to ~10 lines on a large repo
            # (the emathy run: "kept 11/9094"), editing blind. Keep it above the
            # truncate-first tier with a real floor so the top-ranked symbols
            # survive; truncation still trims the long tail under pressure.
            budget.set("topo_context", topo_context, priority=2, min_lines=40)
            budget.set("error_logs", error_logs or "", priority=1, min_lines=20)
            budget.set("scratchpad", scratchpad_context, priority=3)
            budget.set("solved_priors", solved_priors, priority=3, min_lines=0)
            # The user's own directive for this task must survive truncation.
            budget.set("user_answer", user_answer, priority=1, min_lines=0)
            budget.set("interface_contracts", interface_contracts, priority=2)
            budget.set("recent_changes", recent_changes, priority=2)
            budget.set("consensus_context", consensus, priority=3)
            # Token efficiency: the whole-project outline duplicates what the
            # task-ranked topo_context + reference_sites already give a FOCUSED task
            # (one with explicit target files and a substantial relevant map), so
            # drop it there. Keep it for a bare/localized task (no targets), which
            # needs whole-project awareness to find where to edit.
            outline = (
                "" if (target_files and len(topo_context) > 800) else project_outline
            )
            budget.set("project_outline", outline, priority=3, min_lines=0)
            allocated = budget.allocate()

            full_code_context = allocated["code_context"]
            # Only present the reproduction test as the concrete target when the
            # FULL source survived allocation. Under extreme context pressure the
            # allocator can trim this section to a fragment; a partial test framed
            # as "the exact test to pass" would actively mislead, so in that rare
            # case drop it rather than point the model at a stub.
            if (
                allocated["spec_test"]
                and allocated["spec_test"].strip() == (spec_test_source or "").strip()
            ):
                full_code_context += (
                    "\n\n## Reproduction test — your change MUST make this pass\n"
                    "This executable test defines DONE for this task. Write the "
                    "implementation so this exact test passes; do not edit the "
                    "test itself.\n\n" + allocated["spec_test"]
                )
            if allocated["reference_sites"]:
                full_code_context += "\n\n" + allocated["reference_sites"]
            if allocated["topo_context"]:
                full_code_context += "\n\n" + allocated["topo_context"]
            if allocated["recent_changes"]:
                full_code_context += "\n\n" + allocated["recent_changes"]
            if allocated["solved_priors"]:
                full_code_context += "\n\n" + allocated["solved_priors"]
            if allocated["user_answer"]:
                full_code_context += (
                    "\n\n## The user's answer to your earlier question (follow this)\n"
                    + allocated["user_answer"]
                )
            if allocated["project_outline"]:
                full_code_context += (
                    "\n\n## Project Structure (files and their symbols)\n"
                    + allocated["project_outline"]
                )
            if mcp_gathered:
                full_code_context += mcp_gathered
            if runtime_tool_ctx:
                full_code_context += runtime_tool_ctx
            # Escalation: at the widen rung, re-anchor the model on the FULL,
            # verbatim task spec (description + acceptance) and the exact target
            # files and their dependents, so a fix that kept missing the point
            # under a truncated/partial view sees the whole picture.
            if widen_context:
                full_code_context += self._escalation_spec_block(
                    project, task, target_files
                )
            full_code_context += self._mcp_awareness(project)

            guidance_context = " ".join(
                str(s)
                for s in (
                    task.description,
                    task.acceptance_criteria,
                    allocated["error_logs"],
                    full_code_context,
                )
                if s
            )
            language_guidance = guidance_for_files(
                target_files,
                project.config.get("language") or "",
                context=guidance_context,
            )
            context_dict = {
                "project": project,
                "task": task,
                "code_context": full_code_context,
                "error_logs": allocated["error_logs"] or None,
                "task.description": task.description,
                "task.target_files": ", ".join(target_files)
                if target_files
                else "None explicitly specified",
                "scratchpad": allocated["scratchpad"],
                "acceptance_criteria": task.acceptance_criteria,
                "consensus_context": allocated["consensus_context"],
                "interface_contracts": allocated["interface_contracts"],
                "strategy": strategy.upper(),
                "invariants": (
                    f"Strategy: {strategy.upper()}. Output MUST be syntactically valid. "
                    "Provide certainty indicators."
                    + (f"\n\n{language_guidance}" if language_guidance else "")
                ),
                # Marks the boundary between the cacheable stable context above
                # and the volatile tail below (see PROMPT_TEMPLATES); the client
                # caches everything before it.
                "cache_breakpoint": CACHE_BREAKPOINT,
            }

            system_prompt = prompt_manager.format_prompt("system", context_dict)
            # Token efficiency: the plan's global conventions (shared_context) are
            # BYTE-IDENTICAL across every task, so put them at the very front of the
            # system prompt with a cache split. The prefix is then created once and
            # re-read across ALL tasks (Claude), and its stable leading position
            # engages OpenAI/Gemini automatic prefix caching too — same content the
            # model saw before, just billed once instead of per task.
            if shared_context:
                system_prompt = (
                    "## Plan-wide conventions (apply to every task)\n"
                    + shared_context
                    + SYSTEM_CACHE_SPLIT
                    + system_prompt
                )
            # Use the error-correction template whenever failures are known —
            # including a seeded attempt 0 on a red baseline — so the model always
            # sees the actual failures to fix rather than editing blind.
            if error_logs:
                prompt = prompt_manager.format_prompt(
                    "error_correction_instruction", context_dict
                )
            else:
                prompt = prompt_manager.format_prompt(
                    "task_completion_instruction", context_dict
                )
            # Tool-based edit extraction returns whole-file content via a forced
            # tool call; the SEARCH/REPLACE contract applies only to the plain
            # text-generation path, where the parser expects anchored hunks.
            if not get_setting(project.config, "llm", "use_tools"):
                # After repeated anchor-match failures, switch to a full-file
                # rewrite (no anchoring → always applies), breaking the stall.
                if apply_failures >= 2:
                    prompt += FULL_FILE_FALLBACK_INSTRUCTIONS
                else:
                    prompt += EDIT_FORMAT_INSTRUCTIONS

            # Walk-away escape hatch: let the model ask rather than guess when a
            # human decision is genuinely required. Strictly worded so it does not
            # defer work it could simply do.
            if ask_when_stuck:
                prompt += NEEDS_INPUT_INSTRUCTION

            routed_model = self._select_model(
                project, task, strategy, attempt, attempt_cap
            )
            # Escalation: at the stronger-model rung, override the routed model
            # with the configured stronger one (if any) — the cheap/default model
            # has failed on real code repeatedly, so pay for a more capable one.
            if force_stronger:
                stronger = get_setting(
                    project.config, "orchestrator", "escalation_model"
                )
                if stronger:
                    logger.info(
                        f"Escalation: routing {task.id} to stronger model "
                        f"{stronger} after {code_failures} code failure(s)."
                    )
                    routed_model = stronger
            try:
                with project.llm_client.track_task(task.id):
                    llm_response, aborted, pending_attempt = self._invoke_routed(
                        project,
                        task,
                        prompt,
                        system_prompt,
                        routed_model,
                        attempt,
                        track_models,
                    )
            except Exception as e:
                msg = f"LLM generation failed: {e}"
                logger.error(msg)
                self._ledger_record(
                    project,
                    task,
                    {
                        "model": routed_model
                        or getattr(project.llm_client, "model", ""),
                        "attempt": attempt,
                        "cost_before": 0.0,
                        "latency": 0.0,
                        "aborted": False,
                    },
                    success=False,
                )
                self._abort_task(
                    project, branch_name, base_branch, snapshot, untracked_before
                )
                return self._fail_task(project, task, msg)

            if aborted:
                logger.warning(
                    f"LLM stream aborted for {task.id}; retrying with stricter instruction."
                )
                error_logs = "ERROR: response was not code. Output ONLY file edits as code blocks with file paths."
                if no_output_forgiven < forgiveness_cap:
                    no_output_forgiven += 1
                    attempt_cap += 1
                continue

            # The model can explicitly ask the user rather than guess ("NEEDS_INPUT:
            # <question>"). Honor it immediately: revert this attempt's work and park
            # the task with the question, so the run moves on and the user answers
            # later. Off when walk-away mode is disabled (the marker is then ignored).
            if ask_when_stuck:
                needs = _extract_needs_input(llm_response)
                if needs:
                    logger.info(f"Task {task.id} needs user input: {needs}")
                    self._abort_task(
                        project, branch_name, base_branch, snapshot, untracked_before
                    )
                    return self._defer_task(
                        project, task, f"Model requested input: {needs}", [needs]
                    )

            certainty = CertaintyScorer.compute_score(llm_response)
            logger.info(f"LLM Certainty Score: {certainty:.2f}")

            edits, resolve_error = self._resolve_edits(project, llm_response)
            if resolve_error:
                apply_failures += 1
                logger.warning(
                    f"Surgical edit could not be applied (#{apply_failures}): "
                    f"{resolve_error}"
                )
                # Two strikes -> next attempt is told to emit the whole file
                # (FULL_FILE_FALLBACK_INSTRUCTIONS), which can't miss an anchor.
                if apply_failures >= 2:
                    error_logs = (
                        "ERROR: anchored SEARCH/REPLACE edits keep failing to "
                        "match the file. Output the COMPLETE updated file instead "
                        "(full contents in one code block with the file path)."
                    )
                else:
                    error_logs = (
                        "ERROR: a SEARCH/REPLACE edit did not apply cleanly. "
                        f"{resolve_error} Re-read the file and emit a corrected "
                        "SEARCH block that matches the current content verbatim."
                    )
                if no_output_forgiven < forgiveness_cap:
                    no_output_forgiven += 1
                    attempt_cap += 1
                continue
            edits = self._validate_edit_paths(project, task, edits)
            no_gate = not (task_build_command or typecheck_command or test_command)
            if not edits and not (no_gate and certainty >= certainty_threshold):
                # A response that yields no applicable edit is a stall, not a
                # no-op: without escalating it here the loop re-sends the same
                # prompt and the model emits nothing again, spinning until the
                # budget dies (observed: 9 empty attempts, 446K tokens). Treat it
                # like an anchor miss — count it, and after two force a full-file
                # rewrite via the apply_failures>=2 branch above — then retry
                # immediately rather than gating an unchanged tree. The exception
                # is a legitimately-editless task: no gate to satisfy and the
                # model is confident the work already holds — fall through to the
                # certainty-completion path below instead of spinning.
                apply_failures += 1
                logger.warning(
                    f"No applicable file edit in LLM response (#{apply_failures})."
                )
                if apply_failures >= 2:
                    error_logs = (
                        "ERROR: you produced no applicable file edit. Output the "
                        "COMPLETE updated file in a single code block whose opening "
                        "fence carries the file path."
                    )
                else:
                    error_logs = (
                        "ERROR: no file edit was detected in your response. Emit the "
                        "change as an anchored SEARCH/REPLACE hunk (or the complete "
                        "file) in a code block whose fence carries the file path."
                    )
                if no_output_forgiven < forgiveness_cap:
                    no_output_forgiven += 1
                    attempt_cap += 1
                continue
            if not edits:
                # Legitimately editless (guarded above): skip the edits-present
                # work and fall through to the gate/certainty logic, which
                # completes a no-gate high-certainty task.
                pass
            else:
                stall_risk = self.stall_detector.push_edit(edits)
                if stall_risk > 0.7:
                    logger.warning(f"High stall risk detected ({stall_risk:.2f}).")
                    if attempt > 1:
                        error_logs = "ERROR: Stalling detected. Try a fundamentally different approach."
                        continue

                validation_failed = False
                for file_path, content in edits.items():
                    lang = _detect_language(file_path)
                    valid, error = CodeValidator.validate_code(content, language=lang)
                    if not valid:
                        logger.error(f"Validation failed for {file_path}: {error}")
                        error_logs = f"SYNTAX ERROR in {file_path}:\n{error}"
                        validation_failed = True
                        break

                if validation_failed:
                    continue

                tamper = self._detect_test_tampering(project, edits)
                if tamper:
                    logger.error(f"Test tampering rejected: {tamper}")
                    error_logs = (
                        "ERROR: test files were weakened. Do not delete "
                        "tests, weaken assertions, or add skip/ignore markers "
                        "to make the suite pass. Fix the real code instead. "
                        f"Detected: {tamper}"
                    )
                    continue

                dangling = self._detect_dangling_references(project, edits)
                if dangling:
                    logger.warning(f"Incomplete refactor — dangling refs: {dangling}")
                    error_logs = (
                        "ERROR: this edit removed or renamed a symbol but left "
                        "references to it in files you did not change. Update EVERY "
                        "one of these call sites in this attempt (do not fix them "
                        f"one at a time): {dangling}"
                    )
                    continue

                destructive = self._detect_destructive_rewrite(project, edits)
                if destructive:
                    logger.warning(f"Destructive rewrite rejected: {destructive}")
                    error_logs = (
                        "ERROR: this edit deletes real functionality — it removes "
                        "definitions and collapses the file, which passes the "
                        "immediate test only by stripping behavior other code relies "
                        "on. Make the SMALLEST change that fixes the problem: keep "
                        "every existing public definition and its implementation, and "
                        f"add or adjust only what is needed. Detected: {destructive}"
                    )
                    continue

                # Independent adversarial critique BEFORE applying. A rejection
                # feeds concrete objections back as the next attempt's context
                # (regenerate), bounded by critic_max_rejections so an
                # over-zealous critic can't starve the loop — after that the edit
                # flows to the authoritative build/test gates. SKIP/APPROVE fall
                # through and apply as normal; off path is unchanged.
                if critic_enabled and critic_rejections < critic_max_rejections:
                    verdict = self._run_edit_critic(project, task, edits)
                    if verdict.rejected:
                        critic_rejections += 1
                        logger.warning(
                            f"Adversarial critic rejected the edit on attempt "
                            f"{attempt + 1} ({critic_rejections}/"
                            f"{critic_max_rejections}): {verdict.objections}"
                        )
                        error_logs = self._build_critic_error_context(
                            verdict.objections
                        )
                        continue

                # Record each touched file's ORIGINAL content once — first touch
                # wins so the baseline survives across attempts (see task_pre_edit
                # above). Used by the optional post-pass suite-strength check.
                for p in edits:
                    if p not in task_pre_edit:
                        fp = project.path / p
                        task_pre_edit[p] = (
                            fp.read_text(encoding="utf-8") if fp.exists() else ""
                        )
                self._apply_edits(project, edits)
                self._run_formatters(project, edits.keys())
                edited_files.update(edits.keys())

            # Whether an OBJECTIVE compile/type gate passed this attempt. A green
            # build or typecheck IS verification, so a typecheck-only target (a
            # frontend with no unit tests) must NOT then also be gated on the
            # LLM's self-reported certainty — that wrongly rejects good edits.
            gate_verified = False
            build_cmd = task_build_command
            if build_cmd:
                success, output = self._run_gate(
                    project, build_cmd, build_timeout, task_cwd
                )
                if not success:
                    logger.warning(f"Build failed on attempt {attempt + 1}")
                    if should_count_failure(output):
                        code_failures += 1
                    locations = resolver.resolve_errors(output)
                    attributed_error = resolver.format_for_llm(locations)
                    classified = format_classified_error(output)
                    error_logs = self._build_error_context(
                        prior_errors,
                        attempt,
                        output,
                        classified,
                        attributed_error,
                        project=project,
                        test_command=test_command,
                        language=project.config.get("language"),
                        cwd=task_cwd,
                    )
                    continue
                gate_verified = True

            if typecheck_command:
                success, output = self._run_gate(
                    project, typecheck_command, build_timeout, task_cwd
                )
                if not success:
                    logger.warning(f"Type check failed on attempt {attempt + 1}")
                    if should_count_failure(output):
                        code_failures += 1
                    locations = resolver.resolve_errors(output)
                    attributed_error = resolver.format_for_llm(locations)
                    classified = format_classified_error(output)
                    error_logs = self._build_error_context(
                        prior_errors,
                        attempt,
                        output,
                        classified,
                        attributed_error,
                        project=project,
                        test_command=test_command,
                        language=project.config.get("language"),
                        cwd=task_cwd,
                    )
                    continue
                gate_verified = True

            if test_command:
                success, output = self._run_gate(
                    project, test_command, test_timeout, task_cwd
                )
                # Whether the command genuinely exited zero on THIS tree — kept
                # separate from a flake rescue below so the acceptance short-circuit
                # only ever skips a re-run of a real green (not a flaked-then-passed).
                test_exited_green = success
                if not success and self._confirm_flaky(
                    project, test_command, output, test_timeout, task_cwd, flaky_reruns
                ):
                    success = True
                accepted, post_failures = self._gate_accepts(
                    success, output, baseline_failures
                )
                if accepted:
                    if success:
                        logger.info("Tests passed successfully.")
                    else:
                        logger.info(
                            f"Suite still red ({post_failures} failing) but not "
                            f"worse than the baseline ({baseline_failures}); "
                            "accepting incremental progress."
                        )
                        if post_failures < baseline_failures:
                            # Ratchet the baseline down so a later task can't
                            # re-introduce what this one fixed.
                            project.baseline_test_failures = post_failures
                            baseline_failures = post_failures
                    acc_ok, acc_output = self._verify_acceptance(
                        project,
                        task,
                        verify_acceptance,
                        llm_acceptance_judge,
                        test_timeout,
                        cwd=task_cwd,
                        # Only a genuinely GREEN test run (exit 0) lets acceptance
                        # skip an identical command; a red-but-accepted-under-
                        # baseline run OR a flake-rescued one must still be
                        # re-checked, so key on the raw result, not the flake flip.
                        already_passed=test_command if test_exited_green else None,
                    )
                    if not acc_ok:
                        logger.warning(
                            f"Acceptance criteria not met on attempt {attempt + 1}."
                        )
                        if should_count_failure(acc_output):
                            code_failures += 1
                        classified = format_classified_error(acc_output)
                        error_logs = self._build_acceptance_error_context(
                            prior_errors, attempt, task, classified
                        )
                        continue
                    # Spec-as-tests: the pre-written failing test must now pass.
                    spec_status, spec_detail = self._run_spec_test(
                        project, spec_test_path, test_timeout
                    )
                    if spec_status == "red" and spec_test_block:
                        logger.warning(
                            f"Spec test still fails on attempt {attempt + 1}; "
                            "implementation does not satisfy the spec."
                        )
                        error_logs = self._build_acceptance_error_context(
                            prior_errors,
                            attempt,
                            task,
                            format_classified_error(spec_detail),
                        )
                        continue
                    if spec_status == "red":
                        logger.warning(
                            f"Spec test for {task.id} still fails (advisory; not "
                            "blocking). The implementation may not fully satisfy "
                            "the acceptance criterion."
                        )
                        task.processor_data.setdefault("spec_test_gaps", []).append(
                            spec_detail[:200]
                        )
                    self._ledger_record(project, task, pending_attempt, success=True)
                    self._cache_store(
                        project,
                        system_prompt,
                        prompt,
                        llm_response,
                        pending_attempt["model"],
                    )
                    pending_attempt = None
                    self._record_success(task, target_files)
                    # Optional suite-strength check: with the tests green, mutate the
                    # fix's changed region and confirm the suite actually kills the
                    # mutants (advisory unless a floor is configured).
                    self._changed_region_mutation_check(
                        project, task_pre_edit, test_command, task_cwd
                    )
                    # Persist status BEFORE committing so the source markdown's
                    # status:completed is part of the task commit and survives the
                    # merge; otherwise the next task's checkout discards it.
                    project.task_manager.update_task_status(task.id, "completed")
                    self._commit_task(
                        project,
                        branch_name,
                        base_branch,
                        task,
                        sorted(set(target_files) | edited_files),
                    )
                    return self._complete_task(
                        project, task, "Task completed and tests passed.", output
                    )
                else:
                    logger.warning(f"Tests failed on attempt {attempt + 1}.")
                    if should_count_failure(output):
                        code_failures += 1
                    locations = resolver.resolve_errors(output)
                    attributed_error = resolver.format_for_llm(locations)
                    classified = format_classified_error(output)
                    error_logs = self._build_error_context(
                        prior_errors,
                        attempt,
                        output,
                        classified,
                        attributed_error,
                        project=project,
                        test_command=test_command,
                        language=project.config.get("language"),
                        cwd=task_cwd,
                    )
            elif certainty < certainty_threshold and not gate_verified:
                # No test gate AND no compile/type gate ran, so completion rests
                # entirely on the LLM's word. With low certainty that's not enough
                # to trust: force a retry (which escalates to surgical once
                # attempts run out) instead of silently accepting unverified code.
                # A green build/typecheck (gate_verified) is objective proof and
                # bypasses this — the common frontend case (typecheck, no tests).
                logger.warning(
                    f"No tests/build verified this and certainty {certainty:.2f} < "
                    f"{certainty_threshold:.2f}; refusing silent completion."
                )
                error_logs = (
                    "ERROR: no tests verify this change and the response "
                    "expressed low certainty. Provide a higher-confidence, "
                    "syntactically valid edit with explicit verification."
                )
                continue
            else:
                acc_ok, acc_output = self._verify_acceptance(
                    project,
                    task,
                    verify_acceptance,
                    llm_acceptance_judge,
                    test_timeout,
                    cwd=task_cwd,
                )
                if not acc_ok:
                    logger.warning(
                        f"Acceptance criteria not met on attempt {attempt + 1}."
                    )
                    if should_count_failure(acc_output):
                        code_failures += 1
                    classified = format_classified_error(acc_output)
                    error_logs = self._build_acceptance_error_context(
                        prior_errors, attempt, task, classified
                    )
                    continue
                self._ledger_record(project, task, pending_attempt, success=True)
                self._cache_store(
                    project,
                    system_prompt,
                    prompt,
                    llm_response,
                    pending_attempt["model"],
                )
                pending_attempt = None
                self._record_success(task, target_files)
                # A task can merge on certainty/acceptance WITHOUT any test gate —
                # exactly where a weak suite most needs surfacing. Score the fix
                # against the project suite (the seam falls back to it).
                self._changed_region_mutation_check(
                    project, task_pre_edit, test_command, task_cwd
                )
                # Persist status BEFORE committing so status:completed rides into
                # the task commit and survives the merge (see the tests-passed
                # path above).
                project.task_manager.update_task_status(task.id, "completed")
                self._commit_task(
                    project,
                    branch_name,
                    base_branch,
                    task,
                    sorted(set(target_files) | edited_files),
                )
                return self._complete_task(
                    project, task, "Task completed (no tests run).", llm_response
                )

        # The final attempt fell through without succeeding; record it before
        # escalation/failure so its model gets the failing outcome it earned.
        if pending_attempt is not None:
            self._ledger_record(project, task, pending_attempt, success=False)
            pending_attempt = None

        # Strategy escalation: if current strategy failed, try one more attempt
        # with "surgical". Guarded by _depth so escalation can never recurse more
        # than once, even if the strategy-selection logic changes later.
        current_strategy = task.processor_data.get("strategy", "iterative")
        if current_strategy != "surgical" and _depth < 1:
            logger.info(
                f"Escalating strategy from {current_strategy} to surgical for final attempt"
            )
            self._abort_task(
                project, branch_name, base_branch, snapshot, untracked_before
            )

            task.processor_data["strategy"] = "surgical"
            task.processor_data["invariants"] = (
                "ESCALATED: Previous strategy failed. Use SURGICAL approach: "
                "make the smallest possible change to fix the immediate error. "
                "Do not refactor or restructure. Minimal, targeted fix only."
            )
            # Recursive single attempt with surgical strategy
            return self.execute(
                task, project, use_git_branch=use_git_branch, _depth=_depth + 1
            )
        elif _depth >= 1:
            logger.warning(
                f"Strategy escalation blocked at depth {_depth}: already exhausted all strategies"
            )

        logger.warning(
            f"Task {task.id} failed after all attempts including escalation. Reverting."
        )
        self._abort_task(project, branch_name, base_branch, snapshot, untracked_before)
        self.scratchpad.record(
            category="pitfall",
            discovery=(
                f"Failed after {max_retries} attempts + escalation: "
                f"{error_logs[:200] if error_logs else 'unknown'}"
            ),
            task_id=task.id,
            files=target_files,
        )
        # Walk-away mode: don't dead-end on a stuck task — park it with a specific
        # question (missing credential, judgment call, or genuine inability) so the
        # run keeps going and the user resolves it later. The work is already
        # reverted above, so a follow-up run redoes it once answered.
        if ask_when_stuck:
            reason, question = self._deferral_reason(task, error_logs, has_gate)
            logger.warning(f"Task {task.id} parked (needs input): {reason}")
            return self._defer_task(project, task, reason, [question], error_logs)
        return self._fail_task(
            project,
            task,
            f"Task failed after {max_retries} attempts + escalation.",
            error_logs,
        )

    def _escalation_rung(self, project: Project, code_failures: int) -> str:
        """The escalation rung for the next attempt, from the config thresholds."""
        return choose_rung(
            code_failures,
            widen_after=get_setting(
                project.config, "orchestrator", "escalation_widen_after"
            ),
            model_after=get_setting(
                project.config, "orchestrator", "escalation_model_after"
            ),
            decompose_after=get_setting(
                project.config, "orchestrator", "escalation_decompose_after"
            ),
        )

    def _escalation_spec_block(self, project: Project, task: Task, target_files) -> str:
        """The verbatim task spec injected at the widen rung.

        A repeatedly-failing attempt was likely editing against a truncated view
        or drifting from the goal, so re-anchor it on the FULL objective and
        acceptance criteria plus the exact target files (shown in full above)."""
        parts = ["\n\n## Escalation — full task spec (re-read; do not deviate)"]
        if task.description:
            parts.append(f"### Objective\n{task.description}")
        criteria = (getattr(task, "acceptance_criteria", "") or "").strip()
        if criteria:
            parts.append(f"### Acceptance criteria (must be satisfied)\n{criteria}")
        if target_files:
            listed = "\n".join(f"- {f}" for f in target_files)
            parts.append(
                "### Target files (edit ONLY these; they are shown in full above "
                f"and their call sites are listed)\n{listed}"
            )
        return "\n\n".join(parts)

    @staticmethod
    def _decompose_substeps(task: Task) -> list:
        """Named sub-steps to split a stuck task into. Prefers the task's own
        acceptance criteria (each line becomes a sub-step); falls back to a generic
        isolate-then-extend split when there is nothing structured to lean on."""
        criteria = (getattr(task, "acceptance_criteria", "") or "").strip()
        items = []
        for line in criteria.splitlines():
            s = line.strip().lstrip("-*0123456789.) ").strip()
            if len(s) > 8:
                items.append(f"Implement and verify: {s}")
        if len(items) >= 2:
            return items[:5]
        label = task.title or (task.description or "")[:60] or task.id
        return [
            f"Isolate the smallest failing part of '{label}' and make it pass alone",
            f"Wire the remaining behavior of '{label}' on top, keeping the suite green",
        ]

    def _escalate_decompose(self, project: Project, task: Task, error_logs):
        """Top escalation rung: stop retrying and request a decomposition.

        Records named sub-steps on the task and parks it (deferred) with a
        decomposition request, so the convergence loop re-decomposes it into
        runnable sub-tasks instead of the run dead-ending on an over-large task.
        The caller has already reverted this task's work, so nothing is left behind.
        """
        substeps = self._decompose_substeps(task)
        task.processor_data["_escalation_decomposed"] = True
        task.processor_data["escalation_substeps"] = substeps
        reason = (
            f"escalated to decomposition: '{task.title or task.id}' failed "
            "repeatedly and is too large to land in one edit. Split it into the "
            "sub-steps below and run them."
        )
        return self._defer_task(project, task, reason, substeps, error_logs)
