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
from misterdev.llm.client import CACHE_BREAKPOINT
from misterdev.llm.prompt_manager import PromptManager
from misterdev.config import get_setting

from .helpers import (
    logger,
    _is_golden_path,
    _detect_language,
    EDIT_FORMAT_INSTRUCTIONS,
    FULL_FILE_FALLBACK_INSTRUCTIONS,
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
        prior_errors: List[str] = []
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
        # baseline). After the task's gates pass, it is run scoped and must now
        # pass (red -> green). Advisory unless spec_as_tests_block.
        spec_test_path = self._maybe_generate_spec_test(project, task)
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

        # Optional, off-by-default agentic pre-edit gathering: when enabled and
        # an MCP manager with tools exists, the model may request bounded MCP
        # tool calls to gather information. The result is prepended to the task
        # context; off → "" and the path below is byte-identical to today.
        mcp_gathered = self._mcp_gather(project, task)

        for attempt in range(max_retries):
            logger.info(f"Attempt {attempt + 1}/{max_retries} for task {task.id}")
            if pending_attempt is not None:
                self._ledger_record(project, task, pending_attempt, success=False)
                pending_attempt = None

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
            # topo_context is the task-ranked (relevant-first) symbol-graph map
            # that cross-file tasks (delete-all-refs, wire call-sites) need. At
            # priority=3/min_lines=10 it collapsed to ~10 lines on a large repo
            # (the emathy run: "kept 11/9094"), editing blind. Keep it above the
            # truncate-first tier with a real floor so the top-ranked symbols
            # survive; truncation still trims the long tail under pressure.
            budget.set("topo_context", topo_context, priority=2, min_lines=40)
            budget.set("error_logs", error_logs or "", priority=1, min_lines=20)
            budget.set("scratchpad", scratchpad_context, priority=3)
            budget.set("interface_contracts", interface_contracts, priority=2)
            budget.set("recent_changes", recent_changes, priority=2)
            budget.set("consensus_context", consensus, priority=3)
            budget.set("project_outline", project_outline, priority=3, min_lines=0)
            allocated = budget.allocate()

            full_code_context = allocated["code_context"]
            if allocated["reference_sites"]:
                full_code_context += "\n\n" + allocated["reference_sites"]
            if allocated["topo_context"]:
                full_code_context += "\n\n" + allocated["topo_context"]
            if allocated["recent_changes"]:
                full_code_context += "\n\n" + allocated["recent_changes"]
            if allocated["project_outline"]:
                full_code_context += (
                    "\n\n## Project Structure (files and their symbols)\n"
                    + allocated["project_outline"]
                )
            if mcp_gathered:
                full_code_context += mcp_gathered
            full_code_context += self._mcp_awareness(project)

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
                ),
                # Marks the boundary between the cacheable stable context above
                # and the volatile tail below (see PROMPT_TEMPLATES); the client
                # caches everything before it.
                "cache_breakpoint": CACHE_BREAKPOINT,
            }

            system_prompt = prompt_manager.format_prompt("system", context_dict)
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

            routed_model = self._select_model(
                project, task, strategy, attempt, max_retries
            )
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
                continue

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
                continue
            edits = self._validate_edit_paths(project, task, edits)
            if not edits:
                logger.warning("No file edits detected in LLM response.")
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
                success, output = self._run_command(
                    project, build_cmd, timeout=build_timeout, cwd=task_cwd
                )
                if not success:
                    logger.warning(f"Build failed on attempt {attempt + 1}")
                    locations = resolver.resolve_errors(output)
                    attributed_error = resolver.format_for_llm(locations)
                    classified = format_classified_error(output)
                    error_logs = self._build_error_context(
                        prior_errors, attempt, output, classified, attributed_error
                    )
                    continue
                gate_verified = True

            if typecheck_command:
                success, output = self._run_command(
                    project, typecheck_command, timeout=build_timeout, cwd=task_cwd
                )
                if not success:
                    logger.warning(f"Type check failed on attempt {attempt + 1}")
                    locations = resolver.resolve_errors(output)
                    attributed_error = resolver.format_for_llm(locations)
                    classified = format_classified_error(output)
                    error_logs = self._build_error_context(
                        prior_errors, attempt, output, classified, attributed_error
                    )
                    continue
                gate_verified = True

            if test_command:
                success, output = self._run_command(
                    project, test_command, timeout=test_timeout, cwd=task_cwd
                )
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
                    )
                    if not acc_ok:
                        logger.warning(
                            f"Acceptance criteria not met on attempt {attempt + 1}."
                        )
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
                    locations = resolver.resolve_errors(output)
                    attributed_error = resolver.format_for_llm(locations)
                    classified = format_classified_error(output)
                    error_logs = self._build_error_context(
                        prior_errors, attempt, output, classified, attributed_error
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
        return self._fail_task(
            project,
            task,
            f"Task failed after {max_retries} attempts + escalation.",
            error_logs,
        )
