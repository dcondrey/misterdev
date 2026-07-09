"""Test-gate acceptance, acceptance-criteria verification, and error context."""

import re
from typing import List, NamedTuple, Optional, Tuple

from misterdev.core.models import Task
from misterdev.core.execution.project import Project
from misterdev.core.execution.error_classifier import classify_error, ErrorCategory
from misterdev.core.execution.failure_view import (
    extract_failures,
    render_failure_view,
)
from misterdev.config import get_setting

from .helpers import logger, _extract_acceptance_command, JUDGE_MIN_BUDGET_FRACTION


class _AttemptFailure(NamedTuple):
    """One failed attempt: a human-readable label plus a stable error signature.

    The signature lets the retry loop detect when an attempt reproduced an
    earlier failure verbatim (the model's fix changed nothing) so it can escalate
    to a different strategy instead of looping on the same error.
    """

    label: str
    signature: str


def _error_signature(output: str) -> str:
    """A fingerprint that stays identical across attempts for the SAME failure.

    Folds out the volatile parts (line/column numbers, counts, file paths) that
    differ run-to-run without changing the underlying error, then keys on the
    classified category plus a bounded normalized slice. Two attempts that hit
    the same wall fingerprint the same; a genuinely different error does not.
    """
    norm = re.sub(r"\d+", "#", (output or "").lower())
    norm = re.sub(r"[/\\][\w./\\-]+", "<path>", norm)
    norm = re.sub(r"\s+", " ", norm).strip()
    return f"{classify_error(output)}:{norm[:200]}"


class GatesMixin:
    @staticmethod
    def _prior_failures_history(prior_errors: List["_AttemptFailure"]) -> str:
        """Render all-but-the-latest prior attempt errors as a retry-context
        header, or '' when there is no earlier failure to summarize."""
        if len(prior_errors) <= 1:
            return ""
        past = "\n".join(f"- {e.label}" for e in prior_errors[:-1])
        return (
            "### Previous Attempt Failures (a different approach is required)\n"
            f"{past}\n\n"
        )

    @staticmethod
    def _repeat_escalation(prior_errors: List["_AttemptFailure"]) -> str:
        """Banner shown when the newest failure repeats an earlier one verbatim.

        A recurring identical signature means the last fix changed nothing — the
        approach is not converging — so instruct the model to change strategy
        rather than re-edit the same way (the actionable half of reflection
        scoring: detect that reflecting did not help). Additive: it only prepends
        a directive, never removes context, so it cannot degrade a good attempt.
        """
        if len(prior_errors) < 2:
            return ""
        latest = prior_errors[-1].signature
        repeats = sum(1 for e in prior_errors if e.signature == latest)
        if repeats < 2:
            return ""
        return (
            f"### Repeated failure ({repeats}x) — CHANGE YOUR APPROACH\n"
            "Your last attempts produced the SAME error; the current approach is "
            "not working. Do NOT make the same kind of edit again. Step back: "
            "re-read the referenced code and interface contracts, question the "
            "assumption that led to this error, and try a fundamentally different "
            "fix.\n\n"
        )

    def _apply_reflection(
        self, project: Project, task: Task, error_logs: str, reflections: List[str]
    ) -> str:
        """Reflect on the previous failure and prepend the running reflections.

        Generates one short root-cause reflection from ``error_logs`` (the
        Reflexion pattern), appends it to the accumulated ``reflections``, and
        returns ``error_logs`` with a REFLECTION header prepended so the next
        attempt debugs the underlying cause. Best-effort and off when
        ``orchestrator.reflection`` is false; a skip/error leaves ``error_logs``
        unchanged, so the retry path is never blocked or worsened.
        """
        if not get_setting(project.config, "orchestrator", "reflection"):
            return error_logs
        from misterdev.core.verification.reflection import reflect_on_failure

        reflect_model = (project.config.get("critic") or {}).get("model")
        new_reflection = reflect_on_failure(
            task.description,
            error_logs,
            prior_reflections=reflections,
            llm_client=project.llm_client,
            reflect_model=reflect_model,
            timeout=get_setting(project.config, "orchestrator", "reflection_timeout"),
        )
        if new_reflection:
            reflections.append(new_reflection)
        if not reflections:
            return error_logs
        header = "### Reflection on why previous attempts failed\n" + "\n\n".join(
            reflections
        )
        return f"{header}\n\n{error_logs}"

    def _build_error_context(
        self,
        prior_errors: List["_AttemptFailure"],
        attempt: int,
        output: str,
        classified: str,
        attributed_error: str,
        *,
        project: Optional[Project] = None,
        test_command: Optional[str] = None,
        language: Optional[str] = None,
        cwd: Optional[str] = None,
    ) -> str:
        """Combine the current error with a summary of prior failed attempts.

        Surfacing what already failed (and how it was classified) stops the LLM
        from re-submitting the same broken fix across retries. When
        ``orchestrator.failure_probe`` is on and the gate command is known, also
        re-run the first failing test in isolation for a fresh trace.
        """
        prior_errors.append(
            _AttemptFailure(
                f"Attempt {attempt + 1}: {classify_error(output)}",
                _error_signature(output),
            )
        )
        escalation = self._repeat_escalation(prior_errors)
        history = self._prior_failures_history(prior_errors)
        # Lead with the exact failing assertions (expected vs actual, per test)
        # parsed straight from the runner output — the model fixes what it can
        # precisely see. Empty for unrecognized output, so the compressed
        # classified/attributed view below is preserved as the fallback.
        failures = extract_failures(output)
        view = render_failure_view(failures)
        lead = f"{view}\n\n" if view else ""
        probe = self._failure_probe(project, test_command, language, cwd, failures)
        return f"{escalation}{history}{lead}{probe}{classified}\n\n{attributed_error}"

    def _failure_probe(self, project, test_command, language, cwd, failures) -> str:
        """Fresh isolated re-run of the FIRST failing test (D4), when enabled.

        Off unless ``orchestrator.failure_probe`` is set; a no-op (empty string)
        with no failures, no gate command, an unrecognized runner, or a failed
        re-run — so it can only add signal, never break the context. Never raises.
        """
        if not (project is not None and test_command and failures):
            return ""
        try:
            from misterdev.core.execution.probe import (
                isolate_command,
                run_probe,
                should_auto_probe,
            )

            # Fire when explicitly enabled, OR auto-fire for fast interpreted
            # runners (pytest/jest/vitest) where a single-test re-run is cheap;
            # compiled runners (cargo/swift/dotnet) stay opt-in behind the flag.
            enabled = get_setting(
                project.config, "orchestrator", "failure_probe"
            ) or should_auto_probe(test_command, language or "")
            if not enabled:
                return ""
            iso = isolate_command(test_command, failures[0].test, language or "")
            if not iso:
                return ""
            fresh = run_probe(cwd or str(project.path), iso)
            if not fresh:
                return ""
            return (
                "### Fresh isolated re-run of the first failing test\n"
                f"$ {iso}\n{fresh.strip()}\n\n"
            )
        except Exception as e:  # a probe must never break the retry context
            logger.debug(f"Failure probe skipped (non-fatal): {e}")
            return ""

    @staticmethod
    def _gate_accepts(
        success: bool, output: str, baseline_failures: int
    ) -> Tuple[bool, Optional[int]]:
        """Whether the test gate accepts this attempt, plus the parsed failure count.

        A green suite always passes (returns ``(True, 0)``). On a RED baseline
        (``baseline_failures`` > 0), an attempt that leaves the suite no worse —
        parsed failures <= baseline — also passes, so a multi-failure project can
        be reduced one task at a time instead of requiring a single task to make
        the whole suite green. An unparseable red result stays strict (rejected),
        since we will not accept on a number we cannot read.
        """
        if success:
            return True, 0
        if baseline_failures <= 0:
            return False, None
        from misterdev.core.verification.validator import (
            _parse_test_counts,
        )

        total, post = _parse_test_counts(output)
        if total > 0 and post <= baseline_failures:
            return True, post
        return False, post if total > 0 else None

    def _verify_acceptance(
        self,
        project: Project,
        task: Task,
        verify_acceptance: bool,
        llm_acceptance_judge: bool,
        timeout: int,
        cwd=None,
    ) -> Tuple[bool, str]:
        """Verify the task's acceptance_criteria after build/test gates pass.

        Returns (passed, output). Deterministic primary path: extract an
        explicit runnable command from acceptance_criteria and run it; a
        non-zero exit fails acceptance. When acceptance_criteria is empty, the
        gate is disabled, or no command can be confidently extracted, this is a
        no-op that passes (behaviour identical to before this gate) unless the
        default-off ``orchestrator.llm_acceptance_judge`` flag is set, in which
        case an LLM judge is consulted. Never blocks on un-parseable free text.
        """
        if not verify_acceptance:
            return True, ""
        criteria = (task.acceptance_criteria or "").strip()
        if not criteria:
            return True, ""
        command = _extract_acceptance_command(criteria)
        if command:
            logger.info(f"Verifying acceptance criteria via command: {command}")
            success, output = self._run_command(
                project, command, timeout=timeout, cwd=cwd
            )
            if success:
                logger.info("Acceptance criteria command passed.")
                return True, ""
            # A command that can't locate the project manifest is a BROKEN
            # acceptance command, not failing code: acceptance runs only AFTER
            # the build/test gates pass, so the manifest demonstrably exists —
            # a MANIFEST error here means the extracted command is malformed
            # (the emathy run lost `--manifest-path` and every such task
            # false-failed on `cargo test` from the repo root). Treat it as a
            # pass-through. A genuine missing test path (FILE_NOT_FOUND) is left
            # as a real failure.
            if classify_error(output) == ErrorCategory.MANIFEST:
                logger.warning(
                    "Acceptance command could not locate the project manifest; "
                    "the build/test gates already passed, so treating acceptance "
                    f"as satisfied rather than failing on a broken command: {command}"
                )
                return True, ""
            return False, (
                f"Acceptance criterion not met: `{criteria}`\n"
                f"Ran: {command}\n"
                f"Command exited non-zero:\n{output}"
            )
        if llm_acceptance_judge and self._judge_affordable(project):
            return self._llm_acceptance_judge(project, task, criteria)
        return True, ""

    def _judge_affordable(self, project: Project) -> bool:
        """True while enough budget remains to spend on the LLM acceptance judge.

        Cost control for the (now default-on) judge: once the run has burned
        through all but ``JUDGE_MIN_BUDGET_FRACTION`` of the budget, stop paying
        for free-text judging and let those criteria pass, reserving the last
        funds for actually fixing code. Fail-open when budget can't be read.
        """
        client = project.llm_client
        remaining = getattr(client, "budget_remaining", None)
        total = getattr(client, "_budget", None)
        if not isinstance(remaining, (int, float)) or not isinstance(
            total, (int, float)
        ):
            return True
        if total <= 0:
            return True
        return remaining > total * JUDGE_MIN_BUDGET_FRACTION

    def _judge_generate(self, project: Project, prompt: str) -> str:
        """Run an acceptance-judge prompt, on the INDEPENDENT ``judge.model`` when
        configured (so the judge doesn't share the generator's blind spots), else
        on the generator's own model. Routed through ``with_model`` when possible.
        """
        from misterdev.core.verification.independent import (
            generate_independent,
        )

        judge_model = (project.config.get("judge") or {}).get("model")
        return generate_independent(project.llm_client, prompt, "", model=judge_model)

    def _llm_acceptance_judge(
        self, project: Project, task: Task, criteria: str
    ) -> Tuple[bool, str]:
        """Default-off LLM fallback judging free-text acceptance criteria.

        Only reached when ``orchestrator.llm_acceptance_judge`` is true and no
        runnable command could be extracted. A failure to reach a confident
        verdict passes (fail-open) so an unreliable judge never blocks a task.
        """
        try:
            prompt = (
                "A code task has just passed its build and test gates. Judge "
                "ONLY whether the stated acceptance criterion is satisfied by "
                "the task's implementation. Reply with PASS or FAIL on the "
                "first line, then a brief reason.\n\n"
                f"Task: {task.description}\n"
                f"Acceptance criterion: {criteria}\n"
            )
            verdict = self._judge_generate(project, prompt)
        except Exception as e:
            logger.warning(f"LLM acceptance judge failed, passing open: {e}")
            return True, ""
        first = (verdict or "").strip().splitlines()
        if first and first[0].strip().upper().startswith("FAIL"):
            return False, (
                f"Acceptance criterion not met (LLM judge): `{criteria}`\n{verdict}"
            )
        return True, ""

    def _build_acceptance_error_context(
        self,
        prior_errors: List["_AttemptFailure"],
        attempt: int,
        task: Task,
        classified: str,
    ) -> str:
        """Format an acceptance failure into the same retry context as other gates.

        Makes the unmet criterion explicit so the next attempt targets it rather
        than re-submitting a change that only satisfies the build/test gates.
        """
        prior_errors.append(
            _AttemptFailure(
                f"Attempt {attempt + 1}: acceptance criteria not met",
                "acceptance:not_met",
            )
        )
        escalation = self._repeat_escalation(prior_errors)
        history = self._prior_failures_history(prior_errors)
        return (
            f"{escalation}{history}### Acceptance criterion not met\n"
            f"The build and tests passed, but the task's acceptance criterion "
            f"was not satisfied:\n{task.acceptance_criteria}\n\n{classified}"
        )
