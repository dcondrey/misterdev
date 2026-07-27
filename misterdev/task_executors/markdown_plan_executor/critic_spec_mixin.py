"""Adversarial edit critic and spec-as-tests generation/execution."""

import shlex
from typing import Dict, List, Optional, Tuple

from misterdev.core.models import Task
from misterdev.core.execution.project import Project
from misterdev.config import get_setting

from .helpers import logger


def _write_spec_conftest(spec_dir) -> None:
    """Write a pytest conftest that prevents discovery of spec test files.

    collect_ignore_glob only suppresses bare-discovery; explicit paths
    (as used by _run_spec_test) are unaffected. Best-effort: OSError is silent.
    """
    conftest = spec_dir / "conftest.py"
    if not conftest.exists():
        try:
            conftest.write_text(
                "collect_ignore_glob = ['spec_*.py']\n", encoding="utf-8"
            )
        except OSError:
            pass


class CriticSpecMixin:
    def _critic_enabled_for(self, project: Project, task: Task) -> bool:
        """Whether the adversarial critic runs for this task.

        ``adversarial_critic`` True/False forces it; "auto" (default) enables it
        only for the cross-cutting categories in ``critic_auto_categories`` —
        where symptom-fixes, incomplete refactors, and duplication cluster — AND
        only when the task has no objective test gate. A real test suite catches
        the same implementation bugs authoritatively and far more cheaply than an
        extra model call on every attempt, so the critic (whose unique value is
        reviewing what a test cannot) defers to it; set ``adversarial_critic:
        true`` to force the critic even alongside tests.
        """
        setting = get_setting(project.config, "orchestrator", "adversarial_critic")
        if isinstance(setting, str) and setting.strip().lower() == "auto":
            categories = get_setting(
                project.config, "orchestrator", "critic_auto_categories"
            )
            if getattr(task, "category", "") not in (categories or []):
                return False
            return not self._has_objective_test_gate(project, task)
        return bool(setting)

    def _has_objective_test_gate(self, project: Project, task: Task) -> bool:
        """True when this task will run a real test command (the authoritative gate)."""
        if (getattr(task, "processor_data", None) or {}).get("test_command"):
            return True
        try:
            from misterdev.core.planning.targets import (
                select_target,
                target_commands,
            )

            files = list(getattr(task, "files_to_modify", []) or []) + list(
                getattr(task, "context_files", []) or []
            )
            routed = select_target(project.config.get("targets") or [], files)
            return bool(target_commands(routed, project.config).get("test_command"))
        except (KeyError, AttributeError, TypeError):
            return False

    def _run_edit_critic(self, project: Project, task: Task, edits: Dict[str, str]):
        """Run the independent adversarial critic over a candidate edit.

        Reads the (optional) independent ``critic.model`` and timeout from config
        and delegates to the never-raising, timeout-bounded gate. Returns a
        :class:`~misterdev.core.verification.critic.CritiqueVerdict`; a SKIP
        (no client, unparseable, timeout) is treated by the caller as "proceed".
        """
        from misterdev.core.verification.critic import run_edit_critic

        critic_cfg = project.config.get("critic") or {}
        timeout = get_setting(project.config, "orchestrator", "critic_timeout")
        return run_edit_critic(
            task.description,
            task.acceptance_criteria,
            edits,
            llm_client=project.llm_client,
            critic_model=critic_cfg.get("model"),
            candidate_diffs=self._critic_diffs(project, edits),
            panel=critic_cfg.get("panel", 1),
            timeout=timeout,
        )

    @staticmethod
    def _critic_diffs(project: Project, edits: Dict[str, str]) -> Dict[str, str]:
        """Unified diff of each candidate edit vs its current on-disk content.

        Lets the critic review WHAT CHANGED (with a little context) instead of
        whole files — sharper and far smaller for a small edit to a large file.
        A new file (no original) diffs against empty, i.e. all-additions. Reading
        the original is best-effort: an unreadable file falls back to empty.
        """
        import difflib

        diffs: Dict[str, str] = {}
        for path, new_content in edits.items():
            fp = project.path / path
            try:
                original = fp.read_text(encoding="utf-8") if fp.exists() else ""
            except OSError:
                original = ""
            diff = "".join(
                difflib.unified_diff(
                    original.splitlines(keepends=True),
                    (new_content or "").splitlines(keepends=True),
                    fromfile=f"a/{path}",
                    tofile=f"b/{path}",
                )
            )
            diffs[path] = diff or "(no textual change)"
        return diffs

    def _maybe_generate_spec_test(
        self, project: Project, task: Task, validate_timeout: Optional[int] = None
    ) -> Tuple[Optional[str], Optional[str]]:
        """Generate + write a failing spec test for the task, or (None, None).

        Off unless ``orchestrator.spec_as_tests`` and the task has acceptance
        criteria. The test is written under ``.orchestrator/spec_tests/`` — NOT
        the project's own test directory — so it is never collected by the
        project suite and so cannot flip the integration-gate baseline red. Run
        scoped later by :meth:`_run_spec_test`. Best-effort: any failure (no
        client, model error, unwritable path) yields (None, None).

        Returns ``(path, source)``: the path is run as a gate, and the SOURCE is
        injected into the edit context as the concrete reproduction target the
        model must make pass — turning the spec test from a passive after-the-fact
        check into the directed objective (reproduction-first / TDD).

        When ``validate_timeout`` is given, the generated test is run once on the
        CLEAN (pre-edit) tree and KEPT ONLY IF IT ACTUALLY FAILS there — i.e. it
        genuinely reproduces the gap. A test that passes pre-implementation
        encodes nothing the code must satisfy, so trusting it as the gate/target
        is worse than having none (a false green that a wrong edit also passes);
        such a test is discarded. A run we cannot score (no scoped runner for the
        language -> ``skip``) is kept, since we can't disprove it reproduces.
        """
        if not get_setting(project.config, "orchestrator", "spec_as_tests"):
            return None, None
        if not getattr(task, "acceptance_criteria", ""):
            return None, None
        from misterdev.core.verification.spec_tests import (
            _LANG_EXT,
            safe_task_id,
            generate_spec_test,
        )

        language = (project.config.get("language") or "python").lower()
        try:
            source = generate_spec_test(task, project.llm_client, language=language)
        except Exception as e:  # generation is best-effort
            logger.debug(f"Spec-test generation skipped: {e}")
            return None, None
        if not source:
            return None, None
        # Reuse the generator's canonical language->extension map so a compiled
        # language (rust/go/java) gets its real suffix instead of a .txt stub.
        ext = _LANG_EXT.get(language, ".txt")
        safe_id = safe_task_id(task)
        path = project.path / ".orchestrator" / "spec_tests" / f"spec_{safe_id}{ext}"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source, encoding="utf-8")
        except OSError as e:
            logger.debug(f"Spec-test write skipped: {e}")
            return None, None
        if validate_timeout is not None:
            status, _ = self._run_spec_test(project, str(path), validate_timeout)
            if status == "green":
                # Passes without the change: it reproduces nothing, so it is a
                # false gate. Discard it rather than mislead the edit and the gate.
                logger.info(
                    f"Spec-as-test for {task.id} PASSES pre-implementation "
                    "(reproduces nothing); discarded."
                )
                try:
                    path.unlink()
                except OSError:
                    pass
                return None, None
            _write_spec_conftest(path.parent)
            logger.info(
                "Spec-as-test written (validated: "
                f"{'reproduces' if status == 'red' else 'unscored'}; injected as "
                f"target, run as gate): {path}"
            )
            return str(path), source
        _write_spec_conftest(path.parent)
        logger.info(f"Spec-as-test written (injected as target, run as gate): {path}")
        return str(path), source

    def _run_spec_test(
        self, project: Project, spec_path: Optional[str], timeout: int
    ) -> Tuple[str, str]:
        """Run the generated spec test scoped to its file. Returns (status, detail).

        ``status`` is ``green`` (passes — the spec is satisfied), ``red`` (still
        fails), or ``skip`` (no spec test, or no scoped runner for this project's
        language — we only run a single file for pytest/jest-style suites). Never
        raises.
        """
        if not spec_path:
            return "skip", ""
        test_cmd = project.config.get("test_command") or ""
        if "pytest" in test_cmd or spec_path.endswith(".py"):
            runner = f"pytest -q {shlex.quote(spec_path)}"
        elif "jest" in test_cmd:
            runner = f"jest {shlex.quote(spec_path)}"
        elif "vitest" in test_cmd:
            runner = f"npx --yes vitest run {shlex.quote(spec_path)}"
        elif "node --test" in test_cmd or spec_path.endswith(
            (".test.ts", ".test.js", ".test.mjs")
        ):
            # Node's built-in runner strips TS at runtime, so a generated
            # `.test.ts` spec runs with no extra toolchain — this is what makes
            # TDD spec-as-tests work for a typecheck-only frontend target.
            runner = f"node --test {shlex.quote(spec_path)}"
        else:
            return "skip", "no scoped spec-test runner for this project"
        try:
            ok, out = self._run_command(project, runner, timeout=timeout)
        except Exception as e:  # a runner failure must not sink the task
            logger.debug(f"Spec-test run skipped: {e}")
            return "skip", ""
        return ("green" if ok else "red"), out

    @staticmethod
    def _build_critic_error_context(objections: List[str]) -> str:
        """Format critic objections into the same retry context shape as a gate.

        The next attempt sees the concrete problems an independent reviewer found
        in the rejected change and is told to address each before resubmitting.
        """
        listed = "\n".join(f"- {o}" for o in objections)
        return (
            "An independent reviewer rejected your previous change BEFORE it was "
            "applied. Each objection below is a concrete defect — address every "
            "one in your next attempt, then resubmit the corrected edit:\n"
            f"{listed}"
        )
