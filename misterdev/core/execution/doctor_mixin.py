"""DoctorMixin — diagnostic preflight for ProjectOrchestrator.

Extracted from agent.py so the god-module stays bounded. ProjectOrchestrator
inherits this alongside ParallelExecutionMixin and IntegrationGateMixin.
"""

import shlex
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from misterdev.core.gitcmd import run_git
from misterdev.config import get_setting


class DoctorMixin:
    def run_doctor(self, project_path: str | Path) -> Dict[str, Any]:
        """Preflight a project for an unattended run: gather the environment facts
        and route them through the pure ``doctor`` checks. Returns the aggregate
        (counts + exit_code) with the per-check results under ``checks``. Never
        raises — a gathering error degrades that one check, it does not crash."""
        from misterdev.core.execution import doctor as dr

        project = self._get_or_register(project_path)
        if not project:
            err = dr.Check(
                "load project", dr.FAIL, "could not load the project", "check the path"
            )
            out = dr.aggregate([err])
            out["checks"] = [err]
            return out

        checks: list = []
        is_git = (Path(project.path) / ".git").exists()
        checks.append(dr.check_git_repo(is_git))
        base = self._doctor_base_branch(project)
        if is_git:
            checks.append(dr.check_clean_tree(self._working_tree_dirty(project)))
            checks.append(
                dr.check_on_base_branch(self._doctor_current_branch(project), base)
            )
            checks.append(
                dr.check_leftover_task_branches(self._doctor_task_branches(project))
            )
            checks.append(dr.check_dangling_worktrees(self._doctor_worktrees(project)))
        try:
            ok, detail = project.llm_client.health_check()
        except Exception as e:
            ok, detail = False, str(e)
        checks.append(dr.check_models(ok, detail))
        if is_git:
            checks.append(self._doctor_worktree_probe(project))
        checks.append(
            dr.check_requirements(self._doctor_unsatisfied_requirements(project))
        )
        checks.append(
            dr.check_evolution_configured(
                get_setting(project.config, "evolution", "benchmark_dir")
            )
        )

        out = dr.aggregate(checks)
        out["checks"] = checks
        return out

    @staticmethod
    def _doctor_base_branch(project) -> str:
        """The repo's base branch: ``main`` or ``master`` if present, else HEAD."""
        for name in ("main", "master"):
            proc = run_git(
                f"git rev-parse --verify --quiet {shlex.quote(name)}", project.path
            )
            if proc is not None and proc.returncode == 0 and proc.stdout.strip():
                return name
        return DoctorMixin._doctor_current_branch(project) or "main"

    @staticmethod
    def _doctor_current_branch(project) -> Optional[str]:
        proc = run_git("git rev-parse --abbrev-ref HEAD", project.path)
        if proc is None or proc.returncode != 0:
            return None
        return proc.stdout.strip() or None

    @staticmethod
    def _doctor_task_branches(project) -> list:
        proc = run_git("git branch --list task/*", project.path)
        if proc is None or proc.returncode != 0:
            return []
        return [ln.strip(" *").strip() for ln in proc.stdout.splitlines() if ln.strip()]

    @staticmethod
    def _doctor_worktrees(project) -> list:
        """Registered worktrees under the orchestrator's worktree dir (excludes the
        main checkout), i.e. leftovers a prior run did not clean up."""
        proc = run_git("git worktree list --porcelain", project.path)
        if proc is None or proc.returncode != 0:
            return []
        found = []
        marker = str(Path(project.path) / ".orchestrator" / "worktrees")
        for ln in proc.stdout.splitlines():
            if ln.startswith("worktree ") and marker in ln:
                found.append(ln[len("worktree ") :].strip())
        return found

    def _doctor_worktree_probe(self, project):
        """Create a throwaway worktree, prime deps + run the healthcheck, remove it.
        Best-effort — any failure becomes a WARN, never crashes the doctor."""
        from misterdev.core.execution import doctor as dr
        from misterdev.tools.command import CommandTool
        from misterdev.tools.git_tool import GitTool

        setup_cmd = self._worktree_setup_command(project)
        health_cmd = self._worktree_healthcheck_command(project)
        if not setup_cmd and not health_cmd:
            return dr.check_worktree_prime(None, None)

        git = GitTool({})
        wt_root = Path(project.path) / ".orchestrator" / "worktrees"
        wt_root.mkdir(parents=True, exist_ok=True)
        git.worktree_prune(project)
        run_id = uuid.uuid4().hex[:6]
        branch = f"doctor/{run_id}"
        wt_path = wt_root / f"doctor-{run_id}"
        ok, out = git.worktree_add(project, str(wt_path), branch, new_branch=True)
        if not ok:
            return dr.Check(
                "worktree prime + healthcheck",
                dr.WARN,
                f"could not create a throwaway worktree: {out[-120:]}",
                "check git worktree support and disk space",
            )
        timeout = get_setting(project.config, "orchestrator", "worktree_setup_timeout")
        cmd = CommandTool({})
        prime_ok = None
        health_ok = None
        detail = ""
        try:
            if setup_cmd:
                prime_ok, pout = cmd.execute(
                    project, setup_cmd, cwd=str(wt_path), timeout=timeout
                )
                if not prime_ok:
                    detail = pout[-160:]
            if health_cmd:
                health_ok, hout = cmd.execute(
                    project, health_cmd, cwd=str(wt_path), timeout=timeout
                )
                if not health_ok:
                    detail = hout[-160:]
        finally:
            git.worktree_remove(project, str(wt_path))
            git.branch_delete(project, branch)
        return dr.check_worktree_prime(prime_ok, health_ok, detail)

    def _doctor_unsatisfied_requirements(self, project) -> list:
        """Keys in .orchestrator/REQUIREMENTS.md marked MISSING with no typed answer.
        Empty when the file is absent (nothing reviewed yet) or all are provided."""
        md = Path(project.path) / ".orchestrator" / "REQUIREMENTS.md"
        if not md.exists():
            return []
        try:
            from misterdev.core.planning.requirements import RequirementsBook

            answers = RequirementsBook(
                Path(project.path) / ".orchestrator"
            ).load_answers()
            text = md.read_text(encoding="utf-8")
        except OSError:
            return []
        unsatisfied = []
        key = ""
        missing = False
        for line in text.splitlines():
            if line.startswith("## "):
                if key and missing and key not in answers:
                    unsatisfied.append(key)
                key = line[3:].split("—", 1)[0].strip().strip("`")
                missing = False
            elif line.strip().lower().startswith("- status:"):
                missing = "missing" in line.lower()
        if key and missing and key not in answers:
            unsatisfied.append(key)
        return unsatisfied
