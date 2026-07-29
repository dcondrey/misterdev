"""TargetsMixin — completeness-claim verification and multi-target validation.

Extracted from agent.py to keep ProjectOrchestrator bounded. All methods
access dependencies via self, resolved through the MRO at runtime:
- self._project_file_map: stays in agent.py (shared with _run_pipeline/_analyze)
- self._suite_failures / self._target_regressed: IntegrationGateMixin
"""

import re
from pathlib import Path
from typing import Optional

from misterdev.config import get_setting
from misterdev.logging_setup import setup_logger

logger = setup_logger(__name__)


class TargetsMixin:
    @staticmethod
    def _resolve_claim_file(root: Path, label: str, file_map: str) -> Optional[Path]:
        """Best-effort map a claim label to a real file for evidence.

        A label that is itself a path wins. Otherwise its DISTINCTIVE identifier
        tokens (length >= 5, or CamelCase) are matched as WHOLE WORDS against the
        file+symbol map, longest first, and the first file that mentions one is
        used. The word-boundary + distinctiveness rules stop a generic token like
        "backend" from matching an unrelated ``backend_registry.py``. Returns None
        when nothing resolves — the caller then leaves the claim unverified rather
        than judging it against the wrong file.
        """
        if label:
            direct = root / label
            if direct.is_file():
                return direct
        tokens = sorted(
            {
                t
                for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", label or "")
                if len(t) >= 5 or any(c.isupper() for c in t)
            },
            key=len,
            reverse=True,
        )
        for token in tokens:
            pattern = re.compile(rf"\b{re.escape(token)}\b")
            for line in file_map.splitlines():
                if pattern.search(line):
                    cand = root / line.split(":", 1)[0].strip()
                    if cand.is_file():
                        return cand
        return None

    def _verify_completeness_claims(self, project, assessment, report) -> None:
        """Drop completeness claims a second component refutes against the source.

        The analyzer flags "incomplete"/"stub" items from a lossy overview, so it
        can mislabel deliberate design (graceful-degradation, platform no-ops,
        parity shims) as work. Before the spec and tasks are built, recheck each
        claim against the REAL file plus the verified build/test state with an
        independent verifier and prune only the ones it refutes WITH evidence;
        unsure / skip / timeout keeps the claim, so genuine work is never lost.
        No-op when disabled or when no LLM verifier is available.
        """
        if not get_setting(project.config, "orchestrator", "verify_claims"):
            return
        feats = assessment.features
        if not feats.incomplete and not feats.stubs:
            logger.info(
                "Completeness-claim verifier: no incomplete/stub claims flagged; "
                "nothing to verify."
            )
            return

        from misterdev.analyzers.project_analyzer import _health_ground_truth
        from misterdev.core.verification.claim_verifier import Claim, verify_claims

        root = project.path
        health = _health_ground_truth(assessment.health)
        file_map = self._project_file_map(project)

        def read_body(path: Optional[Path]) -> str:
            if path is None or not path.is_file():
                return ""
            try:
                return path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                return ""

        entries = []
        unverified = 0
        for fi in feats.incomplete:
            path = self._resolve_claim_file(root, fi.name, file_map)
            body = read_body(path)
            if not body:
                unverified += 1
                continue
            try:
                rel = path.relative_to(root)
            except ValueError:
                rel = path
            entries.append(
                (
                    Claim(
                        kind="incomplete",
                        label=fi.name,
                        description=f"{fi.description} (source file: {rel})",
                        evidence=f"{health}\n\n{body}",
                    ),
                    "incomplete",
                    fi,
                )
            )
        for sp in feats.stubs:
            if not isinstance(sp, str):
                continue
            body = read_body(root / sp)
            if not body:
                unverified += 1
                continue
            entries.append(
                (
                    Claim(
                        kind="stub",
                        label=sp,
                        description=f"flagged as a stub file (source file: {sp})",
                        evidence=f"{health}\n\n{body}",
                    ),
                    "stub",
                    sp,
                )
            )

        if not entries:
            logger.info(
                "Completeness-claim verifier: no claims with readable source to "
                f"verify ({unverified} kept unverified)."
            )
            return

        judge_model = (project.config.get("judge") or {}).get("model")
        timeout = get_setting(project.config, "orchestrator", "verify_claims_timeout")
        logger.info(
            f"Verifying {len(entries)} completeness claim(s) against the real source..."
        )
        try:
            verdicts = verify_claims(
                [claim for claim, _, _ in entries],
                llm_client=project.llm_client,
                model=judge_model,
                timeout=timeout,
            )
        except Exception as e:
            logger.warning(f"Completeness-claim verification failed (non-fatal): {e}")
            report.degraded_subsystems.append(f"Claim verifier: {e}")
            return

        drop_incomplete: set[int] = set()
        drop_stubs: set[str] = set()
        for (claim, kind, origin), v in zip(entries, verdicts):
            if not v.refuted:
                continue
            if kind == "incomplete":
                drop_incomplete.add(id(origin))
            else:
                drop_stubs.add(origin)
            msg = f"Dropped phantom completeness claim '{claim.label}': {v.reason}"
            logger.info(msg)
            report.key_decisions.append(msg)

        dropped = len(drop_incomplete) + len(drop_stubs)
        logger.info(
            f"Completeness-claim verification: {len(entries) - dropped} kept, "
            f"{dropped} dropped ({unverified} unverified)."
        )
        if dropped:
            feats.incomplete = [
                fi for fi in feats.incomplete if id(fi) not in drop_incomplete
            ]
            feats.stubs = [sp for sp in feats.stubs if sp not in drop_stubs]

    def _resolve_targets(self, project) -> list:
        """Explicit ``targets`` if declared, else auto-discovered when enabled.

        Discovered targets are written back into ``project.config['targets']`` so
        the executor's per-task routing and per-target validation see them too.
        """
        explicit = project.config.get("targets") or []
        if explicit:
            return explicit
        if not get_setting(project.config, "orchestrator", "auto_targets"):
            return []
        from misterdev.core.planning.targets import discover_targets

        discovered = discover_targets(str(project.path))
        if discovered:
            names = ", ".join(t["name"] for t in discovered)
            logger.info(
                f"Auto-discovered {len(discovered)} polyglot target(s): {names}"
            )
            project.config["targets"] = discovered
        return discovered

    def _validate_targets(self, project, env_activate: Optional[str]) -> list:
        """Validate each declared target with its OWN toolchain, vs its baseline.

        Closes the multi-target gap where the end-of-run GateKeeper only ran the
        top-level commands. Crucially this compares against each target's baseline
        (measured before the run, stored on ``project.target_baselines``), so a
        target that was ALREADY broken (e.g. a frontend with pre-existing errors)
        is not counted as a failure for a run that never touched it — only a
        genuine REGRESSION fails. Returns [] when no targets are declared, so
        single-target builds are unaffected.
        """
        from misterdev.task_executors.markdown_plan_executor import MarkdownPlanExecutor

        targets = project.config.get("targets") or []
        if not targets:
            return []
        executor = getattr(self, "_validate_executor", None) or MarkdownPlanExecutor()
        target_baselines = getattr(project, "target_baselines", {}) or {}
        build_to = get_setting(project.config, "build", "build_timeout")
        test_to = get_setting(project.config, "build", "test_timeout")
        results: list = []
        for t in targets:
            gate_cmd = t.get("test_command") or t.get("build_command")
            if not gate_cmd:
                continue
            name = t.get("name") or t.get("path") or "?"
            tp = (t.get("path") or "").strip("/")
            run_dir = project.path / tp if tp else project.path
            timeout = test_to if t.get("test_command") else build_to
            after = self._suite_failures(
                project, executor, gate_cmd, timeout, cwd=run_dir
            )
            baseline = target_baselines.get(name)
            regressed = self._target_regressed(after, baseline)
            ok = not regressed
            detail = "ok" if ok else f"regressed (baseline={baseline}, after={after})"
            if ok and (t.get("web") or t.get("vision")):
                ok, rt_detail = self._run_target_runtime_gates(project, t, run_dir)
                if not ok:
                    detail = rt_detail
            results.append({"name": name, "ok": ok, "detail": detail})
        return results

    def _run_target_runtime_gates(
        self, project, target: dict, run_dir
    ) -> tuple[bool, str]:
        """Run a target's opt-in web/vision behavioral gates in its directory.

        Mirrors the GateKeeper's G4.7/G4.8 but scoped to a sub-project: the web
        gate renders + screenshots, the vision gate judges that screenshot. Both
        are best-effort and timeout-bounded — only a RED (a real failed check)
        fails the target; a SKIP (no browser/model/config) passes.
        """
        evidence = None
        web_cfg = target.get("web")
        if web_cfg:
            from misterdev.core.verification.web_verify import run_web_gate

            web = run_web_gate(run_dir, web_cfg)
            evidence = getattr(web, "evidence", None)
            if web.status == "red":
                return False, f"web verify failed ({web.reason or 'no detail'})"
        vision_cfg = target.get("vision")
        if vision_cfg:
            from misterdev.core.verification.vision_verify import run_vision_gate

            vc = dict(vision_cfg)
            if not vc.get("capture") and evidence:
                vc["capture"] = evidence
            vision = run_vision_gate(
                run_dir, vc or None, llm_client=getattr(project, "llm_client", None)
            )
            if vision.status == "red":
                return False, f"vision verify failed ({vision.reason or 'no detail'})"
        return True, "ok"
