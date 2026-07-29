"""Real adapters binding the loop to live misterdev + the benchmark harness.

The loop's decision spine (fitness/archive/guardrail/loop) is pure and tested; the
steps that spend money or mutate source are here, bound to verified APIs and kept
thin. The pure helpers (patch application, JSON result loading) are unit-tested;
the git-worktree / gate-suite / benchmark-subprocess binds are exercised by a real
benchmark run, not by the suite (they need a checkout and real spend).

Layering: the benchmark runs as a SUBPROCESS (``python -m evaluation.polyglot``),
so ``core.evolution`` never imports ``evaluation`` — per-instance results come back
as JSON and are reconstructed into a local record.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from misterdev.llm.responses.apply import apply_search_replace
from misterdev.llm.responses.parsing import LLMResponseParser
from misterdev.logging_setup import setup_logger
from misterdev.task_executors.markdown_plan_executor.helpers import (
    EDIT_FORMAT_INSTRUCTIONS,
)

from .fitness import FitnessScore
from .guardrail import assert_mutation_allowed
from .loop import Mutation
from .proposer import LLMProposer

logger = setup_logger(__name__)


@dataclass
class BenchResult:
    """One benchmark instance's outcome, reconstructed from harness JSON — the
    duck-typed shape attribution and regression counting consume."""

    name: str
    language: str
    resolved: bool
    error: str = ""


def results_from_report(report: dict) -> List[BenchResult]:
    """Reconstruct per-instance results from a harness ``to_dict`` payload."""
    out: List[BenchResult] = []
    for r in report.get("results", []):
        out.append(
            BenchResult(
                name=str(r.get("name", "")),
                language=str(r.get("language", "unknown")),
                resolved=bool(r.get("resolved", False)),
                error=str(r.get("error", "")),
            )
        )
    return out


def apply_patch_to_worktree(root: str, patch: str) -> List[str]:
    """Apply a patch of anchored SEARCH/REPLACE blocks under ``root``; return the
    written paths.

    Guardrail-checks every target (walls off the evaluator/gates/tests and refuses
    traversal), so a patch can only ever touch mutable repo source. Raises
    ``ValueError`` on an empty/unparseable patch and ``EditConflictError`` (from
    the applier) when a SEARCH anchor does not match — both leave the tree as the
    applier left it, which the caller discards by tearing down the worktree.
    """
    edits = LLMResponseParser.parse_search_replace_blocks(patch)
    if not edits:
        raise ValueError("patch contained no SEARCH/REPLACE blocks")
    by_path: dict = {}
    for e in edits:
        by_path.setdefault(e.path, []).append(e)
    assert_mutation_allowed(by_path.keys())
    written: List[str] = []
    base = Path(root)
    for rel, group in by_path.items():
        target = base / rel
        original = target.read_text(encoding="utf-8") if target.exists() else ""
        new_content = apply_search_replace(original, group)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(new_content, encoding="utf-8")
        written.append(rel)
    return written


def run_benchmark(
    cwd: str,
    benchmark_dir: str,
    workdir: str,
    *,
    limit: Optional[int] = None,
    languages: Optional[List[str]] = None,
    model: Optional[str] = None,
    build_args: Optional[str] = None,
    only: Optional[List[str]] = None,
    timeout: int = 7200,
) -> Tuple[List[BenchResult], float, dict]:
    """Run the polyglot suite as a subprocess in ``cwd`` and return
    (per-instance results, cost, raw report).

    Runs from ``cwd`` so a mutated checkout's misterdev is the one under test.
    ``only`` restricts the run to specific exercise slugs — the cheap, targeted
    evaluation the micro-eval screen depends on (run the handful of cases a
    mutation targets, not the whole suite). Raises on a non-zero exit or timeout.
    """
    with tempfile.TemporaryDirectory(prefix="evo-bench-") as tmp_dir:
        out_json = Path(tmp_dir) / "report.json"
        cmd = [
            sys.executable,
            "-m",
            "evaluation.polyglot",
            "--benchmark",
            benchmark_dir,
            "--workdir",
            workdir,
            "--json",
            str(out_json),
        ]
        if limit is not None:
            cmd += ["--limit", str(limit)]
        if languages:
            cmd += ["--languages", *languages]
        if model:
            cmd += ["--model", model]
        if build_args:
            cmd += ["--build-args", build_args]
        if only:
            cmd += ["--only", *only]
        env = {
            **os.environ,
            "PYTHONPATH": cwd + os.pathsep + os.environ.get("PYTHONPATH", ""),
        }
        logger.info(
            f"Evolution: running benchmark subprocess in {cwd} (limit={limit})."
        )
        subprocess.run(cmd, cwd=cwd, env=env, timeout=timeout, check=True)
        report = json.loads(out_json.read_text(encoding="utf-8"))
    # Real per-run spend when the harness reports it (best-effort); a suite from
    # before the cost field was added simply reports 0.0, which only disables the
    # cost tie-breaker, not the load-bearing resolved-rate/regression objectives.
    try:
        cost = float(report.get("cost", 0.0) or 0.0)
    except (TypeError, ValueError):
        cost = 0.0
    return results_from_report(report), cost, report


def make_proposer(project) -> LLMProposer:
    """An :class:`LLMProposer` bound to the project's edit-generating LLM client.

    The editor call appends misterdev's own edit-format contract so the model
    replies with anchored SEARCH/REPLACE blocks the applier understands, and runs
    under a system prompt that frames the task as a self-edit.
    """
    system = (
        "You are editing misterdev's OWN source to improve how it solves coding "
        "benchmarks. Make the smallest correct change. Never edit tests, the "
        "benchmark harness, or the gate suite."
    )

    def generate(instruction: str) -> str:
        prompt = instruction + "\n" + EDIT_FORMAT_INSTRUCTIONS
        return project.llm_client.generate_edits(prompt, system).content

    return LLMProposer(generate=generate, repo_root=project.path)


class RealSandbox:
    """The loop's ``evaluate`` adapter, bound to a git worktree + gate suite +
    benchmark subprocess. Its ``apply``/``gates``/``benchmark`` share one worktree.

    Off the hot path of the suite: constructed and driven only by an explicit,
    opt-in live evolution run.
    """

    def __init__(
        self,
        project,
        benchmark_dir: str,
        workdir: str,
        gate_commands: dict,
        *,
        limit: Optional[int] = None,
        languages: Optional[List[str]] = None,
        model: Optional[str] = None,
        build_args: Optional[str] = None,
        only: Optional[List[str]] = None,
        gate_timeout: int = 600,
    ):
        from misterdev.tools.git_tool import GitTool

        self.project = project
        self.repo_root = str(project.path)
        self.benchmark_dir = benchmark_dir
        self.workdir = workdir
        self.gate_commands = gate_commands
        self.limit = limit
        self.languages = languages
        self.model = model
        self.build_args = build_args
        self.only = only
        self.gate_timeout = gate_timeout
        self._git = GitTool({})
        self._worktree: Optional[str] = None
        self._branch: Optional[str] = None

    def apply(self, mutation: Mutation) -> Callable[[], None]:
        assert_mutation_allowed(mutation.paths)
        wt = tempfile.mkdtemp(prefix="evo-wt-")
        branch = f"evo/{Path(wt).name}"
        ok, out = self._git.worktree_add(self.project, wt, "HEAD", new_branch=True)
        if not ok:
            shutil.rmtree(wt, ignore_errors=True)
            raise RuntimeError(f"worktree_add failed: {out}")
        self._worktree, self._branch = wt, branch
        try:
            apply_patch_to_worktree(wt, mutation.patch)
        except Exception:
            self._teardown(wt)
            raise
        return lambda: self._teardown(wt)

    def _teardown(self, wt: str) -> None:
        try:
            self._git.worktree_remove(self.project, wt)
        finally:
            self._worktree = None

    def gates(self) -> bool:
        from misterdev.core.verification.gatekeeper import GateKeeper

        keeper = GateKeeper(
            self._worktree,
            build_timeout=self.gate_timeout,
            test_timeout=self.gate_timeout,
        )
        ok, issues, _health = keeper.run_gates(self.gate_commands)
        if not ok:
            logger.info(f"Evolution: sandbox gates failed: {issues[:3]}")
        return ok

    def benchmark(self) -> Tuple[List[BenchResult], float]:
        results, cost, _raw = run_benchmark(
            self._worktree,
            self.benchmark_dir,
            self.workdir,
            limit=self.limit,
            languages=self.languages,
            model=self.model,
            build_args=self.build_args,
            only=self.only,
        )
        report = _DuckReport(results)
        return report, cost

    def benchmark_only(self, only: List[str]) -> Tuple[List[BenchResult], float]:
        """Run ONLY the named exercise slugs in the current worktree — the cheap,
        targeted evaluation the micro-eval screen uses (a handful of cases, not the
        whole suite). ``limit``/``languages`` are dropped: ``only`` is the selector.
        """
        results, cost, _raw = run_benchmark(
            self._worktree,
            self.benchmark_dir,
            self.workdir,
            model=self.model,
            only=only,
        )
        return results, cost


@dataclass
class _DuckReport:
    """A ``FitnessScore.from_report`` / regression-counting compatible view over a
    list of :class:`BenchResult` (``.resolved``, ``.total``, ``.results``)."""

    results: List[BenchResult]

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def resolved(self) -> int:
        return sum(1 for r in self.results if r.resolved)


def baseline_passed(results: List[BenchResult]) -> set:
    """Instance ids (name) that passed at baseline — the regression reference set."""
    return {r.name for r in results if r.resolved and r.name}


def score_of(results: List[BenchResult], cost: float = 0.0, regressions: int = 0):
    """A :class:`FitnessScore` over a result list."""
    return FitnessScore.from_report(
        _DuckReport(results), cost=cost, regressions=regressions
    )
