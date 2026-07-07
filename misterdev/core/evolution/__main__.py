"""CLI: python -m misterdev.core.evolution --benchmark <checkout> --workdir /tmp/evo

Runs one self-improvement pass over misterdev's own repo. Safe by default
(dry-run): measures the baseline benchmark, finds the highest-blame niche, and
proposes a single targeted edit WITHOUT applying or promoting anything. Add
``--live`` to run the full apply/gate/benchmark/promote loop (spends real budget
and self-edits in an isolated worktree). Being an explicit invocation IS the
off-by-default gate — the normal build loop never triggers this.
"""

import argparse
import os
import sys

from .driver import run_evolution


def _emit(line: str) -> None:
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def _gate_commands(repo: str) -> dict:
    """misterdev's own build/test/lint gate commands, detected from the repo, run
    in the sandbox to prove a self-edit did not break misterdev before scoring."""
    from misterdev.analyzers.project_analyzer.detection import (
        detect_build_command,
        detect_lint_command,
        detect_test_command,
    )
    from pathlib import Path

    p = Path(repo)
    return {
        "build_command": detect_build_command(p),
        "test_command": detect_test_command(p),
        "lint_command": detect_lint_command(p),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="misterdev.core.evolution")
    parser.add_argument(
        "--benchmark", required=True, help="polyglot-benchmark checkout"
    )
    parser.add_argument("--workdir", required=True, help="scratch dir for exercises")
    parser.add_argument(
        "--repo", default=None, help="misterdev repo to improve (default: cwd)"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="cap exercises per benchmark run"
    )
    parser.add_argument(
        "--languages", nargs="*", default=None, help="languages to benchmark"
    )
    parser.add_argument(
        "--model", default=None, help="pin a single model (e.g. a ':free' id)"
    )
    parser.add_argument(
        "--steps", type=int, default=1, help="live: mutation steps to try"
    )
    parser.add_argument(
        "--noise-band", type=float, default=0.05, help="min rate delta to promote"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="apply/gate/benchmark/promote self-edits (spends budget); default is dry-run",
    )
    args = parser.parse_args(argv)

    from misterdev.config import ConfigManager
    from misterdev.core.execution.project import Project

    repo = args.repo or os.getcwd()
    config = ConfigManager().load_project_config(repo)
    project = Project(repo, config)

    result = run_evolution(
        project,
        args.benchmark,
        args.workdir,
        steps=args.steps,
        noise_band=args.noise_band,
        limit=args.limit,
        languages=args.languages,
        model=args.model,
        live=args.live,
        gate_commands=_gate_commands(repo) if args.live else None,
    )

    _emit(
        f"baseline: {result.baseline.resolved}/{result.baseline.total} "
        f"({result.baseline.resolved_rate:.1%})"
    )
    if result.blame:
        _emit(
            f"top blame: {result.blame.niche} "
            f"({result.blame.failures}/{result.blame.total})"
        )
    for m in result.proposals:
        _emit(f"proposal [{m.note}] touches: {', '.join(m.paths) or '(none)'}")
    for i, s in enumerate(result.steps, 1):
        _emit(
            f"step {i}: {s.reason}"
            + (f" -> {s.score.resolved}/{s.score.total}" if s.score else "")
        )
    if result.champion:
        c = result.champion
        _emit(f"champion: {c.id} in {c.niche} ({c.resolved}/{c.total})")
    _emit(f"note: {result.note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
