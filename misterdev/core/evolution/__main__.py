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


def gate_commands_for_repo(repo: str) -> dict:
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


def failure_target_for_repo(repo: str):
    """Highest-weight niche from the repo's real-build failure stream, or None.

    Reads ``.orchestrator/failures.jsonl`` (written by finished builds) and ranks
    niches by recency-decayed, recurrence-amplified weight so evolution aims at
    what is still breaking in real use. Best-effort: an unreadable/empty stream
    yields None, which the caller treats as "nothing to do".
    """
    from pathlib import Path

    from misterdev.core.learning.failure_log import FailureLog
    from misterdev.core.learning.targeting import top_stream_target

    records = FailureLog(Path(repo) / ".orchestrator" / "failures.jsonl").load()
    return top_stream_target(records)


def add_evolve_arguments(parser) -> None:
    """Add the evolution-run flags to ``parser`` (an ArgumentParser or subparser).

    Shared by this module's own CLI and ``misterdev evolve`` so the flag surface
    is defined once and can't drift between the two entry points.
    """
    parser.add_argument(
        "--benchmark",
        default=None,
        help="polyglot-benchmark checkout (default: evolution.benchmark_dir in "
        "project.yaml)",
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
        "--noise-band",
        type=float,
        default=None,
        help="min rate delta to promote (default: evolution.noise_band in "
        "project.yaml, else 0.05)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="apply/gate/benchmark/promote self-edits (spends budget); default is dry-run",
    )
    parser.add_argument(
        "--from-failures",
        action="store_true",
        help="aim the mutation at the repo's REAL-build failure stream "
        "(.orchestrator/failures.jsonl) instead of the benchmark's worst niche; "
        "the benchmark still gates promotion",
    )
    parser.add_argument(
        "--screen",
        action="store_true",
        help="live: cheaply screen each candidate on only the targeted + guard "
        "cases (from the reproduction corpus) before spending the full benchmark",
    )
    parser.add_argument(
        "--beam",
        type=int,
        default=1,
        help="live: propose this many candidates per step and keep the best "
        "screened survivor (implies --screen; widens search without widening cost)",
    )
    parser.add_argument(
        "--scheduled",
        action="store_true",
        help="live only: run under run_scheduled_evolution's exclusive lock + "
        "circuit breaker instead of a plain one-shot call (for cron/CI use, where "
        "an overlapping or repeatedly-failing trigger must be a clean no-op)",
    )


def run_evolve(args) -> int:
    """Run one evolution pass from a parsed ``args`` namespace (see
    :func:`add_evolve_arguments`) and print a summary. Returns the process exit
    code (0 unless the run itself raises)."""
    from misterdev.config import ConfigManager, get_setting
    from misterdev.core.execution.project import Project

    repo = args.repo or os.getcwd()
    config = ConfigManager().load_project_config(repo)
    project = Project(repo, config)

    benchmark = args.benchmark or get_setting(config, "evolution", "benchmark_dir")
    if not benchmark:
        _emit(
            "no --benchmark given and evolution.benchmark_dir is not set in "
            "project.yaml"
        )
        return 1
    noise_band = (
        args.noise_band
        if args.noise_band is not None
        else get_setting(config, "evolution", "noise_band")
    )

    target = failure_target_for_repo(repo) if args.from_failures else None
    if args.from_failures and target is None:
        _emit("from-failures: no logged real failures to target; nothing to do")
        return 0

    run_kwargs = dict(
        steps=args.steps,
        noise_band=noise_band,
        limit=args.limit,
        languages=args.languages,
        model=args.model,
        target=target,
        screen=args.screen or args.beam > 1,
        beam=max(1, args.beam),
    )

    if getattr(args, "scheduled", False):
        if not args.live:
            _emit("--scheduled requires --live (a dry-run has nothing to schedule)")
            return 1
        from misterdev.core.evolution.scheduled import run_scheduled_evolution

        result = run_scheduled_evolution(
            project,
            benchmark,
            args.workdir,
            gate_commands=gate_commands_for_repo(repo),
            **run_kwargs,
        )
        if result is None:
            _emit("scheduled: skipped (lock held or circuit breaker open)")
            return 0
    else:
        result = run_evolution(
            project,
            benchmark,
            args.workdir,
            live=args.live,
            gate_commands=gate_commands_for_repo(repo) if args.live else None,
            **run_kwargs,
        )

    for line in format_evolution_result(result):
        _emit(line)
    return 0


def format_evolution_result(result) -> list:
    """Render an ``EvolutionResult`` as the lines the CLI and MCP tool both show.

    Shared so ``run_evolve`` and ``mcp_server.evolve_async`` can't drift on what
    a run's outcome looks like to a caller.
    """
    lines = [
        f"baseline: {result.baseline.resolved}/{result.baseline.total} "
        f"({result.baseline.resolved_rate:.1%})"
    ]
    if result.blame:
        lines.append(
            f"top blame: {result.blame.niche} "
            f"({result.blame.failures}/{result.blame.total})"
        )
    for m in result.proposals:
        lines.append(f"proposal [{m.note}] touches: {', '.join(m.paths) or '(none)'}")
    for i, s in enumerate(result.steps, 1):
        lines.append(
            f"step {i}: {s.reason}"
            + (f" -> {s.score.resolved}/{s.score.total}" if s.score else "")
        )
    if result.champion:
        c = result.champion
        lines.append(f"champion: {c.id} in {c.niche} ({c.resolved}/{c.total})")
    lines.append(f"note: {result.note}")
    return lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="misterdev.core.evolution")
    add_evolve_arguments(parser)
    args = parser.parse_args(argv)
    return run_evolve(args)


if __name__ == "__main__":
    sys.exit(main())
