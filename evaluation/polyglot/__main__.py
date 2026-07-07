"""CLI: python -m evaluation.polyglot --benchmark ./polyglot-benchmark --workdir /tmp/poly

Clone the exercises first (git clone https://github.com/Aider-AI/polyglot-benchmark),
then point --benchmark at the checkout. Use --languages python to run a
toolchain you have locally, and misterdev's free-model routing for a ~$0 run.
"""

import argparse
import json
import sys

from .harness import run_suite


def _emit(line: str) -> None:
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="evaluation.polyglot")
    parser.add_argument(
        "--benchmark", required=True, help="polyglot-benchmark checkout"
    )
    parser.add_argument("--workdir", required=True, help="scratch dir for exercises")
    parser.add_argument(
        "--languages", nargs="*", default=None, help="languages to run (default: all)"
    )
    parser.add_argument("--limit", type=int, default=None, help="cap exercises run")
    parser.add_argument(
        "--model",
        default=None,
        help="pin a single model (e.g. a ':free' id) with no paid escalation",
    )
    parser.add_argument(
        "--build-args",
        default="--budget 2 --allow-dirty --no-suggest",
        help="args passed to misterdev build",
    )
    parser.add_argument("--env-activate", default=None, help="env prefix for tests")
    parser.add_argument(
        "--json", default=None, help="also write the full report as JSON to this path"
    )
    args = parser.parse_args(argv)

    report = run_suite(
        args.benchmark,
        args.workdir,
        languages=args.languages,
        limit=args.limit,
        model=args.model,
        build_args=args.build_args,
        env_activate=args.env_activate,
        progress=lambda r: _emit(
            f"  [{'PASS' if r.resolved else 'FAIL'}] {r.language}/{r.name}"
        ),
    )
    _emit(report.summary())
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report.to_dict(), fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
