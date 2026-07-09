"""CLI: python -m evaluation.native --root evaluation/native/exercises --workdir /tmp/native

Points --root at a tree laid out ``<language>/<slug>/`` (the bundled fixtures
follow this). Swift and C# had no empirical validation in the polyglot benchmark;
this drives misterdev over each stub and grades with the toolchain's own
``swift test`` / ``dotnet test``. Use ``--languages swift`` to run just the
toolchain you have locally, and ``--build-args "--budget 0.5 ..."`` to cap spend.
"""

import argparse
import json
import sys

from .harness import run_suite


def _emit(line: str) -> None:
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="evaluation.native")
    parser.add_argument("--root", required=True, help="exercises tree (<lang>/<slug>/)")
    parser.add_argument("--workdir", required=True, help="scratch dir for exercises")
    parser.add_argument(
        "--languages",
        nargs="*",
        default=None,
        help="swift and/or csharp (default: all)",
    )
    parser.add_argument("--limit", type=int, default=None, help="cap exercises run")
    parser.add_argument(
        "--only", nargs="*", default=None, help="run only these exercise slugs"
    )
    parser.add_argument(
        "--build-args",
        default="--budget 2 --allow-dirty --no-suggest",
        help="args passed to misterdev build (e.g. cap spend via --budget)",
    )
    parser.add_argument("--env-activate", default=None, help="env prefix for tests")
    parser.add_argument("--json", default=None, help="also write the report as JSON")
    args = parser.parse_args(argv)

    report = run_suite(
        args.root,
        args.workdir,
        languages=args.languages,
        limit=args.limit,
        only=args.only,
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
