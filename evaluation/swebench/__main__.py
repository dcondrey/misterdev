"""CLI: python -m evaluation.swebench --dataset tasks.jsonl --workdir /tmp/swe

Loads a JSONL export of SWE-bench instances, runs misterdev on each, and prints
the resolved rate. Point --dataset at a Lite/Verified export (or a handful of
instances) to measure whether a change actually moves the number.
"""

import argparse
import sys

from .harness import run_suite
from .instance import SWEBenchInstance


def _emit(line: str) -> None:
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="evaluation.swebench")
    parser.add_argument("--dataset", required=True, help="JSONL of SWE-bench records")
    parser.add_argument("--workdir", required=True, help="scratch dir for repos")
    parser.add_argument("--limit", type=int, default=None, help="cap instances run")
    parser.add_argument(
        "--build-args",
        default="--budget 5 --allow-dirty --no-suggest",
        help="args passed to misterdev build",
    )
    parser.add_argument("--env-activate", default=None, help="env prefix for tests")
    args = parser.parse_args(argv)

    instances = SWEBenchInstance.load_jsonl(args.dataset)
    report = run_suite(
        instances,
        args.workdir,
        limit=args.limit,
        build_args=args.build_args,
        env_activate=args.env_activate,
        progress=lambda r: _emit(
            f"  [{'PASS' if r.resolved else 'FAIL'}] {r.instance_id}"
        ),
    )
    _emit(report.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
