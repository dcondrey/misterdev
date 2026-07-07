"""Best-practice rules for Python edits, selected by relevance at inject time.

See :mod:`._rules` for the model. ``core`` rules are always emitted; the rest
gate on trigger substrings found in the task context.
"""

from ._rules import Rule

PYTHON_RULES = [
    # --- core baseline (always emitted) ---
    Rule(
        'PEP 8, let ruff/black format; snake_case funcs/vars, PascalCase classes, UPPER_SNAKE constants. Short single-purpose functions (a name with "and" is two functions).',
        core=True,
    ),
    Rule(
        "Explicit > implicit, never `import *`. dataclasses/pydantic > __init__ boilerplate; pathlib > os.path; comprehensions/generators > map/filter.",
        core=True,
    ),
    Rule(
        "DRY: factor shared logic into helpers, delete dead code (don't comment it out). Name things in domain terms.",
        core=True,
    ),
    # --- typing / correctness ---
    Rule(
        "Annotate public signatures + run mypy/pyright; built-in generics (`list[str]` not `List[str]`); Protocol for structural typing > forced inheritance.",
        triggers=(
            "type",
            "annotation",
            "mypy",
            "pyright",
            "generic",
            "protocol",
            "hint",
        ),
    ),
    # --- errors / validation ---
    Rule(
        "Catch specific exceptions, never bare `except`; fail fast at boundaries + validate inputs; context managers (`with`) for files/locks/connections; never mutable default args (`def f(x=[])`).",
        triggers=(
            "except",
            "exception",
            "error",
            "raise",
            "try",
            "validation",
            "boundary",
            "input",
        ),
    ),
    # --- performance (keep work out of the interpreter) ---
    Rule(
        'Profile first (cProfile/py-spy/scalene) — Python\'s cost model is unintuitive. Vectorize hot paths with numpy/polars; `"".join(parts)` not `+=` in loops; set/dict for membership, `collections.deque` for FIFO, `functools.cache` for pure recompute, `__slots__` for bulk instances.',
        triggers=(
            "perf",
            "slow",
            "optimize",
            "loop",
            "alloc",
            "numpy",
            "polars",
            "cache",
            "profile",
            "hot",
            "memory",
        ),
    ),
    # --- concurrency ---
    Rule(
        "GIL: threads only help I/O; multiprocessing or a native ext for CPU-bound. asyncio (httpx/asyncpg) for I/O concurrency, bounded by a Semaphore. Batch DB writes and network calls.",
        triggers=(
            "thread",
            "async",
            "gil",
            "multiprocess",
            "concurren",
            "await",
            "asyncio",
            "parallel",
        ),
    ),
    # --- security ---
    Rule(
        "`hmac.compare_digest` for secrets (`==` is timing-attackable); never unpickle untrusted data; `yaml.safe_load` only; `subprocess` with arg lists, never `shell=True`+user input; parameterize SQL; `secrets` not `random` for tokens/nonces; keep secrets out of logs/repr/exceptions.",
        triggers=(
            "secret",
            "password",
            "pickle",
            "yaml",
            "subprocess",
            "sql",
            "crypto",
            "token",
            "hash",
            "random",
            "deserialize",
        ),
    ),
    # --- operational ---
    Rule(
        "Structured logging with correlation IDs; log levels mean things (ERROR pages, WARNING reviewed, INFO traces). Idempotent retryable network/DB ops; bound every queue and retry loop.",
        triggers=(
            "log",
            "logging",
            "print",
            "observability",
            "retry",
            "queue",
        ),
    ),
]
