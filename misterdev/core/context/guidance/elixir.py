"""Best-practice rules for Elixir edits, selected by relevance at inject time.

See :mod:`._rules` for the model. ``core`` rules are always emitted; the rest
gate on trigger substrings found in the task context.
"""

from ._rules import Rule

ELIXIR_RULES = [
    # --- core baseline (always emitted) ---
    Rule(
        "Pattern-match in function heads (multiple clauses) → not a `case` in the body; the dispatch reads as a specification. Immutable data + the pipe `|>` for left-to-right composition (`data |> transform() |> validate()`).",
        core=True,
    ),
    Rule(
        "Tagged tuples `{:ok, value}`/`{:error, reason}` as the return convention; chain fallible ops with `with ... else` rather than nested conditionals or promise chains.",
        core=True,
    ),
    # --- OTP / processes ---
    Rule(
        "Everything with state, identity, or work-over-time is a process: `GenServer`/`Task`/`Agent` over the raw `spawn`/`send`/`receive`. Supervision trees for fault tolerance — 'let it crash': handle expected cases in code, let the unexpected crash and restart in a known-good state (choose the supervisor strategy & scope deliberately). Never leak unsupervised processes.",
        triggers=(
            "process",
            "genserver",
            "supervisor",
            "supervision",
            "spawn",
            "otp",
            "agent",
            "task",
            "state",
            "crash",
            "receive",
        ),
    ),
    # --- concurrency / distribution / pipelines ---
    Rule(
        "Distribution is native (`Node.connect`, libcluster/Horde for discovery & distributed registries). For data pipelines use `GenStage`/`Broadway`/`Flow` with backpressure — not unbounded producer/consumer.",
        triggers=(
            "distributed",
            "node",
            "cluster",
            "broadway",
            "genstage",
            "flow",
            "concurren",
            "pipeline",
            "backpressure",
        ),
    ),
    # --- Phoenix / Ecto ---
    Rule(
        "Phoenix LiveView (server-held state + WebSocket diffs) over a hand-rolled SPA where it fits. Ecto is a data mapper, not an ORM: compose queries, and use changesets to separate raw input from a validated update (`cast`/`validate_*`).",
        triggers=(
            "phoenix",
            "liveview",
            "ecto",
            "changeset",
            "repo",
            "web",
            "database",
            "query",
            "migration",
        ),
    ),
    # --- errors ---
    Rule(
        "Tagged tuples + `with` for expected failures; `raise`/`rescue` for the exceptional; let-it-crash + supervisor restart for the unexpected. Don't defensively rescue what a supervisor should restart.",
        triggers=(
            "error",
            "rescue",
            "raise",
            "try",
            "exception",
            "fail",
        ),
    ),
    # --- tooling ---
    Rule(
        "`mix` for build/tasks; `credo` (style/complexity), `dialyzer` via `dialyxir` (success-typing bugs), `sobelow` (Phoenix security). `ExUnit` tests run `async: true` in parallel — immutability + process isolation makes that safe.",
        triggers=(
            "mix",
            "credo",
            "dialyzer",
            "dialyxir",
            "sobelow",
            "test",
            "exunit",
            "format",
        ),
    ),
]
