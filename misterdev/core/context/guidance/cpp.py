"""Best-practice rules for C++ edits (C++20/23), selected by relevance at inject time.

See :mod:`._rules` for the model. ``core`` rules are always emitted; the rest
gate on trigger substrings found in the task context.
"""

from ._rules import Rule

CPP_RULES = [
    # --- core baseline (always emitted) ---
    Rule(
        "RAII every resource. Raw pointers non-owning ONLY; unique_ptr owns; shared_ptr only for genuinely shared ownership; weak_ptr breaks cycles. make_unique/make_shared — no manual new/delete in app code.",
        core=True,
    ),
    Rule(
        "const-correctness is not decorative: const methods that don't mutate, const& params not consumed/mutated. constexpr everything evaluable at compile time.",
        core=True,
    ),
    Rule(
        "Prefer the modern subset: Concepts > SFINAE, ranges > raw iterator dances, string_view/span for non-owning views. Writing C++ at all → be sure you need what it uniquely offers (else Rust).",
        core=True,
    ),
    # --- move semantics ---
    Rule(
        "std::move to transfer, std::forward for perfect forwarding — never move a const& or a named return. Know when a copy elides vs a move happens vs neither; return by value and let RVO work. The difference is C-speed vs Java-speed.",
        triggers=("move", "rvalue", "forward", "copy", "elision", "&&"),
    ),
    # --- generics / templates ---
    Rule(
        "Concepts (C++20) + requires clauses > SFINAE/enable_if/tag dispatch — legible errors, documented intent. Constrain templates so callers see what they must satisfy, not a wall of substitution failures.",
        triggers=("template", "concept", "requires", "sfinae", "enable_if", "generic"),
    ),
    # --- ranges / coroutines ---
    Rule(
        "ranges/views compose lazily, no intermediate allocations (closer to Rust iterators) — pipe transform/filter/take, don't hand-roll iterator loops. Coroutines are a compiler primitive with thin stdlib support → reach for cppcoro/libunifex/Boost, don't roll your own promise types.",
        triggers=(
            "range",
            "view",
            "iterator",
            "algorithm",
            "pipeline",
            "coroutine",
            "co_await",
            "co_yield",
        ),
    ),
    # --- UB & safety ---
    Rule(
        "UB is exploitable, not merely 'implementation-defined': uninit reads, use-after-free, data races, signed overflow, iterator invalidation. Run ASan/UBSan/TSan/MSan in CI; fuzz every parser & deserializer.",
        triggers=(
            "undefined",
            "ub",
            "sanitizer",
            "asan",
            "ubsan",
            "tsan",
            "msan",
            "race",
            "overflow",
            "fuzz",
            "security",
            "pointer",
            "invalidation",
        ),
    ),
    # --- tooling ---
    Rule(
        "clang-tidy with modernize-* + bugprone-* checks; include-what-you-use to kill transitive-include rot; C++20 modules to cut build times; warnings-as-errors (-Werror) in CI.",
        triggers=(
            "clang-tidy",
            "include",
            "module",
            "build",
            "warning",
            "lint",
        ),
    ),
]
