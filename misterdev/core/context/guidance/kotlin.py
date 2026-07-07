"""Best-practice rules for Kotlin edits, selected by relevance at inject time.

See :mod:`._rules` for the model. ``core`` rules are always emitted; the rest
gate on trigger substrings found in the task context.
"""

from ._rules import Rule

KOTLIN_RULES = [
    # --- core baseline (always emitted) ---
    Rule(
        "Expression-oriented: if/when/try are expressions → assign to val. Keep when over a sealed type/enum exhaustive (no else, so a new variant fails to compile). Immutability default: val > var, read-only List/Map in signatures (not MutableList); data class for value carriers, never mutable state in one (breaks equals/hashCode).",
        core=True,
    ),
    Rule(
        "Null-safety as design: ?./?:/let compose over the Optional dance; every !! is a documented invariant or a bug → if the invariant is real, encode it in the type. Wrap Java platform types (String!) at the boundary, don't let them leak nullability inward.",
        core=True,
    ),
    Rule(
        "Make illegal states unrepresentable: sealed interface/sealed class + exhaustive when for closed sums (data object for unit variants). @JvmInline value class UserId(val value: Long) for zero-cost typed IDs > raw Long/String.",
        core=True,
    ),
    # --- coroutines / structured concurrency ---
    Rule(
        "Structured concurrency: launch/async inside a coroutineScope/supervisorScope, never GlobalScope (leaks, ignores cancellation). Dispatchers.Default (CPU) / IO (blocking) / Main; withContext to switch. runBlocking only in main/tests. Cooperative cancellation: ensureActive/yield/isActive in loops. Flow is cold; StateFlow/SharedFlow hot. Mutex.withLock (suspends) not synchronized in suspend code.",
        triggers=(
            "coroutine",
            "suspend",
            "async",
            "launch",
            "flow",
            "dispatcher",
            "scope",
            "channel",
            "await",
            "concurren",
        ),
    ),
    # --- error handling ---
    Rule(
        "Sealed hierarchies for domain errors (return them), exceptions for exceptional conditions (throw them). runCatching/Result caveats: it captures CancellationException, and has coroutine/variance/interop rough edges → don't wrap suspend calls blindly. ALWAYS rethrow CancellationException — catching+swallowing it breaks structured concurrency.",
        triggers=(
            "exception",
            "throw",
            "runcatching",
            "result",
            "try",
            "catch",
            "cancellation",
        ),
    ),
    # --- functional / collections / perf ---
    Rule(
        "Collection ops (map/filter/groupBy/associateBy) > manual loops. Sequence for lazy long/short-circuiting chains — not free per element, eager is faster for short chains on small collections. inline fun for zero-overhead lambdas; reified type params to keep type info at runtime; function references (::foo) > trivial lambdas.",
        triggers=(
            "map",
            "filter",
            "sequence",
            "inline",
            "lambda",
            "reified",
            "collection",
            "fold",
            "groupby",
        ),
    ),
    # --- class design ---
    Rule(
        "Final by default — open deliberately; composition > inheritance. value class for wrappers; object for singletons, companion object for statics/factories. Property delegation (by lazy) for cached init. operator/infix used restrained (only where the symbol reads unambiguously).",
        triggers=(
            "class",
            "interface",
            "sealed",
            "open",
            "object",
            "companion",
            "delegate",
            "operator",
            "infix",
            "inheritance",
        ),
    ),
    # --- multiplatform ---
    Rule(
        "KMP: shared logic in commonMain, platform bits behind expect/actual. Prefer kotlinx.serialization/datetime/io over JVM-only libs; Ktor for multiplatform HTTP. Keep expect surface minimal — every expect is a per-target actual burden.",
        triggers=(
            "multiplatform",
            "kmp",
            "expect",
            "actual",
            "commonmain",
            "ktor",
            "kotlinx",
            "serialization",
        ),
    ),
    # --- compose ---
    Rule(
        "Composables are pure functions of state (hoist state up, emit events down). remember + mutableStateOf for local UI state; derivedStateOf for computed state. Side effects only in LaunchedEffect/DisposableEffect. @Stable/@Immutable to unlock skipping — but mutable state in an immutable-looking class breaks recomposition correctness.",
        triggers=(
            "compose",
            "composable",
            "remember",
            "recomposition",
            "mutablestateof",
        ),
    ),
    # --- tooling ---
    Rule(
        "ktlint (format) + detekt (static analysis) in CI; allWarningsAsErrors on. -Xexplicit-api=strict for libraries (public API must be typed & annotated). kotlinx.serialization (compile-time, no reflection) > runtime reflection. runTest for coroutine tests (virtual time, no real delays).",
        triggers=(
            "gradle",
            "ktlint",
            "detekt",
            "build",
            "warning",
        ),
    ),
]
