"""Best-practice rules for Swift edits, selected by relevance at inject time.

See :mod:`._rules` for the model. ``core`` rules are always emitted; the rest
gate on trigger substrings found in the task context.
"""

from ._rules import Rule

SWIFT_RULES = [
    # --- core baseline (always emitted) ---
    Rule(
        "Value semantics are the design language: struct by default → thread safety, predictable mutation, honest equality; class only when identity matters (lifecycle/observed mutation). CoW keeps them cheap (isKnownUniquelyReferenced for large-payload wrappers); mutating/inout for in-place mutation.",
        core=True,
    ),
    Rule(
        "No force-unwrap !/try!/as! & no implicitly-unwrapped optionals → if the invariant is real encode it in the type, else guard let at the top; enum-with-associated-values > a struct of several Options + a bool (exhaustive switch); phantom generics for zero-cost typed IDs; wrap platform (ObjC/NSError) types at the boundary.",
        core=True,
    ),
    # --- concurrency (Swift Concurrency, not GCD/Combine for new code) ---
    Rule(
        "async/await isn't free → design for meaningful suspensions; actors serialize their state (don't over-actor); @MainActor for UI only; Sendable enforced in Swift 6 (class needs final + immutable state). Structured concurrency (async let/TaskGroup, no detached Task{}); cancellation cooperative (Task.checkCancellation); AsyncStream(bufferingPolicy:) bounded; batch cross-actor hops; state can change across await (reentrancy).",
        triggers=(
            "async",
            "await",
            "actor",
            "task",
            "sendable",
            "dispatcher",
            "mainactor",
            "concurren",
            "asyncstream",
        ),
    ),
    # --- memory / ARC ---
    Rule(
        "ARC is deterministic; cycles are the cost. [weak self] default / [unowned self] only when the referent provably outlives the reference in closures capturing self; deinit for observation not real cleanup (make cleanup explicit & idempotent); ~Copyable/consuming/borrowing to shed ARC on single-owner resources & hot paths.",
        triggers=(
            "arc",
            "weak",
            "unowned",
            "retain",
            "cycle",
            "deinit",
            "closure",
            "capture",
            "copyable",
        ),
    ),
    # --- performance ---
    Rule(
        "Value types on the stack beat heap references (ARC tax = atomic swift_retain/release, visible in Instruments) → fix structurally. @inlinable hot generic library APIs. String is O(n)-indexed → Substring/utf8/Array<UInt8>, reserveCapacity; ContiguousArray to skip ObjC bridging; cache DateFormatter/Regex/Calendar; Swift Collections (Deque/OrderedSet). WMO on for release; profile Instruments + os_signpost; defer work out of launch.",
        triggers=(
            "perf",
            "string",
            "substring",
            "array",
            "inlinable",
            "instruments",
            "alloc",
            "hot",
            "optimize",
            "slow",
        ),
    ),
    # --- errors ---
    Rule(
        "throws for recoverable / precondition/fatalError for programmer errors & invariants (assert debug-only). Custom Error enums carrying context — never throw strings or NSError; Result for stored/async-boundary errors; typed throws only where the type is meaningful to callers.",
        triggers=(
            "throw",
            "error",
            "result",
            "precondition",
            "fatalerror",
            "try",
            "catch",
        ),
    ),
    # --- protocols / generics ---
    Rule(
        "Small composed protocols for real polymorphism (multiple impls / mockable boundary), not fashion. Generic constraints <T: P> > existentials any P (static dispatch inlines; any boxes + witness-table lookup); some P in return positions; conditional conformance.",
        triggers=(
            "protocol",
            "generic",
            "existential",
            "any",
            "some",
            "conformance",
            "dispatch",
        ),
    ),
    # --- SwiftUI ---
    Rule(
        "SwiftUI: body fast & pure; @State local / @Binding external / @Observable (5.9+) over @ObservableObject; stable Identifiable IDs in ForEach.",
        triggers=(
            "swiftui",
            "view",
            "@state",
            "binding",
            "observable",
            "body",
            "foreach",
        ),
    ),
]
