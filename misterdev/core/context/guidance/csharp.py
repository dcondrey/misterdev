"""Best-practice rules for C# edits, selected by relevance at inject time.

See :mod:`._rules` for the model. ``core`` rules are always emitted; the rest
gate on trigger substrings found in the task context.
"""

from ._rules import Rule

CSHARP_RULES = [
    # --- core baseline (always emitted) ---
    Rule(
        "record/record struct for value semantics (structural equality, with-expr), class only when identity matters; readonly record struct UserId(Guid Value) for strongly-typed IDs — compiler catches swaps at zero cost.",
        core=True,
    ),
    Rule(
        "Nullable reference types ON project-wide (<Nullable>enable</Nullable>), treat ?/! as API surface; required+init for immutable-but-constructible; sealed by default, internal for anything not deliberately public.",
        core=True,
    ),
    Rule(
        "Pattern matching / switch expressions + property/positional/list patterns > if-else pyramids; no DUs in the language → model closed sets with sealed hierarchies + exhaustive patterns (or OneOf). Composition > inheritance.",
        core=True,
    ),
    # --- async ---
    Rule(
        "async/await all the way, never .Result/.Wait()/.GetAwaiter().GetResult() (deadlock, pool starvation); ConfigureAwait(false) in libs; CancellationToken on every cancellable method — thread it through. ValueTask<T> only for usually-sync hot paths (await once, don't store); never async void except event handlers; IAsyncEnumerable<T> + bounded System.Threading.Channels for streams/backpressure — never unbounded.",
        triggers=(
            "async",
            "await",
            "task",
            "valuetask",
            "cancellation",
            "channel",
            "concurren",
        ),
    ),
    # --- memory / performance ---
    Rule(
        "GC: Gen2 & LOH (>85KB) hurt. Span<T>/stackalloc for zero-alloc slicing/parsing, ArrayPool<T>.Shared/ObjectPool<T> to reuse buffers, string.Create for known-length; in/ref readonly for large readonly structs, watch boxing (struct→object/interface). Struct generics specialize: foreach over List<T> beats a .Where().Select().ToList() chain on hot paths; CollectionsMarshal.AsSpan/GetValueRefOrNullRef for in-place; sealed enables JIT devirtualization. Return IReadOnlyList<T>/concrete not IEnumerable<T> when enumeration must be stable; materialize EF queries (ToList, AsNoTracking) at the boundary.",
        triggers=(
            "perf",
            "span",
            "stackalloc",
            "alloc",
            "gc",
            "struct",
            "boxing",
            "linq",
            "hot",
            "optimize",
            "buffer",
        ),
    ),
    # --- DI / config ---
    Rule(
        "Microsoft.Extensions.DependencyInjection with ValidateScopes+ValidateOnBuild; constructor injection only (no service-locator IServiceProvider), thin constructors (assign fields, no I/O); bind strongly-typed IOptions<T> + ValidateOnStart() — a misconfigured app fails to start, not at 3am.",
        triggers=(
            "di",
            "inject",
            "service",
            "ioptions",
            "scope",
            "configuration",
            "singleton",
        ),
    ),
    # --- serialization / crypto ---
    Rule(
        "System.Text.Json source generator (JsonSerializerContext) > reflection/Newtonsoft; Utf8JsonReader/Writer + Span<byte> all the way down. SHA256.HashData(span) one-shot; CryptographicOperations.FixedTimeEquals (not ==/SequenceEqual); RandomNumberGenerator.GetBytes (not Random).",
        triggers=(
            "json",
            "serialize",
            "crypto",
            "hash",
            "random",
            "security",
            "newtonsoft",
        ),
    ),
    # --- errors ---
    Rule(
        "Exceptions for exceptional/programmer errors, Result types for expected failures; custom exceptions carrying structured context, not stringified state; throw; not throw ex; (preserves stack); catch only what you can handle, let the rest reach a top-level boundary.",
        triggers=(
            "exception",
            "throw",
            "result",
            "try",
            "catch",
        ),
    ),
    # --- tooling ---
    Rule(
        "Analyzers as errors (<AnalysisLevel>latest-recommended, Meziantou/SonarAnalyzer), <TreatWarningsAsErrors>true, nullable warnings as errors, checked-in .editorconfig, dotnet format, dotnet list package --vulnerable, BenchmarkDotNet (not Stopwatch); Native AOT for CLIs/containers.",
        triggers=(
            "analyzer",
            "warning",
            "editorconfig",
            "format",
            "benchmark",
            "aot",
            "build",
        ),
    ),
]
