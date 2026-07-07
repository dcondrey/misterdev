"""Best-practice rules for TypeScript (.ts) edits, selected by relevance.

See :mod:`._rules` for the model. ``core`` rules are always emitted; the rest
gate on trigger substrings found in the task context. This is NON-React
TypeScript — React/TSX has its own rule set (see ``.react``).
"""

from ._rules import Rule

TYPESCRIPT_RULES = [
    # --- core baseline (always emitted) ---
    Rule(
        "strict: true from day one (every flag: noImplicitAny, strictNullChecks, strictFunctionTypes, …). any = abandoning the type system → unknown then narrow. Return types on public/exported fns; infer for internal helpers.",
        core=True,
    ),
    Rule(
        "TypeScript is a design language, not a linter with autocomplete: model the domain in types. readonly / ReadonlyArray<T> for immutability at the type level; prefer readonly props by default.",
        core=True,
    ),
    Rule(
        "Make illegal states unrepresentable: discriminated unions (a kind/type/_tag field) for anything with modes; exhaustive switch with a `never` assertion in default (const _exhaustive: never = x) → compile-time exhaustiveness.",
        core=True,
    ),
    # --- advanced type system ---
    Rule(
        "Branded types (string & { __brand: 'UserId' }) for nominal typing at zero runtime cost; template-literal types for stringly-typed APIs. Conditional/mapped types + built-in utilities (Partial/Pick/Omit/Record/ReturnType) > reinvention. Generic constraints (<T extends X>) > any-typed generics. `satisfies` (4.9+) to conform without widening; `as const` for literal inference.",
        triggers=(
            "brand",
            "template literal",
            "conditional",
            "mapped",
            "utility",
            "generic",
            "infer",
            "satisfies",
            "as const",
            "<t",
        ),
    ),
    # --- runtime validation at trust boundaries ---
    Rule(
        "Types are erased at runtime → validate at EVERY trust boundary (network, form, env, files) with Zod/Valibot/ArkType. Keep schema + type in one place: type T = z.infer<typeof schema>. Parse, don't assert (`as`).",
        triggers=(
            "validation",
            "zod",
            "valibot",
            "arktype",
            "parse",
            "boundary",
            "api",
            "input",
            "json",
            "schema",
            "env",
        ),
    ),
    # --- build / runtime ---
    Rule(
        "tsc for type-checking (noEmit) + a fast bundler (esbuild/swc/Vite) for compilation — separate the two. Run .ts directly with tsx / Bun / node --experimental-strip-types; don't ship untyped emit.",
        triggers=(
            "build",
            "bundler",
            "esbuild",
            "swc",
            "vite",
            "tsc",
            "bun",
            "ts-node",
            "tsx",
            "strip-types",
        ),
    ),
    # --- testing ---
    Rule(
        "Vitest (Vite projects) / Jest / node:test for runtime tests. Library authors: add type-level tests via expect-type / tsd so the public types can't silently regress.",
        triggers=(
            "test",
            "vitest",
            "jest",
            "node:test",
            "expect-type",
            "tsd",
        ),
    ),
    # --- async / errors ---
    Rule(
        "Typed errors / Result-union return types > any-typed catches. catch binds unknown → narrow before use (instanceof / discriminant). No floating promises — await or void them; type Promise<T> results, don't swallow rejections.",
        triggers=(
            "async",
            "await",
            "promise",
            "error",
            "result",
            "throw",
        ),
    ),
]
