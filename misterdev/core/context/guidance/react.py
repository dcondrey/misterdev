"""Best-practice rules for React/JSX/TSX edits, selected by relevance at inject time.

See :mod:`._rules` for the model. ``core`` rules are always emitted; the rest
gate on trigger substrings found in the task context.
"""

from ._rules import Rule

REACT_RULES = [
    # --- core baseline (always emitted) ---
    Rule(
        "Components are pure: same props+state → same output; no Date.now()/Math.random()/mutation/side effects during render (Strict Mode + React Compiler assume purity). Function components + hooks only. State is a snapshot → use the updater setCount(c => c+1) when basing on prior state.",
        core=True,
    ),
    Rule(
        "State lives at the lowest common ancestor of its readers; derived state is NOT state → compute inline / useMemo, never useState + a syncing useEffect. Server state → query cache (TanStack Query/SWR), URL state → the URL, form state → a form lib, UI state → useState. Context is not state management.",
        core=True,
    ),
    # --- effects ---
    Rule(
        "useEffect only to SYNC with outside-React (subscriptions, DOM measurement, non-React libs) — not for derived state, event handling, or 'after render'. Every resource-acquiring effect returns a cleanup (unsubscribe/abort). Never disable react-hooks/exhaustive-deps. useLayoutEffect only to measure/mutate DOM before paint.",
        triggers=(
            "effect",
            "useeffect",
            "subscription",
            "cleanup",
            "layouteffect",
            "sync",
        ),
    ),
    # --- performance ---
    Rule(
        "Re-renders usually aren't the bottleneck — measure first. memo/useMemo/useCallback help only with referentially-stable props (React Compiler auto-memoizes → prefer pure components). Stable key from data identity (item.id), never index. Virtualize long lists; startTransition/useDeferredValue for non-urgent updates. Images dominate: loading=lazy, srcset, reserve dims; code-split via React.lazy. Targets LCP<2.5s / INP<200ms / CLS<0.1.",
        triggers=(
            "perf",
            "memo",
            "render",
            "rerender",
            "key",
            "list",
            "image",
            "lazy",
            "bundle",
            "virtualize",
        ),
    ),
    # --- suspense / async ---
    Rule(
        "<Suspense> boundaries where the user's 'loading' mental model lives (nest for staged loads), each paired with an error boundary. use() (React 19) reads promises/context. Server Components ship zero client JS for their subtrees.",
        triggers=(
            "suspense",
            "async",
            "server component",
            "use(",
            "streaming",
            "boundary",
        ),
    ),
    # --- forms / actions ---
    Rule(
        "React 19: <form action={fn}> + useActionState/useFormStatus for mutations; ref is just a prop (forwardRef is legacy); document metadata/stylesheets in components are hoisted+deduped.",
        triggers=(
            "form",
            "action",
            "mutation",
            "useactionstate",
            "useformstatus",
            "submit",
        ),
    ),
    # --- component design ---
    Rule(
        "Composition over configuration: compound components (<Card.Header>) via context, not a hundred boolean props. Custom hooks for logic, presentational components for UI. Controlled OR uncontrolled per input, not both. Portals (createPortal) for modals/tooltips/dropdowns. DRY: extract reusable components + hooks.",
        triggers=(
            "component",
            "prop",
            "context",
            "portal",
            "compound",
            "hook",
            "controlled",
        ),
    ),
    # --- types / a11y / security ---
    Rule(
        "Type components as plain functions (not FC), discriminated unions for prop modes, no any. Semantic HTML (<button>, not <div onClick>); manage focus on route/modal changes; accessible primitives (Radix/React Aria) + jsx-a11y. Never dangerouslySetInnerHTML with untrusted data (XSS).",
        triggers=(
            "typescript",
            "type",
            "aria",
            "a11y",
            "accessib",
            "focus",
            "semantic",
            "xss",
        ),
    ),
]
