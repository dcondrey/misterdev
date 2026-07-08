"""Best-practice rules for CSS edits, selected by relevance at inject time.

See :mod:`._rules` for the model. ``core`` rules are always emitted; the rest
gate on trigger substrings found in the task context.
"""

from ._rules import Rule

CSS_RULES = [
    # --- core baseline (always emitted) ---
    Rule(
        "Design tokens = custom properties at :root (--color-*/--space-*/--type-*), consumed via var() → never copy-paste literals. Theme by overriding the variables at a scope, not by forking rule blocks. oklch()/color-mix() for perceptually-uniform color.",
        core=True,
    ),
    Rule(
        "Class selectors for almost everything; avoid !important & deep nesting. @layer to order cross-codebase specificity; scope via CSS Modules/scoped styles. data-* attributes for state (not class-toggling) and JS hooks.",
        core=True,
    ),
    # --- layout ---
    Rule(
        "Flexbox = 1D, Grid = 2D. grid-template named areas > div-nesting; repeat(auto-fit, minmax()) for responsive grids without media queries; subgrid to align nested items. gap (works on flex now) > margins; position: sticky natively; logical properties (margin-inline/padding-block) for i18n.",
        triggers=(
            "layout",
            "grid",
            "flex",
            "flexbox",
            "column",
            "row",
            "sticky",
            "align",
            "gap",
            "position",
        ),
    ),
    # --- responsive ---
    Rule(
        "Container queries (@container + container-type) for component-driven breakpoints — media queries are for the page. Fluid clamp()/min()/max() over stepped breakpoints; rem for scalable type, unitless line-height; aspect-ratio > padding hacks.",
        triggers=(
            "responsive",
            "container",
            "media query",
            "breakpoint",
            "clamp",
            "viewport",
            "mobile",
            "aspect-ratio",
            "font-size",
        ),
    ),
    # --- modern selectors ---
    Rule(
        ":has() for parent/relational selection; :is()/:where() to group — :where() = zero specificity for resets. Native nesting (&) → mind the implicit-descendant compat gotcha.",
        triggers=(
            "selector",
            ":has",
            ":is",
            ":where",
            "nesting",
            "specificity",
            "parent",
        ),
    ),
    # --- a11y ---
    Rule(
        ":focus-visible for keyboard focus rings — never outline:none. WCAG contrast (4.5:1 text, 3:1 UI); prefers-reduced-motion fallback for every animation; prefers-color-scheme + color-scheme for dark mode & native controls; prefers-contrast.",
        triggers=(
            "focus",
            "contrast",
            "motion",
            "animation",
            "dark mode",
            "prefers",
            "accessib",
            "wcag",
            "outline",
        ),
    ),
    # --- performance ---
    Rule(
        "CSS is render-blocking → inline critical, load the rest async. Animate only transform/opacity (GPU, no layout); content-visibility: auto + contain-intrinsic-size for offscreen; will-change sparingly (add before, remove after); batch layout reads before writes to avoid thrashing.",
        triggers=(
            "perf",
            "animation",
            "transition",
            "will-change",
            "content-visibility",
            "render",
            "paint",
            "reflow",
            "critical",
        ),
    ),
    # --- typography / print ---
    Rule(
        "text-wrap: balance (headings)/pretty (body); font-display: swap + preload + subset the font. @media print + break-inside: avoid for printable/PDF/reader views.",
        triggers=(
            "font",
            "typography",
            "text-wrap",
            "print",
            "pdf",
            "line-height",
            "letter",
        ),
    ),
    # --- Sass / SCSS (.scss/.sass fold into this module) ---
    Rule(
        "Sass: @use/@forward (the namespaced module system) — never the deprecated @import (global leakage, repeated re-parsing). One concern per _partial.scss. Keep anything themed at RUNTIME in CSS custom properties (var()), NOT Sass $variables: $vars compile away and cannot react to :root overrides, media queries, or JS.",
        triggers=(
            "sass",
            "scss",
            "@use",
            "@forward",
            "@import",
            "partial",
            "$variable",
            "preprocessor",
        ),
    ),
    Rule(
        "Sass: @mixin/@include for parameterized reuse; @extend/%placeholder only WITHIN one file (cross-file @extend reorders the cascade and bloats output). math.div() not the deprecated /; keep nesting ≤3 levels so compiled selectors don't explode in length/specificity — & is for state/variants, not deep trees.",
        triggers=(
            "sass",
            "scss",
            "@mixin",
            "@include",
            "@extend",
            "placeholder",
            "math.div",
            "nesting",
        ),
    ),
]
