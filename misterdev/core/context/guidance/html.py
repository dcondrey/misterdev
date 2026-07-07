"""Best-practice rules for HTML edits, selected by relevance at inject time.

See :mod:`._rules` for the model. ``core`` rules are always emitted; the rest
gate on trigger substrings found in the task context.
"""

from ._rules import Rule

HTML_RULES = [
    # --- core baseline (always emitted) ---
    Rule(
        "Pick the element that MEANS the thing: <button> for actions, <a href> for navigation, <ul>/<ol> for lists, <main>/<article>/<section>/<nav>/<aside> for landmarks — never <div> soup. Semantic HTML gives keyboard, focus, screen-reader & reader-mode behavior for free.",
        core=True,
    ),
    Rule(
        "Headings reflect structure, not size → one <h1>, no skipped levels, style with CSS. Set <html lang> and mark any multilingual content with lang.",
        core=True,
    ),
    # --- forms ---
    Rule(
        "Every input gets a real <label> (placeholder is NOT a label). Use the specific type (email/tel/url/date/number/search) + inputmode + autocomplete tokens; group with <fieldset>/<legend>; give buttons an explicit type=submit|button. Prefer required + :user-invalid + setCustomValidity over JS-reinvented validation.",
        triggers=(
            "form",
            "input",
            "label",
            "submit",
            "fieldset",
            "autocomplete",
            "validation",
            "checkbox",
            "radio",
            "select",
        ),
    ),
    # --- native widgets ---
    Rule(
        "Prefer native over reinventing: <details>/<summary> for disclosure, <dialog> for modals (built-in focus trap + Esc + inert background), <figure>/<figcaption> for captioned media, <time datetime> for dates, <picture>/<source> for art-direction/format switching.",
        triggers=(
            "dialog",
            "modal",
            "accordion",
            "disclosure",
            "details",
            "summary",
            "tooltip",
            "dropdown",
            "popover",
        ),
    ),
    # --- ARIA / accessibility ---
    Rule(
        "ARIA rule #1 = don't (use the semantic element). role=button on a <div> announces a lie — no Enter/Space/focus/disabled for free. Use aria-label for unlabeled controls, aria-live=polite for async updates, aria-current/expanded/controls/selected for state; follow the ARIA Authoring Practices for any custom widget.",
        triggers=(
            "aria",
            "role",
            "screen reader",
            "accessib",
            "a11y",
            "widget",
            "live region",
        ),
    ),
    # --- media / performance ---
    Rule(
        "Images need intrinsic width/height (or aspect-ratio) to prevent CLS; loading=lazy below the fold, fetchpriority=high on the LCP image; srcset/sizes + modern formats (AVIF/WebP). Scripts get defer/async; add rel=noopener to every target=_blank.",
        triggers=(
            "img",
            "image",
            "picture",
            "video",
            "script",
            "lazy",
            "lcp",
            "cls",
            "srcset",
            "loading",
        ),
    ),
    # --- security ---
    Rule(
        "Escape/validate ALL user content (XSS) — never inject raw HTML (innerHTML with untrusted data). Prefer a Content-Security-Policy; no inline event handlers or inline scripts.",
        triggers=(
            "xss",
            "sanitiz",
            "innerhtml",
            "csp",
            "escape",
            "user content",
        ),
    ),
]
