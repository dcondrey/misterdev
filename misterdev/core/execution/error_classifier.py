"""Build and test error classification.

Categorizes raw compiler/test output into actionable error types so the
LLM gets targeted fix guidance instead of raw error dumps.

Categories:
  syntax       - Parse/syntax errors (fix the malformed code)
  missing_import - Unresolved import/use (add the import or dependency)
  wrong_type   - Type mismatch, wrong argument count (fix signature or call site)
  missing_export - Symbol exists but isn't pub/exported (add visibility modifier)
  missing_symbol - Function/struct/field doesn't exist yet (create it or fix the name)
  test_assertion - Test assertion failed (fix logic, not syntax)
  link_error   - Linker/unresolved external (fix Cargo.toml or feature flags)
  unknown      - Unclassifiable
"""

from typing import Dict, List, Tuple

from misterdev.core.execution.error_log_compressor import compress_error_log
from misterdev.logging_setup import setup_logger

logger = setup_logger(__name__)


class ErrorCategory:
    SYNTAX = "syntax"
    MISSING_IMPORT = "missing_import"
    WRONG_TYPE = "wrong_type"
    MISSING_EXPORT = "missing_export"
    MISSING_SYMBOL = "missing_symbol"
    TEST_ASSERTION = "test_assertion"
    LINK_ERROR = "link_error"
    MANIFEST = "manifest_error"
    FILE_NOT_FOUND = "file_not_found"
    UNKNOWN = "unknown"


# (category, indicator_strings)
_INDICATORS: List[Tuple[str, List[str]]] = [
    (
        ErrorCategory.SYNTAX,
        [
            "syntax error",
            "SyntaxError",
            "unexpected token",
            "unexpected eof",
            "unterminated",
            "invalid syntax",
            "expected one of",
            "expected `,`",
            "expected `{`",
            "expected `;`",
            # clang/gcc/swiftc
            "expected ';'",
            "expected expression",
            "expected '}'",
            "expected declaration",
            "extraneous",
            "expected identifier",
        ],
    ),
    (
        ErrorCategory.MISSING_IMPORT,
        [
            "unresolved import",
            "ModuleNotFoundError",
            "No module named",
            "could not find crate",
            "use of undeclared crate",
            "cannot find module",
            "ImportError",
            # swift / C / C++
            "no such module",
            "cannot find interface declaration",
        ],
    ),
    (
        ErrorCategory.MISSING_EXPORT,
        [
            "is private",
            "not accessible",
            "module is private",
            "function is private",
            "struct is private",
            "pub(crate)",
            "inaccessible",
            # swift / C++ / C#
            "is inaccessible due to",
            "marked private",
            "not visible",
            "inaccessible due to its protection level",
        ],
    ),
    (
        ErrorCategory.WRONG_TYPE,
        [
            "type mismatch",
            "expected type",
            "mismatched types",
            "TypeError",
            "incompatible type",
            "cannot convert",
            "expected struct",
            "expected enum",
            "wrong number of",
            "arguments were supplied",
            "the trait bound",
            "is not satisfied",
            # swift / clang
            "cannot convert value of type",
            "cannot initialize",
            "no viable conversion",
            "no matching function for call",
            "argument of type",
            "incompatible pointer",
            # C# / roslyn
            "cannot implicitly convert type",
            "cannot convert from",
        ],
    ),
    (
        ErrorCategory.MISSING_SYMBOL,
        [
            "not found in this scope",
            "cannot find value",
            "cannot find function",
            "cannot find type",
            "no field",
            "no method named",
            "no variant",
            "NameError",
            "AttributeError",
            "has no member",
            "unknown field",
            # swift / clang
            "use of undeclared identifier",
            "use of unresolved identifier",
            "no member named",
            "has no member named",
            # C# / roslyn
            "does not contain a definition for",
            "does not exist in the current context",
        ],
    ),
    (
        ErrorCategory.TEST_ASSERTION,
        [
            "assertion failed",
            "AssertionError",
            "assert_eq",
            "assert_ne",
            "panicked at",
            "test result: FAILED",
            "left:",
            "right:",
            # XCTest / ctest
            "XCTAssert",
            "XCTFail",
            "failed (",
            "tests failed out of",
        ],
    ),
    (
        ErrorCategory.LINK_ERROR,
        [
            "linker",
            "undefined reference",
            "unresolved external",
            "link error",
            "multiple definition",
            "duplicate symbol",
            # apple ld / clang
            "undefined symbols for architecture",
            "symbol(s) not found",
            "ld: ",
            "linker command failed",
        ],
    ),
    (
        ErrorCategory.MANIFEST,
        [
            "failed to parse manifest",
            "could not find `cargo.toml`",
            "missing either a `[package]`",
            "virtual manifest",
            "invalid toml",
            "expected value",
            "duplicate key",
            "error parsing pyproject.toml",
            "invalid package.json",
            # swiftpm / meson / cmake
            "could not find package.swift",
            "manifest parse error",
            "neither directory contains a build file",
            "cmake error",
            "does not appear to contain cmakelists.txt",
        ],
    ),
    (
        ErrorCategory.FILE_NOT_FOUND,
        [
            "no such file or directory",
            "file not found",
            "cannot open",
            "enoent",
        ],
    ),
]

# Fix guidance per category
FIX_GUIDANCE = {
    ErrorCategory.SYNTAX: (
        "This is a SYNTAX error. The code is malformed. "
        "Fix the specific line indicated. Do not restructure; just correct the syntax."
    ),
    ErrorCategory.MISSING_IMPORT: (
        "This is a MISSING IMPORT error. A module, crate, or package is referenced but not imported. "
        "Add the correct 'use' statement or dependency. Check Cargo.toml or import statements."
    ),
    ErrorCategory.MISSING_EXPORT: (
        "This is a VISIBILITY error. The symbol exists but is not public. "
        "Add 'pub' to the declaration, or change the import path to use a re-export."
    ),
    ErrorCategory.WRONG_TYPE: (
        "This is a TYPE MISMATCH error. The function signature or argument types don't match the call site. "
        "Check the interface contracts above. Match the exact types expected."
    ),
    ErrorCategory.MISSING_SYMBOL: (
        "This is a MISSING SYMBOL error. A function, struct, field, or method is referenced but doesn't exist. "
        "Either create it, fix the spelling, or check which module it should come from."
    ),
    ErrorCategory.TEST_ASSERTION: (
        "This is a TEST ASSERTION failure. The code compiles and runs but produces wrong results. "
        "Focus on the LOGIC, not the syntax. Check the algorithm, boundary conditions, and data flow."
    ),
    ErrorCategory.LINK_ERROR: (
        "This is a LINKER error. Check that all referenced symbols are defined and that the "
        "needed libraries/dependencies and feature flags are declared in the build manifest "
        "(Cargo.toml, CMakeLists.txt, package config) and linked."
    ),
    ErrorCategory.MANIFEST: (
        "This is a MANIFEST/CONFIG error. The project manifest (Cargo.toml, pyproject.toml, "
        "package.json) is malformed or missing required sections. Ensure required sections exist "
        "(e.g. [package] with name and version for Cargo.toml) and that the file is valid TOML/JSON."
    ),
    ErrorCategory.FILE_NOT_FOUND: (
        "This is a FILE NOT FOUND error. A referenced file or path does not exist. "
        "Create the missing file or correct the path; check for typos and relative-path assumptions."
    ),
    ErrorCategory.UNKNOWN: (
        "Error type could not be classified. Read the full error output carefully and fix the root cause."
    ),
}


# Rust compiler error codes -> category (authoritative, no fuzzy matching needed)
_RUST_ERROR_CODES: Dict[str, str] = {
    "E0061": ErrorCategory.WRONG_TYPE,  # wrong number of arguments
    "E0106": ErrorCategory.SYNTAX,  # missing lifetime specifier
    "E0277": ErrorCategory.WRONG_TYPE,  # trait bound not satisfied
    "E0308": ErrorCategory.WRONG_TYPE,  # mismatched types
    "E0369": ErrorCategory.WRONG_TYPE,  # binary operation not supported
    "E0382": ErrorCategory.WRONG_TYPE,  # use of moved value
    "E0412": ErrorCategory.MISSING_SYMBOL,  # cannot find type
    "E0422": ErrorCategory.MISSING_SYMBOL,  # cannot find struct/variant
    "E0425": ErrorCategory.MISSING_SYMBOL,  # cannot find value/function
    "E0432": ErrorCategory.MISSING_IMPORT,  # unresolved import
    "E0433": ErrorCategory.MISSING_IMPORT,  # failed to resolve: use of undeclared crate
    "E0603": ErrorCategory.MISSING_EXPORT,  # private item
    "E0614": ErrorCategory.WRONG_TYPE,  # cannot dereference
    "E0615": ErrorCategory.MISSING_SYMBOL,  # attempted to take value of method
    "E0624": ErrorCategory.MISSING_EXPORT,  # associated item is private
}


# C# / Roslyn compiler error codes -> category (authoritative)
_CSHARP_ERROR_CODES: Dict[str, str] = {
    "CS0103": ErrorCategory.MISSING_SYMBOL,  # name does not exist in context
    "CS0117": ErrorCategory.MISSING_SYMBOL,  # type has no definition for member
    "CS1061": ErrorCategory.MISSING_SYMBOL,  # no definition / extension method
    "CS0246": ErrorCategory.MISSING_IMPORT,  # type/namespace not found (using?)
    "CS0234": ErrorCategory.MISSING_IMPORT,  # namespace member does not exist
    "CS0029": ErrorCategory.WRONG_TYPE,  # cannot implicitly convert
    "CS1503": ErrorCategory.WRONG_TYPE,  # argument cannot convert from
    "CS0019": ErrorCategory.WRONG_TYPE,  # operator cannot be applied
    "CS1002": ErrorCategory.SYNTAX,  # ; expected
    "CS1513": ErrorCategory.SYNTAX,  # } expected
    "CS1519": ErrorCategory.SYNTAX,  # invalid token
    "CS0122": ErrorCategory.MISSING_EXPORT,  # inaccessible protection level
}


# Tie-break priority (lower rank = wins a tie). Categories that block the build
# outright and are more fundamental come first: a syntax/manifest failure that
# also trips a downstream type/symbol indicator is the syntax/manifest error.
# Mirrors the order categories are declared in ``_INDICATORS``.
_TIE_BREAK_ORDER: List[str] = [cat for cat, _ in _INDICATORS]


def _tie_break_rank(category: str) -> int:
    try:
        return _TIE_BREAK_ORDER.index(category)
    except ValueError:
        return len(_TIE_BREAK_ORDER)


def classify_error(error_output: str) -> str:
    """Classify build/test error output into a category.

    Checks structured compiler error codes first (Rust, then C#/Roslyn) for an
    exact, authoritative match, then falls back to keyword scoring for other
    languages and test output.
    """
    # Fast path: Rust structured error codes
    for code, category in _RUST_ERROR_CODES.items():
        if f"error[{code}]" in error_output:
            return category
    # Fast path: C#/Roslyn error codes (e.g. "error CS0103:")
    for code, category in _CSHARP_ERROR_CODES.items():
        if f"{code}:" in error_output:
            return category

    lower = error_output.lower()

    # Score each category by number of matching indicators
    scores: Dict[str, int] = {}
    for category, indicators in _INDICATORS:
        score = sum(1 for ind in indicators if ind.lower() in lower)
        if score > 0:
            scores[category] = score

    if not scores:
        return ErrorCategory.UNKNOWN

    # Highest indicator count wins; ties resolve by an explicit priority (more
    # fundamental, build-blocking categories first) rather than relying on the
    # implicit dict-insertion order, so the result is stable across refactors.
    return max(scores, key=lambda c: (scores[c], -_tie_break_rank(c)))


def classify_and_guide(error_output: str) -> Tuple[str, str]:
    """Classify an error and return (category, fix_guidance)."""
    category = classify_error(error_output)
    guidance = FIX_GUIDANCE.get(category, FIX_GUIDANCE[ErrorCategory.UNKNOWN])
    logger.info(f"Error classified as: {category}")
    return category, guidance


def format_classified_error(error_output: str, max_lines: int = 50) -> str:
    """Classify error and format with guidance for LLM prompt."""
    category, guidance = classify_and_guide(error_output)

    # Structure-aware compression first (drops source echo / caret art / explain
    # hints, dedups, caps to the first errors); the line cap is then a belt-and-
    # suspenders bound in case the compressor degraded to a raw-head fallback.
    compact = compress_error_log(error_output) or error_output
    lines = compact.splitlines()
    if len(lines) > max_lines:
        compact = (
            "\n".join(lines[:max_lines])
            + f"\n... ({len(lines) - max_lines} more lines)"
        )

    return (
        f"### Error Classification: {category.upper()}\n"
        f"**Fix Strategy**: {guidance}\n\n"
        f"### Raw Error Output\n```\n{compact}\n```"
    )
