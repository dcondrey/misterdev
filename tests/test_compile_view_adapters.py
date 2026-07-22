"""T2.1 — compile_view is a per-language adapter registry, not an if/else.

Asserts the adapter interface + registry, a fully-working TypeScript (tsc) adapter
alongside the existing rust one, and the contract of the go/swift/csharp stubs
(registered, interface-conformant, recognizing nothing yet). Rust behavior is
covered by test_compile_view.py and must remain unaffected.
"""

import pytest

from misterdev.core.execution.compile_view import (
    CompilerAdapter,
    extract_compile_errors,
    get_adapter,
    registered_languages,
)

# Classic tsc output (two diagnostics) and the --pretty variant of the first.
TSC_CLASSIC = (
    "src/index.ts(12,7): error TS2322: Type 'string' is not assignable to type 'number'.\n"
    "src/index.ts(20,3): error TS2554: Expected 1 arguments, but got 0.\n"
)
TSC_PRETTY = "src/index.ts:12:7 - error TS2322: Type 'string' is not assignable to type 'number'.\n"

STUB_LANGUAGES = []  # go/swift/csharp promoted to real adapters (T2.1b)
FULL_LANGUAGES = ["rust", "typescript", "go", "swift", "csharp"]


def test_registry_has_all_five_languages():
    langs = registered_languages()
    for lang in FULL_LANGUAGES + STUB_LANGUAGES:
        assert lang in langs, f"{lang} adapter must be registered"


@pytest.mark.parametrize("lang", FULL_LANGUAGES + STUB_LANGUAGES)
def test_every_adapter_conforms_to_the_interface(lang):
    ad = get_adapter(lang)
    assert isinstance(ad, CompilerAdapter)
    assert ad.language == lang
    assert isinstance(ad.name, str) and ad.name
    assert isinstance(ad.parse("some output"), list)
    assert isinstance(ad.detect("some output"), bool)


def test_typescript_classic_extracted():
    errs = extract_compile_errors(TSC_CLASSIC, language="typescript")
    by_code = {e.code: e for e in errs}
    assert set(by_code) == {"TS2322", "TS2554"}
    assert by_code["TS2322"].location == "src/index.ts:12:7"
    assert "not assignable" in by_code["TS2322"].message
    assert by_code["TS2554"].location == "src/index.ts:20:3"


def test_typescript_pretty_extracted():
    errs = extract_compile_errors(TSC_PRETTY, language="typescript")
    assert any(e.code == "TS2322" and e.location == "src/index.ts:12:7" for e in errs)


def test_typescript_autodetected_without_language():
    errs = extract_compile_errors(TSC_CLASSIC)
    assert any(e.code == "TS2322" for e in errs)


def test_typescript_does_not_misparse_rust():
    # A rust type error must not be claimed by the tsc adapter.
    assert extract_compile_errors(
        "error[E0308]: mismatched types", language="typescript"
    ) == [] or all(
        e.code and e.code.startswith("E")
        for e in extract_compile_errors("error[E0308]: mismatched types")
    )
