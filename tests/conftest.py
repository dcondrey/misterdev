"""Shared test fixtures.

The suite is fully offline — no test makes a real LLM call. But constructing a
``Project`` (or a provider client) reads the provider API key from the
environment and raises if it is absent, so on a machine without a real key set
(CI, a fresh checkout) those constructions fail with a spurious
``API key ... not set`` error. Provide a dummy key for every test so client
construction succeeds; tests that exercise real behavior inject a fake client or
monkeypatch as needed. Never overrides a real key that is already present.
"""

import os

import pytest


@pytest.fixture(autouse=True)
def _dummy_provider_keys(monkeypatch):
    for var in ("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY"):
        if not os.environ.get(var):
            monkeypatch.setenv(var, "test-key")
    yield
