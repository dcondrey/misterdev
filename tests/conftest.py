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

from misterdev.llm.client.rate_coordinator import clear as _clear_rate_coordinator


@pytest.fixture(autouse=True)
def _reset_rate_coordinator():
    """Clear global rate-coordinator state between tests to prevent cooldown bleed."""
    _clear_rate_coordinator()
    yield
    _clear_rate_coordinator()


@pytest.fixture(autouse=True)
def _dummy_provider_keys(monkeypatch):
    for var in ("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY"):
        if not os.environ.get(var):
            monkeypatch.setenv(var, "test-key")
    yield


@pytest.fixture(autouse=True)
def _git_identity(monkeypatch):
    """Give git a commit identity for tests that init a repo and commit.

    Those tests never set ``user.name``/``user.email``, so ``git commit`` fails
    ("Please tell me who you are") on a machine with no global git identity — CI,
    or a fresh dev box — while a locally-configured identity masks it. The
    ``GIT_*`` env vars supply an identity without mutating any global config, so
    the fix is portable and side-effect-free. Never overrides a value already set.
    """
    identity = {
        "GIT_AUTHOR_NAME": "misterdev-tests",
        "GIT_AUTHOR_EMAIL": "tests@misterdev.local",
        "GIT_COMMITTER_NAME": "misterdev-tests",
        "GIT_COMMITTER_EMAIL": "tests@misterdev.local",
    }
    for var, value in identity.items():
        if not os.environ.get(var):
            monkeypatch.setenv(var, value)
    yield
