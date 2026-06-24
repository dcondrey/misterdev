"""Spec-as-tests helper (opt-in, wired per-task in the executor).

Idea: before a task is implemented, turn its acceptance criteria into a failing
test (an executable spec), so "done" means "this test now passes" rather than a
model's self-report. This is the strongest possible oracle for a single task.

This module is the generation primitive (generate + extract + path). The wiring
lives in :meth:`MarkdownPlanExecutor._maybe_generate_spec_test` /
``_run_spec_test``: when ``orchestrator.spec_as_tests`` is on, the generated test
is written under ``.orchestrator/spec_tests/`` — deliberately OUTSIDE the
project's own test directory, so it is never collected by the project suite and
therefore can never flip the integration-gate baseline red. After the task's own
gates pass, it is run scoped to that one file (pytest/jest-style); a red result
is advisory by default and blocking under ``orchestrator.spec_as_tests_block``.

``write_spec_test``/``spec_test_path`` remain for callers that want the test in
the conventional location; the executor does NOT use them (it writes to the
baseline-safe ``.orchestrator/`` lane instead).

Everything here is best-effort and timeout-bounded: a missing client, an empty
criterion, or a model error yields ``None`` (no test generated), never a raise.
"""

import re
from pathlib import Path
from typing import Callable, Optional

from my_project_orchestrator.core.bounded import run_bounded
from my_project_orchestrator.logging_setup import setup_logger

logger = setup_logger(__name__)

# Generator call: takes the assembled prompt and returns the model's text.
# Injected in tests; defaulted to the project client's generate_code path.
GeneratorCall = Callable[[str], str]

_PROMPT = (
    "You are writing a SINGLE failing test that encodes an acceptance criterion "
    "for a feature that is NOT yet implemented. The test MUST fail today (because "
    "the feature is absent) and pass once the feature is built correctly.\n\n"
    "## Task\n{description}\n\n"
    "## Acceptance Criteria\n{criteria}\n\n"
    "## Target language\n{language}\n\n"
    "Output ONLY the test file's source code inside a single fenced code block. "
    "No prose, no explanation, no setup instructions."
)

# Default per-language test-directory + filename conventions. Used only to place
# the generated file; callers may override the directory.
_LANG_TEST_DIR = {
    "python": "tests",
    "javascript": "tests",
    "typescript": "tests",
    "rust": "tests",
    "go": ".",
    "java": "src/test/java",
}
_LANG_EXT = {
    "python": ".py",
    "javascript": ".test.js",
    "typescript": ".test.ts",
    "rust": ".rs",
    "go": "_test.go",
    "java": ".java",
}

_FENCE = re.compile(r"```[a-zA-Z0-9_+-]*\n(.*?)```", re.DOTALL)


def generate_spec_test(
    task,
    llm_client=None,
    language: str = "python",
    generator: Optional[GeneratorCall] = None,
    timeout: float = 60,
) -> Optional[str]:
    """Generate a failing test source from a task's acceptance criteria.

    Returns the extracted test source string, or ``None`` when there is no
    acceptance criterion, no usable generator/client, the model returns nothing
    parseable, or the call errors/times out. Timeout-bounded via a daemon worker
    so a slow model can never hang a caller. Best-effort: never raises.
    """
    criteria = (getattr(task, "acceptance_criteria", "") or "").strip()
    if not criteria:
        return None

    call = generator or _default_generator(llm_client)
    if call is None:
        return None

    prompt = _PROMPT.format(
        description=(getattr(task, "description", "") or "").strip(),
        criteria=criteria,
        language=language,
    )

    def _work() -> Optional[str]:
        try:
            return extract_code(call(prompt) or "")
        except Exception as e:  # any model/IO failure is non-fatal -> no test
            logger.debug(f"Spec-test generation unavailable: {e}")
            return None

    return run_bounded(_work, timeout, None, "Spec-test generation")


def extract_code(text: str) -> Optional[str]:
    """Extract the source inside the first fenced code block, or None.

    Falls back to the whole trimmed text when the model returned bare code with
    no fence. Returns None for empty input.
    """
    if not text or not text.strip():
        return None
    match = _FENCE.search(text)
    if match:
        body = match.group(1).strip()
        return body or None
    return text.strip() or None


def spec_test_path(project_root: Path, task, language: str = "python") -> Path:
    """Compute the path the generated spec test would be written to.

    Deterministic from the task id and language; lives under the language's
    conventional test directory. Pure (no I/O), so callers can reason about the
    location before deciding to write.
    """
    lang = (language or "python").lower()
    test_dir = _LANG_TEST_DIR.get(lang, "tests")
    ext = _LANG_EXT.get(lang, ".txt")
    safe_id = re.sub(r"[^A-Za-z0-9_]", "_", str(getattr(task, "id", "task")))
    name = f"spec_{safe_id}{ext}"
    return project_root / test_dir / name


def write_spec_test(
    project_root: Path,
    task,
    source: str,
    language: str = "python",
) -> Path:
    """Write ``source`` to the task's spec-test path, creating the dir.

    Returns the written path. Raises OSError on a genuine write failure (the
    caller decides whether that is fatal); generation itself is already
    error-swallowing, so this stays strict so a wiring caller is never told a
    file was written when it was not.
    """
    path = spec_test_path(project_root, task, language)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    logger.info(f"Wrote spec-as-test (expected to fail pre-implementation): {path}")
    return path


def _default_generator(llm_client) -> Optional[GeneratorCall]:
    """Build a generator call from the project's LLM client, or None if unusable.

    Tolerant of client shape so an absent/limited client yields no generator
    (the caller gets None) rather than raising. No network until invoked.
    """
    if llm_client is None or not hasattr(llm_client, "generate_code"):
        return None

    system = (
        "You write a single failing test that encodes one acceptance criterion. "
        "Output only the test source inside one fenced code block."
    )

    def _call(prompt: str) -> str:
        return llm_client.generate_code(prompt, system) or ""

    return _call
