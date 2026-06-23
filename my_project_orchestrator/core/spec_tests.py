"""Spec-as-tests helper (CONSERVATIVE, opt-in, DEFERRED from the build loop).

Idea: before a task is implemented, turn its acceptance criteria into a failing
test (an executable spec), so "done" means "this test now passes" rather than a
model's self-report. This is the strongest possible oracle for a single task.

Status — DEFERRED, NOT wired into the build loop. This module is a standalone,
tested helper plus a documented seam. The generation + write + fail-expectation
path is implemented and tested here, but :func:`generate_spec_test` is NOT called
from ``ProjectOrchestrator._execute_tasks``.

Why deferred (per Phase-4 scope discipline — additive or not at all):
  The natural injection point is the per-task "Prepare tasks with context" block
  inside the wave loop of ``_execute_tasks``. Writing a *failing* test there is
  not control-flow-neutral: the integration gate first runs the suite to
  establish a green baseline (``baseline_ok``) and DISABLES itself if the suite
  is red. A freshly written failing spec test would flip that baseline red and
  silently turn off the per-wave regression gate for the rest of the build —
  changing default behavior of an unrelated gate. Cleanly wiring it therefore
  requires teaching the integration-gate baseline (and the end-of-build gate) to
  exclude or expect these pending spec tests, which is a change to the existing
  loop's control flow, not a pure pre-step. Rather than risk that regression, the
  generation primitive ships tested and the wiring is left as the seam below.

Seam to wire later (additively, once the baseline-exclusion is handled):
  In ``_execute_tasks``, inside the per-task ``for task in ready:`` preparation
  block, when ``get_setting(config, "orchestrator", "spec_as_tests")`` is true
  and the task has acceptance_criteria, call ::

      from my_project_orchestrator.core.spec_tests import (
          generate_spec_test, write_spec_test,
      )
      gen = generate_spec_test(task, project.llm_client, language=lang)
      if gen is not None:
          path = write_spec_test(project.path, task, gen, language=lang)
          task.processor_data["spec_test_path"] = str(path)
          task.processor_data["spec_test_expected_fail"] = True

  AND extend the integration-gate baseline so a known-pending spec test does not
  count against ``baseline_ok`` (e.g. run the spec test separately, or mark it
  skipped until its task completes). Only then is the addition control-flow-safe.

Everything here is best-effort and timeout-bounded: a missing client, an empty
criterion, or a model error yields ``None`` (no test generated), never a raise.
"""

import re
import threading
from pathlib import Path
from typing import Callable, Optional

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

    box: dict = {"source": None}

    def _run() -> None:
        try:
            raw = call(prompt) or ""
            box["source"] = extract_code(raw)
        except Exception as e:  # any model/IO failure is non-fatal -> no test
            logger.debug(f"Spec-test generation unavailable: {e}")
            box["source"] = None

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        logger.warning(f"Spec-test generation exceeded {timeout}s; skipping (no test).")
        return None
    return box["source"]


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
