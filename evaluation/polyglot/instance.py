"""The Aider polyglot exercise record."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

# Per-language test command, run from the exercise directory. Mirrors how the
# Aider polyglot benchmark grades each language.
TEST_COMMANDS: Dict[str, str] = {
    "python": "python -m pytest -q",
    "rust": "cargo test",
    "go": "go test ./...",
    "javascript": "npm test",
    "java": "./gradlew test",
    "cpp": "cmake -B build && cmake --build build && ctest --test-dir build",
}


@dataclass
class PolyglotInstance:
    """One polyglot exercise.

    ``solution_files`` are the stubs misterdev edits; ``test_files`` are the
    graded tests it must make pass (and must NOT weaken — misterdev's test-tamper
    gate protects them). ``instructions`` is the exercise prompt shown as the
    goal; the test files are never described to the model.
    """

    name: str
    language: str
    instructions: str
    solution_files: List[str]
    test_files: List[str]
    test_command: str = ""

    def __post_init__(self):
        if not self.test_command:
            self.test_command = TEST_COMMANDS.get(self.language, "python -m pytest -q")


def load_local_exercise(exercise_dir: str, language: str) -> PolyglotInstance:
    """Build an instance from a checked-out polyglot exercise directory.

    Reads ``.meta/config.json`` for the solution/test file lists and
    ``.docs/instructions.md`` (plus the optional append) for the prompt. Raises
    if the required layout is missing, so a malformed exercise fails loudly at
    load rather than scoring as an empty task.
    """
    import json

    root = Path(exercise_dir)
    config = json.loads((root / ".meta" / "config.json").read_text(encoding="utf-8"))
    files = config.get("files", {})
    docs = root / ".docs"
    instructions = (docs / "instructions.md").read_text(encoding="utf-8")
    append = docs / "instructions.append.md"
    if append.exists():
        instructions += "\n\n" + append.read_text(encoding="utf-8")
    return PolyglotInstance(
        name=root.name,
        language=language,
        instructions=instructions,
        solution_files=list(files.get("solution", [])),
        test_files=list(files.get("test", [])),
    )
