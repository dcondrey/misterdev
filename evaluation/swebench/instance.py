"""The SWE-bench task record."""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List


def _as_list(value: Any) -> List[str]:
    """Coerce a SWE-bench test list to List[str].

    The dataset stores FAIL_TO_PASS / PASS_TO_PASS either as a real list or as a
    JSON-encoded string; accept both (and a bare string) so a record loads
    whichever export it came from.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            try:
                return [str(v) for v in json.loads(text)]
            except (ValueError, TypeError):
                return []
        return [text] if text else []
    return []


@dataclass
class SWEBenchInstance:
    """One SWE-bench task.

    ``fail_to_pass`` are the tests the fix must turn green; ``pass_to_pass`` are
    tests that must stay green (no regression). ``test_patch`` adds/updates the
    tests that judge the fix and is applied by the grader AFTER the model's patch
    — the model never sees it. ``test_command`` is the base runner the specific
    test node ids are appended to.
    """

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    test_patch: str = ""
    fail_to_pass: List[str] = field(default_factory=list)
    pass_to_pass: List[str] = field(default_factory=list)
    test_command: str = "python -m pytest -rA -p no:cacheprovider"
    language: str = "python"
    setup_commands: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SWEBenchInstance":
        """Build from a raw SWE-bench record (tolerant of field-name casing)."""
        return cls(
            instance_id=str(d.get("instance_id") or d.get("id") or ""),
            repo=str(d.get("repo") or ""),
            base_commit=str(d.get("base_commit") or ""),
            problem_statement=str(d.get("problem_statement") or ""),
            test_patch=str(d.get("test_patch") or ""),
            fail_to_pass=_as_list(d.get("FAIL_TO_PASS", d.get("fail_to_pass"))),
            pass_to_pass=_as_list(d.get("PASS_TO_PASS", d.get("pass_to_pass"))),
            test_command=str(
                d.get("test_command") or "python -m pytest -rA -p no:cacheprovider"
            ),
            language=str(d.get("language") or "python"),
            setup_commands=_as_list(d.get("setup_commands")),
        )

    @classmethod
    def load_jsonl(cls, path: str) -> List["SWEBenchInstance"]:
        """Load instances from a JSONL export (one record per line)."""
        out: List[SWEBenchInstance] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(cls.from_dict(json.loads(line)))
        return out
