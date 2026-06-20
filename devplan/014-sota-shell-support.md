---
category: fix
complexity: small
depends_on: []
files_to_modify:
- my_project_orchestrator/core/sota_validator.py
status: completed
test_command: uv run pytest tests/ -x -q
title: Add shell script support to SOTA validator
---

The SOTA validator rejected a valid shell script (`scripts/gen_inventory.sh`) with "Unclosed delimiter '(' at index 3839". This happened because the validator's bracket-matching logic doesn't account for shell syntax where `$(...)` command substitution, `((...))` arithmetic, and here-documents use parentheses differently than programming languages.

Read `sota_validator.py` and fix:

1. Add file extension detection: if the file ends in `.sh`, `.bash`, `.zsh`, or has no extension, use shell-specific validation rules.

2. For shell scripts, skip the bracket-matching validation entirely (shell syntax is too complex for simple delimiter matching). Instead, only validate:
   - File is valid UTF-8
   - No obviously broken syntax (unclosed single/double quotes across lines, but allow here-docs)
   - No banned markers (TODO, FIXME, etc.) if that gate is active

3. Alternatively, if `shellcheck` is available on the system, use it for shell validation:
   ```python
   def _validate_shell(self, content: str, file_path: str) -> ValidationResult:
       # Try shellcheck first
       result = subprocess.run(
           ["shellcheck", "-s", "bash", file_path],
           capture_output=True, text=True, timeout=10
       )
       if result.returncode == 0:
           return ValidationResult(valid=True)
       # Fall back to basic checks
   ```

4. Add `.sh` and `.bash` to the language detection map if not present.