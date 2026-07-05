---
category: feat
complexity: small
depends_on: []
files_to_modify:
- misterdev/core/error_classifier.py
status: completed
test_command: uv run pytest tests/ -x -q
title: Improve error classifier to handle manifest and config errors
---

The error classifier returned "unknown" for a broken `Cargo.toml` (missing `[package]` section) during the crosstalk run. When the classifier returns "unknown", the retry prompt to the LLM lacks targeted guidance, wasting attempts.

Read `error_classifier.py` and add detection for these common error patterns:

1. **Manifest/config errors** (new category: `manifest_error`):
   - `"failed to parse manifest"` (Cargo.toml)
   - `"missing either a `[package]` or a `[workspace]`"` (Cargo.toml)
   - `"invalid toml"` / `"expected value"` (TOML parse errors)
   - `"could not find `Cargo.toml`"` (missing manifest)
   - `"SyntaxError"` in `package.json`, `pyproject.toml`

2. **Dependency/import errors** (if not already handled):
   - `"unresolved import"` / `"cannot find"` (Rust)
   - `"ModuleNotFoundError"` / `"ImportError"` (Python)
   - `"Cannot find module"` (Node)

3. **File not found errors**:
   - `"No such file or directory"`
   - `"file not found"`

For each new category, include a `suggestion` field that gets injected into the retry prompt, e.g.:
- `manifest_error` → "The Cargo.toml or project manifest file is malformed. Check that all required sections ([package], name, version) are present and valid."

Follow the existing classifier pattern for adding new error types.