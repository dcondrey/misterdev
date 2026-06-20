---
category: feat
complexity: medium
depends_on: []
files_to_create:
- tests/test_topography_ts_go.py
files_to_modify:
- my_project_orchestrator/core/topography.py
status: completed
test_command: uv run pytest tests/test_topography_ts_go.py -v
title: Add TypeScript and Go support to topography engine
---

The topography engine currently supports Python and Rust via tree-sitter. Extend it to support TypeScript and Go.

1. Add dependencies: `uv add tree-sitter-typescript tree-sitter-go`

2. In `_get_ts_parsers()`, load the new grammars:
   ```python
   import tree_sitter_typescript
   import tree_sitter_go
   parsers["typescript"] = ts.Parser(ts.Language(tree_sitter_typescript.language_typescript()))
   parsers["go"] = ts.Parser(ts.Language(tree_sitter_go.language()))
   ```

3. Extend `_EXT_TO_LANG`:
   ```python
   ".ts": "typescript", ".tsx": "typescript",
   ".go": "go",
   ```

4. Add `_traverse_typescript(tree, source, file_path)`:
   - `function_declaration` → function
   - `class_declaration` → class
   - `method_definition` → method (nested under class)
   - `interface_declaration` → class (TypeScript interfaces)
   - `type_alias_declaration` → class
   - `export_statement` wrapping any of the above

5. Add `_traverse_go(tree, source, file_path)`:
   - `function_declaration` → function
   - `method_declaration` → method (receiver type as parent)
   - `type_declaration` → `type_spec` children:
     - `struct_type` → struct
     - `interface_type` → class (Go interfaces)

6. Route new languages in `_parse_file()`.

7. Create `tests/test_topography_ts_go.py` with inline code snippets:
   - TypeScript: class with methods, standalone function, interface, type alias
   - Go: function, method with receiver, struct, interface
   - Verify correct symbol types and counts