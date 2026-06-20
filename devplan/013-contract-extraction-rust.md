---
category: feat
complexity: medium
context_files:
- my_project_orchestrator/core/topography.py
depends_on:
- '010'
files_to_modify:
- my_project_orchestrator/core/contracts.py
status: completed
test_command: uv run pytest tests/ -x -q
title: Improve Rust contract extraction with tree-sitter instead of line parsing
---

`ContractRegistry._extract_rust_symbols()` parses Rust code line-by-line with string matching (`line.strip().startswith("pub fn")`, etc.). This is fragile: it misses multi-line signatures, attributes, generic bounds, and where clauses. The crosstalk run showed "Extracted 0 contracts" for several tasks that modified Rust files.

Replace the line-by-line parser with tree-sitter, reusing the parsers from the topography engine:

1. Import `_get_ts_parsers` from `topography.py`.

2. Replace `_extract_rust_symbols()` with a tree-sitter-based version:
   ```python
   def _extract_rust_symbols(self, content: str) -> List[Dict]:
       parsers = _get_ts_parsers()
       if "rust" not in parsers:
           return self._extract_rust_symbols_fallback(content)
       parser = parsers["rust"]
       tree = parser.parse(content.encode())
       # Walk tree for pub items...
   ```

3. Extract these symbol types from the tree:
   - `function_item` with `pub` visibility → full signature including generics and return type
   - `struct_item` with `pub` → struct name + pub fields
   - `enum_item` with `pub` → enum name + variants
   - `trait_item` with `pub` → trait name + method signatures
   - `impl_item` → type name + pub method signatures
   - `type_item` with `pub` → type alias

4. Keep the old line-based parser as `_extract_rust_symbols_fallback()` for when tree-sitter is unavailable.

5. Also improve Python extraction to use tree-sitter when available, falling back to the existing AST-based parser.