---
category: fix
complexity: small
depends_on: []
files_to_modify:
- my_project_orchestrator/task_executors/markdown_plan_executor.py
- my_project_orchestrator/tools/formatter.py
status: completed
test_command: uv run pytest tests/ -x -q
title: Fix formatter to run once per task instead of once per file
---

`_run_formatters()` at line 364 iterates over every modified file and calls the formatter command for each one. For project-wide formatters like `cargo fmt`, `black .`, or `ruff format .`, this is wasteful (T-003 in crosstalk triggered 7 separate `cargo fmt` invocations). Some formatters like `rustfmt` DO take individual files, but `cargo fmt` does not.

Fix by changing the formatter execution model:

1. In `_run_formatters()`, remove the inner `for file_path in files` loop. Instead, call the formatter once with no file path (project-wide):
   ```python
   def _run_formatters(self, project: Project, files: List[str]):
       for tool_name, tool in project.tool_manager.tools.items():
           if getattr(tool, 'type', None) == 'formatter':
               tool.execute(project)
   ```

2. In `FormatterTool.execute()`, make `file_path` default to `"."` and only substitute `{path}` if the template contains it. If the template has no `{path}` placeholder (like `cargo fmt`), just run the command as-is:
   ```python
   def execute(self, project, file_path="."):
       command_template = self.config.get("command")
       if "{path}" in command_template:
           command = command_template.format(path=file_path)
       else:
           command = command_template
       ...
   ```

This preserves per-file formatters (if someone configures `rustfmt {path}`) while fixing project-wide ones.