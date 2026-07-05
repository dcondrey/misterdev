# Plugins

A plugin is an ordinary Python package that declares entry points in the `misterdev.*` groups. Install it alongside misterdev and it is discovered at runtime — no edits to misterdev's own code. `pip install misterdev-plugin-x` adds a capability; nothing else.

Discovery is lazy (on first lookup) and safe: a plugin that fails to import is logged and skipped, never fatal, and a plugin can never shadow a built-in of the same name. Registration is thread-safe, since capabilities are resolved from parallel executor workers.

There are three extensible kinds, one entry-point group each:

| Kind | Entry-point group | Contract |
| --- | --- | --- |
| Tool | `misterdev.tools` | a class with `execute(...)` |
| Gate | `misterdev.gates` | a callable `GateContext -> GateOutcome` |
| Target | `misterdev.targets` | an object with `markers` and `commands(dir)` |

## Tools

A tool is a **class**. Its instance exposes:

```python
def execute(self, project, **kwargs) -> tuple[bool, str]:
    # returns (ok, output)
    ...
```

`execute` returns a `(ok, output)` pair: a bool for success and a string result. The constructor receives the tool's config mapping from `project.yaml` (`__init__(self, config: dict)`), so a tool can be parameterized per project.

Two optional class attributes opt a read-only tool into the agentic gathering loop (see [mcp.md](mcp.md)):

- `gather_safe = True` — the model may call this tool mid-task to gather information before editing.
- `gather_description = "..."` — the one-line description shown to the model.

```python
class HelloTool:
    gather_safe = True
    gather_description = "Return a friendly greeting for a name."

    def __init__(self, config: dict):
        self.name = config.get("name", "hello")

    def execute(self, project, name: str = "world", **_ignored):
        return True, f"Hello, {name}!"
```

To let the model call it mid-task, enable the loop and configure the tool in `project.yaml`:

```yaml
orchestrator:
  mcp_tool_use: true
tools:
  - name: hello
    type: hello
```

## Gates

A gate is a **callable** taking a `GateContext` and returning a `GateOutcome`. It runs automatically inside the gate sequence. A `RED` outcome blocks the change; `GREEN` and `SKIP` do not.

`GateContext` fields:

| Field | Meaning |
| --- | --- |
| `project_path` | Repo root, or the target subdir root for a routed target. |
| `commands` | Resolved `build_command` / `test_command` / `lint_command` / typecheck. |
| `env_activate` | Optional host-venv activation prefix (may be `None`). |

`GateOutcome(status, reason="")` where `status` is one of the module constants `GREEN`, `RED`, `SKIP` (import them from `misterdev.core.execution.outcomes`). `GREEN` passed; `RED` failed and blocks the build; `SKIP` is "no opinion" (missing config/dependency, unparseable, or timeout) and never blocks. `outcome.passed` is `status == GREEN`; `outcome.skipped` is `status == SKIP`. `reason` is a short human-readable explanation.

```python
from misterdev.core.execution.outcomes import GateOutcome, GREEN, RED

def no_shouting_gate(ctx) -> GateOutcome:
    build = (ctx.commands or {}).get("build_command") or ""
    if build and build.isupper():
        return GateOutcome(RED, "build_command is ALL CAPS; please calm down")
    return GateOutcome(GREEN)
```

## Targets

A target teaches misterdev a language or build system its built-in markers don't cover, so polyglot gate routing can own the matching sub-project. A target plugin is any object exposing:

- `markers` — an iterable of filenames or globs (e.g. `("Gemfile", "*.gemspec")`). The first registered target whose markers match a directory owns it.
- `commands(dir) -> dict` — returns a mapping with `build_command` / `test_command` (and optionally lint/typecheck) keys for that directory.

```python
class RubyTarget:
    markers = ("Gemfile", "*.gemspec")

    def commands(self, d):
        return {
            "build_command": "bundle exec rake build",
            "test_command": "bundle exec rspec",
        }
```

An explicit `targets:` block in `project.yaml` always wins; target plugins fill in the gaps for auto-detection.

## Entry points

Register each capability under the matching group in the plugin's `pyproject.toml`. The entry points are the whole contract:

```toml
[project]
name = "misterdev-plugin-hello"
dependencies = ["misterdev"]

[project.entry-points."misterdev.tools"]
hello = "misterdev_plugin_hello:HelloTool"

[project.entry-points."misterdev.gates"]
no_shouting = "misterdev_plugin_hello:no_shouting_gate"

[project.entry-points."misterdev.targets"]
ruby = "misterdev_plugin_hello:RubyTarget"
```

Then `pip install` the package alongside misterdev. Verify discovery:

```python
from misterdev.plugins import TOOLS, GATES, TARGETS
assert TOOLS.get("hello") is not None
assert GATES.get("no_shouting") is not None
```

A full, runnable example — tool, gate, `pyproject.toml`, and notes — lives at [`examples/misterdev-plugin-hello`](../examples/misterdev-plugin-hello).

## Stability

The three entry-point group names (`misterdev.tools`, `misterdev.gates`, `misterdev.targets`) and the tool, gate, and target contracts described above are misterdev's **public plugin API**. They follow semantic versioning: breaking changes to these names or contracts happen only on a major version bump. Build against them with confidence.
