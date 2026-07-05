# misterdev-plugin-hello

A minimal example that shows how a third party extends **misterdev** with **zero
edits to misterdev itself** — a tool and a gate, discovered through Python entry
points.

## What it adds

- **`hello` tool** — a read-only, gather-safe tool (`local.hello`) the model can
  call during the agentic gathering loop.
- **`no_shouting` gate** — a plugin gate that fails a build whose `build_command`
  is ALL CAPS.

## Install

```bash
pip install -e examples/misterdev-plugin-hello
```

That's it. The entry points in `pyproject.toml` register the capabilities:

```toml
[project.entry-points."misterdev.tools"]
hello = "misterdev_plugin_hello:HelloTool"

[project.entry-points."misterdev.gates"]
no_shouting = "misterdev_plugin_hello:no_shouting_gate"
```

## Verify

```python
from misterdev.plugins import TOOLS, GATES
assert TOOLS.get("hello") is not None      # tool discovered
assert GATES.get("no_shouting") is not None  # gate discovered
```

The gate runs automatically inside `GateKeeper.run_gates`. To let the model call
the `hello` tool mid-task, enable the agentic loop and configure the tool:

```yaml
# project.yaml
orchestrator:
  mcp_tool_use: true      # enables the bounded gathering loop
tools:
  - name: hello
    type: hello
```

## Write your own

1. Implement a **tool** (a class with `execute(self, project, **kwargs) -> (ok, output)`;
   add `gather_safe = True` for a read-only tool the agentic loop may call), a
   **gate** (`callable(GateContext) -> GateOutcome`), or a **target** (an object
   with `markers` and `commands(dir)`).
2. Register it under the matching entry-point group: `misterdev.tools`,
   `misterdev.gates`, or `misterdev.targets`.
3. `pip install` your package alongside misterdev. Done.
