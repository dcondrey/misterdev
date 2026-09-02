<!-- repo-header:start -->
<img src="https://github.com/dcondrey.png?size=160" alt="misterdev-plugin-hello logo" width="120" align="left">

<h1>misterdev-plugin-hello</h1>

<p><strong>Documentation for misterdev-plugin-hello in Misterdev.</strong></p>

<br clear="left">

[![CI](https://img.shields.io/github/actions/workflow/status/dcondrey/misterdev/ci.yml?style=flat-square&labelColor=20232a&branch=main&label=CI)](https://github.com/dcondrey/misterdev/actions/workflows/ci.yml) [![OpenSSF Best Practices](https://www.bestpractices.dev/projects/14406/badge)](https://www.bestpractices.dev/projects/14406) [![License](https://img.shields.io/github/license/dcondrey/misterdev?style=flat-square&labelColor=20232a&color=007ec6&label=license)](https://github.com/dcondrey/misterdev/blob/main/LICENSE) [![Code of Conduct](https://img.shields.io/badge/code%20of%20conduct-Contributor%20Covenant%202.1-6a4c93?style=flat-square&labelColor=20232a)](https://github.com/dcondrey/misterdev/blob/main/CODE_OF_CONDUCT.md) [![GitHub Sponsors](https://img.shields.io/badge/GitHub%20Sponsors-Sponsor-EA4AAA?style=flat-square&labelColor=20232a)](https://github.com/sponsors/dcondrey)
<!-- repo-header:end -->

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
