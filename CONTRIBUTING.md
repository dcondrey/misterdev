# Contributing to misterdev

Thanks for your interest in improving misterdev. This guide covers setup, the
quality gate every change must pass, and how to add an extension without
touching the core.

## Development setup

misterdev uses [`uv`](https://docs.astral.sh/uv/) for environment and
dependency management.

```bash
git clone https://github.com/dcondrey/misterdev
cd misterdev
uv sync
```

`uv sync` installs the project plus the `dev` dependency group (pytest, the MCP
SDK the suite exercises, etc.). Python 3.10–3.13 are supported.

## The quality gate

Every change must keep the suite green and the linter clean:

```bash
uv run ruff check .
uv run pytest -q
```

The test suite is ~1300 tests and runs fully offline. A PR that leaves either
command failing will not be merged. Run both locally before you push.

If you fix a bug, add a regression test that fails before your change and passes
after it.

## Commit convention

Commits and PR titles follow [Conventional Commits](https://www.conventionalcommits.org/),
one line, imperative mood:

```
<type>: <description>
```

Type is one of: `fix`, `feat`, `refactor`, `chore`, `docs`, `test`, `perf`,
`security`. Keep the subject terse and specific.

## Pull requests

- Keep the change set minimal and focused on one thing.
- The full quality gate above must pass.
- Explain the *why* in the description, not just the *what*.
- Note any user-visible behavior change in `CHANGELOG.md` under `[Unreleased]`.

## Adding an extension

misterdev is extended through Python entry points — no edits to misterdev
itself. Three groups are discovered at runtime:

| Group                | Adds a…                                                        |
| -------------------- | -------------------------------------------------------------- |
| `misterdev.tools`    | tool the agentic loop can call (`execute(self, project, **kwargs) -> (ok, output)`; set `gather_safe = True` for read-only tools) |
| `misterdev.gates`    | gate (`callable(GateContext) -> GateOutcome`) run by `GateKeeper` |
| `misterdev.targets`  | build target (an object with `markers` and `commands(dir)`)    |

Register your capability under the matching group in your package's
`pyproject.toml`, then `pip install` it alongside misterdev. See the worked
example in [`examples/misterdev-plugin-hello`](examples/misterdev-plugin-hello),
which ships a tool and a gate in a single module.

## Code of Conduct

Participation is governed by our [Code of Conduct](CODE_OF_CONDUCT.md).

## Licensing of contributions (CLA)

misterdev is licensed under **AGPL-3.0-or-later**. By submitting a contribution
you agree that it is licensed under the project's AGPL-3.0 license. You also
grant the author, David Condrey, permission to offer your contribution under the
project's separate commercial license. You retain copyright to your work.
