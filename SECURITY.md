# Security Policy

## Supported versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

misterdev is pre-1.0. Security fixes land on the latest 0.1.x release.

## Reporting a vulnerability

Please report vulnerabilities privately. Do **not** open a public issue for a
security problem.

Two private channels, either is fine:

- **GitHub Security Advisories** — open a draft advisory at
  https://github.com/dcondrey/misterdev/security/advisories/new
- **Email** — david@writerslogic.com

We aim to **acknowledge your report within 48 hours** and will keep you updated
as we investigate and prepare a fix. Coordinated disclosure is appreciated:
please give us a reasonable window to ship a fix before any public disclosure.

**Do not include real secrets** (API keys, tokens, private keys, customer data)
in your report. Redact them, or share a minimal synthetic reproduction instead.

## Scope

misterdev drives an LLM to edit and verify code, calls external tools, and loads
third-party extensions. Reports touching the following are in scope and
especially valued:

- **Model-generated code execution / sandbox escape** — a build, test, or gate
  command, or model-produced code, escaping its intended execution boundary or
  running commands it should not.
- **Path traversal in the file tools** — reads or writes escaping the project
  root (e.g. `../`, absolute paths, symlink tricks) through the editing or
  file-access tools.
- **Secret exposure in logs or gates** — API keys, tokens, or environment
  secrets leaking into logs, gate output, reports, error messages, or LLM
  prompts.
- **MCP server / gateway trust** — the agentic MCP tool-use path (including
  remote gateways) invoking arbitrary or unallowlisted remote tools, or trusting
  malicious tool responses in a way that compromises the host.
- **Plugin / entry-point trust** — capabilities discovered via the
  `misterdev.tools`, `misterdev.gates`, and `misterdev.targets` entry points
  gaining unexpected privileges or bypassing safety controls.

Reports outside this list are still welcome; these are simply the areas where a
finding has the highest impact.
