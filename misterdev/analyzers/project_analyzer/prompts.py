"""Prompt templates for the Phase 1 project analyzers."""

STRUCTURE_PROMPT = """Analyze the project at the path below. Return a JSON object with these exact fields:
  project_type (string: web-api, web-app, cli, library, worker, static-site, monorepo, or unknown),
  languages (array of strings),
  frameworks (array of strings),
  build_command (string or null),
  test_command (string or null),
  lint_command (string or null),
  package_manager (string or null),
  entry_points (array of file path strings),
  directory_structure (string, tree of src dirs, max 30 lines)

Project files:
{file_listing}

Key config files:
{config_contents}

Return ONLY valid JSON, no markdown fences."""

COMPLETENESS_PROMPT = """Analyze this project for completeness. Return a JSON object with:
  existing (array of objects with "name" and "description"),
  incomplete (array of objects with "name", "description", and "complexity"),
  missing (array of objects with "name", "description", and "complexity"),
  dead_code (array of file path strings),
  stubs (array of file path strings),
  broken (array of file path strings),
  todos (array of objects with "file", "line", "text")

{health_ground}
Treat code that builds and is covered by passing tests as IMPLEMENTED. Docs may
describe a from-scratch plan; do NOT report already-built, tested capabilities
as missing or incomplete. Base "missing"/"broken" on the source, not the docs.

Code documented as intentional is COMPLETE, not incomplete: graceful-degradation
and fallback paths, platform-gated no-ops (e.g. a wasm/no-filesystem backend that
returns empty by design), and shims "retained for parity". A function returning an
empty or default value is NOT a stub when a comment or the design states that empty
result is the contract — check the "File intents" section before flagging. For each
item you place in "incomplete"/"stubs"/"missing", name the specific file and the
concrete unmet behavior; if you cannot, omit it. Never infer incompleteness from a
symbol name or a default return alone.

Project docs:
{docs}

Project source overview:
{source_overview}

Return ONLY valid JSON, no markdown fences."""

CONTEXT_PROMPT = """Gather context for this project. Return a JSON object with:
  purpose (string, 1-2 sentences),
  goals (string),
  conventions (string, coding conventions found),
  constraints (string, requirements or compatibility needs),
  recent_activity (string, summary of recent work),
  stated_requirements (string, from spec/design docs)

README:
{readme}

CLAUDE.md / config:
{config}

Recent git log:
{git_log}

Return ONLY valid JSON, no markdown fences."""


DEBT_RISK_PROMPT = """Analyze this project for technical debt and implementation risk. Return a JSON object with:
  tech_debt (object with "score" [0-100], "description", "critical_issues"),
  risk (object with "level" [low, medium, high, critical], "factors", "mitigations")

Project assessment so far:
{assessment_summary}

Source code overview:
{source_overview}

Return ONLY valid JSON, no markdown fences."""
