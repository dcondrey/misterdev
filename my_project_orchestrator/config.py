import yaml
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

from my_project_orchestrator.logging_setup import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Typed config schema (single source of truth).
#
# Each knob's name, type, and default live in exactly one place: a dataclass
# field. DEFAULT_CONFIG is GENERATED from these below, so the dict that the rest
# of the codebase consumes can never drift from the schema, and a default can no
# longer be copy-pasted (and diverge) across call sites. Stdlib dataclasses, no
# pydantic: config is loaded from trusted local YAML, so attribute typing +
# unknown-key validation is enough; we don't need runtime coercion.
# ---------------------------------------------------------------------------


@dataclass
class LLMSettings:
    provider: str = "openrouter"
    model: str = "anthropic/claude-sonnet-4.6"
    temperature: float = 0.1
    api_key_env_var: str = "OPENROUTER_API_KEY"
    streaming: bool = False
    failover: List[Dict[str, Any]] = field(default_factory=list)
    # Per-task model routing: routing maps complexity/strategy -> tier name,
    # models maps tier name -> model id. Empty = use the default model.
    routing: Dict[str, Any] = field(default_factory=dict)
    models: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BuildSettings:
    max_tasks: int = 30
    max_consecutive_failures: int = 3
    build_timeout: int = 120
    test_timeout: int = 180
    lint_timeout: int = 120
    parallel_analysis: bool = True
    # Global spend ceiling for a run; the master budget constraint.
    budget: float = 100.0


@dataclass
class OrchestratorSettings:
    max_consecutive_failures: int = 3
    max_workers: int = 4
    context_budget_tokens: int = 100000
    max_task_attempts: int = 3
    integration_gate: bool = True
    # "auto" keys are budget-driven (BaseLLMClient / convergence loop) so the
    # global build.budget is the single master constraint. Typed Any because
    # they accept "auto", a number, or None.
    max_build_iterations: Any = "auto"
    certainty_threshold: float = 0.5
    max_cost_per_task: Any = "auto"
    allow_test_edits: bool = False
    verify_acceptance: bool = True
    llm_acceptance_judge: bool = True
    # Golden suite: files the model never sees and may never edit, plus a
    # blocking-gate command. Empty/None = feature off.
    golden_paths: List[str] = field(default_factory=list)
    golden_command: Optional[str] = None
    # AB-MCTS spec refinement fires several serial LLM calls before any work;
    # off by default (marginal value, large latency/cost).
    enable_ab_mcts: bool = False
    # "auto" (worktree isolation on a git repo, else shared), "shared" (one
    # working tree), or "worktree" (always isolate each parallel task).
    parallel_mode: str = "auto"
    auto_detect_dependencies: bool = False


PROMPT_TEMPLATES = {
    "system": (
        "You are an expert developer. Follow the strategy, honor interface contracts exactly, "
        "and ensure your output is syntactically valid.\n"
        "{invariants}\n{consensus_context}"
    ),
    "task_completion_instruction": (
        "## Task\n{task.description}\n\n"
        "## Acceptance Criteria\n{acceptance_criteria}\n\n"
        "## Files to Edit\n{task.target_files}\n\n"
        "## Interface Contracts (MUST honor exact signatures)\n{interface_contracts}\n\n"
        "## Recent Changes to Related Files\n{recent_changes}\n\n"
        "## Scratchpad (learnings from previous tasks)\n{scratchpad}\n\n"
        "## Code Context\n{code_context}\n\n"
        "Output your changes as markdown code blocks with file paths."
    ),
    "error_correction_instruction": (
        "## Previous Attempt Failed\n{error_logs}\n\n"
        "## Task\n{task.description}\n\n"
        "## Interface Contracts\n{interface_contracts}\n\n"
        "## Code Context\n{code_context}\n\n"
        "Fix the error. Output corrected code as markdown code blocks with file paths."
    ),
}

# Sections whose keys are schema-validated (used for both DEFAULT_CONFIG
# generation and project.yaml typo detection).
_SECTION_SCHEMAS = {
    "llm": LLMSettings,
    "build": BuildSettings,
    "orchestrator": OrchestratorSettings,
}

# Generated from the schemas above — do not hand-edit section contents; change
# the dataclass fields instead.
DEFAULT_CONFIG = {
    "orchestrator_version": "1.0",
    "llm": asdict(LLMSettings()),
    "tools": [],
    "environment": {"type": "none"},
    "prompt_templates": PROMPT_TEMPLATES,
    "build": asdict(BuildSettings()),
    "orchestrator": asdict(OrchestratorSettings()),
    "build_command": None,
    "test_command": None,
    "lint_command": None,
}


_SCHEMA_FIELDS = {
    section: frozenset(f.name for f in fields(schema))
    for section, schema in _SECTION_SCHEMAS.items()
}


def get_section_setting(section: str, section_cfg: Dict[str, Any], key: str) -> Any:
    """Like :func:`get_setting`, but for a caller that already holds the section
    sub-dict (e.g. the LLM client receives only ``config["llm"]``).

    Same two guarantees: single-source default (from DEFAULT_CONFIG) and a typo
    in ``key`` raises instead of silently returning a default.
    """
    known = _SCHEMA_FIELDS.get(section)
    if known is not None and key not in known:
        raise KeyError(
            f"Unknown config key '{section}.{key}' is not a field of "
            f"{_SECTION_SCHEMAS[section].__name__}. Known: {sorted(known)}"
        )
    if key in (section_cfg or {}):
        return section_cfg[key]
    return DEFAULT_CONFIG.get(section, {}).get(key)


def get_setting(config: Dict[str, Any], section: str, key: str) -> Any:
    """Read config[section][key], falling back to the canonical DEFAULT_CONFIG.

    Two guarantees:
    - Single source of truth for defaults: call sites pass no literal default,
      so a key's default lives in exactly one place (the dataclass schema).
    - Typo-safe: for a schema-validated section, an unknown ``key`` raises
      immediately (caught in tests/dev) instead of silently returning a default.
      This is the full-sweep typo protection without replacing the config dict.
    """
    return get_section_setting(section, config.get(section) or {}, key)


def warn_unknown_keys(project_config: Dict[str, Any]) -> List[str]:
    """Warn about keys in a schema-validated section that the schema doesn't know.

    Catches a user's typo in project.yaml (e.g. ``buildtimeout: 300``) which the
    deep-merge would otherwise silently ignore. Returns the unknown keys found.
    """
    unknown: List[str] = []
    for section, schema in _SECTION_SCHEMAS.items():
        sub = project_config.get(section)
        if not isinstance(sub, dict):
            continue
        known = {f.name for f in fields(schema)}
        for key in sub:
            if key not in known:
                unknown.append(f"{section}.{key}")
                logger.warning(
                    f"Unknown config key '{section}.{key}' in project.yaml is "
                    f"ignored (typo? known {section} keys: {sorted(known)})"
                )
    return unknown


class ConfigManager:
    def __init__(self):
        import copy

        self.global_config = copy.deepcopy(DEFAULT_CONFIG)

    def _deep_update(self, d: Dict[Any, Any], u: Dict[Any, Any]) -> Dict[Any, Any]:
        """Deep merge two dictionaries."""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k] = self._deep_update(d[k], v)
            else:
                d[k] = v
        return d

    def load_project_config(self, project_path: str | Path) -> Dict[str, Any]:
        """Loads and merges project.yaml with global defaults."""
        project_dir = Path(project_path)
        yaml_path = project_dir / "project.yaml"

        project_config = {}
        if yaml_path.exists():
            try:
                with open(yaml_path, "r", encoding="utf-8") as f:
                    loaded_config = yaml.safe_load(f)
                    if isinstance(loaded_config, dict):
                        project_config = loaded_config
                logger.info(f"Loaded project configuration from {yaml_path}")
            except Exception as e:
                logger.error(f"Failed to load project config at {yaml_path}: {e}")
        else:
            logger.warning(f"No project.yaml found at {project_dir}")

        # Surface typo'd keys before they get silently ignored by the merge.
        warn_unknown_keys(project_config)

        # Merge defaults with project config (deep copy to avoid mutating defaults)
        import copy

        merged_config = copy.deepcopy(self.global_config)
        merged_config = self._deep_update(merged_config, project_config)
        return merged_config
