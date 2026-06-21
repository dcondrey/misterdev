import yaml
from pathlib import Path
from typing import Dict, Any

from my_project_orchestrator.logging_setup import setup_logger

logger = setup_logger(__name__)

# Basic global defaults
DEFAULT_CONFIG = {
    "orchestrator_version": "1.0",
    "llm": {
        "provider": "openrouter",
        "model": "anthropic/claude-sonnet-4.6",
        "temperature": 0.1,
    },
    "tools": [],
    "environment": {"type": "none"},
    "prompt_templates": {
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
    },
    "build": {
        "max_tasks": 30,
        "max_attempts_per_task": 3,
        "max_consecutive_failures": 3,
        "build_timeout": 120,
        "test_timeout": 180,
        "lint_timeout": 120,
        "parallel_analysis": True,
    },
    "orchestrator": {
        "max_consecutive_failures": 3,
        "max_workers": 4,
        "context_budget_tokens": 100000,
        "max_task_attempts": 3,
        "integration_gate": True,
        # Hardened defaults: everything opted in. "auto" keys are budget-driven
        # (see BaseLLMClient / agent convergence loop) so the global build.budget
        # is the single master constraint.
        "max_build_iterations": "auto",
        "certainty_threshold": 0.5,
        "max_cost_per_task": "auto",
        "allow_test_edits": False,
        "verify_acceptance": True,
        "llm_acceptance_judge": True,
        # Golden suite: files the model never sees and may never edit, plus a
        # command to run them as a blocking gate. Empty/None = feature off.
        "golden_paths": [],
        "golden_command": None,
        # AB-MCTS spec refinement fires several serial LLM calls before any work;
        # off by default (marginal value, large latency/cost).
        "enable_ab_mcts": False,
    },
    "build_command": None,
    "test_command": None,
    "lint_command": None,
}


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

        # Merge defaults with project config (deep copy to avoid mutating defaults)
        import copy

        merged_config = copy.deepcopy(self.global_config)
        merged_config = self._deep_update(merged_config, project_config)
        return merged_config
