"""Sovereign-tier orchestration capabilities.

Inspired by June 2026 arXiv research:
- Agentic Engineering: Code as transient resource (arXiv:2606.05608)
- Long-Horizon Reasoning: AB-MCTS (Sakana AI / Marlin, June 2026)
- Discordance-Aware Reasoning (CyberDrift, arXiv:2606.15101)
"""

import json
import re
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from my_project_orchestrator.logging_setup import setup_logger
from my_project_orchestrator.llm.client import BaseLLMClient

logger = setup_logger(__name__)


class EphemeralCodeManager:
    """Manages code as a transient, instrumental resource.

    In 'Agentic Engineering', code is often generated just to solve a specific
    reasoning step and can be discarded once its goal is achieved.
    """

    def __init__(self, project_path: Path):
        self.session_id = uuid.uuid4().hex[:8]
        self.ephemeral_dir = (
            project_path / ".orchestrator" / "ephemeral" / self.session_id
        )
        self.ephemeral_dir.mkdir(parents=True, exist_ok=True)

    def run_ephemeral_script(self, code: str, name: str = "temp") -> Tuple[bool, str]:
        """Executes a transient script and returns its output."""
        # Sanitize the (LLM-supplied) name: a '/' or other path char would turn
        # the filename into a non-existent subdir and raise on write, crashing
        # the whole build from this best-effort probe step.
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", name)[:60] or "temp"
        script_path = self.ephemeral_dir / f"{safe}_{uuid.uuid4().hex[:4]}.py"
        logger.info(f"Executing ephemeral logic: {script_path.name}")
        try:
            self.ephemeral_dir.mkdir(parents=True, exist_ok=True)
            script_path.write_text(code, encoding="utf-8")
            res = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=60,
            )
            output = res.stdout
            if res.stderr:
                output += f"\nERR: {res.stderr}"
            return res.returncode == 0, output
        except subprocess.TimeoutExpired:
            return False, f"Ephemeral script timed out after 60s"
        except Exception as e:
            return False, str(e)

    def cleanup(self):
        """Discards all transient resources from this session."""
        if self.ephemeral_dir.exists():
            shutil.rmtree(self.ephemeral_dir, ignore_errors=True)
            logger.info(f"Cleaned up ephemeral session {self.session_id}")

    def __enter__(self) -> "EphemeralCodeManager":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.cleanup()
        return False


class ABMCTSPlanner:
    """Adaptive Branching Monte Carlo Tree Search for task planning.

    Explores multiple reasoning branches to find the optimal execution path.
    Skips branching for trivial/small tasks to save LLM calls.
    """

    def __init__(self, llm_client: BaseLLMClient):
        self.llm = llm_client

    def branch_and_evaluate(self, task: str, context: str, branches: int = 3) -> str:
        """Simulates multiple implementation paths and selects the winner.

        Returns the original task unchanged if branching is not worthwhile.
        """
        # Skip branching for short specs (trivial work)
        if len(task.split()) < 50:
            logger.info("AB-MCTS: Skipping branching (spec too short to benefit)")
            return task

        logger.info(f"AB-MCTS: Branching (n={branches})...")

        simulations = []
        for i in range(branches):
            prompt = (
                f"Simulation Path {i + 1}: Propose a unique implementation strategy "
                f"for this task.\nContext: {context}\nTask: {task}"
            )
            strategy = self.llm.generate_code(
                prompt, "You are a competitive systems architect."
            )
            simulations.append(strategy)

        # Self-Evaluation / Discordance-Aware Selection
        numbered = "\n".join(
            f"--- PATH {i + 1} ---\n{s}" for i, s in enumerate(simulations)
        )
        eval_prompt = (
            f"Evaluate these {branches} competing implementation strategies.\n"
            f"Select the one with the highest 'Verifiability' and lowest 'Regression Risk'.\n\n"
            f"Strategies:\n{numbered}\n\n"
            f"Return ONLY the content of the selected strategy."
        )
        logger.info("AB-MCTS: Evaluating branches...")
        return self.llm.generate_code(
            eval_prompt, "You are a discordance-aware evaluator."
        )


class ProbeGenerator:
    """Synthesizes empirical fact-finding probes with high-rigor reflection.

    Identifies 'assumptions' in the spec and generates scripts to verify them
    using Python's 'inspect', 'ast', and 'dir' modules for reflective analysis.
    """

    def __init__(self, llm_client: BaseLLMClient):
        self.llm = llm_client

    def generate_probes(
        self, spec: str, assessment_summary: str
    ) -> List[Dict[str, str]]:
        """Analyzes spec and generates a list of {name, purpose, script} probes."""
        prompt = f"""Analyze this technical spec and project assessment.
Identify 1-3 critical 'assumptions' or 'unknowns' about the live codebase or environment.

Spec: {spec}
Assessment: {assessment_summary}

For each unknown, synthesize a transient Python 'Probe' script that uses REFLECTIVE techniques:
- Use 'inspect.signature()' to verify function/method arguments.
- Use 'inspect.getsource()' to see the actual logic.
- Use 'dir()' and 'getattr()' to probe runtime objects.
- Use 'ast.parse()' on local files to check structural definitions.

The script MUST print its findings clearly to stdout.

Return a JSON array: [{{"name": "...", "purpose": "...", "script": "..."}}]
Return ONLY the JSON array.
"""
        logger.info("Generating empirical probes...")
        try:
            response = self.llm.generate_code(
                prompt, "You are a senior empirical researcher."
            )
            text = response.strip()
            start = text.find("[")
            end = text.rfind("]")
            if start >= 0 and end > start:
                return json.loads(text[start : end + 1])
        except Exception as e:
            logger.error(f"Failed to generate probes: {e}")

        return []


class ToolSynthesizer:
    """Synthesizes project-local helper tools on-the-fly."""

    def __init__(self, project_path: Path):
        self.tools_dir = project_path / ".orchestrator" / "synthesized_tools"
        self.tools_dir.mkdir(parents=True, exist_ok=True)

    def synthesize_tool(
        self, name: str, purpose: str, llm_client: BaseLLMClient
    ) -> str:
        """Asks LLM to generate a Python script for a specific purpose."""
        prompt = (
            f"Synthesize a standalone Python script to serve as a local tool.\n"
            f"Purpose: {purpose}\nTool Name: {name}\n\n"
            f"Requirements:\n"
            f"- Single file, standard library or project-available dependencies.\n"
            f"- Executable as 'python tool.py'.\n"
            f"- Return ONLY the code in a ```python code block."
        )
        logger.info(f"Synthesizing tool: {name}")
        response = llm_client.generate_code(prompt, "You are a senior tools engineer.")

        code = _extract_code_block(response)
        tool_path = self.tools_dir / f"{name}.py"
        tool_path.write_text(code, encoding="utf-8")
        logger.info(f"Tool {name} saved to {tool_path}")
        return str(tool_path)


class StrategyOptimizer:
    """Optimizes agentic workflows via strategy simulation.

    Caches strategy per task category to avoid redundant LLM calls.
    """

    STRATEGIES = {
        "surgical": "Search-then-Edit (Fast, minimal context)",
        "iterative": "Standard Try-Test-Fix Loop (Thorough)",
        "architectural": "Spec-first Redesign (High impact)",
        "agentic": "Sovereign Agentic Engineering (Transient code, multi-agent)",
    }

    def __init__(self):
        self._cache: Dict[str, str] = {}

    def select_best_strategy(
        self,
        task_description: str,
        task_category: str,
        project_summary: str,
        llm_client: BaseLLMClient,
    ) -> str:
        """Select strategy, using cache for repeated categories."""
        if task_category in self._cache:
            cached = self._cache[task_category]
            logger.info(f"Strategy (cached for {task_category}): {cached.upper()}")
            return cached

        strategy_list = "\n".join(
            f"{i + 1}. {k}: {v}" for i, (k, v) in enumerate(self.STRATEGIES.items())
        )
        prompt = (
            f"Select the most efficient execution strategy.\n\n"
            f"Project: {project_summary}\n"
            f"Task category: {task_category}\n"
            f"Task: {task_description}\n\n"
            f"Available Strategies:\n{strategy_list}\n\n"
            f"Return ONLY the key (surgical, iterative, architectural, or agentic)."
        )
        best = (
            llm_client.generate_code(prompt, "You are a workflow optimizer.")
            .strip()
            .lower()
        )
        if best not in self.STRATEGIES:
            best = "iterative"

        self._cache[task_category] = best
        logger.info(f"Strategy selected for {task_category}: {best.upper()}")
        return best


class RealTimeAligner:
    """Ensures multi-step alignment via a 'Shared Certified Repository' pattern."""

    def __init__(self, project_path: Path):
        self.cert_file = project_path / ".orchestrator" / "consensus.json"
        self.cert_file.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self):
        if self.cert_file.exists():
            try:
                self.data = json.loads(self.cert_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self.data = {"invariants": [], "decisions": []}
        else:
            self.data = {"invariants": [], "decisions": []}

    def certify_decision(self, decision: str, rationale: str):
        """Records a certified project decision to keep future tasks aligned."""
        self.data["decisions"].append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "decision": decision,
                "rationale": rationale,
            }
        )
        self.cert_file.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        logger.info(f"Certified Decision: {decision}")

    def get_consensus_context(self) -> str:
        """Returns the current project consensus as a prompt string."""
        if not self.data["decisions"]:
            return "No prior decisions recorded."

        lines = ["## Project Consensus (Certified)"]
        for d in self.data["decisions"]:
            lines.append(f"- {d['decision']} (Rationale: {d['rationale']})")
        return "\n".join(lines)


def _extract_code_block(response: str) -> str:
    """Extract the first code block from an LLM response."""
    lines = response.split("\n")
    in_block = False
    code_lines = []
    for line in lines:
        if line.strip().startswith("```") and not in_block:
            in_block = True
            # Skip the opening fence line
            continue
        if line.strip().startswith("```") and in_block:
            break
        if in_block:
            code_lines.append(line)
    return "\n".join(code_lines) if code_lines else response.strip()
