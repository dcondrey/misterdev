# 🚀 Project Orchestrator

**State-of-the-Art (SOTA) Autonomous Development Framework**

Project Orchestrator is a modular, extensible Python framework designed to automate the entire software development lifecycle. By combining advanced LLM orchestration with local tool execution and deterministic state management, it enables autonomous "Try-Test-Fix" development loops that scale across complex, multi-module projects.

---

## ✨ Core Pillars

### 🧠 Deep Contextual Intelligence
- **Phase-Based Analysis:** Employs a multi-phase assessment (Structure, Completeness, Context) to build a high-fidelity model of your codebase before writing a single line of code.
- **Scratchpad Memory:** Tasks share an intra-session "Scratchpad" to pass learnings (patterns, pitfalls, environment quirks) forward, preventing redundant failures.

### 🏗️ Deterministic Orchestration
- **6-Phase Build Workflow:** Operates through a rigorous cycle: `Analyze -> Spec -> Decompose -> Execute -> Validate -> Report`.
- **Topological Task Execution:** Automatically resolves task dependencies and executes them in an optimal, safe order.
- **Revert-on-Failure:** Atomic task execution with automatic file restoration if tests or builds fail after retries.

### 🛠️ Extensible Tooling
- **Pluggable Architecture:** Easily add new LLM providers (OpenRouter, Anthropic, etc.), custom tools (Git, Formatters, Test Runners), and environment managers (Venv, Docker).
- **Project-Centric Config:** All behavior is controlled by a declarative `project.yaml` in your repository root.

---

## 🚦 Quick Start

### 1. Installation
```bash
git clone https://github.com/your-repo/project-orchestrator.git
cd project-orchestrator
pip install -e .
```

### 2. Configuration
Create a `project.yaml` in your project root:
```yaml
name: "My SOTA App"
llm:
  provider: "openrouter"
  model: "anthropic/claude-3.5-sonnet"
tools:
  - name: "Git"
    type: "git"
  - name: "Pytest"
    type: "test_runner"
    command: "pytest"
```

### 3. Usage
```bash
# Scan and register a workspace
project-orchestrator scan ./projects

# Run a SOTA build
project-orchestrator build ./projects/my-app "add a robust login system with OAuth2"
```

---

## 📐 Architecture

- **`core/`**: The brain of the system. Handles the state machine, models, and task decomposition.
- **`llm/`**: Abstraction layer for LLM interactions with structured response parsing.
- **`task_executors/`**: Implementation of the "Inner Loop" (Try-Test-Fix).
- **`analyzers/`**: Deep project inspection engines.
- **`tools/`**: Interfaces for interacting with the local system (Git, CLI, Files).

---

## 🛡️ Safety & Integrity
- **Configurable Budgets:** Prevent runaway LLM costs with token and dollar limits.
- **Validation Gates:** Final validation phase ensures no regressions before completion.
- **Interactive Mode:** Optionally confirm every task before execution with `--interactive`.

---

## 🤝 Contributing
We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on our SOTA development standards.

---
*Built with ❤️ for the next generation of software engineers.*
