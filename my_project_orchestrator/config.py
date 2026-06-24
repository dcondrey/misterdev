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
    # Extra sampling parameters (top_p, top_k, min_p, repetition_penalty,
    # frequency_penalty, presence_penalty, seed, ...). Each is sent only to
    # models whose OpenRouter supported_parameters include it, so unsupported
    # knobs never cause a 400. temperature is filtered the same way.
    sampling: Dict[str, Any] = field(default_factory=dict)
    api_key_env_var: str = "OPENROUTER_API_KEY"
    streaming: bool = False
    # Extract edits via a structured function-call (apply_edits) when the model
    # supports `tools`, instead of regex-parsing markdown fences. On by default;
    # falls back to markdown parsing for models without tool support.
    use_tools: bool = True
    failover: List[Dict[str, Any]] = field(default_factory=list)
    # Per-task model routing: routing maps complexity/strategy -> tier name,
    # models maps tier name -> model id (or a list of candidate ids). Empty =
    # use the default model.
    routing: Dict[str, Any] = field(default_factory=dict)
    models: Dict[str, Any] = field(default_factory=dict)
    # Ledger-driven dynamic model selection. False = off (default), True = on
    # using selection_posture, "auto" = self-activating: explore cheap/free
    # models on easy tasks while a (category, complexity) cell is immature, then
    # settle into conservative cheap-first per cell once it has matured. Typed
    # Any because it accepts a bool or the string "auto".
    # escalation is the capability ladder, cheapest tier first; each tier name
    # resolves through `models`. The policy uses a cheaper model on early
    # attempts and climbs to the strongest tier by the final attempt.
    dynamic_selection: Any = "auto"
    escalation: List[str] = field(default_factory=list)
    # A cheaper model is trusted for a first attempt only once it has at least
    # min_observations recorded first-try attempts and a first-try success rate
    # at or above first_try_floor.
    min_observations: int = 5
    first_try_floor: float = 0.5
    # "auto" mode treats a (category, complexity) cell as matured once it has at
    # least this many recorded attempts, after which it stops exploring and
    # behaves conservatively.
    maturity_threshold: int = 12
    # Exploration aggressiveness: "conservative" (cheap only once proven),
    # "balanced" (explore cheap on low/medium-complexity first attempts), or
    # "aggressive" (always try cheapest first). The final attempt is always the
    # strongest tier regardless of posture.
    selection_posture: str = "conservative"
    # Per-complexity reasoning effort, sent only to models that support a
    # reasoning budget. Default leverages reasoning where it pays off (hard
    # tasks) without adding token cost to easy ones; a complexity absent from
    # the map gets no reasoning. Effort values: minimal|low|medium|high|xhigh.
    reasoning_effort: Dict[str, str] = field(
        default_factory=lambda: {"large": "high", "medium": "medium"}
    )
    # Harvest OpenRouter's rotating free models into the cheapest tier. On by
    # default for out-of-box cost savings; the quality floor (gates) plus the
    # always-strong final attempt keep output safe. Set false to keep code off
    # third-party free endpoints entirely.
    use_free_models: bool = True
    # Semantic context retrieval: rank candidate code symbols by embedding
    # similarity to the task and keep the most relevant when they exceed the
    # context cap, instead of truncating in arbitrary order. On by default and
    # self-regulating (only embeds when selection is actually needed); degrades
    # to arbitrary order if no embedding model is reachable.
    semantic_retrieval: bool = True
    # Which embedder backs semantic retrieval. "auto" = OpenRouter for an
    # OpenRouter provider, else a local fastembed model (free, offline, no key);
    # "local" forces fastembed; "openrouter" forces the API; "none" disables the
    # dense signal (lexical-only ranking).
    embedding_backend: str = "auto"
    # fastembed model used by the local backend (downloaded once, then cached).
    local_embedding_model: str = "BAAI/bge-small-en-v1.5"
    # Empty = auto-pick the cheapest (free-preferred) embedding model from
    # OpenRouter's embeddings catalog, like free-model harvesting; set an id to
    # pin one. Ranking is forgiving, so cheapest is a fine default.
    embedding_model: str = ""
    # Id substrings preferred when auto-selecting among equally-priced embedding
    # models; defaults to code-aware models since we rank code.
    embedding_prefer: List[str] = field(default_factory=lambda: ["code"])
    # Output dimensionality; 0 leaves the model default (param omitted).
    embedding_dimensions: int = 0
    # Weight of the lexical identifier-overlap signal vs dense cosine when
    # ranking context (0 = pure dense, 1 = pure lexical). Lexical also drives
    # ranking on its own when no embedding model is reachable.
    lexical_weight: float = 0.3
    # Whether to use models/providers that train on your inputs. Off by default:
    # OpenRouter routing is constrained to providers that do not store or train
    # on inputs (provider data_collection="deny"), which is what makes harvesting
    # free models safe. Set true to permit training providers (more/cheaper free
    # models, but your code may be used for training).
    allow_training_models: bool = False
    # Memoize gate-passing LLM outputs keyed by the full prompt, so an identical
    # request reuses the prior result instead of calling a model. On by default:
    # it is content-hashed (auto-invalidates when inputs change) and every hit is
    # re-validated through the gates, so it can only save cost, never ship stale
    # code.
    cache: bool = True


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
    # Target files at or below this many lines are sent in full; larger files
    # are sent as a symbol outline plus verbatim windows of the task-relevant
    # symbols, so context (and cost) scales with the edit, not the file size.
    large_file_line_threshold: int = 800
    max_task_attempts: int = 3
    integration_gate: bool = True
    # "auto" keys are budget-driven (BaseLLMClient / convergence loop) so the
    # global build.budget is the single master constraint. Typed Any because
    # they accept "auto", a number, or None.
    max_build_iterations: Any = "auto"
    certainty_threshold: float = 0.5
    max_cost_per_task: Any = "auto"
    allow_test_edits: bool = False
    # Optional LSP semantic gate: when on, a language server checks edited files
    # for errors a syntax check misses (undefined names, type errors). Off by
    # default and timeout-bounded so it can never block a build; lsp_timeout
    # caps how long the whole check may take before it is skipped.
    lsp_diagnostics: bool = False
    lsp_timeout: int = 30
    # Optional runtime smoke gate: when on, the built artifact is launched,
    # probed, and asserted to respond before the build is accepted. Off by
    # default and timeout-bounded (runs in a daemon thread) so it can never
    # block a build; missing/incomplete runtime.smoke config makes it a SKIP.
    # The smoke spec itself lives under the top-level ``runtime.smoke`` key.
    runtime_smoke: bool = False
    # Optional web verification gate: when on, a headless browser (Playwright)
    # drives the running web artifact and runs declarative checks (DOM/text
    # presence, no console errors, axe accessibility, screenshot diff). Off by
    # default and timeout-bounded (daemon thread) so it can never block a build;
    # missing config or a missing Playwright/browser makes it a SKIP. The spec
    # lives under the top-level ``runtime.web`` key.
    web_verify: bool = False
    # Optional vision verification gate: when on, a vision model judges whether a
    # captured screenshot satisfies a stated visual requirement. Off by default
    # and timeout-bounded; no config / no model / no network makes it a SKIP. The
    # spec lives under the top-level ``runtime.vision`` key.
    vision_verify: bool = False
    # Optional mutation-score gate: when on, the project's configured mutation
    # command (top-level ``mutation.command``) is run and its parsed score must
    # meet ``mutation.min_score`` — proving the suite kills injected faults, not
    # just passes. Off by default and timeout-bounded; no config / an unparseable
    # score / a timeout is a SKIP, only a score below the floor is a RED.
    mutation_gate: bool = False
    verify_acceptance: bool = True
    llm_acceptance_judge: bool = True
    # Optional goal-completion check: when on, an LLM judge reads the goal,
    # acceptance criteria, and the build's cumulative diff and reports whether the
    # work actually satisfies the goal (gates green != goal met). Off by default.
    # ADVISORY: it records gaps into the report and logs them but does NOT fail
    # the build, unless block_on_goal_gap is also true. Timeout-bounded; no
    # goal/criteria/client, an unparseable verdict, or a judge error is a SKIP.
    goal_check: bool = False
    block_on_goal_gap: bool = False
    goal_check_timeout: int = 60
    # Spec-as-tests (CONSERVATIVE, opt-in, currently DEFERRED): generate a failing
    # test from a task's acceptance criteria before it is implemented. Off by
    # default. The generation primitive lives in core/spec_tests.py and is tested,
    # but it is NOT wired into the execute loop yet: writing a failing test inside
    # the wave loop would flip the integration-gate baseline red and silently
    # disable that gate, which is not control-flow-neutral. When set true today it
    # only logs that the feature is staged-but-not-wired (see the seam in
    # core/spec_tests.py); it never alters the build loop.
    spec_as_tests: bool = False
    # When spec_as_tests is on, a per-task generated spec test is run (scoped,
    # from .orchestrator/spec_tests/) after the task's gates pass. ADVISORY by
    # default: a still-failing spec test is logged/recorded but does not fail the
    # task (the generated test may itself be imperfect). Set this true to make a
    # red spec test fail acceptance and force a retry — strict TDD.
    spec_as_tests_block: bool = False
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
    # Optional MCP (Model Context Protocol) tool awareness: when on, the tools
    # discovered from the servers in the top-level ``mcp.servers`` list are
    # described to the model in the task context so it knows they exist. Off by
    # default and additive only — it never changes the single-shot build loop.
    # The substrate (connect/discover/call) is always available via project.mcp;
    # this flag gates only the awareness injection. Timeout-bounded throughout.
    mcp_enabled: bool = False
    # Optional agentic MCP tool use: when on (and an MCP manager with discovered
    # tools exists), a BOUNDED pre-edit loop lets the model request MCP tool
    # calls to gather information; results are prepended to the task context and
    # the existing edit-generation path runs unchanged. Off by default and purely
    # additive — when off the executor path is byte-identical to today. Each round
    # is timeout-bounded (the tool call goes through MCPManager.call_tool); the
    # loop is hard-capped by ``mcp_max_tool_rounds``. Implies ``mcp_enabled`` for
    # the awareness/registry, but the gathering loop is gated by this flag alone.
    mcp_tool_use: bool = False
    # Hard ceiling on the agentic gathering loop's rounds (see ``mcp_tool_use``).
    # Each round is at most one model turn plus one tool call; the loop always
    # stops at this count even if the model keeps requesting tools.
    mcp_max_tool_rounds: int = 3
    # Optional governance layer: when on, a risk classifier gates risky commands
    # (destructive/irreversible/paid) at the command seam and an append-only
    # audit trail is written. Off by default and additive — when off the command
    # seam is byte-identical to today. In autonomous (non-interactive) mode a
    # risky command is BLOCKED with an escalation record unless
    # ``governance.auto_approve`` is set; in interactive mode it prompts. Ordinary
    # build/test/lint commands classify as SAFE and always run. The policy spec
    # lives under the top-level ``governance`` key.
    governance: bool = False
    # Bounded continuation-on-truncation for the plain text-generation path.
    # When the model cuts a response off at its output-token limit (finish_reason
    # "length"/"max_tokens") — which truncates a large NEW file's full content and
    # fails the edit gate — issue up to this many follow-up calls asking the model
    # to continue exactly where it stopped, concatenating the text before parsing.
    # Activates ONLY on a truncated response, so when a response finishes normally
    # the path is byte-identical to before. 0 disables it (never continues).
    # Default 2: a pure-win correctness fix that costs nothing on untruncated
    # responses and recovers files up to ~3x the single-shot output cap.
    max_continuations: int = 2
    # Optional adversarial edit critic (independent second component). When on, a
    # SECOND component reviews each CANDIDATE edit before it is applied — ideally
    # a DIFFERENT model (``critic.model``) so it does not inherit the generator's
    # blind spots — and either approves it or returns concrete objections that are
    # fed back to the generator for the next attempt. Off by default and purely
    # additive: when off the executor path is byte-identical to today. Best-effort
    # and timeout-bounded (daemon thread); no client, an unparseable verdict, or a
    # timeout is a SKIP that lets the edit proceed to the real gates. The critic is
    # advisory, never authoritative — the build/test gates remain the ground
    # truth; ``critic_max_rejections`` caps how many regenerations it may force per
    # task before deferring to those gates. The independent model id lives under
    # the top-level ``critic.model`` key.
    adversarial_critic: bool = False
    critic_timeout: int = 60
    critic_max_rejections: int = 2


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
    # Runtime gate specs (each off unless its orchestrator flag is true and this
    # carries the matching mapping): `smoke` (runtime_smoke), `web` (web_verify),
    # `vision` (vision_verify). Free-form so projects can describe any
    # launch/url/checks/capture/assert/timeout; not schema-validated like the
    # dataclass sections, mirroring how `environment` is an open dict.
    "runtime": {},
    # MCP (Model Context Protocol) servers to connect to for tool discovery.
    # ``servers`` is a list of {name, command, args, transport} mappings (stdio
    # transport). Empty by default; awareness injection is gated separately by
    # orchestrator.mcp_enabled. Open dict like ``runtime``, not schema-validated.
    "mcp": {"servers": []},
    # Mutation-score gate spec (off unless orchestrator.mutation_gate is true and
    # this carries a ``command``): ``command`` (the mutation-testing command),
    # ``min_score`` (floor, a fraction or percentage), ``timeout`` (seconds).
    # Open dict like ``runtime``, not schema-validated, so any tool/command fits.
    "mutation": {},
    # Governance policy spec (off unless orchestrator.governance is true):
    # ``approval_required`` (extra risky regex patterns), ``auto_approve`` (bool,
    # let risky commands run unattended in autonomous mode), ``network``
    # ("none"|"default", container egress control). Open dict like ``runtime``,
    # not schema-validated, so the pattern list and knobs stay free-form.
    "governance": {"network": "default"},
    # Adversarial-critic spec (off unless orchestrator.adversarial_critic is
    # true): ``model`` (an INDEPENDENT critic model id — different from the
    # generator so it doesn't share its blind spots). Empty leaves the critic on
    # the generator's own model with adversarial framing (weaker independence,
    # logged). Open dict like ``runtime``, not schema-validated.
    "critic": {},
    # Shared INDEPENDENT judge model for the post-gate goal-completion check and
    # the LLM acceptance judge: ``model`` (an id different from the generator so
    # those judgments don't share its blind spots). Empty leaves each judge on the
    # generator's own model (weaker independence, logged). Open dict like
    # ``runtime``, not schema-validated. (The edit-time critic has its own
    # ``critic.model``; this covers the two post-implementation judges.)
    "judge": {},
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
