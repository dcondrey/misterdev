import threading
from pathlib import Path
from typing import Optional

from misterdev.config import get_setting
from misterdev.llm.client import BaseLLMClient, create_llm_client
from misterdev.environments.base_env import BaseEnvironmentManager
from misterdev.environments.venv_env import VenvEnvironmentManager
from misterdev.core.task import TaskManager
from misterdev.core.context.topography import TopographyEngine
from misterdev.logging_setup import setup_logger

logger = setup_logger(__name__)


class ToolManager:
    """Manages initialization and execution of tools."""

    def __init__(self, tools_config: list):
        self.tools = {}
        # Importing the package registers the built-in tools; the registry also
        # discovers third-party tools from the ``misterdev.tools`` entry-point
        # group, so a plugin adds a tool type with no change here.
        import misterdev.tools  # noqa: F401 - registration side effect
        from misterdev.plugins import TOOLS

        for tc in tools_config:
            tool_type = tc.get("type")
            tool_cls = TOOLS.get(tool_type)
            if tool_cls is None:
                if tool_type:
                    logger.warning(
                        f"Unknown tool type {tool_type!r}; falling back to command"
                    )
                tool_cls = TOOLS.get("command")
            tool = tool_cls(tc)
            self.tools[tool.name] = tool

    def get_tool(self, name: str):
        return self.tools.get(name)


class Project:
    """Represents an active project with all its dependencies initialized."""

    def __init__(self, path: str | Path, config: dict):
        self.path = Path(path)
        self.config = config

        self.name = config.get("name", self.path.name)
        self.description = config.get("description", "")

        self.llm_client = self._init_llm_client()
        self.env_manager = self._init_env_manager()
        self.tool_manager = ToolManager(config.get("tools", []))
        self.task_manager = TaskManager(self)
        # Model ledger/selector are built lazily on first use: they touch the
        # .orchestrator dir and only matter when dynamic_selection is enabled.
        self._model_ledger = None
        self._model_selector = None
        self._llm_cache = None
        self._semantic_ranker = None
        self._ranker_built = False
        self._mcp = None
        self._mcp_built = False
        # One Project is shared by tasks that run concurrently in a thread pool
        # (orchestrator parallel_mode). Guard the lazy MCP build so two threads
        # racing first access cannot each run discovery and double-launch servers.
        self._mcp_lock = threading.Lock()
        self._audit_trail = None
        self._governance_policy = None
        self._governance_built = False
        # Topography (symbol graph) is built lazily on first use, not here:
        # every CLI command registers all known projects, and eagerly scanning
        # each one's whole tree just to list/status is wasted work. The executor
        # calls initialize() (idempotent) before it needs the graph.
        self.topography = TopographyEngine(
            self.path,
            self.llm_client,
            golden_paths=get_setting(config, "orchestrator", "golden_paths"),
        )

    @property
    def model_ledger(self):
        """Persistent per-model performance store (lazy, file-backed)."""
        if self._model_ledger is None:
            from misterdev.core.economics.model_ledger import ModelLedger

            self._model_ledger = ModelLedger(
                self.path / ".orchestrator" / "model_stats.json"
            )
        return self._model_ledger

    @property
    def model_selector(self):
        """Ledger-driven model selection policy (lazy)."""
        if self._model_selector is None:
            from misterdev.core.economics.model_selector import (
                ModelSelector,
            )

            self._model_selector = ModelSelector(
                self.config, self.model_ledger, free_models=self._harvest_free_models()
            )
        return self._model_selector

    @property
    def semantic_ranker(self):
        """Embedding-based context ranker, or None when unavailable/disabled.

        Built at most once (discovery hits the network); a None result is
        remembered so topography falls back to arbitrary order without retrying.
        """
        if not self._ranker_built:
            self._ranker_built = True
            if get_setting(self.config, "llm", "semantic_retrieval"):
                from misterdev.core.economics.embeddings import (
                    EmbeddingCache,
                    SemanticRanker,
                )
                from misterdev.llm.client import create_embedding_client

                weight = get_setting(self.config, "llm", "lexical_weight")
                embedder = create_embedding_client(self.config)
                cache = (
                    EmbeddingCache(
                        self.path / ".orchestrator" / "embeddings.json", embedder.model
                    )
                    if embedder is not None
                    else None
                )
                # Always build a ranker: with no embedder it ranks lexically,
                # which still beats the arbitrary-order slice.
                self._semantic_ranker = SemanticRanker(embedder, cache, weight)
        return self._semantic_ranker

    @property
    def mcp(self):
        """MCP tool-host manager, or None when no servers are configured.

        Built at most once (the manager itself is cheap; discovery is deferred
        to first access of ``.tools`` and is timeout-bounded). A None result is
        remembered so callers can skip MCP entirely without retrying.
        """
        if self._mcp_built:
            return self._mcp
        with self._mcp_lock:
            if self._mcp_built:  # a racing thread already built it
                return self._mcp
            self._mcp = self._build_mcp()
            self._mcp_built = True
        return self._mcp

    def _build_mcp(self):
        mcp_cfg = self.config.get("mcp") or {}
        servers = list(mcp_cfg.get("servers") or [])
        # Curated tier: mount the vetted core stack by load tier (config-gated
        # so keyed servers appear only when their env vars are set). ``true``
        # -> "core"; a string/list selects tiers ("core"/"project"/"task"/"all").
        curated = mcp_cfg.get("curated")
        if curated:
            from misterdev.core.integration.mcp_registry import select_curated

            tiers = (
                ("core",)
                if curated is True
                else ((curated,) if isinstance(curated, str) else tuple(curated))
            )
            servers.extend(select_curated(tiers))
        # On-the-fly discovery: search the free official registry for servers
        # matching each capability query and append the trusted, locally-
        # runnable ones (npx/uvx stdio). Best-effort; never blocks the build.
        # Skipped under network isolation: discovery runs UNVETTED code from the
        # public registry on the host, which would defeat a sandbox the project
        # explicitly chose (see also the on-demand gate in the executor).
        discover = mcp_cfg.get("discover")
        if discover and self._host_exec_isolated():
            logger.warning(
                "MCP discovery skipped: project runs under network=none isolation, "
                "so provisioning unvetted registry servers on the host is refused."
            )
            discover = None
        if discover:
            from misterdev.core.integration.mcp import _server_identity
            from misterdev.core.integration.mcp_registry import (
                DEFAULT_TRUSTED_NAMESPACES,
                RegistryCache,
                discover_servers,
            )

            trusted = mcp_cfg.get("trusted_namespaces") or DEFAULT_TRUSTED_NAMESPACES
            # Don't re-discover a package already mounted (curated/configured):
            # dedup by launch identity so the same server isn't run under two names.
            known = {_server_identity(s) for s in servers} - {None}
            found = discover_servers(
                list(discover),
                trusted_namespaces=trusted,
                max_servers=int(mcp_cfg.get("discover_max_servers", 3)),
                min_trust=float(mcp_cfg.get("min_trust", 0.5)),
                cache=RegistryCache(self._mcp_cache_path()),
                known_identities=known,
                source=str(mcp_cfg.get("discover_source", "cgcone")),
            )
            for cfg in found:
                self.audit_trail.record(
                    "mcp_provision",
                    source="startup_discovery",
                    server=cfg.get("name"),
                    command=cfg.get("command"),
                    args=cfg.get("args"),
                )
            servers.extend(found)
        # On-demand discovery starts from a blank slate: the model FINDs a server
        # mid-task and it is mounted live. That needs a manager to mount INTO even
        # when nothing is preconfigured — but not under host-execution isolation,
        # where provisioning is refused anyway (so a null manager is correct).
        on_demand = bool(
            mcp_cfg.get("discover_on_demand", True) and not self._host_exec_isolated()
        )
        if servers or on_demand:
            from misterdev.core.integration.mcp import MCPManager

            manager = MCPManager(servers, allow_tools=mcp_cfg.get("allow_tools"))
            if manager.enabled or on_demand:
                return manager
        return None

    def _mcp_cache_path(self) -> Path:
        return self.path / ".orchestrator" / "mcp_registry_cache.json"

    def _host_exec_isolated(self) -> bool:
        """True when the project deliberately runs under isolation.

        Two signals, either of which means unvetted host execution would break
        the boundary the build chose: ``governance.network=none`` (host network
        cut off), and a container/docker environment (the build runs inside a
        sandbox, but MCP servers spawn on the HOST — so auto-installing a registry
        package on the host escapes that sandbox entirely).
        """
        gov = self.config.get("governance") or {}
        if gov.get("network") == "none":
            return True
        env_type = (self.config.get("environment") or {}).get("type")
        return env_type in ("docker", "container")

    @property
    def llm_cache(self):
        """Response memoization store, or None when caching is disabled."""
        if self._llm_cache is None and get_setting(self.config, "llm", "cache"):
            from misterdev.core.economics.llm_cache import LLMCache

            self._llm_cache = LLMCache(self.path / ".orchestrator" / "llm_cache")
        return self._llm_cache

    @property
    def audit_trail(self):
        """Append-only JSONL audit trail (lazy, file-backed under .orchestrator).

        Defaults ON: it only appends observability records to a gitignored file
        and degrades to a no-op if the path is unwritable, so it cannot regress a
        build. A run that wants it silent leaves the file unread (no behavioral
        effect either way)."""
        if self._audit_trail is None:
            from misterdev.core.audit import AuditTrail

            self._audit_trail = AuditTrail(self.path, enabled=True)
        return self._audit_trail

    @property
    def governance_policy(self):
        """Risk-classified approval policy, or None when governance is off.

        Built from config in AUTONOMOUS mode (interactive prompting is a deferred
        seam: threading stdin into the wave loop risks a hang, so unattended runs
        BLOCK a risky command and record an escalation unless governance.
        auto_approve is set). None when orchestrator.governance is false, so the
        command seam stays byte-identical to today."""
        if not self._governance_built:
            self._governance_built = True
            from misterdev.core.execution.governance import (
                policy_from_config,
            )

            policy = policy_from_config(
                self.config, interactive=False, audit=self.audit_trail
            )
            self._governance_policy = policy if policy.enabled else None
        return self._governance_policy

    def _harvest_free_models(self) -> list:
        """Current free OpenRouter models when use_free_models is enabled."""
        if not get_setting(self.config, "llm", "use_free_models"):
            return []
        import time

        from misterdev.core.economics.free_models import FreeModelCache

        cache = FreeModelCache(self.path / ".orchestrator" / "free_models.json")
        try:
            return cache.get(time.time())
        except Exception as e:
            logger.warning(f"Free-model harvest skipped: {e}")
            return []

    def _init_llm_client(self) -> BaseLLMClient:
        return create_llm_client(self.config)

    def _init_env_manager(self) -> Optional[BaseEnvironmentManager]:
        env_config = self.config.get("environment", {})
        env_type = env_config.get("type")
        if env_type == "venv":
            return VenvEnvironmentManager(env_config, self.path)
        if env_type in ("docker", "container"):
            from misterdev.environments.container_env import (
                ContainerEnvironmentManager,
            )

            gov = self.config.get("governance") or {}
            network = gov.get("network") if gov.get("network") == "none" else None
            return ContainerEnvironmentManager(
                env_config,
                self.path,
                language=self.config.get("language", ""),
                network=network,
            )
        return None
