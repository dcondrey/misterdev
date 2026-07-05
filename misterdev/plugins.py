"""Extensibility registry for tools, gates, and targets.

Built-in capabilities self-register here, and third-party packages contribute
through ``importlib.metadata`` entry points, so ``pip install misterdev-plugin-x``
adds a tool/gate/target with zero edits to this codebase:

    # in the plugin package's pyproject.toml
    [project.entry-points."misterdev.tools"]
    my_tool = "my_pkg:MyTool"

Entry points are discovered lazily on first lookup and never override a built-in
of the same name (a plugin cannot hijack ``command``); a plugin that fails to
import is logged and skipped, never fatal. Registration is thread-safe because
the orchestrator resolves capabilities from parallel executor workers.
"""

import importlib.metadata
import threading
from typing import Callable, Dict, Generic, List, Optional, TypeVar, Union

from misterdev.logging_setup import setup_logger

logger = setup_logger(__name__)

T = TypeVar("T")


def _iter_entry_points(group: str):
    """Entry points for ``group`` across Python 3.10–3.13 metadata APIs."""
    eps = importlib.metadata.entry_points()
    if hasattr(eps, "select"):  # 3.10+ selectable interface
        return eps.select(group=group)
    return eps.get(group, [])  # pragma: no cover - legacy dict interface (<3.10)


class Registry(Generic[T]):
    """A name -> item map that also discovers external items via entry points.

    ``item`` is whatever the kind needs — for tools/gates it is a class or
    factory; the registry does not construct it, so a caller stays in control of
    how the object is instantiated.
    """

    def __init__(self, kind: str, entry_point_group: str):
        self.kind = kind
        self._group = entry_point_group
        self._items: Dict[str, T] = {}
        self._lock = threading.Lock()
        self._entry_points_loaded = False

    def register(
        self, name: str, item: Optional[T] = None
    ) -> Union[T, Callable[[T], T]]:
        """Register ``item`` under ``name``.

        Usable directly (``TOOLS.register("command", CommandTool)``) or as a
        decorator (``@TOOLS.register("command")``). A later registration of the
        same name replaces the earlier one, so an in-process override is explicit.
        """

        def _add(obj: T) -> T:
            with self._lock:
                self._items[name] = obj
            return obj

        return _add(item) if item is not None else _add

    def unregister(self, name: str) -> None:
        """Remove a registration if present (a no-op otherwise). Lets a plugin be
        disabled at runtime and keeps tests from leaking registrations."""
        with self._lock:
            self._items.pop(name, None)

    def get(self, name: str) -> Optional[T]:
        """Look up an item by name (loading external plugins first), or None."""
        self._ensure_entry_points()
        with self._lock:
            return self._items.get(name)

    def names(self) -> List[str]:
        """All registered names, built-in and plugin, sorted."""
        self._ensure_entry_points()
        with self._lock:
            return sorted(self._items)

    def _ensure_entry_points(self) -> None:
        if self._entry_points_loaded:
            return
        with self._lock:
            if self._entry_points_loaded:
                return
            self._entry_points_loaded = True
            for ep in _iter_entry_points(self._group):
                if ep.name in self._items:
                    # A built-in already claims this name; a plugin must not
                    # silently shadow it.
                    logger.debug(
                        f"{self.kind} plugin {ep.name!r} shadows a built-in; ignored"
                    )
                    continue
                try:
                    self._items[ep.name] = ep.load()
                    logger.info(f"Loaded {self.kind} plugin: {ep.name}")
                except Exception as e:
                    # A broken third-party plugin must never break the build.
                    logger.warning(
                        f"Skipping unloadable {self.kind} plugin {ep.name!r}: {e}"
                    )


# One registry per extensible kind. The entry-point group is the third-party
# contract; keep these strings stable.
TOOLS: Registry = Registry("tool", "misterdev.tools")
GATES: Registry = Registry("gate", "misterdev.gates")
TARGETS: Registry = Registry("target", "misterdev.targets")
