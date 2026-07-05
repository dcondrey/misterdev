"""Shared result type for the optional, non-blocking gates.

The runtime smoke, web, vision, and mutation gates all report the same tri-state
outcome — SKIP / GREEN / RED — with the same ``passed`` and ``skipped`` meaning,
plus a ``reason`` and gate-specific evidence. This centralizes the status
vocabulary and the small boilerplate so each gate's result class only declares
its own evidence fields.
"""

# Tri-state gate status. GREEN passed; RED failed (blocks the build); SKIP is
# "no opinion" (missing config/dep, unparseable, or timeout) and never blocks.
SKIP = "skip"
GREEN = "green"
RED = "red"


class GateOutcome:
    """Base for a gate result: a SKIP/GREEN/RED ``status`` and a ``reason``.

    Subclasses add their own evidence fields and call ``super().__init__``.
    ``passed`` is GREEN; ``skipped`` is SKIP.
    """

    def __init__(self, status: str, reason: str = ""):
        self.status = status
        self.reason = reason

    @property
    def passed(self) -> bool:
        return self.status == GREEN

    @property
    def skipped(self) -> bool:
        return self.status == SKIP


class GateContext:
    """Inputs a registered plugin gate receives.

    ``project_path`` is the repo (or target subdir) root; ``commands`` are the
    resolved build/test/lint/typecheck commands; ``env_activate`` is the optional
    host-venv activation prefix. A plugin gate is ``callable(GateContext) ->
    GateOutcome`` registered on ``misterdev.plugins.GATES``; a RED outcome blocks
    the build, SKIP/GREEN do not.
    """

    def __init__(self, project_path, commands, env_activate=None):
        self.project_path = project_path
        self.commands = commands
        self.env_activate = env_activate
