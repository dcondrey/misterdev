"""Example misterdev plugin: one gather-safe tool and one gate.

Installing this package (``pip install -e examples/misterdev-plugin-hello``)
registers the tool and gate with misterdev via entry points — no edit to
misterdev itself. See this directory's README.
"""

from misterdev.core.execution.outcomes import GateOutcome, GREEN, RED


class HelloTool:
    """A read-only, gather-safe tool: returns a greeting.

    ``gather_safe = True`` opts it into the agentic gathering loop, so when it
    is configured in ``project.yaml`` the model may call ``local.hello`` mid-task.
    """

    gather_safe = True
    gather_description = "Return a friendly greeting for a name."

    def __init__(self, config: dict):
        self.name = config.get("name", "hello")

    def execute(self, project, name: str = "world", **_ignored):
        return True, f"Hello, {name}! (from the misterdev-plugin-hello example)"


def no_shouting_gate(ctx) -> GateOutcome:
    """A trivial example gate: fail if the build command SHOUTS in all caps."""
    build = (ctx.commands or {}).get("build_command") or ""
    if build and build.isupper():
        return GateOutcome(RED, "build_command is ALL CAPS; please calm down")
    return GateOutcome(GREEN)
