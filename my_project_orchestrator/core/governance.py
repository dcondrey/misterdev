"""Governance layer: risk classification + approval gates (opt-in).

Classifies a shell command (or a described action) as *risky* when it matches a
destructive, irreversible, or paid operation — ``rm -rf``, ``git push --force``,
``DROP TABLE``/``DROP DATABASE``, ``kubectl delete``, ``terraform apply``/
``destroy``, cloud-CLI ``delete``/``rm``, ``docker rmi``/``system prune``,
deploy/publish commands, and ``curl ... | sh`` pipe-to-shell. Ordinary build,
test, and lint commands (``cargo test``, ``pytest -q``, ``npm run build``,
``ruff check``, ``true``/``false``) classify as SAFE.

Design mirrors :mod:`my_project_orchestrator.core.lsp` / ``container`` / ``mcp``:
strictly opt-in (``orchestrator.governance`` is off by default), never raises
into the caller, and a no-op when off so behavior is byte-identical to today.

When ON, :class:`GovernancePolicy.authorize` is consulted by the command seam:
- SAFE actions are always authorized.
- A risky action in *autonomous* (non-interactive) mode is REFUSED unless
  ``governance.auto_approve`` is set, and an escalation is recorded.
- A risky action in *interactive* mode prompts the operator (the prompt callable
  is injected so tests stay deterministic and the loop never hangs on stdin).

The classifier is intentionally precise: a false positive on a normal build
command would block real builds, so patterns are anchored to the destructive
*verb* (e.g. ``rm`` with a recursive/force flag, ``delete``/``destroy``
subcommands of known tools), never to incidental substrings.
"""

import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from my_project_orchestrator.logging_setup import setup_logger

logger = setup_logger(__name__)


# Each entry: (compiled regex, human reason). Patterns match against the raw
# command string. They are anchored to a destructive verb/flag combination so
# ordinary build/test/lint commands cannot trip them. `re.IGNORECASE` is applied
# uniformly; SQL keywords and CLI verbs are case-insensitive in practice.
_RISK_RULES: Tuple[Tuple[str, str], ...] = (
    # rm with a recursive OR force flag (combined or split): rm -rf, rm -r -f,
    # rm --recursive, rm -fr. Plain `rm file` is NOT flagged (recoverable enough,
    # and far too common to gate). The flag cluster must contain r or f.
    (
        r"\brm\s+(?:-[a-z]*[rf][a-z]*\b|--recursive\b|--force\b)",
        "recursive/forced file removal (rm -rf)",
    ),
    # git push --force / -f / --force-with-lease (history rewrite is irreversible
    # on the remote). A plain `git push` is gated separately below as a publish.
    (
        r"\bgit\s+push\b.*(?:--force\b|--force-with-lease\b|\s-f\b)",
        "force push rewrites remote history",
    ),
    # plain git push (publishes to a remote; reversible but external side effect).
    (r"\bgit\s+push\b", "git push publishes to a remote"),
    # SQL destructive DDL: DROP TABLE / DROP DATABASE / DROP SCHEMA, TRUNCATE.
    (r"\bdrop\s+(?:table|database|schema)\b", "SQL DROP destroys schema/data"),
    (r"\btruncate\s+table\b", "SQL TRUNCATE deletes all rows"),
    # kubectl delete (any resource).
    (r"\bkubectl\s+delete\b", "kubectl delete removes cluster resources"),
    # terraform apply / destroy (provisions or tears down real infrastructure).
    (
        r"\bterraform\s+(?:apply|destroy)\b",
        "terraform apply/destroy mutates infrastructure",
    ),
    # Cloud CLIs with a delete/rm subcommand: aws ... rm/delete, gcloud ...
    # delete, az ... delete. Anchored to the CLI name + a delete-family verb so
    # read-only cloud commands (list/describe/get) are SAFE.
    (
        r"\baws\b[^|;&]*\b(?:rm|delete|delete-[a-z-]+|terminate-[a-z-]+)\b",
        "aws delete/terminate is destructive",
    ),
    (r"\bgcloud\b[^|;&]*\bdelete\b", "gcloud delete is destructive"),
    (r"\baz\b[^|;&]*\bdelete\b", "az delete is destructive"),
    # docker/podman image removal and prune (reclaims/destroys images/volumes).
    (r"\b(?:docker|podman)\s+rmi\b", "container image removal (rmi)"),
    (
        r"\b(?:docker|podman)\s+system\s+prune\b",
        "docker system prune deletes unused data",
    ),
    (
        r"\b(?:docker|podman)\s+volume\s+(?:rm|prune)\b",
        "container volume removal/prune",
    ),
    # Package publish (paid/irreversible release to a public registry).
    (r"\bnpm\s+publish\b", "npm publish releases a package"),
    (r"\b(?:cargo|yarn|pnpm)\s+publish\b", "package publish releases to a registry"),
    (r"\btwine\s+upload\b", "twine upload publishes to PyPI"),
    (r"\bpip\s+upload\b", "pip upload publishes a package"),
    (r"\bgh\s+release\s+create\b", "gh release create publishes a release"),
    # Deploy verbs of common tools (external side effect, often paid).
    (
        r"\b(?:vercel|netlify|fly|flyctl|wrangler|heroku)\s+deploy\b",
        "deploy command pushes to a hosting provider",
    ),
    (r"\bserverless\s+deploy\b", "serverless deploy provisions cloud resources"),
    (r"\bkubectl\s+apply\b", "kubectl apply mutates cluster state"),
    # curl/wget piped directly into a shell (executes untrusted remote code).
    (
        r"\b(?:curl|wget)\b[^|]*\|\s*(?:sudo\s+)?(?:ba)?sh\b",
        "pipe-to-shell executes untrusted remote code",
    ),
    # dd to a block device (overwrites disks).
    (r"\bdd\b[^|;&]*\bof=/dev/", "dd to a device overwrites a disk"),
    # mkfs / fdisk (formats/partitions a disk).
    (r"\bmkfs\b", "mkfs formats a filesystem"),
)

_COMPILED_RULES: Tuple[Tuple[re.Pattern, str], ...] = tuple(
    (re.compile(pat, re.IGNORECASE), reason) for pat, reason in _RISK_RULES
)


def is_risky(
    command: str, extra_patterns: Optional[List[str]] = None
) -> Tuple[bool, str]:
    """Classify ``command`` as risky (destructive/irreversible/paid) or SAFE.

    Returns ``(True, reason)`` for a risky command, ``(False, "")`` otherwise.
    Never raises: a malformed extra pattern is logged and skipped so a bad policy
    entry can never break the build loop. ``extra_patterns`` are project-supplied
    additional risky regexes (``governance.approval_required``).
    """
    if not command or not isinstance(command, str):
        return False, ""
    for rule, reason in _COMPILED_RULES:
        if rule.search(command):
            return True, reason
    for raw in extra_patterns or []:
        try:
            if re.search(raw, command, re.IGNORECASE):
                return True, f"matches policy pattern: {raw}"
        except re.error as e:
            logger.warning(f"Ignoring invalid governance pattern {raw!r}: {e}")
    return False, ""


@dataclass
class GovernanceDecision:
    """Outcome of an authorization check."""

    allowed: bool
    risky: bool
    reason: str = ""
    escalated: bool = False


@dataclass
class GovernancePolicy:
    """Approval policy consulted by the command seam when governance is ON.

    ``enabled`` mirrors ``orchestrator.governance``; when False the policy is a
    transparent no-op (everything authorized, nothing recorded), so default-off
    behavior is identical to today. ``interactive`` selects prompt-vs-block for a
    risky action. ``prompt`` is injected (defaults to a non-interactive denier)
    so the loop never blocks on stdin in tests/autonomous runs.
    """

    enabled: bool = False
    interactive: bool = False
    auto_approve: bool = False
    approval_required: List[str] = field(default_factory=list)
    audit: Optional["object"] = None  # AuditTrail-like; duck-typed to avoid a cycle
    prompt: Optional[Callable[[str, str], bool]] = None
    escalations: List[dict] = field(default_factory=list)

    def classify(self, command: str) -> Tuple[bool, str]:
        return is_risky(command, self.approval_required)

    def authorize(self, command: str, action: str = "command") -> GovernanceDecision:
        """Decide whether ``command`` may run. SAFE always allowed; risky depends
        on mode + auto_approve. Records an escalation + audit entry on refusal.

        Never raises: any audit/prompt failure degrades to a recorded decision.
        """
        if not self.enabled:
            return GovernanceDecision(allowed=True, risky=False)

        risky, reason = self.classify(command)
        if not risky:
            return GovernanceDecision(allowed=True, risky=False)

        if self.auto_approve:
            self._record("gate", command, action, reason, allowed=True, escalated=False)
            return GovernanceDecision(
                allowed=True, risky=True, reason=reason, escalated=False
            )

        if self.interactive and self.prompt is not None:
            try:
                approved = bool(self.prompt(command, reason))
            except Exception as e:  # prompt must never break the loop
                logger.warning(f"Governance prompt failed, refusing: {e}")
                approved = False
            self._record(
                "gate",
                command,
                action,
                reason,
                allowed=approved,
                escalated=not approved,
            )
            return GovernanceDecision(
                allowed=approved,
                risky=True,
                reason=reason,
                escalated=not approved,
            )

        # Autonomous (non-interactive) + no auto_approve: block and escalate.
        self._record("gate", command, action, reason, allowed=False, escalated=True)
        return GovernanceDecision(
            allowed=False, risky=True, reason=reason, escalated=True
        )

    def _record(
        self,
        kind: str,
        command: str,
        action: str,
        reason: str,
        allowed: bool,
        escalated: bool,
    ) -> None:
        if escalated:
            self.escalations.append(
                {"command": command, "action": action, "reason": reason}
            )
        if self.audit is not None:
            try:
                self.audit.record(
                    "gate",
                    action=action,
                    command=command,
                    reason=reason,
                    allowed=allowed,
                    escalated=escalated,
                )
            except Exception as e:  # audit must never break execution
                logger.debug(f"Governance audit record failed: {e}")


def policy_from_config(
    config: dict,
    interactive: bool = False,
    audit: Optional[object] = None,
    prompt: Optional[Callable[[str, str], bool]] = None,
) -> GovernancePolicy:
    """Build a :class:`GovernancePolicy` from a merged config dict.

    Reads ``orchestrator.governance`` (bool, default False) and the top-level
    ``governance`` section (``approval_required`` list, ``auto_approve`` bool).
    Never raises on a malformed section; missing/odd values degrade to defaults.
    """
    from my_project_orchestrator.config import get_setting

    enabled = bool(get_setting(config, "orchestrator", "governance"))
    gov = config.get("governance") or {}
    if not isinstance(gov, dict):
        gov = {}
    approval_required = gov.get("approval_required") or []
    if not isinstance(approval_required, list):
        approval_required = []
    auto_approve = bool(gov.get("auto_approve", False))
    return GovernancePolicy(
        enabled=enabled,
        interactive=interactive,
        auto_approve=auto_approve,
        approval_required=[str(p) for p in approval_required],
        audit=audit,
        prompt=prompt,
    )
