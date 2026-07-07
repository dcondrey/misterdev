"""Governance layer: risk classification + approval gates (opt-in).

Classifies a shell command (or a described action) as *risky* when it matches a
destructive, irreversible, or paid operation — ``rm -rf``, ``git push --force``,
``DROP TABLE``/``DROP DATABASE``, ``kubectl delete``, ``terraform apply``/
``destroy``, cloud-CLI ``delete``/``rm``, ``docker rmi``/``system prune``,
deploy/publish commands, and ``curl ... | sh`` pipe-to-shell. Ordinary build,
test, and lint commands (``cargo test``, ``pytest -q``, ``npm run build``,
``ruff check``, ``true``/``false``) classify as SAFE.

Design mirrors :mod:`misterdev.core.context.lsp` / ``container`` / ``mcp``:
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
import shlex
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from misterdev.logging_setup import setup_logger

logger = setup_logger(__name__)


# Classification tokenizes the command with ``shlex`` and matches destructive
# rules against a segment's argv (its command word + the tokens that follow),
# never against a raw substring. That means an identifier inside a quoted
# argument (``echo "rm -rf /"``) is an inert single token — never the command
# word — so it cannot trip a rule, and shell control operators are honored
# structurally (``safe && rm -rf x`` is two segments, the second destructive)
# rather than approximated with a ``[^|;&]*`` character class.

# Command wrappers whose FIRST argument is the real command: the destructive
# verb sits after them (``sudo rm -rf``, ``env FOO=bar kubectl delete``).
_WRAPPER_COMMANDS = frozenset(
    {"sudo", "doas", "env", "time", "nohup", "nice", "xargs", "command", "exec"}
)

# A leading ``VAR=value`` assignment precedes the command word (``FOO=bar cmd``).
_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")

# Shell control operators that end one command segment; a token that follows is
# a separate invocation. ``|`` is deliberately absent — it stays inside the
# segment so a pipe-to-shell (``curl ... | sh``) is visible as one unit. Also
# treats subshell parens as boundaries so ``( rm -rf x )`` classifies on ``rm``.
_SEGMENT_SEPARATORS = frozenset({";", "&", "&&", "||", "(", ")"})

# An ``rm`` flag cluster that requests recursive OR force removal: ``-rf``,
# ``-r``, ``-fr``, ``--recursive``, ``--force``. Plain ``rm file`` has no such
# flag and is not gated (recoverable, and far too common).
_RM_FORCE_FLAG_RE = re.compile(
    r"-[a-z]*[rf][a-z]*$|--recursive$|--force$", re.IGNORECASE
)

# Longest command slice matched against user-supplied approval_required patterns;
# bounds worst-case regex backtracking (see is_risky).
_MAX_PATTERN_INPUT = 4096


def _segment_tokens(command: str) -> List[List[str]]:
    """Split ``command`` into per-segment argv token lists, quote-aware.

    Segments break on shell control operators and newlines; a separator inside
    quotes never splits, and a quoted argument stays one token. Raises
    ``ValueError`` (from ``shlex``) on an unbalanced quote so the caller can fall
    back. ``shlex`` treats a newline as whitespace, so lines are split first;
    a backslash-newline continuation is joined so it is not mis-split.
    """
    segments: List[List[str]] = []
    joined = command.replace("\\\n", " ")
    for line in joined.splitlines() or [joined]:
        lex = shlex.shlex(line, posix=True, punctuation_chars=True)
        lex.whitespace_split = True
        lex.commenters = ""  # classify the whole line; never drop a '#...' tail
        current: List[str] = []
        for token in lex:  # may raise ValueError on unbalanced quotes
            if token in _SEGMENT_SEPARATORS:
                if current:
                    segments.append(current)
                current = []
            else:
                current.append(token)
        if current:
            segments.append(current)
    return segments


def _command_word(tokens: List[str]) -> Tuple[Optional[int], Optional[str]]:
    """Index and lowercased value of a segment's real command word.

    Skips leading ``VAR=value`` assignments and wrapper commands (``sudo``,
    ``env``, …) so the verb they front is what gets classified. Returns
    ``(None, None)`` for a segment with no command word (e.g. only operators).
    """
    for i, token in enumerate(tokens):
        if _ASSIGNMENT_RE.match(token):
            continue
        low = token.lower()
        if low in _WRAPPER_COMMANDS:
            continue
        return i, low
    return None, None


def _subcommand(rest: List[str]) -> Optional[str]:
    """The subcommand: the first non-option token after the command word.

    Most CLIs put the destructive verb in this position (``git push``, ``npm
    publish``, ``kubectl delete``), so matching here — not "anywhere in the
    args" — is what keeps ``npm run publish`` (a script named ``publish``) and
    ``git branch push`` (a branch named ``push``) SAFE. Mirrors the old regexes'
    ``\\bcmd\\s+verb\\b`` adjacency.
    """
    for token in rest:
        if not token.startswith("-"):
            return token
    return None


def _aws_destructive(token: str) -> bool:
    return (
        token in ("rm", "delete")
        or token.startswith("delete-")
        or token.startswith("terminate-")
    )


def _classify_segment(tokens: List[str]) -> Optional[str]:
    """Reason string if this segment's argv is destructive/irreversible/paid.

    Every rule keys on the command word plus the tokens that follow it, so a
    read-only invocation (``kubectl get``, ``aws s3 ls``, ``docker run --rm``)
    and any destructive-looking text sitting in a quoted argument stay SAFE.
    """
    idx, cmd = _command_word(tokens)
    if cmd is None:
        return None
    rest = [t.lower() for t in tokens[idx + 1 :]]
    sub = _subcommand(rest)

    # curl/wget piped directly into a shell (executes untrusted remote code).
    # Parity with the old rule: the FIRST pipe must reach sh/bash (optionally via
    # sudo), so ``curl x | grep | sh`` is not treated as pipe-to-shell.
    if cmd in ("curl", "wget") and "|" in tokens:
        after = tokens[tokens.index("|") + 1 :]
        j = 0
        while j < len(after) and after[j].lower() in _WRAPPER_COMMANDS:
            j += 1
        if j < len(after) and after[j].lower() in ("sh", "bash"):
            return "pipe-to-shell executes untrusted remote code"

    # rm is flag-driven, not subcommand-driven: a recursive/force flag in ANY
    # position is destructive (``rm -v -rf x``), so scan all following tokens.
    if cmd == "rm":
        if any(_RM_FORCE_FLAG_RE.match(t) for t in tokens[idx + 1 :]):
            return "recursive/forced file removal (rm -rf)"
        return None

    # Subcommand-anchored tools: the destructive verb must sit at the subcommand
    # position, so ``npm run publish`` / ``git branch push`` / ``gh pr create``
    # (verb present only as a script/branch/flag value) stay SAFE.
    if cmd == "git" and sub == "push":
        if any(f in rest for f in ("--force", "--force-with-lease", "-f")):
            return "force push rewrites remote history"
        return "git push publishes to a remote"
    if cmd == "drop" and sub in ("table", "database", "schema"):
        return "SQL DROP destroys schema/data"
    if cmd == "truncate" and sub == "table":
        return "SQL TRUNCATE deletes all rows"
    if cmd == "kubectl":
        if sub == "delete":
            return "kubectl delete removes cluster resources"
        if sub == "apply":
            return "kubectl apply mutates cluster state"
        return None
    if cmd == "terraform" and sub in ("apply", "destroy"):
        return "terraform apply/destroy mutates infrastructure"
    if cmd in ("docker", "podman"):
        if sub == "rmi":
            return "container image removal (rmi)"
        if sub == "volume" and ("rm" in rest or "prune" in rest):
            return "container volume removal/prune"
        if sub == "system" and "prune" in rest:
            return "docker system prune deletes unused data"
        return None
    if cmd == "npm" and sub == "publish":
        return "npm publish releases a package"
    if cmd in ("cargo", "yarn", "pnpm") and sub == "publish":
        return "package publish releases to a registry"
    if cmd == "twine" and sub == "upload":
        return "twine upload publishes to PyPI"
    if cmd == "pip" and sub == "upload":
        return "pip upload publishes a package"
    if cmd == "gh" and sub == "release" and "create" in rest:
        return "gh release create publishes a release"
    if cmd in ("vercel", "netlify", "fly", "flyctl", "wrangler", "heroku") and (
        sub == "deploy"
    ):
        return "deploy command pushes to a hosting provider"
    if cmd == "serverless" and sub == "deploy":
        return "serverless deploy provisions cloud resources"

    # Cloud CLIs and dd matched the verb ANYWHERE after the command in the old
    # rules (nested subcommands: ``aws s3api delete-object``), so keep that here.
    if cmd == "aws" and any(_aws_destructive(t) for t in rest):
        return "aws delete/terminate is destructive"
    if cmd == "gcloud" and "delete" in rest:
        return "gcloud delete is destructive"
    if cmd == "az" and "delete" in rest:
        return "az delete is destructive"
    if cmd == "dd" and any(t.startswith("of=/dev/") for t in rest):
        return "dd to a device overwrites a disk"
    if cmd == "mkfs" or cmd.startswith("mkfs."):
        return "mkfs formats a filesystem"
    return None


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
    try:
        segments = _segment_tokens(command)
    except ValueError:
        # Unbalanced quotes etc.: shlex cannot tokenize. Degrade to a single
        # whitespace-split segment so an obviously-destructive command word is
        # still caught, rather than silently classifying the input as SAFE.
        segments = [command.split()]
    for tokens in segments:
        reason = _classify_segment(tokens)
        if reason:
            return True, reason
    # Bound the input a user-supplied pattern matches against: an adversarial
    # (or accidentally pathological) approval_required regex can backtrack
    # catastrophically, and its cost grows with input length. Real commands are
    # short, so capping the matched slice removes the ReDoS blowup without
    # affecting classification of legitimate commands.
    probe = command[:_MAX_PATTERN_INPUT]
    for raw in extra_patterns or []:
        try:
            if re.search(raw, probe, re.IGNORECASE):
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
    from misterdev.config import get_setting

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
