"""Requirements preflight: review a plan and gather the inputs it will need first.

The walk-away goal is "start it, come back to a finished project." The thing that
breaks that is a task discovering mid-run it needs something only the user can
supply — a credential, a cloud account, a decision — and parking (which parks its
dependents too). This module reviews the WHOLE plan up front and produces the
consolidated, deduplicated list of such inputs, each tagged with which tasks need
it and its FAN-OUT (how many tasks transitively depend on those). The run can then
surface everything at once (``.orchestrator/REQUIREMENTS.md``) and — under the
"smart gate" — stop before spending only when a MISSING input would cascade widely
(a foundational task), while letting late/leaf needs proceed and park.

A deterministic heuristic scan is the reliable base (credential/env-var names,
real-deploy / publish signals); an optional LLM pass enriches it. Everything is
pure except the env check and the file I/O in :class:`RequirementsBook`.
"""

import os
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional

from misterdev.logging_setup import setup_logger
from misterdev.utils.file_utils import atomic_write

logger = setup_logger(__name__)

# An env-var / secret NAME: UPPER_SNAKE ending in a credential word. High-signal
# and low false-positive (a real assertion or type error never contains one).
_ENV_RE = re.compile(r"\b([A-Z][A-Z0-9_]*(?:TOKEN|KEY|SECRET|PASSWORD|CREDENTIALS?))\b")

# Account/service signals: a task that genuinely needs a live account (not a local
# dry-run). ``(key, pattern, summary, how_to_provide)``.
_ACCOUNT_SIGNALS = (
    (
        "CLOUDFLARE_ACCOUNT",
        # real deploy or REMOTE d1 — a local `wrangler ... --dry-run` / local D1 is
        # excluded (it needs no account), which is how well-formed plans gate.
        re.compile(
            r"wrangler\s+deploy(?!\s+--dry-run)|deploy to cloudflare|deploy button|"
            r"migrate:remote|d1\b[^\n]*--remote|migrations\s+apply[^\n]*--remote",
            re.I,
        ),
        "a Cloudflare account (for a real deploy or remote-D1 migration)",
        "run `wrangler login`, or set CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID",
    ),
    (
        "NPM_TOKEN",
        re.compile(
            r"npm\s+publish|publish[^\n]*npm|npm\s+registry|\bNPM_TOKEN\b", re.I
        ),
        "an npm auth token (to publish the package)",
        "set NPM_TOKEN to an npmjs.com automation/access token",
    ),
)


def scan_requirements(tasks) -> List[Dict]:
    """Heuristic pass: the inputs the plan's text implies, deduplicated by key.

    Each item: ``{key, kind, summary, how_to_provide, task_ids}`` where kind is
    ``env`` (a named secret) or ``account`` (a live service)."""
    reqs: Dict[str, Dict] = {}

    def add(key, kind, summary, how, task_id):
        r = reqs.setdefault(
            key,
            {
                "key": key,
                "kind": kind,
                "summary": summary,
                "how_to_provide": how,
                "task_ids": [],
            },
        )
        r["task_ids"].append(task_id)

    for t in tasks:
        text = " ".join(
            str(x or "")
            for x in (
                getattr(t, "title", ""),
                getattr(t, "description", ""),
                getattr(t, "acceptance_criteria", ""),
            )
        )
        for name in set(_ENV_RE.findall(text)):
            add(
                name,
                "env",
                f"the secret `{name}`",
                f"set {name} in your environment before the run",
                t.id,
            )
        for key, pattern, summary, how in _ACCOUNT_SIGNALS:
            if pattern.search(text):
                add(key, "account", summary, how, t.id)

    for r in reqs.values():
        r["task_ids"] = sorted(set(r["task_ids"]))
    return sorted(reqs.values(), key=lambda r: r["key"])


def check_satisfied(req: Dict) -> bool:
    """True when a requirement is already met. Env/secret: the variable is set in
    the environment. Account/decision requirements are never auto-satisfied — they
    need the user (a login or an answer)."""
    if req.get("kind") == "env":
        return bool(os.environ.get(req.get("key", "")))
    return False


def fanout(task_ids: List[str], tasks) -> int:
    """How many tasks (beyond the named ones) transitively depend on ``task_ids`` —
    the blast radius if those tasks can't run. A leaf need is 0; a foundational one
    is large. Pure over the tasks' declared dependencies."""
    seed = set(task_ids)
    dependents = {t.id: set() for t in tasks}
    for t in tasks:
        for dep in getattr(t, "dependencies", []) or []:
            if dep in dependents:
                dependents[dep].add(t.id)
    seen: set = set()
    frontier = set(seed)
    while frontier:
        nxt: set = set()
        for tid in frontier:
            for child in dependents.get(tid, ()):
                if child not in seen and child not in seed:
                    seen.add(child)
                    nxt.add(child)
        frontier = nxt
    return len(seen)


def gating_requirements(reqs: List[Dict], tasks, threshold: int = 3) -> List[Dict]:
    """The MISSING requirements whose cascade justifies stopping the run before it
    spends: an unsatisfied ``account`` need (a login the user can't work around)
    whose tasks gate at least ``threshold`` downstream tasks. Env-var needs are
    advisory (a build often doesn't need a secret's value to write code), so they
    never gate on their own."""
    out = []
    for r in reqs:
        if r.get("satisfied") or r.get("answered"):
            continue
        if r.get("kind") != "account":
            continue
        if fanout(r.get("task_ids", []), tasks) >= threshold:
            out.append(r)
    return out


def review_requirements(
    tasks, llm: Optional[Callable[[str, str], str]] = None
) -> List[Dict]:
    """The full requirement list: the heuristic scan, optionally enriched by one
    LLM pass over the plan. The LLM may add non-obvious needs (an OAuth app, a DNS
    record) and MUST return the same schema; any failure falls back to the scan, so
    this never raises and never blocks."""
    reqs = scan_requirements(tasks)
    if llm is None:
        return _mark_satisfied(reqs)
    try:
        catalog = "\n".join(
            f"- {t.id}: {getattr(t, 'title', '') or getattr(t, 'description', '')[:80]}"
            f" | done: {getattr(t, 'acceptance_criteria', '')[:100]}"
            for t in tasks
        )[:12000]
        system = (
            "You review a build plan and list ONLY the external inputs the USER "
            "must provide for it to COMPLETE — credentials, cloud accounts, API "
            "tokens, or decisions. Exclude anything the model can do itself or that "
            "is only needed to RUN the finished app. Output a JSON array; each item: "
            '{"key","kind"(env|account|decision),"summary","how_to_provide",'
            '"task_ids"(list)}. Invent nothing.'
        )
        raw = llm(catalog, system)
        extra = _parse_llm_reqs(raw)
        merged = {r["key"]: r for r in reqs}
        for r in extra:
            if r.get("key"):
                merged.setdefault(r["key"], r)
        reqs = sorted(merged.values(), key=lambda r: r["key"])
    except Exception as e:  # review enrichment is best-effort
        logger.warning(f"LLM requirements review skipped ({e}); using heuristic scan.")
    return _mark_satisfied(reqs)


def _mark_satisfied(reqs: List[Dict]) -> List[Dict]:
    for r in reqs:
        r["satisfied"] = check_satisfied(r)
    return reqs


def _parse_llm_reqs(raw: str) -> List[Dict]:
    from misterdev.llm.responses import extract_json_array

    try:
        data = extract_json_array(raw)
    except Exception:
        return []
    out = []
    for d in data or []:
        if isinstance(d, dict) and d.get("key"):
            out.append(
                {
                    "key": str(d["key"]),
                    "kind": str(d.get("kind", "decision")),
                    "summary": str(d.get("summary", "")),
                    "how_to_provide": str(d.get("how_to_provide", "")),
                    "task_ids": [str(x) for x in (d.get("task_ids") or [])],
                }
            )
    return out


_PLACEHOLDER = "_(provide this, or write your decision here)_"


class RequirementsBook:
    """Reads/writes ``.orchestrator/REQUIREMENTS.md`` — the upfront input checklist."""

    def __init__(self, orchestrator_dir: Path):
        self.dir = Path(orchestrator_dir)
        self.md_path = self.dir / "REQUIREMENTS.md"

    def load_answers(self) -> Dict[str, str]:
        """``{req_key: answer}`` for any decision the user typed under an Answer line."""
        answers: Dict[str, str] = {}
        if not self.md_path.exists():
            return answers
        try:
            lines = self.md_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return answers
        current = ""
        for line in lines:
            if line.startswith("## "):
                current = line[3:].split("—", 1)[0].strip().strip("`")
                continue
            low = line.strip().lower()
            if current and low.startswith("- answer:"):
                ans = line.split(":", 1)[1].strip().strip("*_ ")
                if ans and ans != _PLACEHOLDER.strip("*_ "):
                    answers[current] = ans
        return answers

    def write(self, reqs: List[Dict]) -> None:
        """Write the checklist, preserving any answers already typed. No-op when
        there are no requirements."""
        if not reqs:
            return
        existing = self.load_answers()
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            blocks = [
                "# Inputs this build will need\n\n"
                "Reviewed from the plan up front. Provide the missing items — for a "
                "credential, set it in your environment; for a decision, replace the "
                "Answer placeholder — then run the same command. Satisfied items need "
                "nothing.\n"
            ]
            for r in reqs:
                mark = "satisfied" if r.get("satisfied") else "MISSING"
                tids = ", ".join(r.get("task_ids", [])[:8])
                head = f"## {r['key']} — {r.get('summary', '')}"
                lines = [head, f"- Status: {mark}", f"- Needed by: {tids or '(plan)'}"]
                if r.get("how_to_provide"):
                    lines.append(f"- Provide: {r['how_to_provide']}")
                if not r.get("satisfied"):
                    lines.append(f"- Answer: {existing.get(r['key'], _PLACEHOLDER)}")
                blocks.append("\n".join(lines))
            atomic_write(self.md_path, "\n\n".join(blocks) + "\n")
        except OSError as e:
            logger.warning(f"Could not write REQUIREMENTS.md ({e}).")
