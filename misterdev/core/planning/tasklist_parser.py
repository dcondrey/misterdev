"""Robust parser for an EXTERNAL, arbitrarily-formatted task list.

misterdev already executes a task graph with dependency-aware parallel waves,
progress-based resume, and active-task tracking (see ``agent.run_project`` +
``topological_sort`` + ``ProgressTracker``). The missing piece this module fills
is INGESTION: turn a hand-written task list — in whatever shape the author chose
— into normalized :class:`~misterdev.core.models.Task` objects that engine
accepts, so a dependency table in the list unlocks the parallel execution that is
already built.

Supported shapes, deterministically:
  - **JSON / YAML**: a task array, ``{"tasks": [...]}``, or phased
    ``{"phases": [{"name": ..., "tasks": [...]}]}``. Field names are alias-mapped
    (``success_criteria``/``done_when`` -> acceptance, ``blocked_by``/``requires``
    -> dependencies, ``relevant_files``/``files`` -> files_to_modify, ...).
  - **Markdown**: phases from headings, tasks from headings or list items
    (ordered, unordered, or ``- [ ]`` checkboxes), multi-line tasks whose
    sub-bullets/`key: value` lines carry files / criteria / dependencies, and a
    **dependency table** (``| Task | Blocked By |``) merged into the graph.
  - **Plain text**: one task per line, or indented blocks; inline ``key: value``.

Anything the deterministic pass cannot confidently structure is handed to an
optional LLM normalizer (``llm`` callable) which re-emits the same task schema —
so "any format" degrades gracefully rather than failing. Dependency references
(a number, a title, or an id) are resolved to real task ids at the end.

Pure and side-effect free: give it text, get Tasks. Never raises on malformed
input — it returns whatever it could parse (possibly empty).
"""

import json
import re
from typing import Any, Callable, Dict, List, Optional

import yaml

from misterdev.core.models import Task
from misterdev.llm.responses import extract_json_array
from misterdev.logging_setup import setup_logger
from misterdev.utils.file_utils import safe_ref_slug

logger = setup_logger(__name__)

# --- field alias maps: authors name things many ways; normalize to one shape ---
_ALIASES: Dict[str, tuple] = {
    "id": ("id", "task_id", "key", "ref", "number", "num"),
    "title": ("title", "name", "task", "summary", "label"),
    "description": ("description", "details", "body", "notes", "desc", "detail"),
    "acceptance_criteria": (
        "acceptance_criteria",
        "success_criteria",
        "acceptance",
        "criteria",
        "done_when",
        "definition_of_done",
        "dod",
        "success",
        "verify",
        "completion",
        "verification",
        "validation",
    ),
    "files_to_modify": (
        "files_to_modify",
        "files",
        "relevant_files",
        "modify",
        "edit",
        "target_files",
        "paths",
    ),
    "files_to_create": ("files_to_create", "create", "new_files", "creates"),
    "context_files": ("context_files", "context", "references", "reference_files"),
    "dependencies": (
        "dependencies",
        "depends_on",
        "deps",
        "blocked_by",
        "requires",
        "after",
        "needs",
        "prerequisites",
        "prereqs",
    ),
    "category": ("category", "kind", "type", "group"),
    "complexity": ("complexity", "effort", "size", "estimate"),
    "phase": ("phase", "stage", "milestone", "epic"),
}

# key: value / key - value / "**key**:" lines inside a markdown/text task body.
_ATTR_RE = re.compile(
    r"^\s*[-*+]?\s*\**\s*"
    r"(id|task[_ ]?id|files?(?:\s+to\s+(?:modify|edit|create))?|relevant\s+files?|"
    r"context(?:\s+files?)?|success\s+criteria|acceptance(?:\s+criteria)?|criteria|"
    r"done[_ ]when|completion|verification|validation|description|details|notes|"
    r"depends?(?:\s+on)?|blocked\s+by|requires?|needs|after|"
    r"priority|category|complexity|effort|phase)"
    # trailing ``\**`` on BOTH sides of the separator: authors write ``**Key:**``
    # (closing stars after the colon) as often as ``**Key**:``.
    r"\**\s*[:=]\**\s*(.+)$",
    re.IGNORECASE,
)
_LIST_ITEM_RE = re.compile(r"^(\s*)(?:[-*+]|\d+[.)])\s+(?:\[[ xX~-]?\]\s+)?(.+)$")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")
_PHASE_WORD_RE = re.compile(
    r"\b(phase|stage|milestone|epic|part|section|wave|sprint|iteration)\b",
    re.IGNORECASE,
)
_SPLIT_REFS_RE = re.compile(r"[,;]| and ")  # NOT '/': it is a path separator


def _pick(d: Dict[str, Any], field: str) -> Any:
    """First present alias value for a normalized field, else None."""
    lowered = {str(k).lower().replace(" ", "_"): v for k, v in d.items()}
    for alias in _ALIASES[field]:
        if alias in lowered and lowered[alias] not in (None, "", []):
            return lowered[alias]
    return None


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [p.strip() for p in _SPLIT_REFS_RE.split(str(value)) if p.strip()]


def _record_to_task(rec: Dict[str, Any], project_ref: str, index: int) -> Task:
    """One normalized record -> a Task. ``dependencies`` stay as RAW refs here;
    :func:`_resolve_dependencies` maps them to ids once every task exists."""
    tid = safe_ref_slug(str(_pick(rec, "id") or ""), fallback=f"T-{index + 1:03d}")
    title = str(_pick(rec, "title") or "").strip()
    description = str(_pick(rec, "description") or title).strip()
    phase = _pick(rec, "phase")
    processor = {k: v for k, v in rec.items()}
    if phase:
        processor["phase"] = str(phase)
    return Task(
        id=tid,
        title=title,
        description=description or title or tid,
        acceptance_criteria=str(_pick(rec, "acceptance_criteria") or "").strip(),
        files_to_modify=_as_list(_pick(rec, "files_to_modify")),
        files_to_create=_as_list(_pick(rec, "files_to_create")),
        context_files=_as_list(_pick(rec, "context_files")),
        dependencies=_as_list(_pick(rec, "dependencies")),  # raw; resolved later
        complexity=str(_pick(rec, "complexity") or "medium").strip().lower(),
        category=str(_pick(rec, "category") or "feature").strip().lower(),
        type="markdown_planner",
        status="pending",
        project_ref=project_ref,
        processor_data=processor,
    )


# ============================================================================
# Structured (JSON / YAML)
# ============================================================================
def _records_from_structured(obj: Any) -> List[Dict[str, Any]]:
    """Flatten a parsed JSON/YAML doc into a flat list of task records,
    tagging each with its phase when the doc is organised into phases."""
    if isinstance(obj, list):
        return [r for r in obj if isinstance(r, dict)]
    if not isinstance(obj, dict):
        return []
    # phased: {"phases": [{"name": ..., "tasks": [...]}]}
    phases = _pick(obj, "phase") if "phases" not in obj else obj.get("phases")
    if isinstance(phases, list):
        out: List[Dict[str, Any]] = []
        for ph in phases:
            if not isinstance(ph, dict):
                continue
            name = _pick(ph, "title") or _pick(ph, "phase") or ""
            for t in ph.get("tasks") or []:
                if isinstance(t, dict):
                    out.append({**t, "phase": t.get("phase", name)})
        if out:
            return out
    for key in ("tasks", "items", "todo", "todos", "list"):
        if isinstance(obj.get(key), list):
            return [r for r in obj[key] if isinstance(r, dict)]
    return []


# ============================================================================
# Markdown
# ============================================================================
def _extract_dependency_table(text: str) -> Dict[str, List[str]]:
    """Parse a markdown dependency table (``| Task | Blocked By |``) into
    ``{task_ref: [blocking_refs]}``. Returns {} when no such table is present."""
    deps: Dict[str, List[str]] = {}
    rows = [ln for ln in text.splitlines() if ln.strip().startswith("|")]
    if len(rows) < 2:
        return deps
    header = [c.strip().lower() for c in rows[0].strip("|").split("|")]
    dep_col = next(
        (
            i
            for i, h in enumerate(header)
            if any(
                k in h
                for k in ("blocked", "depends", "dependency", "requires", "after")
            )
        ),
        None,
    )
    task_col = next(
        (
            i
            for i, h in enumerate(header)
            if any(
                k in h for k in ("task", "id", "step", "feature", "item", "name", "#")
            )
        ),
        None,
    )
    if dep_col is None:
        return deps
    if task_col is None or task_col == dep_col:
        # No recognizable task header (or it collided with the dep column):
        # take the first column that is not the dependency column rather than
        # blindly keying on index 0 and manufacturing a self-edge.
        task_col = next((i for i in range(len(header)) if i != dep_col), None)
    if task_col is None:
        return deps
    for row in rows[1:]:
        cells = [c.strip() for c in row.strip("|").split("|")]
        if len(cells) <= max(task_col, dep_col) or set(cells[task_col]) <= {
            "-",
            ":",
            " ",
        }:
            continue
        ref = cells[task_col].strip("*` ")
        raw = cells[dep_col].strip("*` ")
        if ref and raw and raw.lower() not in ("-", "–", "—", "none", "n/a", ""):
            deps[ref] = _as_list(raw)
    return deps


# A leading task id on a heading/item title: "T004 — ...", "T-001 - ...",
# "Task 12: ...". The separator class includes en/em dashes (– —), the common
# devplan style, so an id is captured (and stripped) rather than left in the title.
# The trailing ``[a-z]?`` admits sub-task ids like ``T062a`` / ``T062b`` (a common
# way to split one numbered task) so both halves parse instead of being dropped.
_TASK_ID_RE = re.compile(
    r"^(task\s*\d+[a-z]?|T-?\d+[a-z]?)\s*[:.)\-–—]+\s*", re.IGNORECASE
)
# A trailing inline dependency note on a heading title: "... (after T015)",
# "... (depends on T7, T8)". Lets a plan that states edges in the heading — not
# only in a table — still contribute them.
_INLINE_DEPS_RE = re.compile(
    r"\(\s*(?:after|depends?(?:\s+on)?|blocked\s+by|requires?|needs)\b[:\s]*"
    r"([^)]+)\)\s*$",
    re.IGNORECASE,
)


def _split_task_id(title: str) -> tuple:
    """(id, remaining_title) — split a leading task id off a title, or (None, title)."""
    m = _TASK_ID_RE.match(title or "")
    if not m:
        return None, (title or "").strip()
    return m.group(1).strip(), title[m.end() :].strip()


def _extract_inline_deps(title: str) -> tuple:
    """(clean_title, [raw_dep_refs]) — pull a trailing "(after ...)" note and a
    trailing parallel marker ("‖ parallel") off a title."""
    deps: List[str] = []
    t = title or ""
    m = _INLINE_DEPS_RE.search(t)
    if m:
        deps = _as_list(m.group(1))
        t = t[: m.start()].strip()
    t = re.sub(r"\s*‖.*$", "", t).strip()  # drop a trailing "‖ parallel" marker
    return t, deps


def _new_task_record(raw_title: str, phase: str) -> Dict[str, Any]:
    tid, rest = _split_task_id(raw_title)
    title, deps = _extract_inline_deps(rest)
    rec: Dict[str, Any] = {"title": title, "phase": phase}
    if tid:
        rec["id"] = tid
    if deps:
        rec["dependencies"] = deps
    return rec


def _records_from_markdown(text: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    current_phase = ""
    cur: Optional[Dict[str, Any]] = None

    def flush():
        nonlocal cur
        if cur is not None:
            records.append(cur)
            cur = None

    lines = text.splitlines()
    # "Task-id mode": when the doc numbers its tasks (T001, Task 3), ONLY id'd
    # headings are tasks. Non-id headings (a "Global Conventions" preamble, a
    # trailing "Dependency Table" section) and their bullets are then context, not
    # phantom tasks — and everything before the first task/phase is captured as a
    # shared preamble injected into every task.
    has_ids = any(
        _split_task_id(_HEADING_RE.match(ln).group(2))[0]
        for ln in lines
        if _HEADING_RE.match(ln)
    )
    preamble: List[str] = []
    in_preamble = has_ids
    in_table = False
    for line in lines:
        if line.strip().startswith("|"):  # dependency/other table — handled apart
            in_table = True
            continue
        in_table = False
        heading = _HEADING_RE.match(line)
        if heading:
            title = heading.group(2).strip()
            is_phase = bool(_PHASE_WORD_RE.search(title)) and len(heading.group(1)) <= 2
            tid = _split_task_id(title)[0]
            if is_phase:
                flush()
                in_preamble = False
                current_phase = title
                continue
            if has_ids and not tid:
                # A non-task section heading in a numbered plan: preamble (before
                # the first task) or a trailing note (after) — never a task.
                flush()
                if in_preamble:
                    preamble.append(line)
                continue
            flush()
            in_preamble = False
            cur = _new_task_record(title, current_phase)
            continue
        if in_preamble:
            preamble.append(line)
            continue
        item = _LIST_ITEM_RE.match(line)
        attr = _ATTR_RE.match(line)
        if attr and cur is not None:
            _apply_attr(cur, attr.group(1), attr.group(2))
            continue
        if item:
            indent = len(item.group(1))
            body = item.group(2).strip()
            inner = _ATTR_RE.match(body)
            if inner and cur is not None and indent > 0:
                _apply_attr(cur, inner.group(1), inner.group(2))
                continue
            if has_ids:
                # In a numbered plan a bullet is task detail, not a sibling task:
                # fold it into the description (markdown emphasis stripped).
                if cur is not None:
                    cur["description"] = (
                        cur.get("description", "") + " " + re.sub(r"\*+", "", body)
                    ).strip()
                continue
            flush()
            cur = _new_task_record(body, current_phase)
            continue
        if cur is not None and line.strip() and not in_table:
            cur["description"] = (
                cur.get("description", "") + " " + line.strip()
            ).strip()
    flush()

    shared = "\n".join(preamble).strip()
    if shared:
        for rec in records:
            rec.setdefault("shared_context", shared)
    return records


def _clean_path(p: str) -> str:
    """One file path from a prose list: drop a trailing annotation like ``(new)`` /
    ``(generated)``, every backtick (a path never contains one — so a quoted path
    followed by sentence punctuation like ``\\`stats.ts\\`.`` doesn't strand an
    inner backtick), surrounding quotes, and trailing prose punctuation."""
    p = re.sub(r"\s*\([^)]*\)\s*$", "", p.strip())
    p = p.replace("`", "").strip().strip("'\"")
    return p.rstrip(" .,;:)").strip()


def _apply_attr(rec: Dict[str, Any], key: str, value: str) -> None:
    k = key.lower().replace(" ", "_")
    # Keep the raw (backtick-preserving) text for acceptance: stripping only the
    # OUTER backticks off a value like "`a` = 0; `b`" corrupts it into an
    # unbalanced string that later runs as a broken shell command. Other keys
    # clean per-item (files via _clean_path, deps via resolution).
    raw = value.strip()
    value = raw.strip("`")
    if k.startswith("file") or "relevant_file" in k:
        default = "files_to_create" if "create" in k else "files_to_modify"
        for raw in _as_list(value):
            # A per-path "(new)"/"(create)" annotation routes that path to
            # files_to_create even under a generic "Files:" key.
            new = any(w in raw.lower() for w in ("(new", "(create", "(generated"))
            target = "files_to_create" if new else default
            path = _clean_path(raw)
            if path:
                rec.setdefault(target, [])
                rec[target] = _as_list(rec[target]) + [path]
    elif k.startswith("context"):
        rec["context_files"] = _as_list(rec.get("context_files")) + _as_list(value)
    elif any(w in k for w in ("depend", "blocked", "require", "need", "after")):
        rec["dependencies"] = _as_list(rec.get("dependencies")) + _as_list(value)
    elif any(
        w in k
        for w in (
            "success",
            "acceptance",
            "criteria",
            "done_when",
            "completion",
            "verif",
            "validation",
        )
    ):
        rec["acceptance_criteria"] = raw
    elif k.startswith("desc") or k in ("details", "notes", "body"):
        rec["description"] = (rec.get("description", "") + " " + value).strip()
    elif k in ("id", "task_id"):
        rec["id"] = value
    elif k in ("category", "priority", "complexity", "effort", "phase"):
        rec[k] = value


def _records_from_text(text: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None
    for line in text.splitlines():
        if not line.strip():
            continue
        attr = _ATTR_RE.match(line)
        indented = line[0] in " \t"
        if attr and cur is not None and indented:
            _apply_attr(cur, attr.group(1), attr.group(2))
            continue
        item = _LIST_ITEM_RE.match(line)
        title = item.group(2).strip() if item else line.strip()
        if cur is not None:
            records.append(cur)
        cur = _new_task_record(title, "")
    if cur is not None:
        records.append(cur)
    return records


# ============================================================================
# Dependency resolution
# ============================================================================
def _resolve_dependencies(tasks: List[Task]) -> None:
    """Map each raw dependency ref (a number, a title, or an id) to a real task
    id, in place. Unresolvable refs are dropped with a warning."""
    by_id = {t.id: t.id for t in tasks}
    by_slug = {safe_ref_slug(t.id, fallback=""): t.id for t in tasks}
    by_title = {t.title.lower().strip(): t.id for t in tasks if t.title}
    by_ordinal = {str(i + 1): t.id for i, t in enumerate(tasks)}

    def resolve(ref: str) -> Optional[str]:
        r = ref.strip().strip("*`.# ")
        for table in (by_id, by_slug, by_ordinal):
            if r in table:
                return table[r]
        slug = safe_ref_slug(r, fallback="")
        if slug in by_slug:
            return by_slug[slug]
        if r.lower() in by_title:
            return by_title[r.lower()]
        # "task 3" / "T-003" -> 3 / 003
        m = re.search(r"(\d+)", r)
        if m and str(int(m.group(1))) in by_ordinal:
            return by_ordinal[str(int(m.group(1)))]
        return None

    for t in tasks:
        resolved, dropped = [], []
        for ref in t.dependencies:
            rid = resolve(ref)
            if rid and rid != t.id and rid not in resolved:
                resolved.append(rid)
            elif not rid:
                dropped.append(ref)
        if dropped:
            logger.warning(f"Task {t.id}: unresolved dependencies dropped: {dropped}")
        t.dependencies = resolved


# ============================================================================
# Public API
# ============================================================================
def detect_format(source_name: str, text: str) -> str:
    ext = source_name.rsplit(".", 1)[-1].lower() if "." in source_name else ""
    if ext in ("json",):
        return "json"
    if ext in ("yaml", "yml"):
        return "yaml"
    if ext in ("md", "markdown"):
        return "markdown"
    if ext in ("txt", "text"):
        return "text"
    stripped = text.lstrip()
    if stripped.startswith(("[", "{")):
        return "json"
    lines = text.splitlines()
    if (
        any(_HEADING_RE.match(ln) or _LIST_ITEM_RE.match(ln) for ln in lines)
        or "|" in text
    ):
        return "markdown"
    return "text"


def parse_task_list(
    text: str,
    source_name: str,
    project_ref: str,
    llm: Optional[Callable[[str, str], str]] = None,
) -> List[Task]:
    """Parse an arbitrary-format task list into resolved :class:`Task` objects.

    ``llm`` (optional ``(prompt, system) -> reply``) is the normalization
    fallback used only when the deterministic pass yields nothing usable.
    """
    fmt = detect_format(source_name, text)
    records: List[Dict[str, Any]] = []
    dep_table: Dict[str, List[str]] = {}
    try:
        if fmt == "json":
            records = _records_from_structured(json.loads(text))
        elif fmt == "yaml":
            records = _records_from_structured(yaml.safe_load(text))
        elif fmt == "markdown":
            records = _records_from_markdown(text)
            dep_table = _extract_dependency_table(text)
        else:
            records = _records_from_text(text)
    except (json.JSONDecodeError, yaml.YAMLError, ValueError) as e:
        logger.warning(f"Structured parse of {source_name} failed ({e}); trying LLM.")

    records = [r for r in records if _pick(r, "title") or _pick(r, "description")]
    if not records and llm is not None:
        records = _records_from_llm(text, llm)
    if not records:
        logger.error(f"No tasks parsed from {source_name}.")
        return []

    tasks = [_record_to_task(r, project_ref, i) for i, r in enumerate(records)]
    # Merge a dependency table by task title / id / ordinal.
    if dep_table:
        _merge_dependency_table(tasks, dep_table)
    _resolve_dependencies(tasks)
    logger.info(
        f"Parsed {len(tasks)} task(s) from {source_name} ({fmt}); "
        f"{sum(1 for t in tasks if t.dependencies)} with dependencies."
    )
    return tasks


def _merge_dependency_table(tasks: List[Task], table: Dict[str, List[str]]) -> None:
    index = {}
    for i, t in enumerate(tasks):
        index[str(i + 1)] = t
        index[t.id.lower()] = t
        if t.title:
            index[t.title.lower().strip()] = t
    for ref, blockers in table.items():
        key = ref.lower().strip()
        m = re.search(r"(\d+)", ref)
        task = index.get(key) or (index.get(str(int(m.group(1)))) if m else None)
        if task is not None:
            task.dependencies = list(dict.fromkeys([*task.dependencies, *blockers]))


_LLM_SYSTEM = (
    "You normalize a task list into strict JSON. Output ONLY a JSON array; each "
    "element has: id, title, description, acceptance_criteria, files_to_modify "
    "(list), files_to_create (list), context_files (list), dependencies (list of "
    "referenced task ids/titles), category, complexity. Preserve every task and "
    "every stated dependency; invent nothing."
)


def _records_from_llm(
    text: str, llm: Callable[[str, str], str]
) -> List[Dict[str, Any]]:
    try:
        reply = llm(
            f"Normalize this task list to the JSON schema:\n\n{text}", _LLM_SYSTEM
        )
    except Exception as e:  # normalization is best-effort
        logger.warning(f"LLM task-list normalization failed: {e}")
        return []
    arr = extract_json_array((reply or "").strip(), default=None)
    return [r for r in arr if isinstance(r, dict)] if isinstance(arr, list) else []
