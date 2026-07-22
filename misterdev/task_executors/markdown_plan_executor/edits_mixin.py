"""Edit path validation, test-tamper detection, and edit application."""

import re
from pathlib import Path
from typing import Dict, Optional, Tuple

from misterdev.core.models import Task
from misterdev.core.execution.project import Project
from misterdev.llm.responses import (
    EditConflictError,
    LLMResponseParser,
    apply_search_replace,
)
from misterdev.config import get_setting
from misterdev.utils.file_utils import write_file

from .helpers import (
    logger,
    _is_golden_path,
    _is_test_file,
    _diagnose_py_tampering,
    _diagnose_tampering,
)


def _nonblank_len(text: str) -> int:
    """Characters of a file ignoring blank lines and per-line indentation, so a
    reformat (whitespace churn) doesn't register as a size change — only real
    content does."""
    return sum(len(line.strip()) for line in text.splitlines() if line.strip())


class EditsMixin:
    def _validate_edit_paths(
        self, project: Project, task: Task, edits: Dict[str, str]
    ) -> Dict[str, str]:
        """Reject hallucinated or out-of-scope edits before they touch disk.

        Drops edits that escape the project root (absolute paths, ``..``
        traversal) or that would create empty files, and warns when the LLM
        touches files outside the task's declared scope. This is what prevents
        a misrouted edit from clobbering files outside the project.
        """
        project_root = project.path.resolve()
        expected = set(task.files_to_modify + task.files_to_create)
        golden_paths = get_setting(project.config, "orchestrator", "golden_paths")
        valid: Dict[str, str] = {}
        for path, content in edits.items():
            if ".." in Path(path).parts or Path(path).is_absolute():
                logger.error(f"Rejected edit with unsafe path: {path}")
                continue
            if _is_golden_path(path, golden_paths):
                logger.error(f"Rejected edit to protected golden file: {path}")
                continue
            full = (project.path / path).resolve()
            try:
                inside = full.is_relative_to(project_root)
            except ValueError:
                inside = False
            if not inside:
                logger.error(f"Rejected edit to path outside project root: {path}")
                continue
            if not content.strip():
                logger.warning(f"Rejected empty-content edit: {path}")
                continue
            if expected and path not in expected:
                logger.warning(f"LLM modified file outside task scope: {path}")
            valid[path] = content
        return valid

    def _detect_test_tampering(
        self, project: Project, edits: Dict[str, str]
    ) -> Optional[str]:
        """Reject edits that weaken existing test files (deterministic gate).

        For each edited TEST file that already exists on disk, compares the
        current (pre-edit) content against the proposed content. An edit that
        reduces tests/assertions or adds skip markers is tampering: the test
        gate must not be satisfied by gutting the gate. New test files and
        purely additive edits pass. Set ``orchestrator.allow_test_edits`` to
        skip the check (escape hatch); it defaults to off (check enforced).
        Must be called BEFORE ``_apply_edits`` so the on-disk content still
        reflects the pre-edit state.
        """
        if get_setting(project.config, "orchestrator", "allow_test_edits"):
            return None
        reasons = []
        for file_path, content in edits.items():
            if not _is_test_file(file_path):
                continue
            full_path = project.path / file_path
            if not full_path.exists():
                continue
            try:
                before = full_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            # Python uses precise assertion-survival diagnosis (resistant to
            # rename/move/split/merge); other languages fall back to the
            # cross-language regex totals, which can't parse structure.
            if file_path.endswith((".py", ".pyi")):
                reason = _diagnose_py_tampering(before, content)
            else:
                reason = _diagnose_tampering(before, content)
            if reason:
                reasons.append(f"{file_path} ({reason})")
        return "; ".join(reasons) if reasons else None

    def _detect_dangling_references(
        self, project: Project, edits: Dict[str, str]
    ) -> Optional[str]:
        """Reject an edit that removes/renames a symbol but leaves its callers.

        The whack-a-mole failure: an edit deletes or renames a symbol yet leaves
        references in files it didn't touch, so the build fails one missed site
        at a time until attempts run out. This catches it deterministically
        BEFORE a build cycle: for every graph symbol defined in an edited file
        whose name no longer appears in that file's new content (removed or
        renamed), flag any caller — in a file NOT part of this edit — that still
        references the old name on disk. Returns a description of the dangling
        sites, or None when the change is complete. Graph-driven, so a coincidental
        name match without a real reference edge is never flagged.
        """
        topo = getattr(project, "topography", None)
        graph = getattr(topo, "graph", None)
        symbols = getattr(graph, "symbols", None)
        if not symbols:
            return None
        edited = set(edits.keys())
        file_cache: Dict[str, str] = {}
        dangling: list[str] = []
        for sym in symbols.values():
            if sym.file_path not in edited or not sym.incoming_calls:
                continue
            # A method's defining file spells it `fn new`, never the qualified
            # `Type::new` — that form appears only at call sites. Test the edited
            # file with the unqualified definition token so an intact method is
            # not misread as "removed"; callers are still matched on the
            # qualified name below. For a free function the two are identical.
            defined_token = sym.name.rsplit("::", 1)[-1]
            if re.compile(rf"\b{re.escape(defined_token)}\b").search(
                edits.get(sym.file_path, "")
            ):
                continue  # symbol still defined in the edited file -> not removed
            word = re.compile(rf"\b{re.escape(sym.name)}\b")
            for caller_key in sym.incoming_calls:
                caller = symbols.get(caller_key)
                if caller is None or caller.file_path in edited:
                    continue  # caller lives in a file this edit already changes
                content = file_cache.get(caller.file_path)
                if content is None:
                    try:
                        content = (project.path / caller.file_path).read_text(
                            encoding="utf-8"
                        )
                    except (OSError, UnicodeDecodeError):
                        content = ""
                    file_cache[caller.file_path] = content
                if word.search(content):
                    dangling.append(
                        f"`{sym.name}` still referenced in "
                        f"{caller.file_path}:{caller.start_line}"
                    )
        if dangling:
            return "; ".join(sorted(set(dangling))[:40])
        return None

    def _detect_destructive_rewrite(
        self, project: Project, edits: Dict[str, str]
    ) -> Optional[str]:
        """Reject a likely destructive stub: an edit that COLLAPSES a source file
        AND removes definitions it used to declare.

        The observed reward-hack: told to make a failing test pass, a weak model
        deletes the real implementation (e.g. a 171-line module) and leaves an
        11-line stub that satisfies the test but strips functionality other code
        relied on. ``_detect_dangling_references`` catches this only when the graph
        recorded the removed symbol's callers, but cross-file call edges are often
        unresolved (JS especially), so the stub slips through to the late, expensive
        integration gate. This is the graph-edge-INDEPENDENT complement: flag an
        edit that BOTH (a) removes >=1 definition the file used to declare and (b)
        shrinks the file below half its non-blank size. Two conditions together, so
        a legitimate small fix, or a large rewrite that PRESERVES the public API, is
        never caught. Off via ``orchestrator.destructive_edit_guard: false``. Must
        run BEFORE edits are applied, so on-disk content is still the pre-edit
        original.
        """
        if not get_setting(project.config, "orchestrator", "destructive_edit_guard"):
            return None
        graph = getattr(getattr(project, "topography", None), "graph", None)
        symbols = getattr(graph, "symbols", None)
        if not symbols:
            return None
        defs_by_file: Dict[str, set] = {}
        for sym in symbols.values():
            defs_by_file.setdefault(sym.file_path, set()).add(sym.name)

        destructive: list[str] = []
        for path, new_content in edits.items():
            if _is_test_file(path) or path not in defs_by_file:
                continue  # tests are handled by the tamper guard; skip new files
            orig_defs = defs_by_file[path]
            if len(orig_defs) < 2:
                continue  # a single-definition file is small; rewriting it is normal
            # A method is defined as `def name`/`fn name`, never the qualified
            # `Type::name` (that form appears only at call sites), so match on the
            # unqualified token — same rule the dangling-ref guard uses.
            removed = sorted(
                name
                for name in orig_defs
                if not re.search(
                    rf"\b{re.escape(name.rsplit('::', 1)[-1])}\b", new_content or ""
                )
            )
            if not removed:
                continue  # every prior definition still present -> API preserved
            try:
                original = (project.path / path).read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            orig_size = _nonblank_len(original)
            new_size = _nonblank_len(new_content or "")
            if orig_size and new_size / orig_size < 0.5:
                pct = int(new_size / orig_size * 100)
                destructive.append(
                    f"{path} collapses to {pct}% of its size and removes "
                    f"definition(s): {', '.join(removed[:5])}"
                )
        return "; ".join(destructive) if destructive else None

    def _changed_region_mutation_check(
        self, project: Project, pre_edit: Dict[str, str], test_command: str, cwd
    ) -> None:
        """Advisory suite-strength check on a task that just completed.

        When ``orchestrator.changed_region_mutation`` is on, mutate the changed
        region of the largest edited SOURCE file and re-run the test command; log
        how many mutants the suite kills. A low score means the passing test does
        not actually verify the fix — the evasion-resistant complement to the
        shape-based ``_detect_destructive_rewrite``. Best-effort and bounded: never
        raises, never fails the task; set ``mutation.changed_region_min_score`` > 0
        to surface a weak suite as a warning.

        Falls back to the project-level ``test_command`` when the task carried no
        per-task test gate (the common no-test completion path): the project suite
        is what the integration gate runs, so it is the right thing to mutate
        against — and a task that merged WITHOUT a test gate is exactly where a
        weak suite most needs surfacing.
        """
        cmd = test_command or project.config.get("test_command")
        if not (
            cmd
            and pre_edit
            and get_setting(project.config, "orchestrator", "changed_region_mutation")
        ):
            return
        try:
            import subprocess

            from misterdev.core.execution.outcomes import RED, SKIP
            from misterdev.core.verification.changed_region_mutation import (
                run_changed_region_mutation,
            )

            mcfg = project.config.get("mutation") or {}
            floor = float(mcfg.get("changed_region_min_score", 0.0))
            cap = int(mcfg.get("changed_region_max_mutants", 12))
            run_dir = cwd or project.path

            def _runner(cmd, timeout):
                try:
                    p = subprocess.run(
                        cmd,
                        shell=True,
                        cwd=str(run_dir),
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                    )
                except subprocess.TimeoutExpired:
                    return False, ""
                return p.returncode == 0, (p.stdout or "") + (p.stderr or "")

            # Score EVERY edited non-test source file with a mutable changed region,
            # not just the largest: a multi-file fix's crux may be a small file, and
            # a weak suite there is exactly what a single-file check misses. A SKIP
            # (no mutable change, e.g. a whitespace-only or comment edit) is passed
            # over; largest-first only orders the logs.
            scored = 0
            for path in sorted(pre_edit, key=lambda p: len(pre_edit[p]), reverse=True):
                if _is_test_file(path) or not (project.path / path).exists():
                    continue
                new = (project.path / path).read_text(encoding="utf-8")
                res = run_changed_region_mutation(
                    project.path,
                    path,
                    pre_edit[path],
                    new,
                    cmd,
                    runner=_runner,
                    min_score=floor,
                    max_mutants=cap,
                )
                if res.status == SKIP:
                    continue
                scored += 1
                logger.info(f"Changed-region mutation [{path}]: {res.reason}")
                if res.status == RED:
                    logger.warning(
                        f"Weak suite for {path}: the passing test barely constrains "
                        f"the fix — {res.reason}"
                    )
            # Enabled but nothing scorable: keep the seam observable, never silent.
            if scored == 0:
                logger.info(
                    "Changed-region mutation: no mutable source change to score "
                    f"(files: {sorted(pre_edit)})"
                )
        except Exception as e:  # advisory only — never fail a completed task
            logger.debug(f"Changed-region mutation check skipped: {e}")

    def _resolve_edits(
        self, project: Project, llm_response: str
    ) -> Tuple[Dict[str, str], Optional[str]]:
        """Turn the LLM response into ``{path: full_content}`` edits.

        Prefers surgical SEARCH/REPLACE hunks: each is applied against the
        current on-disk file so the model never has to reproduce a large file
        in full (the whole-file path truncates past the output-token limit).
        The resolved full content then flows through the same path/syntax/
        tamper gates as before. Returns ``(edits, error)``; a non-empty error
        means a hunk did not apply and the attempt should retry rather than
        write a partial file. Falls back to the whole-file parser when the
        response contains no SEARCH/REPLACE markers.
        """
        sr_edits = LLMResponseParser.parse_search_replace_blocks(llm_response)
        if not sr_edits:
            return LLMResponseParser.parse_file_edits(llm_response), None
        by_path: Dict[str, list] = {}
        for edit in sr_edits:
            by_path.setdefault(edit.path, []).append(edit)
        resolved: Dict[str, str] = {}
        for path, hunks in by_path.items():
            full_path = project.path / path
            try:
                original = (
                    full_path.read_text(encoding="utf-8") if full_path.exists() else ""
                )
            except (UnicodeDecodeError, OSError) as exc:
                return {}, f"{path}: could not read file to apply edit ({exc})"
            try:
                resolved[path] = apply_search_replace(original, hunks)
            except EditConflictError as exc:
                return {}, str(exc)
        return resolved, None

    def _apply_edits(self, project: Project, edits: Dict[str, str]):
        """Apply a batch of full-file edits atomically.

        Writes are all-or-nothing: the pre-edit content of every file is
        snapshotted first, and if any write raises, the already-written files
        are rolled back to their snapshot (and newly-created files removed)
        before the error propagates. Without this, a failure on file N left
        files 1..N-1 written — a partial, inconsistent tree that only a
        ``BudgetExceededError`` revert would clean up. Reuses ``write_file`` for
        the writes rather than introducing a second write path.
        """
        snapshots: Dict[Path, Optional[str]] = {}
        for file_path in edits:
            full_path = project.path / file_path
            try:
                snapshots[full_path] = (
                    full_path.read_text(encoding="utf-8")
                    if full_path.exists()
                    else None
                )
            except (UnicodeDecodeError, OSError):
                # Unreadable pre-image (binary/permission): treat as "cannot
                # safely roll back", so leave it out of the snapshot and let a
                # failed write on it surface as-is.
                snapshots[full_path] = None
        written: list[Path] = []
        try:
            for file_path, content in edits.items():
                full_path = project.path / file_path
                write_file(full_path, content)
                written.append(full_path)
        except Exception:
            for full_path in written:
                original = snapshots.get(full_path)
                try:
                    if original is None:
                        full_path.unlink(missing_ok=True)
                    else:
                        write_file(full_path, original)
                except OSError as restore_err:
                    logger.error(
                        f"Failed to roll back {full_path} after a partial edit: "
                        f"{restore_err}"
                    )
            raise

    def _run_formatters(self, project: Project, files):
        # Per-file formatters substitute {path}; project-wide formatters run
        # once. Pass each file only to formatters whose command templates use
        # {path}; otherwise invoke the formatter a single time.
        file_list = list(files)
        for tool_name, tool in project.tool_manager.tools.items():
            if getattr(tool, "type", None) != "formatter":
                continue
            template = (
                getattr(tool, "config", {}).get("command", "")
                if hasattr(tool, "config")
                else ""
            )
            if "{path}" in template:
                for file_path in file_list:
                    tool.execute(project, file_path=file_path)
            else:
                tool.execute(project)
