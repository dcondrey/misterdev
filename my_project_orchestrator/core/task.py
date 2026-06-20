import frontmatter
from pathlib import Path
from typing import List, Dict, Any, Optional

from my_project_orchestrator.core.models import Task
from my_project_orchestrator.logging_setup import setup_logger

logger = setup_logger(__name__)

class TaskManager:
    """Manages discovery, loading, and updating of tasks for a project."""
    def __init__(self, project: Any):
        self.project = project
        self.devplan_dir = self.project.config.get("devplan_dir", "devplan")
        self.tasks: Dict[str, Task] = {}

    def discover_tasks(self) -> List[Task]:
        """Scans the devplan directory for markdown files with front-matter."""
        devplan_path = self.project.path / self.devplan_dir
        if not devplan_path.exists():
            logger.warning(f"Devplan directory {devplan_path} does not exist.")
            return []

        self.tasks.clear()
        discovered_tasks = []
        for file_path in sorted(devplan_path.rglob("*.md")):
            try:
                task = self._load_task_from_file(file_path)
                if task:
                    self.tasks[task.id] = task
                    discovered_tasks.append(task)
            except Exception as e:
                logger.error(f"Error loading task from {file_path}: {e}")

        self._resolve_dependency_ids()
        if self.project.config.get("orchestrator", {}).get("auto_detect_dependencies", False):
            self._detect_file_overlaps()
        logger.info(f"Discovered {len(discovered_tasks)} tasks.")
        return discovered_tasks

    def _detect_file_overlaps(self):
        """Add implicit dependencies when tasks touch the same file.

        Two independent tasks that modify (or create-then-modify) the same file
        would conflict if run in the same wave. Chaining them by ID keeps the
        edits serialized in a deterministic order.
        """
        file_to_tasks: Dict[str, List[str]] = {}
        for task in sorted(self.tasks.values(), key=lambda t: t.id):
            for f in list(task.files_to_modify) + list(task.files_to_create):
                file_to_tasks.setdefault(f, []).append(task.id)

        added = 0
        for file_path, task_ids in file_to_tasks.items():
            if len(task_ids) < 2:
                continue
            for i in range(1, len(task_ids)):
                later = self.tasks[task_ids[i]]
                earlier_id = task_ids[i - 1]
                if earlier_id != later.id and earlier_id not in later.dependencies:
                    later.dependencies.append(earlier_id)
                    added += 1
                    logger.info(
                        f"Implicit dependency: {later.id} depends on {earlier_id} "
                        f"(both touch {file_path})"
                    )
        if added:
            logger.info(f"Detected {added} implicit dependencies from file overlaps")

    def _resolve_dependency_ids(self):
        """Resolve short dependency IDs (e.g., '001') to full task IDs (e.g., '001-posting-shard')."""
        all_ids = set(self.tasks.keys())
        for task in self.tasks.values():
            resolved = []
            for dep in task.dependencies:
                if dep in all_ids:
                    resolved.append(dep)
                else:
                    # Try prefix match: "001" matches "001-posting-shard"
                    matches = [tid for tid in all_ids if tid.startswith(dep + "-") or tid == dep]
                    if len(matches) == 1:
                        resolved.append(matches[0])
                    elif len(matches) > 1:
                        logger.warning(f"Ambiguous dependency '{dep}' in {task.id}: matches {matches}")
                        resolved.append(matches[0])
                    else:
                        logger.warning(f"Unresolved dependency '{dep}' in {task.id}")
                        resolved.append(dep)
            task.dependencies = resolved

    def _load_task_from_file(self, file_path: Path) -> Optional[Task]:
        with open(file_path, "r", encoding="utf-8") as f:
            post = frontmatter.load(f)
            
        if not post.metadata or "status" not in post.metadata:
            return None

        task_id = file_path.stem
        
        meta = post.metadata
        depends_on = meta.get("depends_on", [])
        if isinstance(depends_on, str):
            depends_on = [depends_on]

        return Task(
            id=task_id,
            description=post.content,
            type="markdown_planner",
            status=meta.get("status", "pending"),
            source_ref=str(file_path),
            project_ref=str(self.project.path),
            processor_data=meta,
            dependencies=depends_on,
            files_to_modify=meta.get("files_to_modify", []),
            files_to_create=meta.get("files_to_create", []),
            context_files=meta.get("context_files", []),
            category=meta.get("category", "feature"),
            complexity=meta.get("complexity", "medium"),
            title=meta.get("title", task_id),
        )

    def get_pending_tasks(self) -> List[Task]:
        return [t for t in self.tasks.values() if t.status == "pending"]

    def update_task_status(self, task_id: str, new_status: str):
        if task_id not in self.tasks:
            logger.warning(f"Task {task_id} not in task registry, status update skipped.")
            return

        task = self.tasks[task_id]
        task.status = new_status

        # Persist back to the markdown file. Decomposed tasks (from build())
        # have no backing file, so persistence is expected to be skipped.
        if task.source_ref:
            try:
                with open(task.source_ref, "r", encoding="utf-8") as f:
                    post = frontmatter.load(f)
                
                post.metadata['status'] = new_status
                
                with open(task.source_ref, "w", encoding="utf-8") as f:
                    f.write(frontmatter.dumps(post))
                logger.info(f"Updated task {task_id} status to {new_status}")
            except Exception as e:
                logger.error(f"Failed to persist task status for {task_id}: {e}")
