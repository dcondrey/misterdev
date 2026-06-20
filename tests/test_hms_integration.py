"""Integration test against the real HMS devplan.

Verifies that all 23 HMS tasks load correctly, dependencies resolve,
and topological sort produces a valid execution order.

Skips if HMS project is not available at /Volumes/A/hms/.
"""

import pytest
from pathlib import Path

from my_project_orchestrator.config import ConfigManager
from my_project_orchestrator.core.task import TaskManager
from my_project_orchestrator.core.decomposer import topological_sort

HMS_PATH = Path("/Volumes/A/hms")
HMS_AVAILABLE = HMS_PATH.exists() and (HMS_PATH / "project.yaml").exists()


@pytest.mark.skipif(not HMS_AVAILABLE, reason="HMS project not available")
class TestHMSIntegration:

    def _load_tasks(self):
        cfg = ConfigManager().load_project_config(HMS_PATH)

        class FP:
            path = HMS_PATH
            config = cfg

        tm = TaskManager(FP())
        tm.discover_tasks()
        return tm

    def test_discovers_all_task_files(self):
        tm = self._load_tasks()
        task_ids = sorted(tm.tasks.keys())
        assert len(task_ids) >= 23, f"Expected 23+ tasks, got {len(task_ids)}: {task_ids}"

    def test_skips_documentation_files(self):
        tm = self._load_tasks()
        doc_ids = {"ARCHITECTURE", "CONTEXT", "DEBATES", "MODULES", "README", "TASKS"}
        for doc_id in doc_ids:
            assert doc_id not in tm.tasks, f"Doc file {doc_id} loaded as task"

    def test_dependencies_resolve(self):
        tm = self._load_tasks()
        all_ids = set(tm.tasks.keys())
        for task in tm.tasks.values():
            for dep in task.dependencies:
                assert dep in all_ids, (
                    f"Task {task.id} has unresolved dependency '{dep}'. "
                    f"Available: {sorted(all_ids)}"
                )

    def test_group1_has_no_dependencies(self):
        tm = self._load_tasks()
        g1_ids = [tid for tid, t in tm.tasks.items() if t.processor_data.get("group") == "G1"]
        assert len(g1_ids) == 5, f"Expected 5 G1 tasks, got {g1_ids}"
        for tid in g1_ids:
            assert tm.tasks[tid].dependencies == [], f"G1 task {tid} should have no deps"

    def test_group2_depends_on_group1(self):
        tm = self._load_tasks()
        g2_ids = [tid for tid, t in tm.tasks.items() if t.processor_data.get("group") == "G2"]
        assert len(g2_ids) >= 1
        for tid in g2_ids:
            deps = tm.tasks[tid].dependencies
            assert len(deps) > 0, f"G2 task {tid} should depend on G1 tasks"

    def test_topological_sort_valid(self):
        tm = self._load_tasks()
        tasks = list(tm.tasks.values())
        sorted_tasks = topological_sort(tasks)
        assert len(sorted_tasks) == len(tasks)

        # Verify every task appears after its dependencies
        position = {t.id: i for i, t in enumerate(sorted_tasks)}
        for task in sorted_tasks:
            for dep in task.dependencies:
                assert position[dep] < position[task.id], (
                    f"Task {task.id} (pos {position[task.id]}) appears before "
                    f"its dependency {dep} (pos {position[dep]})"
                )

    def test_pending_tasks_exclude_completed(self):
        tm = self._load_tasks()
        pending = tm.get_pending_tasks()
        for task in pending:
            assert task.status == "pending"

    def test_files_to_modify_populated(self):
        tm = self._load_tasks()
        tasks_with_files = [t for t in tm.tasks.values() if t.files_to_modify]
        assert len(tasks_with_files) >= 20, (
            f"Expected most tasks to have files_to_modify, got {len(tasks_with_files)}"
        )

    def test_all_tasks_have_test_command(self):
        tm = self._load_tasks()
        for task in tm.tasks.values():
            tc = task.processor_data.get("test_command")
            assert tc, f"Task {task.id} missing test_command in frontmatter"
