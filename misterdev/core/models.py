from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone


class ExecutionResult(BaseModel):
    status: str  # 'completed', 'failed', 'in_progress', 'deferred'
    message: str
    logs: str = ""
    # For status == 'deferred': the human questions that must be answered before
    # the task can complete (a missing credential, a judgment call, an ambiguity).
    questions: List[str] = Field(default_factory=list)
    start_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None


class Task(BaseModel):
    id: str
    description: str
    type: str = "default"
    status: str = "pending"
    source_ref: Optional[str] = None  # e.g., file path
    project_ref: str
    processor_data: Dict[str, Any] = Field(default_factory=dict)
    execution_history: List[ExecutionResult] = Field(default_factory=list)
    # Fields ported from /build skill task decomposition
    title: str = ""
    acceptance_criteria: str = ""
    files_to_create: List[str] = Field(default_factory=list)
    files_to_modify: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(
        default_factory=list
    )  # task IDs that must complete first
    complexity: str = "medium"  # trivial, small, medium, large, architectural
    category: str = "feature"  # infrastructure, core, feature, fix, test, docs, integration, cleanup
    context_files: List[str] = Field(
        default_factory=list
    )  # Files relevant for context but not modified
