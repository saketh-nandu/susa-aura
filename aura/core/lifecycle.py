"""
SUSA AURA — Core Lifecycle, Planning & Request Contracts.

Defines typed data structures for:
- User intent classification.
- Bounded task planning and step specifications.
- Tool call requests and execution records.
- Grounded evidence and citation records.
- Complete orchestration responses with loop detection.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from aura.connect.retriever import SourceReference


class UserIntent(str, Enum):
    """Classified user intent driving execution strategy."""

    DIRECT_ANSWER = "direct_answer"
    RESEARCH_QA = "research_qa"
    CODE_REASONING = "code_reasoning"
    WORKSPACE_OPERATION = "workspace_operation"
    MULTI_STEP_TASK = "multi_step_task"


@dataclass
class EvidenceRecord:
    """A claim grounded by verified source retrieval or deterministic execution."""

    claim: str
    source_url: str
    snippet: str
    confidence: float = 1.0
    evidence_type: str = "verified_source"


@dataclass
class TaskStep:
    """A single bounded step within an execution plan."""

    step_id: int
    description: str
    tool_name: str | None = None
    tool_args: dict[str, Any] = field(default_factory=dict)
    expected_outcome: str = ""
    status: str = "pending"  # "pending", "running", "completed", "failed", "skipped"
    result: Any = None
    error_message: str | None = None
    verified: bool = False
    duration_sec: float = 0.0


@dataclass
class TaskPlan:
    """A bounded sequence of planned execution steps."""

    plan_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    goal: str = ""
    intent: UserIntent = UserIntent.DIRECT_ANSWER
    steps: list[TaskStep] = field(default_factory=list)
    max_steps: int = 8
    created_at: float = field(default_factory=time.time)

    def is_complete(self) -> bool:
        """True if all steps are in completed or skipped states."""
        return len(self.steps) > 0 and all(s.status in ("completed", "skipped") for s in self.steps)

    def has_failures(self) -> bool:
        """True if any step failed."""
        return any(s.status == "failed" for s in self.steps)


@dataclass
class AuraRequest:
    """Structured request to AURA Core."""

    user_input: str
    intent: UserIntent | None = None
    enable_search: bool = True
    enable_tools: bool = True
    enable_memory: bool = True
    user_approved_operations: bool = False
    max_steps: int = 8
    timeout_sec: float = 30.0


@dataclass
class AuraResponse:
    """Comprehensive, structured response from AURA Core."""

    response_text: str
    intent: UserIntent
    plan: TaskPlan | None = None
    steps_executed: int = 0
    sources: list[SourceReference] = field(default_factory=list)
    evidence: list[EvidenceRecord] = field(default_factory=list)
    context_used: str = ""
    execution_time_sec: float = 0.0
    success: bool = True
    error_message: str | None = None
    limitations: list[str] = field(default_factory=list)
