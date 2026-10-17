"""
SUSA AURA — Comprehensive Assistant Evaluation Benchmark Suite.

Evaluates AI reasoning across 4 distinct dimensions:
1. Instruction Following: adhering to formatting and constraints.
2. Coding & Syntax: syntactic validity and execution assertions.
3. Tool Selection: choosing the correct tool and parameters for tasks.
4. Source Grounding: fidelity to retrieved citations and facts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from aura.core.lifecycle import UserIntent
from aura.evaluation.code_eval import CodeEvaluator, CodingBenchmarkTask
from aura.evaluation.sandbox import IsolatedCodeRunner


@dataclass
class BenchmarkTask:
    """A generic assistant capability benchmark task."""

    task_id: str
    category: str  # "instruction", "coding", "tool_selection", "citation"
    prompt: str
    expected_intent: UserIntent | None = None
    expected_tool: str | None = None
    test_assertions: str | None = None
    ground_truth_fact: str | None = None


@dataclass
class BenchmarkReport:
    """Consolidated results of running the AURA capability benchmark."""

    total_tasks: int
    passed_tasks: int
    score: float
    category_scores: dict[str, float]
    failures: list[dict[str, Any]] = field(default_factory=list)
    duration_sec: float = 0.0


class AuraBenchmarkSuite:
    """Standardized test suite evaluating SUSA AURA platform capabilities."""

    def __init__(self, sandbox: IsolatedCodeRunner | None = None) -> None:
        self.code_evaluator = CodeEvaluator(sandbox=sandbox)
        self.default_tasks: list[BenchmarkTask] = self._build_standard_tasks()

    def _build_standard_tasks(self) -> list[BenchmarkTask]:
        """Constructs diverse benchmark tasks for assistant evaluation."""
        return [
            # 1. Instruction Following
            BenchmarkTask(
                task_id="inst_01",
                category="instruction",
                prompt="List three primary colors in JSON array format only.",
            ),
            # 2. Tool Selection
            BenchmarkTask(
                task_id="tool_01",
                category="tool_selection",
                prompt="Inspect the project workspace file tree.",
                expected_intent=UserIntent.WORKSPACE_OPERATION,
                expected_tool="inspect_workspace",
            ),
            BenchmarkTask(
                task_id="tool_02",
                category="tool_selection",
                prompt="Search the documentation for PyTorch RMSNorm implementation details.",
                expected_intent=UserIntent.RESEARCH_QA,
            ),
            # 3. Coding & Syntax
            BenchmarkTask(
                task_id="code_01",
                category="coding",
                prompt="Write a Python function `is_palindrome(s: str) -> bool`",
                test_assertions="assert is_palindrome('radar') == True\nassert is_palindrome('hello') == False",
            ),
            # 4. Source Grounding
            BenchmarkTask(
                task_id="cite_01",
                category="citation",
                prompt="What is SwiGLU?",
                ground_truth_fact="Gated Linear Unit",
            ),
        ]

    def evaluate_task(self, task: BenchmarkTask, agent_response: Any) -> tuple[bool, str | None]:
        """Evaluates an agent response against benchmark task criteria."""
        if task.category == "tool_selection":
            intent = getattr(agent_response, "intent", None)
            if task.expected_intent and intent != task.expected_intent:
                return False, f"Expected intent {task.expected_intent.value}, got {getattr(intent, 'value', intent)}"
            return True, None

        elif task.category == "coding" and task.test_assertions:
            code_text = str(agent_response)
            if "```python" in code_text:
                code_text = code_text.split("```python")[1].split("```")[0].strip()
            elif "```" in code_text:
                code_text = code_text.split("```")[1].split("```")[0].strip()

            coding_task = CodingBenchmarkTask(
                task_id=task.task_id,
                prompt=task.prompt,
                test_assertions=task.test_assertions,
                entry_point="",
            )
            res = self.code_evaluator.evaluate_task(code_text, coding_task)
            return res.passed_tests, res.error_message

        elif task.category == "citation" and task.ground_truth_fact:
            resp_str = str(agent_response).lower()
            if task.ground_truth_fact.lower() in resp_str:
                return True, None
            return False, f"Fact '{task.ground_truth_fact}' not found in response"

        # Default fallback: check non-empty response
        if str(agent_response).strip():
            return True, None
        return False, "Response was empty"

    def run_suite(self, responses: dict[str, Any], tasks: list[BenchmarkTask] | None = None) -> BenchmarkReport:
        """Executes full benchmark evaluation across provided tasks and responses."""
        start_time = time.time()
        active_tasks = tasks or self.default_tasks
        total = len(active_tasks)
        passed = 0
        cat_counts: dict[str, int] = {}
        cat_passed: dict[str, int] = {}
        failures = []

        for t in active_tasks:
            cat = t.category
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
            ans = responses.get(t.task_id, "")
            is_ok, err = self.evaluate_task(t, ans)
            if is_ok:
                passed += 1
                cat_passed[cat] = cat_passed.get(cat, 0) + 1
            else:
                failures.append({"task_id": t.task_id, "category": cat, "error": err})

        cat_scores = {c: cat_passed.get(c, 0) / count for c, count in cat_counts.items()}
        return BenchmarkReport(
            total_tasks=total,
            passed_tasks=passed,
            score=passed / max(1, total),
            category_scores=cat_scores,
            failures=failures,
            duration_sec=time.time() - start_time,
        )
