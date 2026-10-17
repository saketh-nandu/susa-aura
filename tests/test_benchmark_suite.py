"""
Tests for AuraBenchmarkSuite:
- Standard benchmark task execution
- Category scoring: instruction following, coding, tool selection, citation grounding
"""

from __future__ import annotations

import pytest

from aura.core.lifecycle import UserIntent
from aura.evaluation.benchmark import AuraBenchmarkSuite, BenchmarkTask


def test_benchmark_suite_scoring():
    """Verify benchmark suite accurately assesses agent response criteria."""
    suite = AuraBenchmarkSuite()

    # Synthetic responses matching the 5 default tasks
    responses = {
        "inst_01": '["red", "blue", "yellow"]',
        "tool_01": type("IntentObj", (), {"intent": UserIntent.WORKSPACE_OPERATION})(),
        "tool_02": type("IntentObj", (), {"intent": UserIntent.RESEARCH_QA})(),
        "code_01": "def is_palindrome(s: str) -> bool:\n    return s == s[::-1]\n",
        "cite_01": "SwiGLU is a Gated Linear Unit with Swish activation.",
    }

    report = suite.run_suite(responses)

    assert report.total_tasks == 5
    assert report.passed_tasks == 5
    assert report.score == 1.0
    assert "instruction" in report.category_scores
    assert "coding" in report.category_scores
    assert "tool_selection" in report.category_scores
    assert "citation" in report.category_scores
    assert report.category_scores["coding"] == 1.0
    assert len(report.failures) == 0


def test_benchmark_suite_records_failure():
    """Verify benchmark suite records specific failure details when tasks fail."""
    suite = AuraBenchmarkSuite()

    # Broken responses
    responses = {
        "inst_01": "",  # Empty
        "code_01": "def is_palindrome(s): return False\n",  # Wrong logic
    }

    report = suite.run_suite(responses)
    assert report.passed_tasks < report.total_tasks
    assert len(report.failures) > 0
    fail_ids = [f["task_id"] for f in report.failures]
    assert "inst_01" in fail_ids
    assert "code_01" in fail_ids
