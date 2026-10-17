"""
Unit tests for SUSA-1B CodeMind Evaluation & Sandboxing Subsystem.
"""

from __future__ import annotations

import pytest

from aura.evaluation import (
    CodeEvaluator,
    CodingBenchmarkTask,
    IsolatedCodeRunner,
)


@pytest.fixture
def runner() -> IsolatedCodeRunner:
    return IsolatedCodeRunner(default_timeout_sec=2.0)


@pytest.fixture
def evaluator(runner) -> CodeEvaluator:
    return CodeEvaluator(sandbox=runner)


def test_sandbox_valid_execution(runner):
    """Verify sandboxed code executes and captures stdout cleanly."""
    code = 'print("SANDBOX_SUCCESS")'
    res = runner.run_python(code)
    assert res.success is True
    assert "SANDBOX_SUCCESS" in res.stdout


def test_sandbox_timeout_interruption(runner):
    """Verify infinite loop is safely halted by timeout without hanging."""
    infinite_loop_code = "while True: pass"
    res = runner.run_python(infinite_loop_code, timeout_sec=0.5)
    assert res.success is False
    assert res.timed_out is True
    assert "TimeoutExpired" in res.error_message


def test_sandbox_syntax_error(runner):
    """Verify runtime syntax error returns failure result."""
    bad_code = "def invalid syntax :"
    res = runner.run_python(bad_code)
    assert res.success is False
    assert res.returncode != 0


def test_evaluator_syntax_check(evaluator):
    """Verify AST syntax validation detects malformed Python code."""
    valid_code = "def add(x, y): return x + y"
    invalid_code = "def add(x, y) return x + y"

    ok1, _ = evaluator.validate_syntax(valid_code)
    assert ok1 is True

    ok2, err = evaluator.validate_syntax(invalid_code)
    assert ok2 is False
    assert "SyntaxError" in err


def test_benchmark_task_evaluation(evaluator):
    """Verify evaluation across correct and incorrect coding solutions."""
    task = CodingBenchmarkTask(
        task_id="task_multiply",
        prompt="def multiply(a, b):",
        test_assertions="assert multiply(3, 4) == 12\nassert multiply(0, 5) == 0\n",
        entry_point="multiply",
    )

    correct_solution = "def multiply(a, b):\n    return a * b\n"
    res_correct = evaluator.evaluate_task(correct_solution, task)
    assert res_correct.passed_tests is True
    assert res_correct.syntax_valid is True

    wrong_solution = "def multiply(a, b):\n    return a + b\n"
    res_wrong = evaluator.evaluate_task(wrong_solution, task)
    assert res_wrong.passed_tests is False
    assert res_wrong.syntax_valid is True
