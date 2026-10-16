"""
SUSA-1B CodeMind — Automated Code Evaluation Benchmark Suite.

Evaluates synthesized code for syntax validity, runtime execution success,
and programmatic unit-test assertion verification (pass@1).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from susacodemind.evaluation.sandbox import IsolatedCodeRunner


@dataclass
class CodingBenchmarkTask:
    """A coding task with prompt, test harness, and entry point."""

    task_id: str
    prompt: str
    test_assertions: str
    entry_point: str
    difficulty: str = "basic"


@dataclass
class EvalTaskResult:
    """Result of evaluating a single coding task."""

    task_id: str
    syntax_valid: bool
    passed_tests: bool
    error_message: str | None = None


class CodeEvaluator:
    """Benchmarking suite for measuring code intelligence."""

    def __init__(self, sandbox: IsolatedCodeRunner | None = None) -> None:
        self.sandbox = sandbox or IsolatedCodeRunner(default_timeout_sec=3.0)

    def validate_syntax(self, code_str: str) -> tuple[bool, str | None]:
        """Parses code to verify syntactic validity without executing."""
        try:
            ast.parse(code_str)
            return True, None
        except SyntaxError as e:
            return False, f"SyntaxError at line {e.lineno}: {e.msg}"

    def evaluate_task(self, generated_code: str, task: CodingBenchmarkTask) -> EvalTaskResult:
        """Evaluates generated code against task assertions in the sandbox."""
        # 1. Syntax check
        syntax_ok, syntax_err = self.validate_syntax(generated_code)
        if not syntax_ok:
            return EvalTaskResult(
                task_id=task.task_id,
                syntax_valid=False,
                passed_tests=False,
                error_message=syntax_err,
            )

        # 2. Combine code with test assertions
        full_script = f"{generated_code}\n\n# Benchmark Assertions\n{task.test_assertions}\n"

        # 3. Sandboxed execution
        exec_res = self.sandbox.run_python(full_script)
        if exec_res.success:
            return EvalTaskResult(
                task_id=task.task_id,
                syntax_valid=True,
                passed_tests=True,
                error_message=None,
            )
        else:
            err = exec_res.stderr.strip() or exec_res.error_message or "AssertionError"
            return EvalTaskResult(
                task_id=task.task_id,
                syntax_valid=True,
                passed_tests=False,
                error_message=err,
            )

    def run_benchmark(
        self,
        task_solutions: dict[str, str],
        tasks: list[CodingBenchmarkTask],
    ) -> dict[str, float]:
        """Calculates pass@1 and syntax accuracy across all benchmark tasks."""
        total = len(tasks)
        syntax_passed = 0
        tests_passed = 0

        for t in tasks:
            solution = task_solutions.get(t.task_id, "")
            res = self.evaluate_task(solution, t)
            if res.syntax_valid:
                syntax_passed += 1
            if res.passed_tests:
                tests_passed += 1

        return {
            "total_tasks": total,
            "syntax_accuracy": syntax_passed / max(1, total),
            "pass_at_1": tests_passed / max(1, total),
        }
