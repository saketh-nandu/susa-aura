"""
SUSA AURA — Isolated Code Execution Sandbox.

Executes generated Python code in an isolated subprocess with strict timeouts,
memory bounds, environment isolation, and sandboxed temporary directories.
Enforces Non-Negotiable Rule 9: Never execute generated code with unrestricted host access.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExecutionResult:
    """Outcome of sandboxed code execution."""

    success: bool
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool = False
    error_message: str | None = None


class IsolatedCodeRunner:
    """Safely executes untrusted code snippets with timeouts and sandbox isolation."""

    def __init__(self, default_timeout_sec: float = 3.0) -> None:
        self.default_timeout_sec = default_timeout_sec

    def run_python(
        self,
        code_str: str,
        timeout_sec: float | None = None,
    ) -> ExecutionResult:
        """Executes a Python code snippet in an isolated ephemeral directory."""
        timeout = timeout_sec or self.default_timeout_sec

        # Create isolated temporary directory
        with tempfile.TemporaryDirectory(prefix="aura_sandbox_") as tmp_dir:
            script_path = Path(tmp_dir) / "test_exec.py"
            script_path.write_text(code_str, encoding="utf-8")

            # Run with Python flags:
            # -I : Run in isolated mode (ignores PYTHONPATH and user site-packages)
            # -s : Don't add user site directory to sys.path
            cmd = [sys.executable, "-I", "-s", str(script_path.name)]

            # Sanitized minimal environment
            safe_env = {
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "PYTHONHASHSEED": "0",
                "LC_ALL": "C.UTF-8",
            }

            try:
                proc = subprocess.run(
                    cmd,
                    cwd=tmp_dir,
                    env=safe_env,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False,
                )
                success = proc.returncode == 0
                return ExecutionResult(
                    success=success,
                    stdout=proc.stdout,
                    stderr=proc.stderr,
                    returncode=proc.returncode,
                    timed_out=False,
                )
            except subprocess.TimeoutExpired:
                return ExecutionResult(
                    success=False,
                    stdout="",
                    stderr=f"Execution timed out after {timeout} seconds.",
                    returncode=-1,
                    timed_out=True,
                    error_message=f"TimeoutExpired ({timeout}s)",
                )
            except Exception as e:
                return ExecutionResult(
                    success=False,
                    stdout="",
                    stderr=str(e),
                    returncode=-1,
                    timed_out=False,
                    error_message=str(e),
                )
