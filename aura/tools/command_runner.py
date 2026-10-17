"""
SUSA AURA — Safe Command Execution Subsystem.

Provides allowlisted command execution with strict timeouts, output truncation,
working directory isolation, and permission verification.
Enforces non-negotiable safety rules: never run destructive commands automatically.
"""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ALLOWED_COMMANDS = {
    "pytest",
    "git status",
    "git diff",
    "git log",
    "python -m pytest",
    "python scripts/verify_env.py",
    "python scripts/count_params.py",
    "python scripts/evaluate.py",
    "ls",
    "pwd",
}


@dataclass
class CommandResult:
    """Outcome of an executed command."""

    command: str
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False
    authorized: bool = True
    error_message: str | None = None


class SafeCommandRunner:
    """Safely executes authorized developer commands within the workspace."""

    def __init__(
        self,
        workspace_root: str | Path,
        allowed_commands: set[str] | None = None,
        timeout_sec: float = 30.0,
        max_output_chars: int = 8000,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.allowed_commands = allowed_commands or set(DEFAULT_ALLOWED_COMMANDS)
        self.timeout_sec = timeout_sec
        self.max_output_chars = max_output_chars

    def is_command_allowed(self, cmd_str: str) -> bool:
        """Checks if a command matches or starts with an allowlisted command pattern."""
        cleaned = cmd_str.strip()
        for allowed in self.allowed_commands:
            if cleaned == allowed or cleaned.startswith(f"{allowed} "):
                return True
        return False

    def run(self, cmd_str: str, user_approved: bool = False) -> CommandResult:
        """Executes an allowlisted command or an approved arbitrary command."""
        cleaned = cmd_str.strip()

        if not self.is_command_allowed(cleaned) and not user_approved:
            return CommandResult(
                command=cleaned,
                exit_code=1,
                stdout="",
                stderr="",
                authorized=False,
                error_message=(
                    f"Command '{cleaned}' is not in the safe allowlist and was not explicitly approved. "
                    "Explicit authorization is required for unlisted commands."
                ),
            )

        try:
            tokens = shlex.split(cleaned)
            proc = subprocess.run(
                tokens,
                cwd=str(self.workspace_root),
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
            )

            stdout = proc.stdout
            stderr = proc.stderr

            if len(stdout) > self.max_output_chars:
                stdout = stdout[: self.max_output_chars] + f"\n... [Truncated: exceeded {self.max_output_chars} chars]"
            if len(stderr) > self.max_output_chars:
                stderr = stderr[: self.max_output_chars] + f"\n... [Truncated: exceeded {self.max_output_chars} chars]"

            return CommandResult(
                command=cleaned,
                exit_code=proc.returncode,
                stdout=stdout,
                stderr=stderr,
                timed_out=False,
                authorized=True,
            )
        except subprocess.TimeoutExpired:
            return CommandResult(
                command=cleaned,
                exit_code=-1,
                stdout="",
                stderr=f"Command timed out after {self.timeout_sec} seconds.",
                timed_out=True,
                authorized=True,
                error_message="Execution timed out.",
            )
        except Exception as exc:
            return CommandResult(
                command=cleaned,
                exit_code=1,
                stdout="",
                stderr=str(exc),
                timed_out=False,
                authorized=True,
                error_message=str(exc),
            )
