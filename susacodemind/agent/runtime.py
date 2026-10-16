"""
SUSA-1B CodeMind — Coding Agent Runtime Engine.

A controlled, safety-constrained software layer that inspects project workspaces,
generates unified diffs, runs tests in an isolated sandbox, and enforces strict
approval gates before executing commands or modifying files.
"""

from __future__ import annotations

import difflib
import shutil
from pathlib import Path

from susacodemind.agent.workspace import WorkspaceInspector
from susacodemind.evaluation.sandbox import ExecutionResult, IsolatedCodeRunner
from susacodemind.inference.generator import CodeGenerator


class CodeMindAgent:
    """Safety-constrained coding agent for workspace assistance."""

    def __init__(
        self,
        workspace_root: str | Path,
        generator: CodeGenerator | None = None,
        sandbox: IsolatedCodeRunner | None = None,
    ) -> None:
        self.workspace = WorkspaceInspector(workspace_root)
        self.generator = generator
        self.sandbox = sandbox or IsolatedCodeRunner(default_timeout_sec=5.0)
        self.modified_files: list[str] = []
        self.action_history: list[str] = []

    def inspect_workspace(self) -> str:
        """Returns visual file tree of the target project workspace."""
        tree = self.workspace.get_workspace_tree()
        self.action_history.append(f"Inspected workspace: {self.workspace.workspace_root}")
        return tree

    def propose_edit(self, relative_path: str, new_content: str) -> str:
        """Generates a standard unified diff of proposed changes without writing to disk."""
        target_path = self.workspace.workspace_root / relative_path
        if target_path.exists():
            original_content = target_path.read_text(encoding="utf-8")
        else:
            original_content = ""

        orig_lines = original_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)

        diff = difflib.unified_diff(
            orig_lines,
            new_lines,
            fromfile=f"a/{relative_path}",
            tofile=f"b/{relative_path}",
        )
        diff_text = "".join(diff)
        return diff_text or "No changes detected."

    def apply_edit(
        self,
        relative_path: str,
        new_content: str,
        user_approved: bool = False,
    ) -> bool:
        """Applies proposed edit to disk strictly after explicit user approval.

        Creates an automatic backup (.bak) of the target file.
        """
        if not user_approved:
            raise PermissionError(
                f"Modifying '{relative_path}' was rejected. "
                "Non-Negotiable Rule: Explicit user authorization is required before modifying workspace files."
            )

        target_path = (self.workspace.workspace_root / relative_path).resolve()
        # Prevent path traversal
        if not str(target_path).startswith(str(self.workspace.workspace_root)):
            raise PermissionError(f"Access denied: path outside workspace ({relative_path})")

        # Create backup if file exists
        if target_path.exists():
            backup_path = target_path.with_suffix(target_path.suffix + ".bak")
            shutil.copy2(target_path, backup_path)

        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(new_content, encoding="utf-8")

        if relative_path not in self.modified_files:
            self.modified_files.append(relative_path)

        self.action_history.append(f"Applied edit to: {relative_path}")
        return True

    def run_sandboxed_test(self, python_code: str) -> ExecutionResult:
        """Executes a test snippet safely within the isolated sandbox."""
        self.action_history.append("Ran sandboxed unit test")
        return self.sandbox.run_python(python_code)

    def summary(self) -> str:
        """Produces a report of all actions and file modifications."""
        lines = [
            "=" * 70,
            "  CODEMIND AGENT EXECUTION SUMMARY",
            "=" * 70,
            f"  Workspace:       {self.workspace.workspace_root}",
            f"  Actions Logged:  {len(self.action_history)}",
            f"  Modified Files:  {len(self.modified_files)}",
        ]
        for f in self.modified_files:
            lines.append(f"    - {f}")
        lines.append("=" * 70)
        return "\n".join(lines)
