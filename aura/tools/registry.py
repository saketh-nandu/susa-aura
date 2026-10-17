"""
SUSA AURA — Tool Definition & Registry Subsystem.

Provides declarative tool interfaces, parameter validation, execution tracking,
and strict authorization enforcement.
"""

from __future__ import annotations

import difflib
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from aura.evaluation.sandbox import ExecutionResult, IsolatedCodeRunner
from aura.tools.command_runner import CommandResult, SafeCommandRunner
from aura.tools.workspace import FileInfo, WorkspaceInspector


@dataclass
class ToolDefinition:
    """Metadata describing an AURA tool capability."""

    name: str
    description: str
    parameters: dict[str, Any]
    requires_approval: bool = False


@dataclass
class ToolExecutionResult:
    """Standardized output of any AURA tool execution."""

    tool_name: str
    success: bool
    data: Any
    error_message: str | None = None


class ToolRegistry:
    """Registry coordinating all approved AURA tools."""

    def __init__(
        self,
        workspace_root: str | Path,
        sandbox: IsolatedCodeRunner | None = None,
        command_runner: SafeCommandRunner | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.workspace = WorkspaceInspector(self.workspace_root)
        self.sandbox = sandbox or IsolatedCodeRunner(default_timeout_sec=5.0)
        self.command_runner = command_runner or SafeCommandRunner(self.workspace_root)
        self.modified_files: list[str] = []
        self.action_history: list[str] = []
        self.tools: dict[str, ToolDefinition] = {}
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Registers built-in AURA workspace, inspection, sandbox, and command tools."""
        self.tools["inspect_workspace"] = ToolDefinition(
            name="inspect_workspace",
            description="Returns a formatted tree of files in the workspace.",
            parameters={},
            requires_approval=False,
        )
        self.tools["read_file"] = ToolDefinition(
            name="read_file",
            description="Reads the text content of a file within the workspace.",
            parameters={"relative_path": "str"},
            requires_approval=False,
        )
        self.tools["propose_edit"] = ToolDefinition(
            name="propose_edit",
            description="Generates a standard unified diff of proposed changes without writing to disk.",
            parameters={"relative_path": "str", "new_content": "str"},
            requires_approval=False,
        )
        self.tools["apply_edit"] = ToolDefinition(
            name="apply_edit",
            description="Applies proposed file edit to disk with automatic .bak backup. Requires approval.",
            parameters={"relative_path": "str", "new_content": "str", "user_approved": "bool"},
            requires_approval=True,
        )
        self.tools["run_python_sandbox"] = ToolDefinition(
            name="run_python_sandbox",
            description="Executes a Python code snippet safely within an isolated ephemeral sandbox.",
            parameters={"code": "str"},
            requires_approval=False,
        )
        self.tools["run_command"] = ToolDefinition(
            name="run_command",
            description="Executes an allowlisted terminal command safely within the workspace.",
            parameters={"command": "str", "user_approved": "bool"},
            requires_approval=False,
        )

    def execute(self, tool_name: str, **kwargs: Any) -> ToolExecutionResult:
        """Executes a registered tool by name with arguments."""
        if tool_name not in self.tools:
            return ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                data=None,
                error_message=f"Tool '{tool_name}' is not registered in AURA Tools.",
            )

        try:
            if tool_name == "inspect_workspace":
                tree = self.workspace.get_workspace_tree()
                self.action_history.append(f"Inspected workspace: {self.workspace_root}")
                return ToolExecutionResult(tool_name=tool_name, success=True, data=tree)

            elif tool_name == "read_file":
                rel = kwargs["relative_path"]
                content = self.workspace.read_file(rel)
                self.action_history.append(f"Read file: {rel}")
                return ToolExecutionResult(tool_name=tool_name, success=True, data=content)

            elif tool_name == "propose_edit":
                rel = kwargs["relative_path"]
                new_content = kwargs["new_content"]
                diff = self.propose_edit(rel, new_content)
                return ToolExecutionResult(tool_name=tool_name, success=True, data=diff)

            elif tool_name == "apply_edit":
                rel = kwargs["relative_path"]
                new_content = kwargs["new_content"]
                user_approved = kwargs.get("user_approved", False)
                self.apply_edit(rel, new_content, user_approved=user_approved)
                return ToolExecutionResult(tool_name=tool_name, success=True, data=f"Applied edit to {rel}")

            elif tool_name == "run_python_sandbox":
                code = kwargs["code"]
                res = self.sandbox.run_python(code)
                self.action_history.append("Ran sandboxed Python execution")
                return ToolExecutionResult(tool_name=tool_name, success=res.success, data=res)

            elif tool_name == "run_command":
                cmd = kwargs["command"]
                user_approved = kwargs.get("user_approved", False)
                res = self.command_runner.run(cmd, user_approved=user_approved)
                self.action_history.append(f"Ran command: {cmd}")
                return ToolExecutionResult(tool_name=tool_name, success=(res.exit_code == 0), data=res)

            else:
                return ToolExecutionResult(
                    tool_name=tool_name,
                    success=False,
                    data=None,
                    error_message=f"No execution handler implemented for tool: {tool_name}",
                )

        except Exception as exc:
            return ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                data=None,
                error_message=str(exc),
            )

    def propose_edit(self, relative_path: str, new_content: str) -> str:
        """Generates standard unified diff of proposed changes."""
        target_path = self.workspace_root / relative_path
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
        """Applies proposed edit to disk with user approval and .bak backup."""
        if not user_approved:
            raise PermissionError(
                f"Modifying '{relative_path}' was rejected. "
                "Non-Negotiable Rule: Explicit user authorization is required before modifying workspace files."
            )

        target_path = (self.workspace_root / relative_path).resolve()
        if not str(target_path).startswith(str(self.workspace_root)):
            raise PermissionError(f"Access denied: path outside workspace ({relative_path})")

        if target_path.exists():
            backup_path = target_path.with_suffix(target_path.suffix + ".bak")
            shutil.copy2(target_path, backup_path)

        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(new_content, encoding="utf-8")

        if relative_path not in self.modified_files:
            self.modified_files.append(relative_path)

        self.action_history.append(f"Applied edit to: {relative_path}")
        return True
