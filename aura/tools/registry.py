"""
SUSA AURA — Tool Definition & Registry Subsystem.

Provides declarative tool interfaces, parameter validation, execution tracking,
audit logging, resource constraints, and strict authorization enforcement.
"""

from __future__ import annotations

import difflib
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from aura.evaluation.sandbox import ExecutionResult, IsolatedCodeRunner
from aura.tools.command_runner import CommandResult, SafeCommandRunner
from aura.tools.workspace import FileInfo, WorkspaceInspector


@dataclass
class ToolParameter:
    """Specification of an expected parameter for a tool."""

    name: str
    type_name: str
    description: str
    required: bool = True
    default: Any = None


@dataclass
class ToolDefinition:
    """Metadata and parameter schema describing an AURA tool capability."""

    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)
    requires_approval: bool = False
    category: str = "workspace"

    def to_dict(self) -> dict[str, Any]:
        """Serializes tool definition for inspection or agent context."""
        return {
            "name": self.name,
            "description": self.description,
            "requires_approval": self.requires_approval,
            "category": self.category,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type_name,
                    "description": p.description,
                    "required": p.required,
                }
                for p in self.parameters
            ],
        }


@dataclass
class ToolExecutionResult:
    """Standardized output of any AURA tool execution."""

    tool_name: str
    success: bool
    data: Any
    error_message: str | None = None
    execution_time_sec: float = 0.0
    timestamp: float = field(default_factory=time.time)


class ToolRegistry:
    """Central registry coordinating all approved AURA tools."""

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
        self.execution_audit_log: list[ToolExecutionResult] = []
        self.tools: dict[str, ToolDefinition] = {}
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Registers built-in AURA workspace, inspection, sandbox, and command tools."""
        self.tools["inspect_workspace"] = ToolDefinition(
            name="inspect_workspace",
            description="Returns a formatted tree of files in the workspace.",
            parameters=[],
            requires_approval=False,
            category="workspace",
        )
        self.tools["read_file"] = ToolDefinition(
            name="read_file",
            description="Reads the text content of a file within the workspace.",
            parameters=[
                ToolParameter("relative_path", "str", "Relative file path from workspace root", required=True),
            ],
            requires_approval=False,
            category="workspace",
        )
        self.tools["propose_edit"] = ToolDefinition(
            name="propose_edit",
            description="Generates a standard unified diff of proposed changes without writing to disk.",
            parameters=[
                ToolParameter("relative_path", "str", "Relative file path to modify", required=True),
                ToolParameter("new_content", "str", "Proposed new full content of file", required=True),
            ],
            requires_approval=False,
            category="workspace",
        )
        self.tools["apply_edit"] = ToolDefinition(
            name="apply_edit",
            description="Applies proposed file edit to disk with automatic .bak backup. Requires explicit approval.",
            parameters=[
                ToolParameter("relative_path", "str", "Relative file path to write", required=True),
                ToolParameter("new_content", "str", "New full content of file", required=True),
                ToolParameter("user_approved", "bool", "Explicit user confirmation flag", required=False, default=False),
            ],
            requires_approval=True,
            category="workspace",
        )
        self.tools["run_python_sandbox"] = ToolDefinition(
            name="run_python_sandbox",
            description="Executes a Python code snippet safely within an isolated ephemeral sandbox.",
            parameters=[
                ToolParameter("code", "str", "Complete Python code script to execute", required=True),
            ],
            requires_approval=False,
            category="computation",
        )
        self.tools["run_command"] = ToolDefinition(
            name="run_command",
            description="Executes an allowlisted terminal command safely within the workspace.",
            parameters=[
                ToolParameter("command", "str", "Terminal command string to run", required=True),
                ToolParameter("user_approved", "bool", "Explicit user confirmation flag for non-allowlisted commands", required=False, default=False),
            ],
            requires_approval=False,
            category="system",
        )

    def validate_arguments(self, tool_def: ToolDefinition, kwargs: dict[str, Any]) -> tuple[bool, str | None]:
        """Validates that all required parameters are supplied with appropriate types."""
        for param in tool_def.parameters:
            if param.required and param.name not in kwargs:
                return False, f"Missing required parameter '{param.name}' for tool '{tool_def.name}'"

            if param.name in kwargs:
                val = kwargs[param.name]
                if param.type_name == "str" and not isinstance(val, str):
                    return False, f"Parameter '{param.name}' must be a string, got {type(val).__name__}"
                elif param.type_name == "bool" and not isinstance(val, bool):
                    return False, f"Parameter '{param.name}' must be a boolean, got {type(val).__name__}"
                elif param.type_name == "int" and not isinstance(val, int):
                    return False, f"Parameter '{param.name}' must be an integer, got {type(val).__name__}"

        return True, None

    def execute(self, tool_name: str, **kwargs: Any) -> ToolExecutionResult:
        """Executes a registered tool by name with validated arguments and audit tracking."""
        start_time = time.time()

        if tool_name not in self.tools:
            res = ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                data=None,
                error_message=f"Tool '{tool_name}' is not registered in AURA Tools.",
                execution_time_sec=time.time() - start_time,
            )
            self.execution_audit_log.append(res)
            return res

        tool_def = self.tools[tool_name]
        is_valid, validation_err = self.validate_arguments(tool_def, kwargs)
        if not is_valid:
            res = ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                data=None,
                error_message=f"Validation failed: {validation_err}",
                execution_time_sec=time.time() - start_time,
            )
            self.execution_audit_log.append(res)
            return res

        try:
            if tool_name == "inspect_workspace":
                tree = self.workspace.get_workspace_tree()
                self.action_history.append(f"Inspected workspace: {self.workspace_root}")
                res = ToolExecutionResult(
                    tool_name=tool_name,
                    success=True,
                    data=tree,
                    execution_time_sec=time.time() - start_time,
                )

            elif tool_name == "read_file":
                rel = kwargs["relative_path"]
                content = self.workspace.read_file(rel)
                self.action_history.append(f"Read file: {rel}")
                res = ToolExecutionResult(
                    tool_name=tool_name,
                    success=True,
                    data=content,
                    execution_time_sec=time.time() - start_time,
                )

            elif tool_name == "propose_edit":
                rel = kwargs["relative_path"]
                new_content = kwargs["new_content"]
                diff = self.propose_edit(rel, new_content)
                res = ToolExecutionResult(
                    tool_name=tool_name,
                    success=True,
                    data=diff,
                    execution_time_sec=time.time() - start_time,
                )

            elif tool_name == "apply_edit":
                rel = kwargs["relative_path"]
                new_content = kwargs["new_content"]
                user_approved = kwargs.get("user_approved", False)
                self.apply_edit(rel, new_content, user_approved=user_approved)
                res = ToolExecutionResult(
                    tool_name=tool_name,
                    success=True,
                    data=f"Applied edit to {rel}",
                    execution_time_sec=time.time() - start_time,
                )

            elif tool_name == "run_python_sandbox":
                code = kwargs["code"]
                exec_res = self.sandbox.run_python(code)
                self.action_history.append("Ran sandboxed Python execution")
                res = ToolExecutionResult(
                    tool_name=tool_name,
                    success=exec_res.success,
                    data=exec_res,
                    error_message=exec_res.error_message if not exec_res.success else None,
                    execution_time_sec=time.time() - start_time,
                )

            elif tool_name == "run_command":
                cmd = kwargs["command"]
                user_approved = kwargs.get("user_approved", False)
                cmd_res = self.command_runner.run(cmd, user_approved=user_approved)
                self.action_history.append(f"Ran command: {cmd}")
                res = ToolExecutionResult(
                    tool_name=tool_name,
                    success=(cmd_res.exit_code == 0),
                    data=cmd_res,
                    error_message=cmd_res.error_message,
                    execution_time_sec=time.time() - start_time,
                )

            else:
                res = ToolExecutionResult(
                    tool_name=tool_name,
                    success=False,
                    data=None,
                    error_message=f"No execution handler implemented for tool: {tool_name}",
                    execution_time_sec=time.time() - start_time,
                )

        except Exception as exc:
            res = ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                data=None,
                error_message=str(exc),
                execution_time_sec=time.time() - start_time,
            )

        self.execution_audit_log.append(res)
        return res

    def propose_edit(self, relative_path: str, new_content: str) -> str:
        """Generates standard unified diff of proposed changes."""
        target_path = (self.workspace_root / relative_path).resolve()
        try:
            if not target_path.is_relative_to(self.workspace_root.resolve()):
                raise PermissionError(f"Access denied: path outside workspace ({relative_path})")
        except (ValueError, AttributeError):
            raise PermissionError(f"Access denied: path outside workspace ({relative_path})")

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
        try:
            if not target_path.is_relative_to(self.workspace_root.resolve()):
                raise PermissionError(f"Access denied: path outside workspace ({relative_path})")
        except (ValueError, AttributeError):
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

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Returns schemas of all registered tools for reasoning engines."""
        return [t.to_dict() for t in self.tools.values()]
