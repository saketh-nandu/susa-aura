"""
Tests for Advanced AURA Tools Subsystem:
- Typed ToolParameter validation (missing arguments, type mismatches)
- Schema export via get_tool_schemas()
- Path traversal prevention across tools
"""

from __future__ import annotations

import pytest
from pathlib import Path

from aura.tools.registry import ToolParameter, ToolRegistry


def test_tool_parameter_validation_missing_required(tmp_path):
    """Verify tool execution fails gracefully when required parameter is missing."""
    registry = ToolRegistry(workspace_root=tmp_path)

    # Calling read_file without relative_path
    res = registry.execute("read_file")
    assert res.success is False
    assert "Missing required parameter 'relative_path'" in res.error_message


def test_tool_parameter_validation_type_mismatch(tmp_path):
    """Verify tool execution fails when argument type doesn't match schema."""
    registry = ToolRegistry(workspace_root=tmp_path)

    # Calling read_file with an integer instead of string
    res = registry.execute("read_file", relative_path=12345)
    assert res.success is False
    assert "must be a string" in res.error_message


def test_get_tool_schemas_export(tmp_path):
    """Verify tool schemas export correctly with typed parameters."""
    registry = ToolRegistry(workspace_root=tmp_path)
    schemas = registry.get_tool_schemas()

    assert len(schemas) >= 5
    tool_names = [s["name"] for s in schemas]
    assert "inspect_workspace" in tool_names
    assert "read_file" in tool_names
    assert "apply_edit" in tool_names
    assert "run_python_sandbox" in tool_names

    read_schema = next(s for s in schemas if s["name"] == "read_file")
    assert len(read_schema["parameters"]) == 1
    assert read_schema["parameters"][0]["name"] == "relative_path"
    assert read_schema["parameters"][0]["type"] == "str"


def test_path_traversal_forbidden_in_tools(tmp_path):
    """Verify path traversal outside workspace is blocked."""
    registry = ToolRegistry(workspace_root=tmp_path)

    res = registry.execute("read_file", relative_path="../../etc/passwd")
    assert res.success is False
    assert "traversal" in res.error_message.lower() or "denied" in res.error_message.lower()


def test_sibling_path_traversal_forbidden(tmp_path):
    """Verify sibling directory path traversal (prefix matching loophole) is strictly blocked."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sibling = tmp_path / "workspace_sibling"
    sibling.mkdir()
    secret_file = sibling / "secret.txt"
    secret_file.write_text("secret_data", encoding="utf-8")

    registry = ToolRegistry(workspace_root=workspace)

    # Attempt to read sibling file via relative traversal
    res = registry.execute("read_file", relative_path="../workspace_sibling/secret.txt")
    assert res.success is False
    assert "forbidden" in res.error_message.lower() or "denied" in res.error_message.lower()
