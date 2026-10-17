"""
Unit tests for SUSA AURA — AURA Tools Subsystem.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from aura.tools import (
    CommandResult,
    SafeCommandRunner,
    ToolRegistry,
    WorkspaceInspector,
)


@pytest.fixture
def temp_tool_workspace() -> Path:
    tmp = tempfile.mkdtemp(prefix="aura_tools_ws_")
    ws = Path(tmp)
    (ws / "main.py").write_text("print('test tools')\n")
    return ws


def test_command_runner_allowlist(temp_tool_workspace):
    """Verify allowlisted commands execute while unapproved arbitrary commands are blocked."""
    runner = SafeCommandRunner(workspace_root=temp_tool_workspace)

    # Allowlisted command
    res = runner.run("pwd")
    assert res.authorized is True
    assert res.exit_code == 0
    assert str(temp_tool_workspace) in res.stdout

    # Non-allowlisted command without approval -> blocked
    blocked = runner.run("rm -rf /some/random/dir", user_approved=False)
    assert blocked.authorized is False
    assert "not in the safe allowlist" in (blocked.error_message or "")


def test_tool_registry_execution(temp_tool_workspace):
    """Verify tools in ToolRegistry dispatch correctly."""
    registry = ToolRegistry(workspace_root=temp_tool_workspace)

    # 1. Workspace inspection tool
    tree_res = registry.execute("inspect_workspace")
    assert tree_res.success is True
    assert "main.py" in str(tree_res.data)

    # 2. Read file tool
    read_res = registry.execute("read_file", relative_path="main.py")
    assert read_res.success is True
    assert "test tools" in str(read_res.data)

    # 3. Sandboxed Python execution tool
    py_code = "print(10 + 32)"
    sandbox_res = registry.execute("run_python_sandbox", code=py_code)
    assert sandbox_res.success is True
    assert "42" in sandbox_res.data.stdout


def test_tool_registry_approval_gate(temp_tool_workspace):
    """Verify apply_edit tool requires explicit approval."""
    registry = ToolRegistry(workspace_root=temp_tool_workspace)

    # Attempt edit without approval
    res = registry.execute(
        "apply_edit",
        relative_path="main.py",
        new_content="print('updated')\n",
        user_approved=False,
    )
    assert res.success is False
    assert "rejected" in (res.error_message or "").lower() or "authorization" in (res.error_message or "").lower()

    # Attempt edit with approval
    res_ok = registry.execute(
        "apply_edit",
        relative_path="main.py",
        new_content="print('updated')\n",
        user_approved=True,
    )
    assert res_ok.success is True
    assert (temp_tool_workspace / "main.py.bak").exists()
    assert (temp_tool_workspace / "main.py").read_text() == "print('updated')\n"
