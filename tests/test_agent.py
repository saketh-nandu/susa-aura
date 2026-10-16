"""
Unit tests for SUSA-1B CodeMind Coding Agent Runtime.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import pytest

from susacodemind.agent import CodeMindAgent, WorkspaceInspector


@pytest.fixture
def temp_workspace() -> Path:
    tmp = tempfile.mkdtemp(prefix="agent_ws_")
    ws = Path(tmp)
    # Populate mock project files
    (ws / "main.py").write_text("def run():\n    print('hello')\n")
    (ws / "utils").mkdir()
    (ws / "utils" / "helper.py").write_text("def helper():\n    return 42\n")
    return ws


def test_workspace_scanning(temp_workspace):
    """Verify agent inspects workspace tree and finds files."""
    inspector = WorkspaceInspector(temp_workspace)
    files = inspector.scan_files()
    rel_paths = [f.relative_path for f in files]

    assert "main.py" in rel_paths
    assert "utils/helper.py" in rel_paths


def test_agent_propose_diff(temp_workspace):
    """Verify unified diff generation without disk mutation."""
    agent = CodeMindAgent(workspace_root=temp_workspace)
    new_code = "def run():\n    print('hello world updated')\n"

    diff = agent.propose_edit("main.py", new_code)
    assert "--- a/main.py" in diff
    assert "+++ b/main.py" in diff
    assert "-    print('hello')" in diff
    assert "+    print('hello world updated')" in diff

    # Ensure file on disk was NOT mutated
    orig_disk = (temp_workspace / "main.py").read_text()
    assert "hello world updated" not in orig_disk


def test_agent_approval_gate_and_backup(temp_workspace):
    """Verify edits require explicit user authorization and create backups."""
    agent = CodeMindAgent(workspace_root=temp_workspace)
    new_code = "def run():\n    print('authorized change')\n"

    # Attempt without approval -> must raise PermissionError
    with pytest.raises(PermissionError, match="Explicit user authorization is required"):
        agent.apply_edit("main.py", new_code, user_approved=False)

    # Apply with approval -> must succeed, create backup, and update file
    applied = agent.apply_edit("main.py", new_code, user_approved=True)
    assert applied is True
    assert (temp_workspace / "main.py.bak").exists()
    assert "authorized change" in (temp_workspace / "main.py").read_text()
    assert "main.py" in agent.modified_files
