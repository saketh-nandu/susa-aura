"""
Unit tests for SUSA AURA — AURA Core Subsystem.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from aura.core import AuraAgent, AuraCore, CodeMindAgent, SessionState


@pytest.fixture
def temp_core_workspace() -> Path:
    tmp = tempfile.mkdtemp(prefix="aura_core_ws_")
    ws = Path(tmp)
    (ws / "app.py").write_text("print('core app')\n")
    return ws


def test_core_initialization_and_aliases(temp_core_workspace):
    """Verify AuraCore instantiates and aliases AuraAgent and CodeMindAgent correctly."""
    core = AuraCore(workspace_root=temp_core_workspace)
    assert isinstance(core, AuraAgent)
    assert isinstance(core, CodeMindAgent)
    assert core.session.turn_count == 0
    assert core.session.system_prompt.startswith("You are SUSA AURA")


def test_core_ask_and_memory_integration(temp_core_workspace):
    """Verify core ask method integrates with memory subsystem."""
    core = AuraCore(workspace_root=temp_core_workspace)

    # Save a fact into memory
    core.memory.store_fact(key="project_goal", content="Build an original 1B model from scratch")

    # Ask a question that triggers relevant memory retrieval
    result = core.ask("What is the project goal for the 1B model?")
    assert "response" in result
    assert result["turn"] == 1
    assert "project_goal" in result["context_used"]


def test_core_ask_with_search(temp_core_workspace):
    """Verify core ask method integrates with AURA Connect web research."""
    core = AuraCore(workspace_root=temp_core_workspace)
    result = core.ask("What is PyTorch RMSNorm?", enable_search=True)

    assert "response" in result
    assert len(result["sources"]) > 0
    assert any("pytorch" in s.url.lower() for s in result["sources"])


def test_core_diff_and_approval_compatibility(temp_core_workspace):
    """Verify backward compatibility methods inspect_workspace, propose_edit, apply_edit."""
    core = AuraCore(workspace_root=temp_core_workspace)

    tree = core.inspect_workspace()
    assert "app.py" in tree

    diff = core.propose_edit("app.py", "print('modified app')\n")
    assert "--- a/app.py" in diff
    assert "+++ b/app.py" in diff

    # Ensure disk was not modified without approval
    assert (temp_core_workspace / "app.py").read_text() == "print('core app')\n"

    # Apply with approval
    applied = core.apply_edit("app.py", "print('modified app')\n", user_approved=True)
    assert applied is True
    assert (temp_core_workspace / "app.py").read_text() == "print('modified app')\n"
    assert (temp_core_workspace / "app.py.bak").exists()
