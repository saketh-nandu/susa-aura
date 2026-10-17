"""
Tests for Core Lifecycle, Intent Classification, Planning, Loop Detection, and State Verification.
"""

from __future__ import annotations

import pytest
from pathlib import Path

from aura.core.lifecycle import AuraRequest, UserIntent
from aura.core.runtime import AuraCore, LoopDetectedError
from aura.model.backend import NativeAuraBackend


def test_intent_classification(tmp_path):
    """Verify intent classifier accurately categorizes requests."""
    core = AuraCore(workspace_root=tmp_path)

    assert core.classify_intent("inspect workspace file tree") == UserIntent.WORKSPACE_OPERATION
    assert core.classify_intent("read file main.py") == UserIntent.WORKSPACE_OPERATION
    assert core.classify_intent("search documentation for RMSNorm") == UserIntent.RESEARCH_QA
    assert core.classify_intent("run python script") == UserIntent.CODE_REASONING
    assert core.classify_intent("hello, how are you?") == UserIntent.DIRECT_ANSWER


def test_core_solve_workspace_plan_and_state_verification(tmp_path):
    """Verify AuraCore.solve() builds and executes a plan with state verification."""
    # Create sample file
    test_file = tmp_path / "hello.py"
    test_file.write_text("print('hello')\n", encoding="utf-8")

    core = AuraCore(workspace_root=tmp_path)
    req = AuraRequest(user_input="inspect workspace")
    resp = core.solve(req)

    assert resp.success is True
    assert resp.intent == UserIntent.WORKSPACE_OPERATION
    assert resp.plan is not None
    assert len(resp.plan.steps) == 1
    assert resp.plan.steps[0].tool_name == "inspect_workspace"
    assert resp.plan.steps[0].status == "completed"
    assert resp.plan.steps[0].verified is True


def test_loop_detection_raises_error(tmp_path):
    """Verify loop detection triggers when identical calls repeat without progress."""
    core = AuraCore(workspace_root=tmp_path)

    # Intentionally trigger duplicate tool steps in a plan
    from aura.core.lifecycle import TaskStep
    plan = core.create_plan(AuraRequest(user_input="inspect workspace"))
    # Add repeated identical steps to simulate an agent stuck in a loop
    plan.steps = [
        TaskStep(step_id=1, description="Step 1", tool_name="inspect_workspace", tool_args={}),
        TaskStep(step_id=2, description="Step 2", tool_name="inspect_workspace", tool_args={}),
        TaskStep(step_id=3, description="Step 3", tool_name="inspect_workspace", tool_args={}),
    ]

    # Monkeypatch create_plan to return this looping plan
    core.create_plan = lambda req: plan

    with pytest.raises(LoopDetectedError) as exc_info:
        core.solve("inspect workspace")

    assert "Execution loop detected" in str(exc_info.value)


def test_multi_step_sequential_decomposition(tmp_path):
    """Verify multi-step sequential tasks are decomposed into multiple plan steps."""
    test_file = tmp_path / "hello.py"
    test_file.write_text("print('hello world')\n", encoding="utf-8")

    core = AuraCore(workspace_root=tmp_path)
    req = AuraRequest(user_input="inspect workspace then read hello.py")
    resp = core.solve(req)

    assert resp.success is True
    assert resp.plan is not None
    assert len(resp.plan.steps) == 2
    assert resp.plan.steps[0].tool_name == "inspect_workspace"
    assert resp.plan.steps[1].tool_name == "read_file"
    assert resp.plan.steps[1].tool_args["relative_path"] == "hello.py"
