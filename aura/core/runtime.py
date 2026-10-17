"""
SUSA AURA — AURA Core: Central Orchestration & Planning Engine.

Coordinates the entire SUSA AURA system:
- Request normalization and intent classification.
- Bounded task planning and step-by-step execution.
- Loop detection, retry budgeting, and resource constraint tracking.
- State verification after tool execution.
- Source-grounded web research via AURA Connect.
- Multi-tier memory integration (Working, User, Knowledge, Task).
- Model execution via unified ModelBackend abstraction.
- Strict authorization gates for file modification and commands.
- Full backward compatibility with CodeMindAgent and legacy AuraCore API.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aura.connect.retriever import AuraConnect, RetrievalResponse, SourceReference
from aura.core.lifecycle import (
    AuraRequest,
    AuraResponse,
    EvidenceRecord,
    TaskPlan,
    TaskStep,
    UserIntent,
)
from aura.evaluation.sandbox import ExecutionResult, IsolatedCodeRunner
from aura.inference.generator import AuraGenerator
from aura.memory.manager import AuraMemory, MemoryCategory
from aura.model.backend import (
    GenerationOptions,
    ModelBackend,
    NativeAuraBackend,
)
from aura.tools.command_runner import SafeCommandRunner
from aura.tools.registry import ToolExecutionResult, ToolRegistry
from aura.tools.workspace import WorkspaceInspector


@dataclass
class SessionState:
    """Active conversational and operational session state."""

    session_id: str = "default_session"
    system_prompt: str = (
        "You are SUSA AURA, an Adaptive Universal Reasoning Assistant. "
        "You assist with reasoning, code development, research, and analysis. "
        "Be accurate, concise, grounded in evidence, and honest about limitations."
    )
    turn_count: int = 0
    active_tools_count: int = 0
    research_enabled: bool = True
    memory_enabled: bool = True
    max_steps_per_task: int = 8
    max_retries_per_step: int = 2


class LoopDetectedError(RuntimeError):
    """Raised when an execution loop or unproductive repetition is detected."""

    pass


class AuraCore:
    """Central orchestration runtime for SUSA AURA."""

    def __init__(
        self,
        workspace_root: str | Path = ".",
        backend: ModelBackend | None = None,
        generator: AuraGenerator | None = None,
        tools: ToolRegistry | None = None,
        memory: AuraMemory | None = None,
        connect: AuraConnect | None = None,
        sandbox: IsolatedCodeRunner | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()

        # Model backend initialization
        if backend is not None:
            self.backend = backend
            self.generator = getattr(backend, "generator", generator)
        else:
            self.generator = generator
            self.backend = NativeAuraBackend(generator=self.generator)

        # Initialize subsystems
        self.sandbox = sandbox or IsolatedCodeRunner(default_timeout_sec=5.0)
        self.tools = tools or ToolRegistry(workspace_root=self.workspace_root, sandbox=self.sandbox)
        self.memory = memory or AuraMemory(storage_path=self.workspace_root / ".aura" / "memory.json", enabled=True)
        self.connect = connect or AuraConnect(enable_online=True)

        # Backward compatibility properties
        self.workspace = self.tools.workspace
        self.modified_files = self.tools.modified_files
        self.action_history = self.tools.action_history

        # Session state
        self.session = SessionState()
        if system_prompt:
            self.session.system_prompt = system_prompt

    # --------------------------------------------------------------------------
    # Intent Classification & Planning
    # --------------------------------------------------------------------------

    def classify_intent(self, user_input: str) -> UserIntent:
        """Classifies user intent to determine execution strategy."""
        text = user_input.lower().strip()

        # 1. Multi-step tasks (e.g. 'task A then task B')
        if " then " in text or text.startswith("step 1") or " and verify" in text:
            return UserIntent.MULTI_STEP_TASK

        # 2. Workspace / File operations
        workspace_triggers = [
            "inspect workspace",
            "file tree",
            "read file",
            "list files",
            "propose diff",
            "apply edit",
            "modify file",
            "write file",
            "create file",
        ]
        if any(t in text for t in workspace_triggers) or text.startswith("read ") or text.startswith("cat "):
            return UserIntent.WORKSPACE_OPERATION

        # 3. Code execution / reasoning
        code_triggers = ["run python", "run code", "sandbox", "evaluate python", "syntax check", "execute script"]
        if any(t in text for t in code_triggers) or (text.startswith("def ") or text.startswith("class ")):
            return UserIntent.CODE_REASONING

        # 4. Research / Documentation
        research_triggers = [
            "search",
            "what is",
            "documentation",
            "arxiv",
            "latest",
            "how does",
            "research",
            "find info",
        ]
        if any(t in text for t in research_triggers):
            return UserIntent.RESEARCH_QA

        return UserIntent.DIRECT_ANSWER

    def create_plan(self, request: AuraRequest) -> TaskPlan:
        """Constructs a bounded task execution plan based on classified intent."""
        intent = request.intent or self.classify_intent(request.user_input)
        plan = TaskPlan(goal=request.user_input, intent=intent, max_steps=request.max_steps)

        text = request.user_input.strip()

        # Multi-step sequential decomposition (e.g. "task A then task B")
        if intent == UserIntent.MULTI_STEP_TASK or " then " in text.lower():
            subtasks = [s.strip() for s in re.split(r"\b(?:then|and then)\b", text, flags=re.IGNORECASE) if s.strip()]
            step_id = 1
            for sub in subtasks:
                sub_intent = self.classify_intent(sub)
                if sub_intent != UserIntent.MULTI_STEP_TASK:
                    sub_req = AuraRequest(user_input=sub, intent=sub_intent)
                    sub_plan = self.create_plan(sub_req)
                    for step in sub_plan.steps:
                        step.step_id = step_id
                        plan.steps.append(step)
                        step_id += 1
            return plan

        if intent == UserIntent.WORKSPACE_OPERATION:
            if "inspect" in text.lower() or "tree" in text.lower():
                plan.steps.append(
                    TaskStep(
                        step_id=1,
                        description="Inspect workspace directory layout",
                        tool_name="inspect_workspace",
                        tool_args={},
                        expected_outcome="Directory file tree",
                    )
                )
            elif "read" in text.lower():
                # Extract file path heuristic
                parts = text.split()
                target_file = parts[-1] if len(parts) > 1 else ""
                plan.steps.append(
                    TaskStep(
                        step_id=1,
                        description=f"Read file content: {target_file}",
                        tool_name="read_file",
                        tool_args={"relative_path": target_file},
                        expected_outcome=f"Content of {target_file}",
                    )
                )

        elif intent == UserIntent.CODE_REASONING:
            if "run" in text.lower() or "execute" in text.lower():
                # Extract code if present
                code_snippet = text
                if "```python" in text:
                    code_snippet = text.split("```python")[1].split("```")[0].strip()
                elif "```" in text:
                    code_snippet = text.split("```")[1].split("```")[0].strip()

                plan.steps.append(
                    TaskStep(
                        step_id=1,
                        description="Run Python code in isolated sandbox",
                        tool_name="run_python_sandbox",
                        tool_args={"code": code_snippet},
                        expected_outcome="Successful sandbox execution result",
                    )
                )

        return plan

    # --------------------------------------------------------------------------
    # Central Execution & Verification Engine
    # --------------------------------------------------------------------------

    def solve(self, request: AuraRequest | str) -> AuraResponse:
        """Executes a structured request through planning, tools, verification, and reasoning."""
        start_time = time.time()
        if isinstance(request, str):
            request = AuraRequest(user_input=request)

        self.session.turn_count += 1
        self.memory.add_conversation_turn("user", request.user_input)

        intent = request.intent or self.classify_intent(request.user_input)
        plan = self.create_plan(request)

        sources: list[SourceReference] = []
        evidence: list[EvidenceRecord] = []
        context_parts = []
        limitations = []

        # 1. Retrieve relevant memory facts
        if request.enable_memory and self.session.memory_enabled:
            relevant_memories = self.memory.retrieve_relevant(request.user_input, top_k=2)
            if relevant_memories:
                mem_text = "\n".join(f"[Memory - {m.key}]: {m.content}" for m in relevant_memories)
                context_parts.append(mem_text)

        # 2. Web research grounding if requested or intent is research
        if (request.enable_search or intent == UserIntent.RESEARCH_QA) and self.session.research_enabled:
            retrieval_res: RetrievalResponse = self.connect.research(request.user_input, max_results=3)
            if retrieval_res.is_available and retrieval_res.sources:
                sources = retrieval_res.sources
                context_parts.append(f"[Web Research]:\n{retrieval_res.context_text}")
                for s in sources:
                    evidence.append(
                        EvidenceRecord(
                            claim=s.snippet[:120],
                            source_url=s.url,
                            snippet=s.snippet,
                            confidence=1.0,
                        )
                    )
            elif not retrieval_res.is_available:
                limitations.append(retrieval_res.error_message or "Online research unavailable.")

        # 3. Execute plan steps with loop detection and state verification
        steps_executed = 0
        call_history: list[str] = []

        if request.enable_tools and plan.steps:
            for step in plan.steps:
                if steps_executed >= request.max_steps:
                    limitations.append(f"Reached maximum allowed step budget ({request.max_steps}).")
                    break

                if step.tool_name:
                    # Loop detection: check if identical call has been run repeatedly
                    call_sig = f"{step.tool_name}:{sorted(step.tool_args.items())}"
                    call_history.append(call_sig)
                    if call_history.count(call_sig) > 2:
                        raise LoopDetectedError(
                            f"Execution loop detected: Tool '{step.tool_name}' invoked multiple times "
                            "with identical arguments without state progression."
                        )

                    step.status = "running"
                    step_start = time.time()
                    tool_res: ToolExecutionResult = self.tools.execute(step.tool_name, **step.tool_args)
                    step.duration_sec = time.time() - step_start
                    steps_executed += 1

                    if tool_res.success:
                        step.status = "completed"
                        step.result = tool_res.data
                        # Verify state
                        step.verified = self._verify_step_state(step)
                        context_parts.append(f"[Tool Output - {step.tool_name}]:\n{str(tool_res.data)}")
                    else:
                        step.status = "failed"
                        step.error_message = tool_res.error_message
                        context_parts.append(f"[Tool Failed - {step.tool_name}]: {tool_res.error_message}")
                        limitations.append(f"Step '{step.description}' failed: {tool_res.error_message}")

        # 4. Synthesize final answer via ModelBackend
        full_context = "\n\n".join(context_parts)
        prompt = (
            f"<|bos|>[SYSTEM]: {self.session.system_prompt}\n"
        )
        if full_context:
            prompt += f"[CONTEXT]:\n{full_context}\n"
        prompt += f"[USER]: {request.user_input}\n[ASSISTANT]: "

        generation_opts = GenerationOptions(
            max_new_tokens=160,
            temperature=0.7,
            stop_strings=["[USER]:", "[SYSTEM]:", "<|eos|>"],
        )
        assistant_response = self.backend.generate(prompt=prompt, options=generation_opts)

        # Record assistant turn in working memory
        self.memory.add_conversation_turn("assistant", assistant_response)

        # Store task outcome in task memory if steps were executed
        if steps_executed > 0:
            self.memory.store_fact(
                key=f"task_{plan.plan_id}",
                content=f"Executed plan for '{request.user_input[:50]}': {len(plan.steps)} steps, success: {not plan.has_failures()}",
                category=MemoryCategory.TASK_OUTCOME,
                user_approved=True,
            )

        duration = time.time() - start_time
        return AuraResponse(
            response_text=assistant_response,
            intent=intent,
            plan=plan,
            steps_executed=steps_executed,
            sources=sources,
            evidence=evidence,
            context_used=full_context,
            execution_time_sec=duration,
            success=not plan.has_failures(),
            error_message="; ".join(limitations) if limitations else None,
            limitations=limitations,
        )

    def _verify_step_state(self, step: TaskStep) -> bool:
        """Verifies that an executed step actually altered the target system state."""
        if step.tool_name == "apply_edit":
            rel_path = step.tool_args.get("relative_path")
            if rel_path:
                target = (self.workspace_root / rel_path).resolve()
                return target.exists() and target.read_text(encoding="utf-8") == step.tool_args.get("new_content")
        elif step.tool_name == "run_python_sandbox":
            if isinstance(step.result, ExecutionResult):
                return step.result.success
        return True

    # --------------------------------------------------------------------------
    # Backward Compatibility Methods (Matching CodeMindAgent & AuraCore API)
    # --------------------------------------------------------------------------

    def ask(
        self,
        user_input: str,
        enable_search: bool = False,
        temperature: float = 0.7,
        max_new_tokens: int = 128,
    ) -> dict[str, Any]:
        """Backward-compatible query method returning legacy dict format."""
        req = AuraRequest(user_input=user_input, enable_search=enable_search)
        res = self.solve(req)
        return {
            "response": res.response_text,
            "sources": res.sources,
            "context_used": res.context_used,
            "turn": self.session.turn_count,
        }

    def inspect_workspace(self) -> str:
        """Returns visual file tree of the target project workspace."""
        res = self.tools.execute("inspect_workspace")
        return str(res.data)

    def propose_edit(self, relative_path: str, new_content: str) -> str:
        """Generates a standard unified diff of proposed changes without writing to disk."""
        res = self.tools.execute("propose_edit", relative_path=relative_path, new_content=new_content)
        return str(res.data)

    def apply_edit(
        self,
        relative_path: str,
        new_content: str,
        user_approved: bool = False,
    ) -> bool:
        """Applies proposed edit to disk strictly after explicit user approval with .bak backup."""
        res = self.tools.execute(
            "apply_edit",
            relative_path=relative_path,
            new_content=new_content,
            user_approved=user_approved,
        )
        if not res.success:
            if "rejected" in (res.error_message or "") or "authorization" in (res.error_message or ""):
                raise PermissionError(res.error_message)
            raise RuntimeError(res.error_message)
        return True

    def run_sandboxed_test(self, python_code: str) -> ExecutionResult:
        """Executes a test snippet safely within the isolated sandbox."""
        res = self.tools.execute("run_python_sandbox", code=python_code)
        if isinstance(res.data, ExecutionResult):
            return res.data
        return ExecutionResult(success=False, stdout="", stderr=res.error_message or "Execution failed", returncode=1)

    def summary(self) -> str:
        """Produces a report of all actions and file modifications."""
        lines = [
            "=" * 70,
            "  SUSA AURA — CORE EXECUTION SUMMARY",
            "=" * 70,
            f"  Workspace:       {self.workspace.workspace_root}",
            f"  Actions Logged:  {len(self.action_history)}",
            f"  Modified Files:  {len(self.modified_files)}",
            f"  Memory Records:  {len(self.memory.records)}",
        ]
        for f in self.modified_files:
            lines.append(f"    - {f}")
        lines.append("=" * 70)
        return "\n".join(lines)


# Aliases for backward compatibility
AuraAgent = AuraCore
CodeMindAgent = AuraCore
