"""
SUSA AURA — AURA Core: Conversation Management & Assistant Orchestration Engine.

Orchestrates the entire SUSA AURA system:
- Connects model generation (AuraModel / AuraGenerator) with tools and memory.
- Manages conversational sessions and explicit system instructions.
- Dispatches tool executions through AURA Tools with strict approval gates.
- Grounds answers using retrieved web research from AURA Connect.
- Maintains inspectable session state, history, and actions.
- Provides full backward compatibility with CodeMindAgent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aura.connect.retriever import AuraConnect, RetrievalResponse
from aura.evaluation.sandbox import ExecutionResult, IsolatedCodeRunner
from aura.inference.generator import AuraGenerator
from aura.memory.manager import AuraMemory
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


class AuraCore:
    """Central orchestration runtime for SUSA AURA."""

    def __init__(
        self,
        workspace_root: str | Path = ".",
        generator: AuraGenerator | None = None,
        tools: ToolRegistry | None = None,
        memory: AuraMemory | None = None,
        connect: AuraConnect | None = None,
        sandbox: IsolatedCodeRunner | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.generator = generator

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
    # Backward Compatibility Methods (matching CodeMindAgent API)
    # --------------------------------------------------------------------------

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

    # --------------------------------------------------------------------------
    # AURA Core Orchestration Methods
    # --------------------------------------------------------------------------

    def ask(
        self,
        user_input: str,
        enable_search: bool = False,
        temperature: float = 0.7,
        max_new_tokens: int = 128,
    ) -> dict[str, Any]:
        """Processes a user prompt through memory, retrieval, reasoning, and response generation."""
        self.session.turn_count += 1
        self.memory.add_conversation_turn("user", user_input)

        sources: list[Any] = []
        context_parts = []

        # 1. Retrieve relevant memory facts
        if self.session.memory_enabled:
            relevant_memories = self.memory.retrieve_relevant(user_input, top_k=2)
            if relevant_memories:
                mem_text = "\n".join(f"[Memory - {m.key}]: {m.content}" for m in relevant_memories)
                context_parts.append(mem_text)

        # 2. Retrieve online sources via AURA Connect if requested
        if enable_search and self.session.research_enabled:
            retrieval_res: RetrievalResponse = self.connect.research(user_input, max_results=3)
            if retrieval_res.is_available and retrieval_res.sources:
                sources = retrieval_res.sources
                context_parts.append(f"[Web Research]:\n{retrieval_res.context_text}")

        # 3. Construct prompt
        full_context = "\n\n".join(context_parts)
        constructed_prompt = (
            f"<|bos|>[SYSTEM]: {self.session.system_prompt}\n"
        )
        if full_context:
            constructed_prompt += f"[CONTEXT]:\n{full_context}\n"
        constructed_prompt += f"[USER]: {user_input}\n[ASSISTANT]: "

        # 4. Generate response via model (if loaded) or structured response
        if self.generator is not None:
            generated_text = self.generator.generate(
                prompt=constructed_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                stop_strings=["[USER]:", "[SYSTEM]:", "<|eos|>"],
            )
            # Extract assistant turn
            assistant_response = generated_text.removeprefix(constructed_prompt).strip()
        else:
            assistant_response = (
                f"SUSA AURA received: '{user_input}'. "
                "(Model checkpoint not currently loaded in memory; core orchestrator running in diagnostic mode.)"
            )

        self.memory.add_conversation_turn("assistant", assistant_response)

        return {
            "response": assistant_response,
            "sources": sources,
            "context_used": full_context,
            "turn": self.session.turn_count,
        }


# Aliases for backward compatibility
AuraAgent = AuraCore
CodeMindAgent = AuraCore
