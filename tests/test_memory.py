"""
Unit tests for SUSA AURA — AURA Memory Subsystem.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from aura.memory import AuraMemory, ConversationTurn, MemoryRecord


@pytest.fixture
def temp_memory_file() -> Path:
    tmp = tempfile.mktemp(suffix=".json")
    return Path(tmp)


def test_memory_session_turns():
    """Verify conversational dialogue turns are tracked in session memory."""
    mem = AuraMemory(enabled=True)
    mem.add_conversation_turn("user", "Hello AURA")
    mem.add_conversation_turn("assistant", "Hello! How can I assist you today?")

    assert len(mem.conversation_history) == 2
    assert mem.conversation_history[0].role == "user"
    assert mem.conversation_history[1].content == "Hello! How can I assist you today?"

    context_str = mem.format_context_for_prompt()
    assert "USER: Hello AURA" in context_str
    assert "ASSISTANT: Hello! How can I assist you today?" in context_str

    mem.clear_conversation()
    assert len(mem.conversation_history) == 0


def test_memory_persistence_and_reload(temp_memory_file):
    """Verify facts are persisted to disk and reloaded accurately."""
    mem = AuraMemory(storage_path=temp_memory_file, enabled=True)
    stored = mem.store_fact(
        key="preferred_os",
        content="Fedora Linux",
        category="preference",
        tags=["os", "developer"],
        user_approved=True,
    )
    assert stored is True
    assert temp_memory_file.exists()

    # Create new instance pointing to same file
    mem_reloaded = AuraMemory(storage_path=temp_memory_file, enabled=True)
    fact = mem_reloaded.get_fact("preferred_os")
    assert fact == "Fedora Linux"


def test_memory_permission_enforcement():
    """Verify non-negotiable rule: user consent required before persisting memories."""
    mem = AuraMemory(enabled=True)
    with pytest.raises(PermissionError, match="Explicit consent is required"):
        mem.store_fact(key="secret_note", content="confidential", user_approved=False)


def test_memory_relevant_retrieval():
    """Verify query keyword overlap retrieves appropriate memory records."""
    mem = AuraMemory(enabled=True)
    mem.store_fact(key="model_target", content="AURA 1B Transformer with GQA", tags=["model", "architecture"])
    mem.store_fact(key="favorite_food", content="Pizza", tags=["food"])

    relevant = mem.retrieve_relevant("Tell me about the 1B model architecture", top_k=1)
    assert len(relevant) == 1
    assert relevant[0].key == "model_target"


def test_memory_deletion():
    """Verify memory records can be explicitly edited or deleted."""
    mem = AuraMemory(enabled=True)
    mem.store_fact(key="temp_key", content="temp value")
    assert mem.get_fact("temp_key") == "temp value"

    deleted = mem.delete_fact("temp_key")
    assert deleted is True
    assert mem.get_fact("temp_key") is None
