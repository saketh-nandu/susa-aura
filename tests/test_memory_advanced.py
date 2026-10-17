"""
Tests for Advanced AURA Memory Subsystem:
- Memory categories (Working, User, Knowledge, Task)
- Automatic write-time secret scrubbing
- TTL expiration and auto-purging
- Conflict detection and tracking
"""

from __future__ import annotations

import time
import pytest

from aura.memory.manager import AuraMemory, MemoryCategory


def test_memory_categories_and_filtering(tmp_path):
    """Verify distinct memory categories and category-based retrieval."""
    mem_file = tmp_path / "memory.json"
    memory = AuraMemory(storage_path=mem_file, enabled=True)

    # Store across categories
    memory.store_fact("pref_editor", "VS Code with dark mode", category=MemoryCategory.USER_PREFERENCE)
    memory.store_fact("project_spec", "Target 1B parameters", category=MemoryCategory.USER_FACT)
    memory.store_fact("doc_rmsnorm", "RMSNorm avoids mean centering", category=MemoryCategory.KNOWLEDGE_DOC)
    memory.store_fact("task_01", "Completed baseline tests", category=MemoryCategory.TASK_OUTCOME)

    # List by category
    prefs = memory.list_all(category=MemoryCategory.USER_PREFERENCE)
    assert len(prefs) == 1
    assert prefs[0].key == "pref_editor"

    facts = memory.list_all(category=MemoryCategory.USER_FACT)
    assert len(facts) == 1
    assert facts[0].key == "project_spec"

    tasks = memory.list_all(category=MemoryCategory.TASK_OUTCOME)
    assert len(tasks) == 1
    assert tasks[0].key == "task_01"


def test_write_time_secret_scrubbing_in_memory(tmp_path):
    """Verify credentials and API keys are automatically scrubbed when storing memory."""
    mem_file = tmp_path / "memory.json"
    memory = AuraMemory(storage_path=mem_file, enabled=True)

    # Attempt to store a sensitive credential in user memory
    sensitive_token = "ghp_11223344556677889900aabbccddeeffgghh"
    memory.store_fact(
        "git_token",
        f"User GitHub token is: {sensitive_token}",
        category=MemoryCategory.USER_PREFERENCE,
        user_approved=True,
    )

    stored_val = memory.get_fact("git_token")
    assert stored_val is not None
    assert sensitive_token not in stored_val
    assert "[REDACTED_GITHUB_TOKEN]" in stored_val or "[REDACTED_GITHUB_PAT]" in stored_val


def test_ttl_expiration_and_purging(tmp_path):
    """Verify records with TTL expire and are automatically purged."""
    mem_file = tmp_path / "memory.json"
    memory = AuraMemory(storage_path=mem_file, enabled=True)

    # Store with a very short TTL (0.1 seconds)
    memory.store_fact(
        "temp_code",
        "Verification code 1234",
        category=MemoryCategory.WORKING,
        ttl_seconds=0.1,
    )

    # Immediate access succeeds
    assert memory.get_fact("temp_code") == "Verification code 1234"

    # Wait for expiration
    time.sleep(0.15)

    # Access after expiration returns None
    assert memory.get_fact("temp_code") is None

    # Purge removes it from records
    purged = memory.purge_expired()
    assert purged >= 0
    assert "temp_code" not in memory.records


def test_conflict_detection_and_tracking(tmp_path):
    """Verify that updating a key with conflicting content records conflict history."""
    mem_file = tmp_path / "memory.json"
    memory = AuraMemory(storage_path=mem_file, enabled=True)

    memory.store_fact("status", "System is initializing", category=MemoryCategory.USER_FACT)
    memory.store_fact("status", "System is active and operational", category=MemoryCategory.USER_FACT)

    rec = memory.records["status"]
    assert rec.content == "System is active and operational"
    assert len(rec.conflicts_with) == 1
    assert "Replaced previous content" in rec.conflicts_with[0]
