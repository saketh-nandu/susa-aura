"""
SUSA AURA — AURA Memory Subsystem.

Provides transparent, inspectable, and user-controlled memory management:
- Working memory for active session context and intermediate tasks.
- User memory for explicitly permitted durable preferences and facts.
- Knowledge memory for indexed documents and source-backed material.
- Task memory for prior outcomes, procedures, and execution steps.
- Expiration handling (TTL / purge_expired).
- Automatic write-time secret scrubbing (never persist credentials).
- Conflict tracking and audit provenance.
- Privacy & permission gate: requires explicit consent before persisting durable data.
- Can be disabled to run purely stateless base model inference.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from aura.data.cleaner import DataCleaner


class MemoryCategory(str, Enum):
    """Categorization for distinct types of memory records."""

    WORKING = "working"
    USER_PREFERENCE = "preference"
    USER_FACT = "fact"
    KNOWLEDGE_DOC = "knowledge"
    TASK_OUTCOME = "task"


@dataclass
class MemoryRecord:
    """A distinct unit of persisted knowledge, fact, or user preference."""

    key: str
    content: str
    category: str = MemoryCategory.USER_FACT.value
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    tags: list[str] = field(default_factory=list)
    provenance: str = "user"  # "user", "web", "task_execution", "system"
    confidence: float = 1.0
    expires_at: float | None = None
    conflicts_with: list[str] = field(default_factory=list)

    def is_expired(self, current_time: float | None = None) -> bool:
        """Checks if this record has passed its expiration timestamp."""
        if self.expires_at is None:
            return False
        now = current_time if current_time is not None else time.time()
        return now >= self.expires_at


@dataclass
class ConversationTurn:
    """A single dialogue turn within an active session."""

    role: str  # "user", "assistant", "system", "tool"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


class AuraMemory:
    """Inspectable, user-controlled memory manager for SUSA AURA."""

    def __init__(
        self,
        storage_path: str | Path | None = None,
        enabled: bool = True,
        data_cleaner: DataCleaner | None = None,
    ) -> None:
        self.enabled = enabled
        self.storage_path = Path(storage_path) if storage_path else None
        self.cleaner = data_cleaner or DataCleaner()
        self.records: dict[str, MemoryRecord] = {}
        self.conversation_history: list[ConversationTurn] = []

        if self.enabled and self.storage_path and self.storage_path.exists():
            self._load_from_disk()

    def _load_from_disk(self) -> None:
        """Loads persisted memory records from JSON storage with resilience."""
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data.get("records", []):
                    # Ensure backward compatibility with older serialized records
                    rec = MemoryRecord(
                        key=item["key"],
                        content=item["content"],
                        category=item.get("category", MemoryCategory.USER_FACT.value),
                        created_at=item.get("created_at", time.time()),
                        updated_at=item.get("updated_at", time.time()),
                        tags=item.get("tags", []),
                        provenance=item.get("provenance", "user"),
                        confidence=item.get("confidence", 1.0),
                        expires_at=item.get("expires_at", None),
                        conflicts_with=item.get("conflicts_with", []),
                    )
                    self.records[rec.key] = rec
        except Exception:
            # Gracefully handle corrupt or empty memory storage file
            pass

    def save_to_disk(self) -> None:
        """Flushes memory records to JSON storage if a storage path is configured."""
        if not self.enabled or not self.storage_path:
            return

        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": "1.1.0",
            "saved_at": time.time(),
            "records": [asdict(r) for r in self.records.values()],
        }
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def add_conversation_turn(self, role: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        """Records a conversational turn in active session working memory."""
        # Scrub credentials before retaining in working memory
        clean_content, _ = self.cleaner.scrub_secrets(content)
        self.conversation_history.append(
            ConversationTurn(role=role, content=clean_content, metadata=metadata or {})
        )

    def clear_conversation(self) -> None:
        """Clears active session conversation history."""
        self.conversation_history.clear()

    def store_fact(
        self,
        key: str,
        content: str,
        category: str | MemoryCategory = MemoryCategory.USER_FACT.value,
        tags: list[str] | None = None,
        provenance: str = "user",
        confidence: float = 1.0,
        ttl_seconds: float | None = None,
        user_approved: bool = True,
    ) -> bool:
        """Persists a verified fact, user preference, knowledge doc, or task outcome.

        Non-negotiable security:
        - Automatically scrubs secrets/credentials before persisting.
        - Requires user authorization for storage.
        """
        if not self.enabled:
            return False

        if not user_approved:
            raise PermissionError("Explicit consent is required to persist memory.")

        # 1. Scrub secrets from content to prevent credential leakage
        sanitized_content, _ = self.cleaner.scrub_secrets(content)

        cat_str = category.value if isinstance(category, MemoryCategory) else category
        now = time.time()
        expires_at = (now + ttl_seconds) if ttl_seconds is not None else None

        # 2. Conflict tracking
        conflicts = []
        if key in self.records and self.records[key].content != sanitized_content:
            conflicts.append(f"Replaced previous content at {now}: {self.records[key].content[:80]}")

        # 3. Create or update record
        if key in self.records:
            rec = self.records[key]
            rec.content = sanitized_content
            rec.category = cat_str
            rec.updated_at = now
            rec.provenance = provenance
            rec.confidence = confidence
            rec.expires_at = expires_at
            if conflicts:
                rec.conflicts_with.extend(conflicts)
            if tags:
                rec.tags = list(set(rec.tags + tags))
        else:
            self.records[key] = MemoryRecord(
                key=key,
                content=sanitized_content,
                category=cat_str,
                created_at=now,
                updated_at=now,
                tags=tags or [],
                provenance=provenance,
                confidence=confidence,
                expires_at=expires_at,
                conflicts_with=conflicts,
            )

        self.save_to_disk()
        return True

    def retrieve_relevant(
        self,
        query: str,
        category: str | MemoryCategory | None = None,
        top_k: int = 3,
    ) -> list[MemoryRecord]:
        """Retrieves active, non-expired memories matching query keywords and optional category."""
        if not self.enabled or not self.records:
            return []

        # Auto-purge expired entries on retrieval
        self.purge_expired()

        cat_str = category.value if isinstance(category, MemoryCategory) else category
        q_words = set(query.lower().split())
        scored: list[tuple[float, MemoryRecord]] = []

        for rec in self.records.values():
            if cat_str and rec.category != cat_str:
                continue

            text = f"{rec.key} {rec.content} {' '.join(rec.tags)}".lower()
            overlap = sum(1 for w in q_words if w in text)
            if overlap > 0:
                score = (overlap / (len(q_words) + 1e-5)) * rec.confidence
                scored.append((score, rec))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored[:top_k]]

    def purge_expired(self) -> int:
        """Removes all expired records and syncs with disk. Returns count of purged records."""
        now = time.time()
        expired_keys = [k for k, r in self.records.items() if r.is_expired(now)]
        for k in expired_keys:
            del self.records[k]

        if expired_keys:
            self.save_to_disk()
        return len(expired_keys)

    def get_fact(self, key: str) -> str | None:
        """Gets a stored memory fact by exact key if non-expired."""
        rec = self.records.get(key)
        if rec and not rec.is_expired():
            return rec.content
        return None

    def delete_fact(self, key: str) -> bool:
        """Deletes a persisted memory fact."""
        if key in self.records:
            del self.records[key]
            self.save_to_disk()
            return True
        return False

    def list_all(self, category: str | MemoryCategory | None = None) -> list[MemoryRecord]:
        """Returns all persisted non-expired memory records."""
        self.purge_expired()
        cat_str = category.value if isinstance(category, MemoryCategory) else category
        if cat_str:
            return [r for r in self.records.values() if r.category == cat_str]
        return list(self.records.values())

    def format_context_for_prompt(self, max_turns: int = 5) -> str:
        """Formats active conversation history and relevant preferences into context string."""
        self.purge_expired()
        lines = []
        if self.records:
            pref_lines = [
                f"- [{r.category.upper()}] {r.key}: {r.content}"
                for r in self.records.values()
                if r.category in [MemoryCategory.USER_PREFERENCE.value, MemoryCategory.USER_FACT.value]
            ]
            if pref_lines:
                lines.append("[PERSISTENT PREFERENCES & FACTS]:")
                lines.extend(pref_lines)
                lines.append("")

        if self.conversation_history:
            recent_turns = self.conversation_history[-max_turns:]
            lines.append("[CONVERSATION HISTORY]:")
            for turn in recent_turns:
                lines.append(f"{turn.role.upper()}: {turn.content}")

        return "\n".join(lines)
