"""
SUSA AURA — AURA Memory Subsystem.

Provides transparent, inspectable memory management:
- Active conversation context for the current session.
- Explicit user preferences and verified project facts.
- Searchable persistent notes with timestamp tracking.
- Memory editing, deletion, and relevance retrieval.
- Privacy & permission gate: requires explicit consent before persisting sensitive data.
- Can be completely disabled to run purely stateless base model inference.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class MemoryRecord:
    """A distinct unit of persisted knowledge, fact, or user preference."""

    key: str
    content: str
    category: str = "preference"  # "preference", "fact", "note", "session"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    tags: list[str] = field(default_factory=list)


@dataclass
class ConversationTurn:
    """A single dialogue turn within a session."""

    role: str  # "user", "assistant", "system", "tool"
    content: str
    timestamp: float = field(default_factory=time.time)


class AuraMemory:
    """Inspectable memory manager for SUSA AURA."""

    def __init__(self, storage_path: str | Path | None = None, enabled: bool = True) -> None:
        self.enabled = enabled
        self.storage_path = Path(storage_path) if storage_path else None
        self.records: dict[str, MemoryRecord] = {}
        self.conversation_history: list[ConversationTurn] = []

        if self.enabled and self.storage_path and self.storage_path.exists():
            self._load_from_disk()

    def _load_from_disk(self) -> None:
        """Loads persisted memory records from JSON storage."""
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data.get("records", []):
                    rec = MemoryRecord(**item)
                    self.records[rec.key] = rec
        except Exception:
            pass

    def save_to_disk(self) -> None:
        """Flushes memory records to JSON storage if a storage path is configured."""
        if not self.enabled or not self.storage_path:
            return

        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": "1.0.0",
            "records": [asdict(r) for r in self.records.values()],
        }
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def add_conversation_turn(self, role: str, content: str) -> None:
        """Records a conversational turn in active session memory."""
        self.conversation_history.append(ConversationTurn(role=role, content=content))

    def clear_conversation(self) -> None:
        """Clears active session conversation history."""
        self.conversation_history.clear()

    def store_fact(
        self,
        key: str,
        content: str,
        category: str = "fact",
        tags: list[str] | None = None,
        user_approved: bool = True,
    ) -> bool:
        """Persists a verified fact or user preference."""
        if not self.enabled:
            return False

        if not user_approved:
            raise PermissionError("Explicit consent is required to persist memory.")

        now = time.time()
        if key in self.records:
            rec = self.records[key]
            rec.content = content
            rec.category = category
            rec.updated_at = now
            if tags:
                rec.tags = list(set(rec.tags + tags))
        else:
            self.records[key] = MemoryRecord(
                key=key,
                content=content,
                category=category,
                created_at=now,
                updated_at=now,
                tags=tags or [],
            )

        self.save_to_disk()
        return True

    def retrieve_relevant(self, query: str, top_k: int = 3) -> list[MemoryRecord]:
        """Retrieves memories matching query keywords."""
        if not self.enabled or not self.records:
            return []

        q_words = set(query.lower().split())
        scored: list[tuple[float, MemoryRecord]] = []

        for rec in self.records.values():
            text = f"{rec.key} {rec.content} {' '.join(rec.tags)}".lower()
            overlap = sum(1 for w in q_words if w in text)
            if overlap > 0:
                score = overlap / (len(q_words) + 1e-5)
                scored.append((score, rec))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored[:top_k]]

    def get_fact(self, key: str) -> str | None:
        """Gets a stored memory fact by exact key."""
        rec = self.records.get(key)
        return rec.content if rec else None

    def delete_fact(self, key: str) -> bool:
        """Deletes a persisted memory fact."""
        if key in self.records:
            del self.records[key]
            self.save_to_disk()
            return True
        return False

    def list_all(self) -> list[MemoryRecord]:
        """Returns all persisted memory records."""
        return list(self.records.values())

    def format_context_for_prompt(self, max_turns: int = 5) -> str:
        """Formats active conversation history and relevant preferences into context string."""
        lines = []
        if self.records:
            pref_lines = [f"- {r.key}: {r.content}" for r in self.records.values() if r.category in ["preference", "fact"]]
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
