"""
SUSA AURA — AURA Connect: Online Research, Documentation Retrieval & Source Tracking.

Provides modular web search interfaces, source attribution, URL tracking,
HTML text extraction, prompt injection defense, and offline safety fallbacks.
Enforces non-negotiable safety rules: untrusted web content is treated as data,
never as system instructions, and credentials/secrets are scrubbed automatically.
"""

from __future__ import annotations

import html
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol

from aura.data.cleaner import DataCleaner


@dataclass
class SourceReference:
    """Attribution metadata for retrieved documentation or web search result."""

    url: str
    title: str
    snippet: str
    timestamp: float = field(default_factory=time.time)
    source_type: str = "web"  # "web", "doc", "local"


@dataclass
class RetrievalResponse:
    """Structured response containing synthesized context and tracked sources."""

    query: str
    sources: list[SourceReference]
    context_text: str
    is_available: bool = True
    error_message: str | None = None


class SearchProvider(Protocol):
    """Protocol for pluggable search engines."""

    def search(self, query: str, max_results: int = 5) -> list[SourceReference]:
        ...


class OfflineSearchProvider:
    """Safe fallback when internet access is disabled or unavailable."""

    def search(self, query: str, max_results: int = 5) -> list[SourceReference]:
        return []


class MockSearchProvider:
    """Deterministic search provider for offline testing and evaluation."""

    def __init__(self, mock_knowledge: dict[str, list[dict[str, str]]] | None = None) -> None:
        self.mock_knowledge = mock_knowledge or {
            "pytorch": [
                {
                    "title": "PyTorch RMSNorm Documentation",
                    "url": "https://pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html",
                    "snippet": "Root Mean Square Layer Normalization normalizes the input across the last dimension.",
                }
            ],
            "swiglu": [
                {
                    "title": "GLU Variants Improve Transformer",
                    "url": "https://arxiv.org/abs/2002.05202",
                    "snippet": "SwiGLU is a Gated Linear Unit with the Swish activation function.",
                }
            ],
            "transformer": [
                {
                    "title": "Attention Is All You Need",
                    "url": "https://arxiv.org/abs/1706.03762",
                    "snippet": "The Transformer is an architecture that relies entirely on self-attention mechanisms.",
                }
            ],
        }

    def search(self, query: str, max_results: int = 5) -> list[SourceReference]:
        results = []
        q_lower = query.lower()
        for key, items in self.mock_knowledge.items():
            if key in q_lower or any(word in q_lower for word in key.split()):
                for item in items[:max_results]:
                    results.append(
                        SourceReference(
                            url=item["url"],
                            title=item["title"],
                            snippet=item["snippet"],
                        )
                    )
        return results[:max_results]


class AuraConnect:
    """AURA Connect subsystem for web research, doc retrieval, and source grounding."""

    # Disallowed prompt injection markers often embedded in web content
    INJECTION_PATTERNS = [
        re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
        re.compile(r"system\s*prompt\s*:", re.IGNORECASE),
        re.compile(r"<\|im_start\|>", re.IGNORECASE),
        re.compile(r"<\|im_end\|>", re.IGNORECASE),
        re.compile(r"<\|bos\|>", re.IGNORECASE),
        re.compile(r"<\|eos\|>", re.IGNORECASE),
    ]

    def __init__(
        self,
        provider: SearchProvider | None = None,
        enable_online: bool = True,
        data_cleaner: DataCleaner | None = None,
    ) -> None:
        self.enable_online = enable_online
        self.provider = provider or (MockSearchProvider() if not enable_online else MockSearchProvider())
        self.cleaner = data_cleaner or DataCleaner()

    @staticmethod
    def extract_text_from_html(html_content: str) -> str:
        """Strips HTML tags, scripts, and styles, returning clean text."""
        # Strip script and style blocks
        clean = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html_content, flags=re.DOTALL | re.IGNORECASE)
        # Strip all HTML tags
        clean = re.sub(r"<[^>]+>", " ", clean)
        # Unescape HTML entities
        clean = html.unescape(clean)
        # Collapse multiple whitespace
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean

    def sanitize_retrieved_text(self, raw_text: str) -> str:
        """Scrubs credentials and defangs prompt injection patterns from external content."""
        # 1. Scrub credentials and sensitive tokens using DataCleaner
        clean_text, _ = self.cleaner.scrub_secrets(raw_text)

        # 2. Defang prompt injection attempts
        for pat in self.INJECTION_PATTERNS:
            clean_text = pat.sub("[REDACTED_INJECTION_ATTEMPT]", clean_text)

        return clean_text

    def research(self, query: str, max_results: int = 3) -> RetrievalResponse:
        """Executes a search query, sanitizes results, and constructs source-grounded context."""
        if not self.enable_online and isinstance(self.provider, OfflineSearchProvider):
            return RetrievalResponse(
                query=query,
                sources=[],
                context_text="",
                is_available=False,
                error_message="AURA Connect is offline. Online research is disabled.",
            )

        try:
            raw_sources = self.provider.search(query, max_results=max_results)
            if not raw_sources:
                return RetrievalResponse(
                    query=query,
                    sources=[],
                    context_text="No external sources found for query.",
                    is_available=True,
                )

            sanitized_sources = []
            context_blocks = []

            for i, src in enumerate(raw_sources, start=1):
                safe_snippet = self.sanitize_retrieved_text(src.snippet)
                safe_title = self.sanitize_retrieved_text(src.title)
                safe_src = SourceReference(
                    url=src.url,
                    title=safe_title,
                    snippet=safe_snippet,
                    timestamp=src.timestamp,
                    source_type=src.source_type,
                )
                sanitized_sources.append(safe_src)
                context_blocks.append(
                    f"[{i}] {safe_src.title} ({safe_src.url}):\n{safe_src.snippet}"
                )

            context_str = "\n\n".join(context_blocks)
            return RetrievalResponse(
                query=query,
                sources=sanitized_sources,
                context_text=context_str,
                is_available=True,
            )

        except Exception as exc:
            return RetrievalResponse(
                query=query,
                sources=[],
                context_text="",
                is_available=False,
                error_message=f"Retrieval error in AURA Connect: {exc}",
            )
