"""
SUSA AURA — AURA Connect: Online Research, Documentation Retrieval & Source Tracking.

Provides modular web search interfaces, source attribution, URL tracking,
HTML text extraction, prompt injection defense, caching, and offline safety fallbacks.
Enforces non-negotiable safety rules: untrusted web content is treated as data,
never as system instructions, and credentials/secrets are scrubbed automatically.
"""

from __future__ import annotations

import html
import os
import re
import time
import urllib.parse
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from aura.data.cleaner import DataCleaner


class EvidenceType(str, Enum):
    """Categorization of evidence confidence and provenance."""

    VERIFIED_SOURCE = "verified_source"
    CONTEXTUAL_SNIPPET = "contextual_snippet"
    UNVERIFIED_INFERENCE = "unverified_inference"
    UNKNOWN = "unknown"


@dataclass
class SourceReference:
    """Attribution metadata for retrieved documentation or web search result."""

    url: str
    title: str
    snippet: str
    timestamp: float = field(default_factory=time.time)
    source_type: str = "web"  # "web", "doc", "local"
    evidence_type: str = EvidenceType.VERIFIED_SOURCE.value


@dataclass
class RetrievalResponse:
    """Structured response containing synthesized context and tracked sources."""

    query: str
    sources: list[SourceReference]
    context_text: str
    is_available: bool = True
    error_message: str | None = None
    cached: bool = False


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
                            evidence_type=EvidenceType.VERIFIED_SOURCE.value,
                        )
                    )
        return results[:max_results]


class ResearchCache:
    """In-memory TTL cache for research queries to prevent duplicate network calls."""

    def __init__(self, default_ttl_sec: float = 3600.0) -> None:
        self.default_ttl_sec = default_ttl_sec
        self.cache: dict[str, tuple[float, RetrievalResponse]] = {}

    def get(self, query: str) -> RetrievalResponse | None:
        key = query.strip().lower()
        if key in self.cache:
            created_at, response = self.cache[key]
            if time.time() - created_at <= self.default_ttl_sec:
                response.cached = True
                return response
            else:
                del self.cache[key]
        return None

    def put(self, query: str, response: RetrievalResponse) -> None:
        key = query.strip().lower()
        self.cache[key] = (time.time(), response)

    def clear(self) -> None:
        self.cache.clear()


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
        cache: ResearchCache | None = None,
    ) -> None:
        env_local = os.environ.get("AURA_LOCAL_ONLY", "0").lower() in ("1", "true", "yes")
        env_offline = os.environ.get("AURA_OFFLINE", "0").lower() in ("1", "true", "yes")
        if env_local or env_offline:
            enable_online = False

        self.enable_online = enable_online
        self.provider = provider or (OfflineSearchProvider() if not enable_online else MockSearchProvider())
        self.cleaner = data_cleaner or DataCleaner()
        self.cache = cache or ResearchCache()

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

    def research(self, query: str, max_results: int = 3, bypass_cache: bool = False) -> RetrievalResponse:
        """Executes a search query, sanitizes results, caches outcomes, and constructs source-grounded context."""
        if not bypass_cache:
            cached_resp = self.cache.get(query)
            if cached_resp is not None:
                return cached_resp

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
                resp = RetrievalResponse(
                    query=query,
                    sources=[],
                    context_text="",
                    is_available=True,
                    error_message="No matching results found for query.",
                )
                self.cache.put(query, resp)
                return resp

            sanitized_sources = []
            context_blocks = []
            for s in raw_sources:
                clean_snippet = self.sanitize_retrieved_text(s.snippet)
                sanitized_sources.append(
                    SourceReference(
                        url=s.url,
                        title=s.title,
                        snippet=clean_snippet,
                        timestamp=s.timestamp,
                        source_type=s.source_type,
                        evidence_type=getattr(s, "evidence_type", EvidenceType.VERIFIED_SOURCE.value),
                    )
                )
                context_blocks.append(f"Source: [{s.title}]({s.url})\nSnippet: {clean_snippet}")

            response = RetrievalResponse(
                query=query,
                sources=sanitized_sources,
                context_text="\n\n".join(context_blocks),
                is_available=True,
            )
            self.cache.put(query, response)
            return response

        except Exception as e:
            return RetrievalResponse(
                query=query,
                sources=[],
                context_text="",
                is_available=False,
                error_message=f"Retrieval error: {str(e)}",
            )

    def format_citations(self, sources: list[SourceReference]) -> str:
        """Formats source references into a standardized bibliography."""
        if not sources:
            return "No external sources cited."
        lines = ["[CITATIONS & SOURCES]:"]
        for idx, s in enumerate(sources, start=1):
            lines.append(f"[{idx}] {s.title} — {s.url} ({s.evidence_type})")
        return "\n".join(lines)
