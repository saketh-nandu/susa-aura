"""
Tests for Advanced AURA Connect Subsystem:
- ResearchCache TTL, query hits, and misses
- EvidenceType classification and citation bibliography formatting
"""

from __future__ import annotations

import time
import pytest

from aura.connect.retriever import (
    AuraConnect,
    EvidenceType,
    MockSearchProvider,
    ResearchCache,
    RetrievalResponse,
    SourceReference,
)


def test_research_cache_hit_and_ttl():
    """Verify research cache returns cached response and purges after TTL."""
    cache = ResearchCache(default_ttl_sec=0.1)
    mock_resp = RetrievalResponse(
        query="transformer",
        sources=[],
        context_text="Attention mechanism",
        is_available=True,
    )

    cache.put("transformer", mock_resp)

    # Immediate get hits cache
    cached = cache.get("transformer")
    assert cached is not None
    assert cached.context_text == "Attention mechanism"
    assert cached.cached is True

    # After expiration, returns None
    time.sleep(0.15)
    assert cache.get("transformer") is None


def test_connect_with_cache_integration():
    """Verify AuraConnect transparently utilizes ResearchCache."""
    cache = ResearchCache(default_ttl_sec=3600.0)
    connect = AuraConnect(provider=MockSearchProvider(), cache=cache)

    res1 = connect.research("pytorch rmsnorm")
    assert res1.is_available is True
    assert res1.cached is False

    # Second call should be a cache hit
    res2 = connect.research("pytorch rmsnorm")
    assert res2.cached is True


def test_format_citations():
    """Verify formatting of source bibliography with evidence types."""
    connect = AuraConnect()
    sources = [
        SourceReference(
            url="https://arxiv.org/abs/1706.03762",
            title="Attention Is All You Need",
            snippet="Transformer relies on self-attention.",
            evidence_type=EvidenceType.VERIFIED_SOURCE.value,
        )
    ]
    bib = connect.format_citations(sources)
    assert "[CITATIONS & SOURCES]:" in bib
    assert "Attention Is All You Need" in bib
    assert "https://arxiv.org/abs/1706.03762" in bib
    assert "verified_source" in bib


def test_offline_provider_when_local_only(monkeypatch):
    """Verify AuraConnect enforces OfflineSearchProvider when AURA_LOCAL_ONLY=1."""
    monkeypatch.setenv("AURA_LOCAL_ONLY", "1")
    connect = AuraConnect()
    assert connect.enable_online is False
    from aura.connect.retriever import OfflineSearchProvider
    assert isinstance(connect.provider, OfflineSearchProvider)
    res = connect.research("quantum computing")
    assert res.is_available is False
    assert "offline" in res.error_message.lower()
