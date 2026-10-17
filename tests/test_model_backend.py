"""
Tests for SUSA AURA ModelBackend Abstraction & Privacy Enforcement.
"""

from __future__ import annotations

import os
import pytest

from aura.model.backend import (
    ExternalModelAdapter,
    GenerationOptions,
    ModelBackend,
    NativeAuraBackend,
)


def test_native_backend_diagnostic_mode():
    """Verify NativeAuraBackend operates in diagnostic mode when no generator is provided."""
    backend = NativeAuraBackend(generator=None)
    assert backend.is_local is True
    assert backend.is_available is False
    assert "diagnostic mode" in backend.name

    opts = GenerationOptions(max_new_tokens=50)
    output = backend.generate("Hello AURA", options=opts)
    assert "AURA Diagnostic Backend" in output
    assert "Base model weights are not loaded" in output


def test_external_adapter_local_only_enforcement(monkeypatch):
    """Verify ExternalModelAdapter strictly blocks external network calls in local-only mode."""
    monkeypatch.setenv("AURA_LOCAL_ONLY", "1")
    adapter = ExternalModelAdapter(provider_name="test-cloud", api_key="test-key", local_only=True)

    assert adapter.is_local is False
    assert adapter.is_available is False

    with pytest.raises(PermissionError) as exc_info:
        adapter.generate("Test prompt")
    assert "Local-only mode is active" in str(exc_info.value)
    assert "strictly forbidden" in str(exc_info.value)


def test_external_adapter_missing_api_key(monkeypatch):
    """Verify ExternalModelAdapter fails gracefully when unconfigured."""
    monkeypatch.setenv("AURA_LOCAL_ONLY", "0")
    adapter = ExternalModelAdapter(provider_name="test-cloud", api_key=None, local_only=False)

    assert adapter.is_available is False
    with pytest.raises(RuntimeError) as exc_info:
        adapter.generate("Test prompt")
    assert "requires an API key" in str(exc_info.value)


def test_external_adapter_mock_generation(monkeypatch):
    """Verify ExternalModelAdapter generates mock response when enabled and configured."""
    monkeypatch.setenv("AURA_LOCAL_ONLY", "0")
    adapter = ExternalModelAdapter(provider_name="mock-provider", api_key="sk-valid-key", local_only=False)

    assert adapter.is_available is True
    res = adapter.generate("Test prompt")
    assert "mock-provider" in res
