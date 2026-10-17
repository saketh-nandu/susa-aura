"""
SUSA AURA — Unified Model Backend Abstraction.

Provides a common interface for:
- Native AURA model execution (AuraModel / AuraGenerator).
- Diagnostic fallback when weights are not instantiated.
- Optional external model adapters with strict local-only/privacy guarantees.
Enforces non-negotiable rule: Local-only mode must never transmit prompts or data externally.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from aura.inference.generator import AuraGenerator


@dataclass
class GenerationOptions:
    """Hyperparameters and constraints for text generation."""

    max_new_tokens: int = 128
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.95
    stop_strings: list[str] = field(default_factory=lambda: ["<|eos|>", "[USER]:", "[SYSTEM]:"])


@runtime_checkable
class ModelBackend(Protocol):
    """Protocol defining common interface for all AURA model backends."""

    @property
    def name(self) -> str:
        """Identifier of the backend."""
        ...

    @property
    def is_local(self) -> bool:
        """True if the model runs entirely on local hardware."""
        ...

    @property
    def is_available(self) -> bool:
        """True if the backend is configured, initialized, and ready."""
        ...

    def generate(self, prompt: str, options: GenerationOptions | None = None) -> str:
        """Generates text completion from input prompt."""
        ...


class NativeAuraBackend:
    """Backend running the original SUSA AURA Transformer locally."""

    def __init__(self, generator: AuraGenerator | None = None) -> None:
        self.generator = generator

    @property
    def name(self) -> str:
        if self.generator is not None:
            return f"aura-native ({self.generator.model.count_parameters():,} params)"
        return "aura-native (diagnostic mode)"

    @property
    def is_local(self) -> bool:
        return True

    @property
    def is_available(self) -> bool:
        return self.generator is not None

    def generate(self, prompt: str, options: GenerationOptions | None = None) -> str:
        opts = options or GenerationOptions()
        if self.generator is None:
            return (
                f"[AURA Diagnostic Backend]: Received input prompt ({len(prompt)} chars). "
                "Base model weights are not loaded into memory. Operating in deterministic reasoning mode."
            )

        full_output = self.generator.generate(
            prompt=prompt,
            max_new_tokens=opts.max_new_tokens,
            temperature=opts.temperature,
            top_k=opts.top_k,
            top_p=opts.top_p,
            stop_strings=opts.stop_strings,
        )
        return full_output.removeprefix(prompt).strip()


class ExternalModelAdapter:
    """Optional external model adapter with strict privacy and local-only enforcement."""

    def __init__(
        self,
        provider_name: str = "mock-external",
        api_key: str | None = None,
        local_only: bool = True,
    ) -> None:
        self.provider_name = provider_name
        self.api_key = api_key or os.environ.get("AURA_EXTERNAL_API_KEY")
        # Global environment flag takes precedence
        env_local = os.environ.get("AURA_LOCAL_ONLY", "1").lower() in ("1", "true", "yes")
        self.local_only = local_only or env_local

    @property
    def name(self) -> str:
        return f"external-adapter ({self.provider_name})"

    @property
    def is_local(self) -> bool:
        return False

    @property
    def is_available(self) -> bool:
        return not self.local_only and bool(self.api_key)

    def generate(self, prompt: str, options: GenerationOptions | None = None) -> str:
        if self.local_only:
            raise PermissionError(
                "Local-only mode is active (AURA_LOCAL_ONLY=1). "
                "Transmitting prompts or data to external model providers is strictly forbidden."
            )

        if not self.api_key:
            raise RuntimeError(
                f"External provider '{self.provider_name}' requires an API key. "
                "Please configure AURA_EXTERNAL_API_KEY or use the native local model."
            )

        # Fallback deterministic response for configured test adapter
        return f"[External Provider {self.provider_name}]: Synthesized response for prompt."
