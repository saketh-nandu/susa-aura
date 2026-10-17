"""
SUSA AURA — Model Architecture Configuration & Parameter Verification.

Provides strongly typed configuration with comprehensive dimension validation
and exact parameter count calculations for decoder-only Transformers.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    """Configuration for SUSA AURA decoder-only Transformer."""

    vocab_size: int = 32000
    max_seq_len: int = 2048
    d_model: int = 2048
    n_heads: int = 16
    n_kv_heads: int | None = None
    n_layers: int = 20
    d_ff: int | None = None
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = True
    dropout: float = 0.0
    profile_name: str = "custom"
    description: str = ""

    def __post_init__(self) -> None:
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads

        if self.d_ff is None:
            # Standard SwiGLU hidden dimension formula: ~8/3 * d_model rounded to multiple of 256
            raw_d_ff = int(2 * (4 * self.d_model) / 3)
            self.d_ff = int(256 * math.ceil(raw_d_ff / 256))

        self.validate()

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def validate(self) -> None:
        """Validate architectural invariants to guarantee mathematical correctness."""
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {self.max_seq_len}")
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {self.n_heads}")
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {self.n_layers}")
        if self.d_ff is not None and self.d_ff <= 0:
            raise ValueError(f"d_ff must be positive, got {self.d_ff}")

        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})."
            )

        if self.n_kv_heads is not None:
            if self.n_kv_heads <= 0:
                raise ValueError(f"n_kv_heads must be positive, got {self.n_kv_heads}")
            if self.n_heads % self.n_kv_heads != 0:
                raise ValueError(
                    f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads}) for Grouped-Query Attention."
                )

    def calculate_exact_parameters(self) -> dict[str, Any]:
        """Compute the exact parameter count derived from the architecture definition.

        Returns a detailed dictionary broken down by functional component.
        """
        head_dim = self.head_dim
        d_ff = self.d_ff or int(2 * (4 * self.d_model) / 3)
        n_kv = self.n_kv_heads if self.n_kv_heads is not None else self.n_heads

        # 1. Embeddings
        token_emb_params = self.vocab_size * self.d_model

        # 2. Per-layer Attention (Q, K, V, O projections)
        q_proj = self.d_model * (self.n_heads * head_dim)
        k_proj = self.d_model * (n_kv * head_dim)
        v_proj = self.d_model * (n_kv * head_dim)
        o_proj = (self.n_heads * head_dim) * self.d_model
        attn_per_layer = q_proj + k_proj + v_proj + o_proj

        # 3. Per-layer Feed-Forward (SwiGLU: gate, up, and down projections)
        gate_proj = self.d_model * d_ff
        up_proj = self.d_model * d_ff
        down_proj = d_ff * self.d_model
        mlp_per_layer = gate_proj + up_proj + down_proj

        # 4. Layer Normalizations (RMSNorm weights per layer: attn_norm + mlp_norm)
        norms_per_layer = 2 * self.d_model

        # 5. Transformer Layer Total
        single_layer_total = attn_per_layer + mlp_per_layer + norms_per_layer
        all_layers_total = self.n_layers * single_layer_total

        # 6. Final Layer Norm
        final_norm = self.d_model

        # 7. LM Output Head (0 additional parameters if tied to token embeddings)
        lm_head = 0 if self.tie_word_embeddings else (self.d_model * self.vocab_size)

        total = token_emb_params + all_layers_total + final_norm + lm_head

        return {
            "token_embeddings": token_emb_params,
            "single_layer": {
                "q_proj": q_proj,
                "k_proj": k_proj,
                "v_proj": v_proj,
                "o_proj": o_proj,
                "attention_subtotal": attn_per_layer,
                "gate_proj": gate_proj,
                "up_proj": up_proj,
                "down_proj": down_proj,
                "mlp_subtotal": mlp_per_layer,
                "layer_norms": norms_per_layer,
                "layer_total": single_layer_total,
            },
            "all_layers_total": all_layers_total,
            "final_norm": final_norm,
            "lm_head": lm_head,
            "tied_embeddings": self.tie_word_embeddings,
            "total_parameters": total,
            "formatted_total": f"{total:,}",
            "millions": round(total / 1e6, 2),
            "billions": round(total / 1e9, 3),
        }

    def print_parameter_summary(self) -> str:
        """Return a formatted string representing the architecture parameter breakdown."""
        stats = self.calculate_exact_parameters()
        lines = [
            "=" * 64,
            "  SUSA AURA — Architecture & Parameter Summary",
            f"  Profile: {self.profile_name}",
            "=" * 64,
            f"  Vocabulary size:       {self.vocab_size:,}",
            f"  Context window:        {self.max_seq_len:,} tokens",
            f"  Hidden dimension:      {self.d_model}",
            f"  Layers:                {self.n_layers}",
            f"  Attention heads:       {self.n_heads} (Query)",
            f"  KV heads:              {self.n_kv_heads} (KV, GQA ratio: {self.n_heads // (self.n_kv_heads or self.n_heads)}:1)",
            f"  Head dimension:        {self.head_dim}",
            f"  FFN hidden dimension:  {self.d_ff} (SwiGLU)",
            f"  Tied embeddings:       {self.tie_word_embeddings}",
            "-" * 64,
            f"  Token Embeddings:      {stats['token_embeddings']:>14,}",
            f"  Attention (all layers):{stats['single_layer']['attention_subtotal'] * self.n_layers:>14,}",
            f"  SwiGLU MLP (all layers):{stats['single_layer']['mlp_subtotal'] * self.n_layers:>13,}",
            f"  Norms (all layers + 1): {(stats['single_layer']['layer_norms'] * self.n_layers + stats['final_norm']):>14,}",
            f"  LM Head:               {stats['lm_head']:>14,} {'(Tied to embeddings)' if self.tie_word_embeddings else ''}",
            "=" * 64,
            f"  TOTAL PARAMETERS:      {stats['total_parameters']:>14,}",
            f"                         ({stats['billions']} Billion / {stats['millions']} Million)",
            "=" * 64,
        ]
        return "\n".join(lines)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelConfig:
        """Construct ModelConfig from dictionary, handling nested structure if present."""
        model_data = data.get("model", data)
        valid_keys = {
            "vocab_size",
            "max_seq_len",
            "d_model",
            "n_heads",
            "n_kv_heads",
            "n_layers",
            "d_ff",
            "norm_eps",
            "rope_theta",
            "tie_word_embeddings",
            "dropout",
        }
        filtered = {k: v for k, v in model_data.items() if k in valid_keys}
        if "profile_name" in data:
            filtered["profile_name"] = data["profile_name"]
        if "description" in data:
            filtered["description"] = data["description"]
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ModelConfig:
        """Load and validate configuration directly from a YAML file."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            content = yaml.safe_load(f)
        return cls.from_dict(content)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
