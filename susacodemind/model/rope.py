"""
SUSA-1B CodeMind — Rotary Position Embeddings (RoPE).

Encodes relative positions by rotating pairs of query and key projections in the
complex plane. RoPE adds zero learnable parameters and supports sequence extrapolation.
"""

from __future__ import annotations

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    """Precomputed and dynamic Rotary Position Embedding for causal attention."""

    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"Rotary embedding dimension must be even, got {dim}")
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Precompute inverted frequency spectrum
        # theta_i = theta^(-2i / d) for i in [0, d/2 - 1]
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute initial cache up to max_seq_len
        self._build_cache(self.max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)  # [seq_len, dim // 2]
        # Duplicate frequencies to match head_dim
        emb = torch.cat((freqs, freqs), dim=-1)  # [seq_len, dim]
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cos and sin tensors shaped for broadcasting with [B, n_heads, S, head_dim].

        Args:
            x: Tensor of shape [B, n_heads, S, head_dim]
            position_ids: Optional explicit position indices [B, S] or [S]
        """
        seq_len = x.shape[2]

        if position_ids is None:
            if seq_len > self.cos_cached.shape[0]:
                self._build_cache(max(seq_len, self.cos_cached.shape[0] * 2))
            cos = self.cos_cached[:seq_len].to(dtype=x.dtype, device=x.device)
            sin = self.sin_cached[:seq_len].to(dtype=x.dtype, device=x.device)
            # Unsqueeze for broadcasting over batch and head dimensions: [1, 1, S, head_dim]
            return cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)

        # Custom / incremental position IDs for inference KV caching
        max_pos = int(position_ids.max().item()) + 1
        if max_pos > self.cos_cached.shape[0]:
            self._build_cache(max(max_pos, self.cos_cached.shape[0] * 2))

        # position_ids shape: [B, S] -> gather cos and sin: [B, S, head_dim] -> unsqueeze head: [B, 1, S, head_dim]
        cos = self.cos_cached[position_ids].unsqueeze(1).to(dtype=x.dtype, device=x.device)
        sin = self.sin_cached[position_ids].unsqueeze(1).to(dtype=x.dtype, device=x.device)
        return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input for RoPE calculation."""
    d_half = x.shape[-1] // 2
    x1 = x[..., :d_half]
    x2 = x[..., d_half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embedding to Query and Key tensors.

    Args:
        q: [B, n_heads, S, head_dim]
        k: [B, n_kv_heads, S, head_dim]
        cos: [1 or B, 1, S, head_dim]
        sin: [1 or B, 1, S, head_dim]
    """
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot
