"""
SUSA AURA — SwiGLU Gated Feed-Forward Network.

Computes the Swish-Gated Linear Unit activation:
    SwiGLU(x) = (SiLU(x * W_gate) * (x * W_up)) * W_down
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SwiGLU(nn.Module):
    """Gated feed-forward network using SwiGLU activation without bias."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gated activation: SiLU(w_gate(x)) * w_up(x)
        gated = F.silu(self.w_gate(x)) * self.w_up(x)
        gated = self.dropout(gated)
        out = self.w_down(gated)
        return out

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, d_ff={self.d_ff}"
