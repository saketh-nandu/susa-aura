"""
SUSA AURA — Root Mean Square Layer Normalization (RMSNorm).

RMSNorm provides numerical stabilization for Transformer activations while
avoiding the mean-centering overhead of standard LayerNorm:
    RMSNorm(x) = (x / sqrt(mean(x^2) + eps)) * weight
"""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # Compute root mean square over the last dimension in float32 for stability
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        # Cast to float32 for accurate norm calculation then cast back
        normed = self._norm(x.float()).to(orig_dtype)
        return normed * self.weight

    def extra_repr(self) -> str:
        return f"{self.weight.shape[0]}, eps={self.eps}"
