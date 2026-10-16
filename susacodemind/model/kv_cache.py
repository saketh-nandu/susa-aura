"""
SUSA-1B CodeMind — Key-Value Cache for Autoregressive Generation.

Manages cached Key and Value states across Transformer layers during incremental
decoding to reduce inference computational complexity from O(N^2) to O(N).
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import torch


class KVCache:
    """Stores past keys and values for all decoder layers."""

    def __init__(self) -> None:
        self.key_cache: List[Optional[torch.Tensor]] = []
        self.value_cache: List[Optional[torch.Tensor]] = []

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update and return accumulated Key and Value tensors for layer_idx.

        Args:
            key_states: [B, n_kv_heads, S_new, head_dim]
            value_states: [B, n_kv_heads, S_new, head_dim]
            layer_idx: index of the Transformer layer
        """
        # Ensure cache lists are large enough
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(None)
            self.value_cache.append(None)

        if self.key_cache[layer_idx] is None:
            # First forward pass for this layer
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            # Append along sequence dimension (dim=2)
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=2
            )

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return the sequence length cached for layer_idx."""
        if len(self.key_cache) > layer_idx and self.key_cache[layer_idx] is not None:
            return self.key_cache[layer_idx].shape[2]
        return 0

    def reset(self) -> None:
        """Clear all cached keys and values."""
        self.key_cache.clear()
        self.value_cache.clear()
