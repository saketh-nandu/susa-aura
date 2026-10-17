"""
SUSA AURA — Grouped-Query Attention (GQA) with RoPE and KV Cache.

Implements Grouped-Query Attention (GQA) where query heads share key/value heads,
substantially reducing KV cache memory footprint while maintaining full expressive capacity.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from aura.model.config import ModelConfig
from aura.model.kv_cache import KVCache
from aura.model.rope import RotaryEmbedding, apply_rotary_pos_emb


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeats KV heads along the head dimension to match the number of query heads.

    Args:
        hidden_states: [B, n_kv_heads, S, head_dim]
        n_rep: repetition factor (n_heads // n_kv_heads)
    """
    if n_rep == 1:
        return hidden_states
    b, n_kv_heads, s, head_dim = hidden_states.shape
    expanded = hidden_states[:, :, None, :, :].expand(b, n_kv_heads, n_rep, s, head_dim)
    return expanded.reshape(b, n_kv_heads * n_rep, s, head_dim)


class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention module with Rotary Positional Embeddings and KV Cache."""

    def __init__(self, config: ModelConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads if config.n_kv_heads is not None else config.n_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.n_heads // self.n_kv_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Linear projections without bias (modern standard)
        self.q_proj = nn.Linear(self.d_model, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, self.d_model, bias=False)

        # Rotary Positional Embedding
        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
        )

        self.dropout = config.dropout

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        """Forward pass for Grouped-Query Attention.

        Args:
            hidden_states: [B, S, d_model]
            attention_mask: Optional [B, 1, S_q, S_k] or [B, S]
            position_ids: Optional [B, S]
            kv_cache: Optional KVCache instance for autoregressive generation
        """
        b_size, q_len, _ = hidden_states.shape

        # 1. Project Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape to [B, n_heads, S, head_dim]
        q = q.view(b_size, q_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b_size, q_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(b_size, q_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # 2. Determine sequence positions and apply RoPE
        if kv_cache is not None and position_ids is None:
            # If KV cache is active, calculate start position from cache
            past_len = kv_cache.get_seq_length(self.layer_idx)
            pos_ids = torch.arange(past_len, past_len + q_len, device=hidden_states.device).unsqueeze(0)
        else:
            pos_ids = position_ids

        cos, sin = self.rotary_emb(q, position_ids=pos_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # 3. Update KV Cache if provided
        if kv_cache is not None:
            k, v = kv_cache.update(k, v, self.layer_idx)

        total_kv_len = k.shape[2]

        # 4. Repeat KV heads for Grouped-Query Attention
        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)

        # 5. Compute Attention (leveraging scaled_dot_product_attention)
        # If no past cache and standard prefill, we can use is_causal=True
        # If past cache is present (q_len == 1, total_kv_len > 1), causal mask is not needed for the single token
        # If custom attention_mask is given, format appropriately
        dropout_p = self.dropout if self.training else 0.0

        if attention_mask is not None:
            # Broadcast mask to [B, 1, q_len, total_kv_len]
            if attention_mask.dim() == 2:
                # Padding mask [B, total_kv_len] -> [B, 1, 1, total_kv_len]
                mask = attention_mask[:, None, None, :].to(dtype=torch.bool)
                attn_output = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=mask,
                    dropout_p=dropout_p,
                    is_causal=(q_len > 1 and q_len == total_kv_len),
                )
            else:
                attn_output = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attention_mask,
                    dropout_p=dropout_p,
                    is_causal=False,
                )
        else:
            is_causal = (q_len > 1 and q_len == total_kv_len)
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=dropout_p,
                is_causal=is_causal,
            )

        # 6. Recombine heads and apply output projection
        # [B, n_heads, S, head_dim] -> [B, S, n_heads * head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(b_size, q_len, self.d_model)
        output = self.o_proj(attn_output)
        return output
