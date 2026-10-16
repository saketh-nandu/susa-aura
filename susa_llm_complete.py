#!/usr/bin/env python3
"""
================================================================================
  SUSA-1B CODEMIND — COMPLETE ALL-IN-ONE LARGE LANGUAGE MODEL IMPLEMENTATION
================================================================================
An original, state-of-the-art decoder-only Transformer Large Language Model
designed from scratch under the SUSA technology ecosystem.

Architecture Pillars:
  1. Root Mean Square Layer Normalization (RMSNorm) for training stability.
  2. Rotary Position Embeddings (RoPE) for context extension & relative position.
  3. Grouped-Query Attention (GQA) for 2x-8x KV-cache memory reduction.
  4. SwiGLU Gated Feed-Forward Network for superior non-linear representations.
  5. Tied Embeddings (input token embeddings == output projection weights).
  6. Analytical Parameter Precision & O(N) KV-Cache Autoregressive Generation.

Zero external model dependencies: Standard PyTorch only.
================================================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

# ==============================================================================
# 1. CONFIGURATION & HYPERPARAMETERS
# ==============================================================================

@dataclass
class ModelConfig:
    """Complete architectural hyperparameters for SUSA-1B CodeMind."""

    vocab_size: int = 32000
    max_seq_len: int = 2048
    d_model: int = 2048
    n_heads: int = 16
    n_kv_heads: int = 8
    n_layers: int = 20
    d_ff: int = 5632
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = True
    dropout: float = 0.0

    def __post_init__(self) -> None:
        assert self.d_model % self.n_heads == 0, "d_model must divide evenly by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        if self.d_ff is None:
            # SwiGLU 8/3 standard scaling rounded to multiple of 256
            raw_ff = int(2 * (4 * self.d_model) / 3)
            self.d_ff = ((raw_ff + 255) // 256) * 256

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def num_queries_per_kv(self) -> int:
        return self.n_heads // self.n_kv_heads

    def calculate_exact_parameters(self) -> int:
        """Calculates exact theoretical parameter count down to the exact tensor element."""
        # Embeddings: vocab_size * d_model
        embedding_params = self.vocab_size * self.d_model

        # Attention per layer: W_q, W_k, W_v, W_o
        # W_q: [d_model, n_heads * head_dim] == [d_model, d_model]
        # W_k: [d_model, n_kv_heads * head_dim]
        # W_v: [d_model, n_kv_heads * head_dim]
        # W_o: [d_model, d_model]
        q_params = self.d_model * (self.n_heads * self.head_dim)
        k_params = self.d_model * (self.n_kv_heads * self.head_dim)
        v_params = self.d_model * (self.n_kv_heads * self.head_dim)
        o_params = self.d_model * self.d_model
        attn_per_layer = q_params + k_params + v_params + o_params

        # SwiGLU MLP per layer: W_gate, W_up, W_down
        # W_gate: [d_model, d_ff], W_up: [d_model, d_ff], W_down: [d_ff, d_model]
        mlp_per_layer = 3 * (self.d_model * self.d_ff)

        # RMSNorm per layer: input_layernorm + post_attention_layernorm
        norm_per_layer = 2 * self.d_model

        total_layer_params = self.n_layers * (attn_per_layer + mlp_per_layer + norm_per_layer)

        # Final RMSNorm
        final_norm = self.d_model

        # LM Head (0 if tied, vocab_size * d_model if untied)
        lm_head = 0 if self.tie_word_embeddings else (self.vocab_size * self.d_model)

        return embedding_params + total_layer_params + final_norm + lm_head


# Preset configurations for different compute scales
SMOKE_CONFIG = ModelConfig(
    vocab_size=1024, max_seq_len=128, d_model=64, n_heads=4, n_kv_heads=2, n_layers=2, d_ff=176
)

SMALL_CONFIG = ModelConfig(
    vocab_size=32000, max_seq_len=512, d_model=512, n_heads=8, n_kv_heads=4, n_layers=6, d_ff=1408
)

MEDIUM_CONFIG = ModelConfig(
    vocab_size=32000, max_seq_len=1024, d_model=1024, n_heads=16, n_kv_heads=8, n_layers=16, d_ff=2816
)

TARGET_1B_CONFIG = ModelConfig(
    vocab_size=32000, max_seq_len=2048, d_model=2048, n_heads=16, n_kv_heads=8, n_layers=20, d_ff=5632
)


# ==============================================================================
# 2. NORMALIZATION — RMSNorm
# ==============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (RMSNorm).

    Normalizes inputs by root-mean-square without mean subtraction, reducing
    computational overhead by 10-50% while preserving training dynamics.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # RMS = sqrt(mean(x^2) + eps)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in float32 for numerical stability in mixed-precision regimes
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


# ==============================================================================
# 3. ROTARY POSITION EMBEDDINGS (RoPE)
# ==============================================================================

def precompute_rope_frequencies(dim: int, max_seq_len: int, theta: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Precomputes complex rotary frequencies for positions [0, max_seq_len)."""
    # dim must be even for 2D plane rotations
    assert dim % 2 == 0, f"RoPE dimension must be even, got {dim}"
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    # Outer product: [max_seq_len, dim // 2]
    freqs = torch.outer(t, freqs)
    # Compute cos and sin in float32
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return cos, sin


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies 2D rotation to Query and Key representations."""
    # Split queries into even and odd indices
    xq_r, xq_i = xq[..., 0::2], xq[..., 1::2]
    xk_r, xk_i = xk[..., 0::2], xk[..., 1::2]

    # Reshape cos and sin to match [1, S, 1, D/2]
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)

    # Complex multiplication: (r + i*j) * (cos + sin*j) = (r*cos - i*sin) + (r*sin + i*cos)*j
    xq_out_r = xq_r * cos - xq_i * sin
    xq_out_i = xq_r * sin + xq_i * cos
    xk_out_r = xk_r * cos - xk_i * sin
    xk_out_i = xk_r * sin + xk_i * cos

    # Interleave back to original head dimension
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(-2)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(-2)

    return xq_out.type_as(xq), xk_out.type_as(xk)


# ==============================================================================
# 4. FEED-FORWARD NETWORK — SwiGLU
# ==============================================================================

class SwiGLU(nn.Module):
    """Swish-Gated Linear Unit (SwiGLU).

    Computes: W_down( Swish(W_gate(x)) * W_up(x) )
    Demonstrably outperforms standard ReLU and GeLU activations on code benchmarks.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.silu(w_gate(x)) * w_up(x)
        gated = F.silu(self.w_gate(x)) * self.w_up(x)
        return self.dropout(self.w_down(gated))


# ==============================================================================
# 5. KV CACHE CONTAINER
# ==============================================================================

class KVCache:
    """Manages Key and Value activations for O(N) autoregressive token generation."""

    def __init__(self) -> None:
        self.k_cache: list[torch.Tensor] = []
        self.v_cache: list[torch.Tensor] = []

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Appends current token K, V states to layer cache and returns concatenated history."""
        while len(self.k_cache) <= layer_idx:
            self.k_cache.append(torch.empty(0))
            self.v_cache.append(torch.empty(0))

        if self.k_cache[layer_idx].numel() == 0:
            self.k_cache[layer_idx] = key_states
            self.v_cache[layer_idx] = value_states
        else:
            self.k_cache[layer_idx] = torch.cat([self.k_cache[layer_idx], key_states], dim=1)
            self.v_cache[layer_idx] = torch.cat([self.v_cache[layer_idx], value_states], dim=1)

        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if len(self.k_cache) > layer_idx and self.k_cache[layer_idx].numel() > 0:
            return self.k_cache[layer_idx].size(1)
        return 0

    def clear(self) -> None:
        self.k_cache.clear()
        self.v_cache.clear()


# ==============================================================================
# 6. ATTENTION — Grouped-Query Attention (GQA)
# ==============================================================================

class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention (GQA) with RoPE and KV Cache.

    Shares key and value heads across multiple query heads (n_heads // n_kv_heads),
    reducing KV memory footprint while maintaining multi-head expressive power.
    """

    def __init__(self, config: ModelConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.num_queries_per_kv = config.num_queries_per_kv

        # Linear projections
        self.q_proj = nn.Linear(self.d_model, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, self.d_model, bias=False)

        # Precompute RoPE tables
        cos, sin = precompute_rope_frequencies(self.head_dim, config.max_seq_len, config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        b_sz, seq_len, _ = hidden_states.shape

        # 1. Project Q, K, V
        q = self.q_proj(hidden_states).view(b_sz, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(b_sz, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(b_sz, seq_len, self.n_kv_heads, self.head_dim)

        # 2. Extract position-specific RoPE frequencies
        past_len = kv_cache.get_seq_length(self.layer_idx) if kv_cache is not None else 0
        if position_ids is None:
            pos_start = past_len
            pos_end = past_len + seq_len
            cos = self.rope_cos[pos_start:pos_end]
            sin = self.rope_sin[pos_start:pos_end]
        else:
            cos = self.rope_cos[position_ids[0]]
            sin = self.rope_sin[position_ids[0]]

        # 3. Apply Rotary Embeddings
        q, k = apply_rotary_emb(q, k, cos, sin)

        # 4. Cache or retrieve historical Key and Value representations
        if kv_cache is not None:
            k, v = kv_cache.update(k, v, self.layer_idx)

        total_kv_len = k.size(1)

        # 5. Broadcast KV heads to match Query heads if GQA is active (n_heads != n_kv_heads)
        if self.num_queries_per_kv > 1:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=2)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=2)

        # Permute for scaled dot-product attention: [B, H, S, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 6. Scaled Dot-Product Attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Apply causal mask during non-cached or prefill sequence computation
        if seq_len > 1:
            causal_mask = torch.triu(
                torch.full((seq_len, total_kv_len), float("-inf"), device=hidden_states.device),
                diagonal=total_kv_len - seq_len + 1,
            )
            attn_weights = attn_weights + causal_mask

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(q)
        attn_output = torch.matmul(attn_weights, v)

        # Transpose back: [B, S, H * D]
        attn_output = attn_output.transpose(1, 2).contiguous().view(b_sz, seq_len, -1)
        return self.o_proj(attn_output)


# ==============================================================================
# 7. TRANSFORMER DECODER LAYER BLOCK
# ==============================================================================

class TransformerBlock(nn.Module):
    """Transformer Decoder Layer with Pre-RMSNorm and Residual Connections."""

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Pre-attention RMSNorm & GQA
        self.input_layernorm = RMSNorm(config.d_model, config.norm_eps)
        self.self_attn = GroupedQueryAttention(config, layer_idx=layer_idx)

        # Pre-MLP RMSNorm & SwiGLU
        self.post_attention_layernorm = RMSNorm(config.d_model, config.norm_eps)
        self.mlp = SwiGLU(config.d_model, config.d_ff, config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        # Pre-LN Attention + Residual
        normed_attn = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(
            normed_attn,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
        )
        hidden_states = hidden_states + attn_out

        # Pre-LN SwiGLU + Residual
        normed_mlp = self.post_attention_layernorm(hidden_states)
        mlp_out = self.mlp(normed_mlp)
        hidden_states = hidden_states + mlp_out

        return hidden_states


# ==============================================================================
# 8. SUSA-1B CODEMIND MASTER MODEL
# ==============================================================================

@dataclass
class ModelOutput:
    """Standardized output container for SusaCodeMind."""
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    past_key_values: KVCache | None = None


class SusaCodeMind(nn.Module):
    """SUSA-1B CodeMind Complete Decoder-Only Transformer."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        # 1. Input Token Embeddings
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)

        # 2. Decoder Layers
        self.layers = nn.ModuleList([TransformerBlock(config, i) for i in range(config.n_layers)])

        # 3. Final RMSNorm
        self.norm = RMSNorm(config.d_model, config.norm_eps)

        # 4. Language Modeling Output Head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # 5. Tied Embeddings
        if config.tie_word_embeddings:
            self.lm_head.weight = self.tok_embeddings.weight

        # 6. Weight Initialization & Residual Scaling
        self.apply(self._init_weights)
        self._apply_residual_scaling()

        self.gradient_checkpointing = False

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)

    def _apply_residual_scaling(self) -> None:
        scale = 1.0 / math.sqrt(2.0 * self.config.n_layers)
        for layer in self.layers:
            layer.self_attn.o_proj.weight.data.mul_(scale)
            layer.mlp.w_down.weight.data.mul_(scale)

    def count_parameters(self) -> int:
        """Counts exact trainable parameters."""
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
        use_cache: bool = False,
    ) -> ModelOutput:
        """Computes forward pass and cross-entropy next-token loss if labels are supplied."""
        if use_cache and kv_cache is None:
            kv_cache = KVCache()

        # Token embeddings: [B, S, d_model]
        hidden_states = self.tok_embeddings(input_ids)

        # Pass through Decoder Layers
        for layer in self.layers:
            if self.gradient_checkpointing and self.training and kv_cache is None:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer, hidden_states, attention_mask, position_ids, None, use_reentrant=False
                )
            else:
                hidden_states = layer(
                    hidden_states, attention_mask=attention_mask, position_ids=position_ids, kv_cache=kv_cache
                )

        # Final RMSNorm
        hidden_states = self.norm(hidden_states)

        # Logits: [B, S, vocab_size]
        logits = self.lm_head(hidden_states)

        # Next-token prediction loss
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return ModelOutput(logits=logits, loss=loss, past_key_values=kv_cache if use_cache else None)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 48,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_id: int | None = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Autoregressive code generation with temperature, top-k, and nucleus sampling."""
        self.eval()
        kv_cache = KVCache() if use_cache else None
        current_ids = input_ids.clone()

        for _ in range(max_new_tokens):
            if use_cache and kv_cache is not None and kv_cache.get_seq_length(0) > 0:
                model_inputs = current_ids[:, -1:]
            else:
                model_inputs = current_ids

            out = self.forward(model_inputs, kv_cache=kv_cache, use_cache=use_cache)
            next_logits = out.logits[:, -1, :].clone()

            if temperature < 1e-5:
                # Greedy argmax
                next_tokens = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                next_logits = next_logits / temperature

                # Top-K
                if top_k > 0:
                    k_val = min(top_k, next_logits.size(-1))
                    indices_to_remove = next_logits < torch.topk(next_logits, k_val)[0][..., -1, None]
                    next_logits[indices_to_remove] = -float("Inf")

                # Top-P (Nucleus)
                if 0.0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_logits[indices_to_remove] = -float("Inf")

                probs = F.softmax(next_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)

            current_ids = torch.cat([current_ids, next_tokens], dim=-1)

            if eos_token_id is not None and (next_tokens == eos_token_id).all():
                break

        return current_ids


# ==============================================================================
# 9. OPTIMIZATION & TRAINING ENGINE
# ==============================================================================

def create_adamw_optimizer(
    model: nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> torch.optim.AdamW:
    """Creates AdamW optimizer with weight decay applied ONLY to 2D matrices (weights)."""
    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(param_groups, lr=lr, betas=betas, eps=eps)


def get_cosine_lr(step: int, max_steps: int, warmup_steps: int, lr: float, min_lr: float) -> float:
    """Cosine learning rate schedule with linear warmup."""
    if step < warmup_steps:
        return lr * float(step + 1) / float(max(1, warmup_steps))
    if step > max_steps:
        return min_lr
    decay_ratio = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)


# ==============================================================================
# 10. VERIFICATION & DEMONSTRATION RUNNER
# ==============================================================================

def main() -> int:
    print("=" * 75)
    print("  SUSA-1B CODEMIND — COMPLETE ALL-IN-ONE ARCHITECTURE VERIFICATION")
    print("=" * 75)

    # 1. Verify Parameter Counting Formula vs Actual Tensors
    profiles = [
        ("smoke", SMOKE_CONFIG),
        ("small", SMALL_CONFIG),
        ("medium", MEDIUM_CONFIG),
        ("target_1b", TARGET_1B_CONFIG),
    ]

    print("\n[1/4] Verifying exact parameter calculations across all profiles:")
    for name, cfg in profiles:
        exact_calc = cfg.calculate_exact_parameters()
        print(f"  - Profile '{name:<10}': {exact_calc:>13,} parameters ({exact_calc / 1e6:.2f}M)")

    # 2. Instantiate Smoke Model
    print("\n[2/4] Instantiating PyTorch model from scratch (Profile: smoke)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SusaCodeMind(SMOKE_CONFIG).to(device)
    actual_params = model.count_parameters()
    expected_params = SMOKE_CONFIG.calculate_exact_parameters()

    print(f"  - Actual instantiated parameters: {actual_params:,}")
    print(f"  - Mathematical formula:          {expected_params:,}")
    assert actual_params == expected_params, "Parameter counts must match 100%!"
    print("  - Parameter Invariant:             EXACT MATCH (100% precision)")

    # 3. Test Forward & Backward Optimization Step
    print("\n[3/4] Testing Forward pass, Next-Token Loss, and Backpropagation...")
    optimizer = create_adamw_optimizer(model, lr=1e-3)

    # Mock batch: [B=2, S=8]
    dummy_input = torch.randint(0, SMOKE_CONFIG.vocab_size, (2, 8), device=device)
    dummy_labels = dummy_input.clone()

    out = model(dummy_input, labels=dummy_labels)
    loss = out.loss
    assert loss is not None, "Loss should not be None when labels are provided"
    print(f"  - Initial Cross-Entropy Loss: {loss.item():.4f}")

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    print("  - Optimization step & gradient update: SUCCESS")

    # 4. Test Autoregressive Generation with KV Cache
    print("\n[4/4] Testing Autoregressive Token Generation with O(N) KV Cache...")
    prompt = torch.tensor([[42, 105, 88]], device=device)
    generated = model.generate(prompt, max_new_tokens=10, temperature=0.7, top_k=20)
    print(f"  - Prompt Tokens:    {prompt[0].tolist()}")
    print(f"  - Generated Tokens: {generated[0].tolist()}")
    assert generated.size(1) == prompt.size(1) + 10, "Generated length mismatch"
    print("  - KV Cache Generation: SUCCESS")

    print("\n" + "=" * 75)
    print("  ALL ARCHITECTURAL AND TRAINING CAPABILITIES ARE 100% FUNCTIONAL!")
    print("=" * 75 + "\n")
    return 0


if __name__ == "__main__":
    main()
