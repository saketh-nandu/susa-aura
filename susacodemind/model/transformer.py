"""
SUSA-1B CodeMind — Core Decoder-Only Transformer Architecture.

Integrates Grouped-Query Attention (GQA), SwiGLU gated MLP, RMSNorm pre-normalization,
Rotary Position Embeddings (RoPE), tied embeddings, and next-token autoregressive generation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from susacodemind.model.attention import GroupedQueryAttention
from susacodemind.model.config import ModelConfig
from susacodemind.model.kv_cache import KVCache
from susacodemind.model.rmsnorm import RMSNorm
from susacodemind.model.swiglu import SwiGLU


@dataclass
class ModelOutput:
    """Standardized output container for SusaCodeMind."""

    logits: torch.Tensor
    loss: torch.Tensor | None = None
    past_key_values: KVCache | None = None


class TransformerBlock(nn.Module):
    """Single Transformer Decoder Layer with Pre-RMSNorm and Residual Connections."""

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # 1. Attention Block
        self.input_layernorm = RMSNorm(config.d_model, config.norm_eps)
        self.self_attn = GroupedQueryAttention(config, layer_idx=layer_idx)

        # 2. Feed-Forward Block (SwiGLU)
        d_ff = config.d_ff or int(2 * (4 * config.d_model) / 3)
        self.post_attention_layernorm = RMSNorm(config.d_model, config.norm_eps)
        self.mlp = SwiGLU(config.d_model, d_ff, config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        # Pre-LN Self-Attention with residual connection
        normed_attn_in = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(
            normed_attn_in,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
        )
        hidden_states = hidden_states + attn_out

        # Pre-LN SwiGLU MLP with residual connection
        normed_mlp_in = self.post_attention_layernorm(hidden_states)
        mlp_out = self.mlp(normed_mlp_in)
        hidden_states = hidden_states + mlp_out

        return hidden_states


class SusaCodeMind(nn.Module):
    """SUSA-1B CodeMind Decoder-Only Transformer Model."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        # 1. Token Embeddings
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)

        # 2. Transformer Decoder Layers
        self.layers = nn.ModuleList(
            [TransformerBlock(config, i) for i in range(config.n_layers)]
        )

        # 3. Final RMSNorm
        self.norm = RMSNorm(config.d_model, config.norm_eps)

        # 4. Language Model Output Head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # 5. Optional Weight Tying
        if config.tie_word_embeddings:
            self.lm_head.weight = self.tok_embeddings.weight

        # 6. Initialize Weights
        self.apply(self._init_weights)
        self._apply_residual_scaling()

        # Gradient checkpointing flag
        self.gradient_checkpointing = False

    def _init_weights(self, module: nn.Module) -> None:
        """Standard Gaussian initialization for Transformer weights."""
        if isinstance(module, nn.Linear):
            std = 0.02
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            std = 0.02
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)

    def _apply_residual_scaling(self) -> None:
        """Scales residual output projections (o_proj and down_proj) by 1 / sqrt(2 * n_layers).

        Follows GPT-2 / LLaMA conventions to prevent variance growth across deep residual paths.
        """
        residual_scale = 1.0 / math.sqrt(2.0 * self.config.n_layers)
        for layer in self.layers:
            # Scale attention output projection
            layer.self_attn.o_proj.weight.data.mul_(residual_scale)
            # Scale MLP output projection
            layer.mlp.w_down.weight.data.mul_(residual_scale)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.tok_embeddings

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.tok_embeddings = value
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.tok_embeddings.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
        use_cache: bool = False,
    ) -> ModelOutput:
        """Forward pass for next-token prediction and generation.

        Args:
            input_ids: [B, S] token indices
            labels: Optional [B, S] target token indices for cross-entropy loss computation
            attention_mask: Optional [B, S] binary mask (1 for valid token, 0 for padding)
            position_ids: Optional [B, S] position indices
            kv_cache: Optional KVCache object
            use_cache: whether to create and return a KVCache if none provided
        """
        if use_cache and kv_cache is None:
            kv_cache = KVCache()

        # 1. Embed tokens: [B, S, d_model]
        hidden_states = self.tok_embeddings(input_ids)

        # 2. Pass through Transformer layers
        for layer in self.layers:
            if self.gradient_checkpointing and self.training and kv_cache is None:
                # Use gradient checkpointing to conserve memory during backward pass
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                    use_reentrant=False,
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    kv_cache=kv_cache,
                )

        # 3. Final RMSNorm
        hidden_states = self.norm(hidden_states)

        # 4. Project to vocabulary logits: [B, S, vocab_size]
        logits = self.lm_head(hidden_states)

        # 5. Compute loss if labels are supplied (Next-token prediction)
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten batch and sequence dimensions for cross entropy
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return ModelOutput(
            logits=logits,
            loss=loss,
            past_key_values=kv_cache if use_cache else None,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_id: int | None = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Autoregressive text generation with temperature, top-k, and nucleus (top-p) sampling.

        Args:
            input_ids: [B, S] prompt tokens
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (1.0 = standard, 0.0 = greedy)
            top_k: Top-k filtering cutoff (0 disables top-k)
            top_p: Nucleus filtering cumulative probability threshold
            eos_token_id: Optional token ID to signal early generation termination
            use_cache: Use KV cache for fast O(N) generation
        """
        self.eval()
        kv_cache = KVCache() if use_cache else None
        current_ids = input_ids.clone()

        for _ in range(max_new_tokens):
            # If using KV cache and past tokens exist, only feed the latest single token
            if use_cache and kv_cache is not None and kv_cache.get_seq_length(0) > 0:
                model_inputs = current_ids[:, -1:]
            else:
                model_inputs = current_ids

            out = self.forward(model_inputs, kv_cache=kv_cache, use_cache=use_cache)
            # Take logits of the last token: [B, vocab_size]
            next_token_logits = out.logits[:, -1, :].clone()

            if temperature == 0.0 or temperature < 1e-6:
                # Greedy decoding
                next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                # Apply temperature
                next_token_logits = next_token_logits / temperature

                # Top-K filtering
                if top_k > 0:
                    k_val = min(top_k, next_token_logits.size(-1))
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, k_val)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float("Inf")

                # Top-P (nucleus) filtering
                if 0.0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above top_p threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift indices to keep the first token above threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    next_token_logits[indices_to_remove] = -float("Inf")

                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)

            # Append generated token
            current_ids = torch.cat([current_ids, next_tokens], dim=-1)

            # Check if all batches produced EOS
            if eos_token_id is not None and (next_tokens == eos_token_id).all():
                break

        return current_ids

    def count_parameters(self) -> int:
        """Count actual trainable parameters in the instantiated PyTorch model."""
        return sum(p.numel() for p in self.parameters())

    def verify_parameter_count(self) -> bool:
        """Verify that actual model parameters match the calculated architectural formula."""
        calc_stats = self.config.calculate_exact_parameters()
        expected = calc_stats["total_parameters"]
        actual = self.count_parameters()
        if expected != actual:
            raise AssertionError(
                f"Parameter mismatch! Calculated: {expected:,}, Actual instantiated: {actual:,}"
            )
        return True
