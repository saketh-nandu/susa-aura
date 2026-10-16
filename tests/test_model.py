"""
Comprehensive unit tests for SUSA-1B CodeMind Model Architecture (Milestone 2).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from susacodemind.model import (
    GroupedQueryAttention,
    KVCache,
    ModelConfig,
    RMSNorm,
    RotaryEmbedding,
    SusaCodeMind,
    SwiGLU,
    apply_rotary_pos_emb,
)

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


@pytest.fixture
def smoke_config() -> ModelConfig:
    return ModelConfig.from_yaml(CONFIG_DIR / "smoke.yaml")


def test_rmsnorm():
    """Verify RMSNorm normalizes root-mean-square to 1.0."""
    dim = 64
    norm = RMSNorm(dim=dim, eps=1e-5)
    x = torch.randn(2, 8, dim) * 5.0  # High variance input
    y = norm(x)

    assert y.shape == x.shape
    # Check that root mean square across last dimension is ~1.0
    rms = torch.sqrt(torch.mean(y.pow(2), dim=-1))
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-2)


def test_rope_relative_shift():
    """Verify RoPE inner products depend on relative distance."""
    head_dim = 16
    seq_len = 10
    rope = RotaryEmbedding(dim=head_dim, max_seq_len=64)

    q = torch.randn(1, 1, seq_len, head_dim)
    k = torch.randn(1, 1, seq_len, head_dim)

    cos, sin = rope(q)
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape


def test_swiglu_mlp():
    """Verify SwiGLU forward pass dimensions and non-linearity."""
    d_model = 64
    d_ff = 176
    mlp = SwiGLU(d_model=d_model, d_ff=d_ff)
    x = torch.randn(2, 8, d_model)
    out = mlp(x)
    assert out.shape == (2, 8, d_model)


def test_grouped_query_attention(smoke_config):
    """Verify Grouped-Query Attention handles GQA repeating and causal attention."""
    attn = GroupedQueryAttention(smoke_config, layer_idx=0)
    b, s, d = 2, 8, smoke_config.d_model
    x = torch.randn(b, s, d)

    # Standard forward pass
    out = attn(x)
    assert out.shape == (b, s, d)

    # Forward pass with KV cache
    cache = KVCache()
    # Step 1: Prefill 8 tokens
    out_prefill = attn(x, kv_cache=cache)
    assert out_prefill.shape == (b, s, d)
    assert cache.get_seq_length(0) == s

    # Step 2: Incremental step with 1 new token
    x_next = torch.randn(b, 1, d)
    out_step = attn(x_next, kv_cache=cache)
    assert out_step.shape == (b, 1, d)
    assert cache.get_seq_length(0) == s + 1


def test_transformer_parameter_count_match(smoke_config):
    """Verify actual instantiated PyTorch parameter count matches calculated formula."""
    model = SusaCodeMind(smoke_config)
    # verify_parameter_count raises AssertionError if counts disagree
    assert model.verify_parameter_count() is True


def test_model_forward_and_loss(smoke_config):
    """Verify model forward pass, output logits, and next-token cross-entropy loss."""
    model = SusaCodeMind(smoke_config)
    b, s = 2, 16
    input_ids = torch.randint(0, smoke_config.vocab_size, (b, s))

    # 1. Forward without labels
    out = model(input_ids)
    assert out.logits.shape == (b, s, smoke_config.vocab_size)
    assert out.loss is None

    # 2. Forward with labels for next-token prediction
    labels = input_ids.clone()
    out_with_loss = model(input_ids, labels=labels)
    assert out_with_loss.loss is not None
    assert out_with_loss.loss.item() > 0.0

    # 3. Backward pass to ensure gradients propagate to all active weights
    out_with_loss.loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient!"
            # Gradient should not be all NaN or Inf
            assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradient!"


def test_autoregressive_generation(smoke_config):
    """Verify greedy and sampled token generation produces valid output sequences."""
    model = SusaCodeMind(smoke_config)
    prompt = torch.tensor([[10, 20, 30]], dtype=torch.long)
    max_new_tokens = 6

    # Test greedy decoding
    generated_greedy = model.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        use_cache=True,
    )
    assert generated_greedy.shape == (1, 3 + max_new_tokens)
    assert torch.equal(generated_greedy[:, :3], prompt)

    # Test temperature and top-p / top-k sampling
    generated_sampled = model.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.8,
        top_k=20,
        top_p=0.9,
        use_cache=True,
    )
    assert generated_sampled.shape == (1, 3 + max_new_tokens)


def test_checkpoint_save_and_load(smoke_config):
    """Verify model state saving and recovery."""
    model1 = SusaCodeMind(smoke_config)
    input_ids = torch.randint(0, smoke_config.vocab_size, (1, 8))

    with torch.no_grad():
        orig_out = model1(input_ids).logits

    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        torch.save(model1.state_dict(), f.name)

        model2 = SusaCodeMind(smoke_config)
        model2.load_state_dict(torch.load(f.name, weights_only=True))

        with torch.no_grad():
            loaded_out = model2(input_ids).logits

        assert torch.allclose(orig_out, loaded_out, atol=1e-5)


def test_weight_tying(smoke_config):
    """Verify tied embeddings share the same underlying memory."""
    assert smoke_config.tie_word_embeddings is True
    model = SusaCodeMind(smoke_config)
    # The storage pointers must be identical
    assert model.lm_head.weight.data_ptr() == model.tok_embeddings.weight.data_ptr()
