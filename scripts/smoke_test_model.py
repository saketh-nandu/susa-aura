#!/usr/bin/env python3
"""
SUSA-1B CodeMind — Interactive Model Smoke Test & Generation Demo.

Demonstrates model instantiation, parameter verification, forward pass,
loss computation, and autoregressive text generation on CPU using the smoke profile.
"""

from __future__ import annotations

import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from susacodemind.model import ModelConfig, SusaCodeMind


def main() -> int:
    print("\n" + "=" * 70)
    print("  SUSA-1B CODEMIND — MODEL ARCHITECTURE SMOKE TEST")
    print("=" * 70)

    config_path = PROJECT_ROOT / "configs" / "smoke.yaml"
    print(f"\n[1/5] Loading architecture configuration: {config_path.name}")
    config = ModelConfig.from_yaml(config_path)
    print(f"      - d_model: {config.d_model}, layers: {config.n_layers}")
    print(f"      - Query heads: {config.n_heads}, KV heads: {config.n_kv_heads}")
    print(f"      - FFN hidden dimension (SwiGLU): {config.d_ff}")
    print(f"      - Max context window: {config.max_seq_len} tokens")

    print(f"\n[2/5] Instantiating PyTorch model from scratch...")
    model = SusaCodeMind(config)
    param_count = model.count_parameters()
    print(f"      - Instantiated parameters: {param_count:,}")

    print(f"\n[3/5] Verifying mathematical parameter formula vs actual PyTorch tensors...")
    model.verify_parameter_count()
    print("      - Parameter count matches expected formula: SUCCESS (100% exact)")

    print(f"\n[4/5] Testing Forward Pass & Next-Token Prediction Loss...")
    b, s = 2, 8
    dummy_input = torch.randint(0, config.vocab_size, (b, s))
    dummy_labels = dummy_input.clone()

    out = model(dummy_input, labels=dummy_labels)
    loss_val = out.loss.item()
    print(f"      - Forward pass logits shape: {list(out.logits.shape)}")
    print(f"      - Cross-entropy loss: {loss_val:.4f}")

    print(f"\n[5/5] Testing Autoregressive Token Generation with KV Cache...")
    prompt = torch.tensor([[42, 105, 88]], dtype=torch.long)
    generated = model.generate(
        prompt,
        max_new_tokens=10,
        temperature=0.8,
        top_k=20,
        top_p=0.9,
        use_cache=True,
    )
    print(f"      - Prompt tokens:    {prompt.tolist()[0]}")
    print(f"      - Generated tokens: {generated.tolist()[0]}")
    print(f"      - KV-cache incremental generation: SUCCESS")

    print("\n" + "=" * 70)
    print("  STATUS: SUSA-1B CODEMIND MODEL ARCHITECTURE VERIFIED SUCCESSFULLY!")
    print("=" * 70 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
