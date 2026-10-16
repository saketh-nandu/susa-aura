# SUSA-1B CodeMind — Architecture Specification

## 1. Overview

**SUSA-1B CodeMind** is a decoder-only autoregressive Transformer designed and trained from scratch with randomly initialized weights. It is purpose-built to excel in code synthesis, debugging, architecture planning, and developer agent workflows.

## 2. Core Architectural Components

### 2.1 Token Embeddings & Positional Encoding
- **Vocabulary Size:** Configurable (default `32,000` tokens).
- **Embeddings:** Learned embedding matrix $\mathbf{W}_e \in \mathbb{R}^{V \times d_{\text{model}}}$.
- **Weight Tying:** The language model output head $\mathbf{W}_{\text{head}}$ is tied to the input embedding matrix $\mathbf{W}_e^T$ by default to reduce parameter overhead by $V \times d_{\text{model}}$ (~65.5M parameters).
- **Positional Representation:** Rotary Position Embeddings (RoPE) applied to query and key representations at each attention layer. RoPE provides relative position encoding with length extrapolation capability and zero learned positional parameters.

### 2.2 Pre-Layer Normalization
- **RMSNorm (Root Mean Square Layer Normalization):** Replaces standard LayerNorm to eliminate mean-centering overhead while preserving stabilization:
$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}} \odot \gamma$$
- Normalized inputs are supplied before attention and feed-forward blocks (Pre-LN residual connections) for stable gradient flow.

### 2.3 Attention Mechanism
- **Grouped-Query Attention (GQA):**
  - Query heads: $n_{\text{heads}}$ (e.g., 16 in target model).
  - Key/Value heads: $n_{\text{kv\_heads}}$ (e.g., 8 in target model).
  - GQA ratio: 2:1 query heads per KV head.
  - Reduces KV cache memory by 50% during inference and autoregressive generation compared to standard Multi-Head Attention (MHA).
- **Causal Masking:** Strict lower-triangular causal attention mask prevents future token leakage during next-token prediction.

### 2.4 Feed-Forward Network (SwiGLU)
- **Activation:** Swish-Gated Linear Unit (SwiGLU):
$$\text{SwiGLU}(x) = (\text{Swish}(x \mathbf{W}_{\text{gate}}) \odot (x \mathbf{W}_{\text{up}})) \mathbf{W}_{\text{down}}$$
- **Hidden Dimension:** $d_{\text{ff}} \approx \frac{8}{3} d_{\text{model}}$, aligned to multiples of 256 for tensor core computational efficiency.

---

## 3. Parameter Profiles

The repository supports four primary architectural profiles configured via YAML:

| Profile | Layers | $d_{\text{model}}$ | Heads (Q) | Heads (KV) | $d_{\text{ff}}$ | Max Seq | Vocab Size | Exact Parameter Count |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Smoke** | 2 | 64 | 4 | 2 | 176 | 128 | 1,024 | **158,016** (0.16M) |
| **Small** | 6 | 512 | 8 | 4 | 1,408 | 512 | 32,000 | **34,085,376** (34.09M) |
| **Medium** | 16 | 1024 | 16 | 8 | 2,816 | 1024 | 32,000 | **221,545,472** (221.55M) |
| **Target 1B**| 20 | 2048 | 16 | 8 | 5,632 | 2048 | 32,000 | **1,009,338,368** (1.009B) |

---

## 4. Parameter Formula Breakdown (Target 1B)

For $V=32000, d=2048, L=20, n_h=16, n_{kv}=8, d_k=128, d_{ff}=5632$:

1. **Token Embeddings:** $32,000 \times 2,048 = 65,536,000$
2. **Attention Projections (per layer):**
   - $W_q: 2048 \times 2048 = 4,194,304$
   - $W_k: 2048 \times 1024 = 2,097,152$
   - $W_v: 2048 \times 1024 = 2,097,152$
   - $W_o: 2048 \times 2048 = 4,194,304$
   - *Subtotal per layer:* $12,582,912$
3. **SwiGLU MLP (per layer):**
   - $W_{\text{gate}}: 2048 \times 5632 = 11,534,336$
   - $W_{\text{up}}: 2048 \times 5632 = 11,534,336$
   - $W_{\text{down}}: 5632 \times 2048 = 11,534,336$
   - *Subtotal per layer:* $34,603,008$
4. **RMSNorm Weights (per layer):** $2 \times 2048 = 4,096$
5. **Per-Layer Total:** $12,582,912 + 34,603,008 + 4,096 = 47,190,016$
6. **20 Layers Subtotal:** $20 \times 47,190,016 = 943,800,320$
7. **Final RMSNorm:** $2,048$
8. **LM Output Head:** $0$ (Tied to Embeddings)
9. **Grand Total:** $65,536,000 + 943,800,320 + 2,048 = \mathbf{1,009,338,368}$ parameters.
