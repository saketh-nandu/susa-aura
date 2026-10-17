# SUSA AURA — Evolution Roadmap

> **Adaptive Universal Reasoning Assistant (SUSA AURA)**
> Technical roadmap from foundation to general-purpose reasoning assistant.

---

## Completed Milestones

- [x] **Milestone 1 — Architecture Foundation:** Original decoder-only Transformer with RoPE, SwiGLU, RMSNorm, Grouped-Query Attention (GQA), and tied embeddings.
- [x] **Milestone 2 — Mathematical Verification:** 100% exact parameter calculation formula matching instantiated PyTorch tensors across `smoke`, `small`, `medium`, and `target_1b`.
- [x] **Milestone 3 — Data & Tokenizer Subsystem:** Byte-level BPE tokenizer with FIM tokens, secret scrubbing, and deduplication.
- [x] **Milestone 4 — Pretraining Engine:** Resumable checkpoints, mixed precision (AMP), cosine decay schedule, and gradient accumulation.
- [x] **Milestone 5 — Supervised Fine-Tuning (SFT):** Instruction tuning pipeline with prompt loss masking.
- [x] **Milestone 6 — Code & Reasoning Evaluation:** Sandboxed AST syntax validation and unit test verification harness (pass@1).
- [x] **Milestone 7 — SUSA AURA Migration:** Unified refactoring from CodeMind to SUSA AURA, modular subsystems (Model, Core, Connect, Memory, Tools, Interface), and backward-compatibility shims.

---

## Active & Upcoming Milestones

- [ ] **Milestone 8 — GPU Scaling Experiments (AURA 34M & 222M):** Benchmark throughput (tokens/sec) and loss convergence on Google Colab free GPU tier.
- [ ] **Milestone 9 — Long-Context Extrapolation:** Extended context evaluation beyond 2,048 tokens using RoPE scaling (YaRN / dynamic NTK-aware RoPE).
- [ ] **Milestone 10 — AURA Connect Live Retrieval:** Integration with search API endpoints (DuckDuckGo HTML, SearXNG, official documentation APIs).
- [ ] **Milestone 11 — AURA Memory Embeddings:** Vector-based dense retrieval for conversational memory using lightweight local embeddings.
- [ ] **Milestone 12 — AURA 1B Pretraining Run:** Full-scale multi-day pretraining run on cloud compute with checkpoint replication and evaluation reports.
