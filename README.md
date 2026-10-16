# SUSA-1B CodeMind

> **An original, coding-focused 1-Billion parameter Large Language Model built from scratch under the SUSA technology ecosystem.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Hardware Cost](https://img.shields.io/badge/Compute%20Cost-%E2%82%B90%20(Free%20Tier)-green.svg)](#zero-cost-philosophy)

---

## 1. Project Overview

**SUSA-1B CodeMind** is an open, modular research initiative to build an original decoder-only Transformer from scratch with randomly initialized weights.

The long-term goal of CodeMind is to specialize in:
- Writing, debugging, understanding, and documenting code.
- Synthesizing web applications, APIs, and microservices.
- Providing project-level structural understanding and automated testing.
- Serving as the model foundation for the **CodeMind Agent Runtime**.

### Architectural Pillars
- **Strictly Original:** Trained from randomly initialized weights. No repurposed external weights, fake wrappers, or API facades.
- **Modern Transformer Architecture:**
  - Grouped-Query Attention (GQA) for memory-efficient KV caching.
  - SwiGLU feed-forward networks for expressive non-linear representations.
  - Root Mean Square Layer Normalization (RMSNorm).
  - Rotary Position Embeddings (RoPE) for relative position encoding and context extrapolation.
  - Tied input/output embeddings to save ~65.5M parameters.
- **Zero-Cost Design:** Engineered specifically for developers working on commodity laptops (e.g. Fedora Linux, 8 GB RAM) combined with free-tier cloud accelerators (Google Colab).

---

## 2. Model Profiles & Parameter Counts

All configurations are validated mathematically and programmatically:

| Profile | Target Hardware | Layers | Hidden Dim | Heads (Q/KV) | FFN Dim | Max Context | Exact Parameter Count |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`smoke`** | Local Laptop CPU | 2 | 64 | 4 / 2 | 176 | 128 | **158,016** (~0.16M) |
| **`small`** | Fast Colab GPU | 6 | 512 | 8 / 4 | 1,408 | 512 | **34,085,376** (~34M) |
| **`medium`** | Benchmark GPU | 16 | 1024 | 16 / 8 | 2,816 | 1,024 | **221,545,472** (~222M) |
| **`target_1b`** | Scaled GPU Training | 20 | 2048 | 16 / 8 | 5,632 | 2,048 | **1,009,338,368** (~1.009B) |

---

## 3. Directory Layout

```text
susa-1b-codemind/
├── configs/               # Reproducible YAML configurations (smoke, small, medium, target_1b)
│   ├── smoke.yaml
│   ├── small.yaml
│   ├── medium.yaml
│   └── target_1b.yaml
├── susacodemind/          # Core package
│   ├── model/             # Config, RoPE, RMSNorm, GQA, Transformer implementation
│   ├── tokenizer/         # Tokenizer training and encoding pipeline
│   ├── data/              # Preprocessing, deduplication, secret filtering
│   ├── training/          # Pretraining & instruction-tuning loops
│   ├── inference/         # Generation, temperature, top-k/top-p sampling
│   ├── evaluation/        # Sandboxed evaluation suites & perplexity tests
│   ├── agent/             # Restricted workspace coding agent runtime
│   └── utils/             # Hardware diagnostics & analytical memory estimation
├── scripts/               # CLI utilities (verify_env, count_params)
├── tests/                 # Automated unit and integration tests
├── notebooks/             # Google Colab training runner
├── docs/                  # Architecture specs, training guides, dataset policies
├── checkpoints/           # Checkpoint storage (Git-ignored)
├── logs/                  # Training logs (Git-ignored)
└── data/                  # Local datasets (Git-ignored)
```

---

## 4. Getting Started (Fedora Linux & Local CPU)

### Prerequisites
- Python 3.10+ (Tested on Fedora 44 with Python 3.14)
- Git 2.40+

### Setup
```bash
# 1. Create and activate a lightweight virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Upgrade pip and install minimal CPU dependencies
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

### Verification & Tests
```bash
# Verify system hardware safety and configuration validity
python scripts/verify_env.py

# Inspect parameter breakdown and memory estimates
python scripts/count_params.py --config configs/smoke.yaml
python scripts/count_params.py --config configs/target_1b.yaml

# Run unit test suite (49 unit tests, 86% coverage)
pytest tests/

# Zero-cost dataset acquisition (Python standard library & curated coding instructions)
python scripts/download_dataset.py --source stdlib --generate-instructions
```

---

## 5. Development Milestones

- [x] **Milestone 1 — Repository Foundation:** Project structure, configuration validator, system checker, resource estimator, test suite, and Colab starter.
- [x] **Milestone 2 — Tiny Model Implementation:** Full decoder-only Transformer with SwiGLU, RMSNorm, RoPE, and GQA tested on CPU.
- [x] **Milestone 3 — Data & Tokenizer:** Tokenizer training pipeline with secret filtering, deduplication, and license tracking.
- [x] **Milestone 4 — Training Engine:** Next-token prediction, gradient accumulation, mixed precision, and checkpoint resumption.
- [x] **Milestone 5 — GitHub & Google Colab Integration:** Cloud replication and Drive checkpoint sync.
- [ ] **Milestone 6 — GPU Benchmarking:** Measure tokens/sec and determine feasible training scale on free Colab tier.
- [ ] **Milestone 7 — Pretraining Runs:** Scaled pretraining experiments with transparent loss curves and checkpoints.
- [x] **Milestone 8 — Instruction Tuning & Evaluation:** Supervised fine-tuning for code instruction following and sandboxed benchmarking.
- [x] **Milestone 9 — Local Inference & Coding Agent Runtime:** Safety-constrained agent for autonomous workspace assistance.

---

## 6. Zero-Cost Philosophy

This project is built by students using entirely free resources:
1. **Local laptop (Fedora, 8GB RAM, 4 CPU threads):** Used exclusively for development, code validation, smoke tests, and lightweight preprocessing.
2. **Google Colab (Free Tier):** Leveraged for GPU benchmarking and model pretraining experiments.
3. **No Paid APIs or Cloud Compute:** Zero cloud expenses are incurred.

---

## 7. License

Licensed under the [Apache License, Version 2.0](LICENSE).
Copyright © 2026 SUSA Ecosystem / SUSA-1B CodeMind Contributors.
