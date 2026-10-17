# SUSA AURA — Adaptive Universal Reasoning Assistant

> **An original, modular AI reasoning assistant and configurable decoder-only Transformer built from scratch under the SUSA ecosystem.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![CLI](https://img.shields.io/badge/CLI-aura-purple.svg)](#aura-cli-interface)
[![Hardware Cost](https://img.shields.io/badge/Compute%20Cost-%E2%82%B90%20(Free%20Tier)-green.svg)](#zero-cost-engineering-philosophy)

---

## 1. Project Overview & Identity

**SUSA AURA** (*Adaptive Universal Reasoning Assistant*, evolved from the foundational CodeMind research project) is an autonomous, general-purpose AI assistant and original language model runtime.

AURA combines an independently implemented decoder-only Transformer with a modular agent runtime that connects multi-turn reasoning, memory retention, safe command execution, and source-grounded web research.

### Core Principles
1. **Strictly Original Model Implementation:** Trained from scratch starting with randomly initialized weights. No third-party pretrained weights, fake wrappers, or API simulations are presented as our own foundation model.
2. **Modular Subsystem Architecture:**
   - **AURA Model:** Configurable Transformer (RoPE, SwiGLU, RMSNorm, GQA, tied embeddings), custom tokenizer, dataset pipeline, training engine, and inference runtime.
   - **AURA Core:** Multi-turn conversation orchestration, session state tracking, tool dispatch, and prompt assembly.
   - **AURA Connect:** Source-grounded research, URL tracking, HTML extraction, credential redaction, and prompt-injection defanging.
   - **AURA Memory:** Inspectable conversation session history, keyword-indexed memory retrieval, and disk persistence.
   - **AURA Tools:** Sandboxed Python code evaluation, workspace inspection, unified diff proposals, and gated file modifications.
   - **AURA Interface:** Interactive CLI REPL with slash commands, streaming generation, and single-turn prompt execution.
3. **Zero-Cost Engineering Philosophy:** Designed to run development, smoke tests, and preprocessing on commodity laptops (e.g., Fedora Linux, 8 GB RAM, CPU-only) while scaling pretraining seamlessly to free cloud GPU environments (Google Colab).
4. **Full Backward Compatibility:** Complete backward-compatibility shims are maintained under `susacodemind/` so legacy imports and workflows continue to function with transparent deprecation notices.

---

## 2. Architectural Subsystems

```
susa-aura/
├── configs/               # Reproducible YAML configurations (smoke, small, medium, target_1b)
│   ├── smoke.yaml
│   ├── small.yaml
│   ├── medium.yaml
│   └── target_1b.yaml
├── aura/                  # SUSA AURA Python package
│   ├── model/             # Transformer architecture (GQA, RoPE, RMSNorm, SwiGLU, KVCache)
│   ├── tokenizer/         # Tokenizer training & encoding pipeline (AuraTokenizer)
│   ├── data/              # Preprocessing, binary memory-mapped shards, secret cleaning, deduplication
│   ├── training/          # Pretraining & instruction-tuning loops (AuraTrainer)
│   ├── inference/         # Generation engine with sampling & stop strings (AuraGenerator)
│   ├── evaluation/        # Sandboxed code execution & benchmark evaluation (IsolatedCodeRunner)
│   ├── core/              # Assistant orchestration runtime & session state (AuraCore)
│   ├── memory/            # Conversation turns & persistent searchable memory (AuraMemory)
│   ├── connect/           # Source-grounded web research & security sanitization (AuraConnect)
│   ├── tools/             # Tool registry, workspace inspection, safe command runner (ToolRegistry)
│   ├── interface/         # Interactive CLI application & entry point (aura.interface.cli)
│   ├── agent/             # Forwarding compatibility layer for legacy agent interfaces
│   └── utils/             # Diagnostics & hardware resource estimation (estimate_resources)
├── susacodemind/          # Full backward-compatibility shims forwarding to aura
├── scripts/               # Production CLI utilities (train, evaluate, download_dataset, verify_env)
├── notebooks/             # Google Colab notebooks (notebooks/colab/aura_training.ipynb)
├── tests/                 # 76 automated unit, integration, and backward-compat tests
└── docs/                  # In-depth architectural specifications and guides
```

---

## 3. Model Profiles & Parameter Verification

All parameter counts are calculated mathematically from tensor shapes and verified by PyTorch unit tests:

| Profile | Target Hardware | Layers | Hidden Dim | Heads (Q/KV) | FFN Dim | Max Context | Exact Parameter Count |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`smoke`** | Local Laptop CPU | 2 | 64 | 4 / 2 | 176 | 128 | **158,016** (~0.16M) |
| **`small`** | Fast Colab GPU | 6 | 512 | 8 / 4 | 1,408 | 512 | **34,085,376** (~34M) |
| **`medium`** | Benchmark GPU | 16 | 1024 | 16 / 8 | 2,816 | 1,024 | **221,545,472** (~222M) |
| **`target_1b`** | Scaled GPU Training | 20 | 2048 | 16 / 8 | 5,632 | 2,048 | **1,009,338,368** (~1.009B) |

> *Note on Parameter Efficiency:* Input token embeddings and output projection heads are tied (`weight_tying: true`), eliminating ~65.5 million redundant parameters in the 1B configuration.

---

## 4. Quick Start (Fedora Linux & Local CPU)

### Prerequisites
- Python 3.10+ (Tested on Fedora Linux with Python 3.14)
- Git 2.40+

### Installation
```bash
# 1. Clone repository and navigate into workspace
git clone https://github.com/saketh-nandu/susa-1b-codemind.git susa-aura
cd susa-aura

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Upgrade pip and install package in editable mode
pip install --upgrade pip
pip install -e ".[dev]"
```

### Run Environment & Parameter Check
```bash
python scripts/verify_env.py
```

### Run Full Test Suite
```bash
pytest tests/ -v
```

---

## 5. AURA CLI Interface

The package exposes the standalone executable `aura` directly in your environment:

```bash
# Display help and available options
aura --help

# Inspect current workspace structure
aura --inspect

# Single-turn prompt execution
aura --prompt "Explain the AURA Core architecture"

# Launch interactive conversational REPL
aura

# Launch with a specific trained checkpoint and configuration
aura --checkpoint checkpoints/checkpoint_step_5.pt \
     --tokenizer-dir tokenizer_cache/smoke \
     --config configs/smoke.yaml
```

Inside the interactive REPL:
- `/help` — List interactive commands
- `/inspect` — Display workspace file tree
- `/mem` — Inspect persistent memory records
- `/remember <key> <fact>` — Store an explicit fact into persistent memory
- `/stats` — Show active session statistics (turns, tool calls, memory count)
- `/clear` — Clear current conversation context
- `/exit` — Exit the REPL

---

## 6. Training Pipeline

### Step 1: Acquire & Prepare Dataset
```bash
# Download Python standard library sources and generate instruction samples
python scripts/download_dataset.py --source stdlib --generate-instructions

# Build memory-mapped binary shards (data/train.bin, data/val.bin)
python scripts/prepare_dataset.py --input-dir data/raw_code --output-dir data --val-ratio 0.1
```

### Step 2: Train Custom Tokenizer
```bash
python scripts/train_tokenizer.py --input-dir data/raw_code --output-dir tokenizer_cache/smoke --vocab-size 1024
```

### Step 3: Run Model Pretraining
```bash
python scripts/train.py \
    --config configs/smoke.yaml \
    --train-bin data/train.bin \
    --val-bin data/val.bin \
    --max-steps 100 \
    --save-interval 50 \
    --checkpoint-dir checkpoints/
```

### Step 4: Supervised Instruction Fine-Tuning
```bash
python scripts/train_instruct.py \
    --base-checkpoint checkpoints/checkpoint_step_100.pt \
    --tokenizer-dir tokenizer_cache/smoke \
    --data-file data/coding_instructions.jsonl \
    --output-dir checkpoints/instruct
```

---

## 7. Google Colab Workflow

For cloud-accelerated training on Google Colab's free GPU tier:

1. Open [`notebooks/colab/aura_training.ipynb`](notebooks/colab/aura_training.ipynb) in Google Colab.
2. Select runtime: **Runtime > Change runtime type > T4 GPU**.
3. Execute the automated cells:
   - Diagnostic hardware verification (GPU model, VRAM limits).
   - Optional Google Drive mounting for checkpoint persistence.
   - Smoke training validation before initiating larger model runs.
   - Continuous loss and learning-rate logging.
   - Automatic checkpoint syncing to Google Drive.

---

## 8. Backward Compatibility Layer

Codebases or scripts written for legacy CodeMind continue to work without modification:

```python
# Legacy import (emits DeprecationWarning with upgrade path to aura)
from susacodemind.model import SusaCodeMind
from susacodemind.agent import CodeMindAgent
from susacodemind.training import Trainer

# Modern SUSA AURA import
from aura.model import AuraModel
from aura.core import AuraCore
from aura.training import AuraTrainer
```

---

## 9. Zero-Cost Engineering Philosophy

This project adheres strictly to ethical, transparent resource management:
- **Local CPU:** Verification, unit tests, tokenization, and architecture smoke tests run on local commodity hardware with zero cloud cost.
- **Free Cloud GPUs:** Cloud training utilizes free Google Colab sessions without incurring subscription or API fees.
- **Responsible AI:** Transparent parameter reporting, verifiable loss metrics, and sandboxed tool execution.

---

## 10. License

Licensed under the [Apache License, Version 2.0](LICENSE).
Copyright © 2026 SUSA Ecosystem / SUSA AURA Contributors.
