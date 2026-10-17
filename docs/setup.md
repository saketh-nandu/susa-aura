# SUSA AURA — Setup & Installation Guide

This guide covers setting up SUSA AURA on **Fedora Linux (Commodity Laptop CPU)** and **Google Colab (Free GPU)**.

---

## 1. Prerequisites

- **Operating System:** Linux (Fedora 40+, Ubuntu 22.04+), macOS, or Windows WSL2.
- **Python:** Python 3.10 to 3.14 (Verified on Python 3.14.8 on Fedora).
- **Git:** Git 2.40+.
- **RAM:** Minimum 4 GB RAM for CPU smoke tests; 8 GB recommended.

---

## 2. Local Environment Setup (Fedora Linux / CPU)

```bash
# 1. Clone the repository
git clone https://github.com/saketh-nandu/susa-1b-codemind.git susa-aura
cd susa-aura

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Upgrade pip and install CPU PyTorch dependencies
pip install --upgrade pip
pip install -e ".[dev]"

# 4. Copy environment configuration template
cp .env.example .env
```

---

## 3. Verification & Diagnostic Commands

```bash
# 1. Check system hardware safety and architecture profiles
python scripts/verify_env.py

# 2. Inspect exact parameter breakdown and memory estimates
python scripts/count_params.py configs/smoke.yaml
python scripts/count_params.py configs/target_1b.yaml

# 3. Run automated test suite (55+ unit & integration tests)
pytest tests/ -v

# 4. Run model architecture smoke test
python scripts/smoke_test_model.py
```

---

## 4. Running the AURA Interactive CLI

Once installed, launch the CLI using the console script or direct runner:

```bash
# Using installed console script
aura

# Or using direct Python script
python -m aura.interface.cli

# Inspect workspace directly
aura --inspect

# Single-turn query mode
aura --prompt "What is RMSNorm in modern Transformers?"
```

---

## 5. Google Colab GPU Setup

1. Open `notebooks/colab/aura_training.ipynb` in Google Colab.
2. Select **Runtime -> Change runtime type -> Hardware accelerator -> T4 GPU**.
3. Run Step 1 (Hardware Diagnostics) to verify CUDA GPU availability.
4. Run through the automated steps to train, fine-tune, and evaluate.
