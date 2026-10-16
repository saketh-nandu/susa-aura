# SUSA-1B CodeMind — Training Guide

## 1. Zero-Cost Hardware Strategy

To respect our strict **₹0 budget** and protect local development laptops (Fedora with 8GB RAM, 4 CPU threads):

### 1.1 Local Laptop Role
- Writing, testing, and debugging architecture code.
- Running unit and functional tests (`pytest tests/`).
- Running ultra-small CPU smoke tests (`configs/smoke.yaml`).
- Processing small dataset samples and building tokenizers.
- Git operations, versioning, and documentation.
- **NEVER attempt full 1B pretraining on laptop CPU** (Will cause system freezes and OOM).

### 1.2 Google Colab GPU Role
- GPU environment verification and VRAM benchmarking.
- Pretraining small and medium profile models (`configs/small.yaml`, `configs/medium.yaml`).
- Model-size scaling tests.
- Storing active checkpoints on local runtime disk (`/content/checkpoints`) and syncing snapshots to Google Drive.

---

## 2. Local Environment Setup

### 2.1 Dependencies Installation
```bash
# Create local virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install CPU PyTorch and minimal dependencies
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install pyyaml pytest psutil numpy
```

### 2.2 System & Config Verification
```bash
# Run environment diagnostics and config validation
python scripts/verify_env.py

# Inspect parameter breakdown for a specific profile
python scripts/count_params.py configs/smoke.yaml
python scripts/count_params.py configs/target_1b.yaml
```

---

## 3. Training Loop Specifications
- **Objective:** Standard autoregressive next-token prediction cross-entropy loss.
- **Optimizer:** AdamW ($\beta_1=0.9, \beta_2=0.95, \epsilon=10^{-8}$, weight decay $0.1$).
- **Learning Rate Schedule:** Cosine decay with linear warmup.
- **Precision:** Mixed precision (bfloat16 on Ampere+ GPUs, float16 on T4, float32 on CPU).
- **Gradient Accumulation:** Used to achieve target global batch size without exceeding GPU VRAM.
- **Gradient Checkpointing:** Recomputes activations in backward pass to drop activation memory by up to 70%.
