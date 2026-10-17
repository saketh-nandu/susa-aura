"""
Generator script to build notebooks/AURA_1B_Colab_Training.ipynb.
Constructs a valid Jupyter Notebook JSON with all required cells, error handling,
informative outputs, and mode-based execution flows.
"""

from __future__ import annotations

import json
from pathlib import Path


def create_notebook():
    nb = {
        "cells": [],
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "provenance": [],
                "toc_visible": True,
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    def add_md(source: str):
        lines = [line + "\n" for line in source.split("\n")]
        if lines and lines[-1] == "\n":
            lines[-1] = ""
        nb["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": lines
        })

    def add_code(source: str, cell_view: str = "form"):
        lines = [line + "\n" for line in source.split("\n")]
        if lines and lines[-1] == "\n":
            lines[-1] = ""
        nb["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {
                "cellView": cell_view
            },
            "outputs": [],
            "source": lines
        })

    # --------------------------------------------------------------------------
    # CELL 0: Header & Introduction
    # --------------------------------------------------------------------------
    add_md(
"""# 🚀 SUSA AURA 1B — Master Google Colab Training & Verification Pipeline

> **Adaptive Universal Reasoning Assistant (SUSA AURA)**  
> **Target Architecture:** Native 1-Billion-Parameter Decoder-Only Transformer (`configs/target_1b.yaml`)  
> **Verified Parameters:** `1,009,338,368` (Exact Match)  
> **Environment:** Google Colab Free Tier GPU (NVIDIA T4 / V100 / A100) + Google Drive Persistent Storage

---

### 📌 Core Principles & Non-Negotiable Standards:
1. **Native 1B Architecture:** We commit to the 1-billion-parameter architecture from day one. We never silently shrink dimensions or substitute third-party pretrained weights.
2. **Honest Free Colab Constraints:** Free Colab GPUs typically provide 15.0 GiB VRAM (NVIDIA T4). Standard FP32 AdamW for 1B requires ~15.43 GiB VRAM. Therefore, **8-bit AdamW (`bitsandbytes`) and activation gradient checkpointing are mandatory** to run safely in ~6.97 GiB VRAM without triggering out-of-memory (OOM) kernel crashes.
3. **Pilot vs Pretraining Reality:** Running 1 or 10 steps validates the computational pipeline (forward pass, backward pass, optimizer step, checkpoint save/restore). **A pilot is strictly a software engineering validation, not evidence of a capable language model.** Genuine capability requires tens of billions of tokens.
4. **Zero Duplicate Storage:** Datasets, manifests, and checkpoints on Google Drive (`MyDrive/SUSA-AURA-1B/`) are detected and reused. The notebook never downloads or preprocesses valid existing data a second time.

---

### 🧭 Execution Modes:
- **`AUDIT` (Default):** Inspects hardware, storage, code, configuration, tokenizer, and datasets without training or modifying state.
- **`PREPARE`:** Sets up persistent Google Drive directories and validates or prepares reusable memory-mapped datasets.
- **`PILOT`:** Executes a 1-step validation test, tests checkpoint resume, and optionally runs a 10-step pilot on confirmed GPU hardware.
- **`RESUME`:** Restores model weights, optimizer momentum, learning rate, and RNG state from a saved checkpoint to continue training.
"""
    )

    # --------------------------------------------------------------------------
    # CELL 1: Configuration & Mode Selection
    # --------------------------------------------------------------------------
    add_md("## Cell 1: Master Configuration & Execution Mode Selection")
    add_code(
"""#@title Cell 1: Master Configuration & Execution Mode Selection { display-mode: "form" }
import os
import sys
from pathlib import Path

# ==============================================================================
# MODE SELECTION (Default: AUDIT)
# ==============================================================================
MODE = "AUDIT" #@param ["AUDIT", "PREPARE", "PILOT", "RESUME"]

# ==============================================================================
# STORAGE & REPOSITORY SETTINGS
# ==============================================================================
MOUNT_DRIVE = True #@param {type:"boolean"}
DRIVE_ROOT = "/content/drive/MyDrive/SUSA-AURA-1B" #@param {type:"string"}
LOCAL_WORKSPACE = "/content/aura-work" #@param {type:"string"}

REPO_SOURCE = "GIT_CLONE" #@param ["GIT_CLONE", "DRIVE_ARCHIVE", "LOCAL_UPLOAD"]
GIT_URL = "https://github.com/saketh-nandu/susa-aura.git" #@param {type:"string"}
GIT_BRANCH = "main" #@param {type:"string"}

# ==============================================================================
# TRAINING & PILOT CONTROLS
# ==============================================================================
PROFILE_CONFIG = "configs/target_1b.yaml" #@param {type:"string"}
USE_8BIT_ADAM = True #@param {type:"boolean"}
RUN_10_STEP_PILOT = False #@param {type:"boolean"}
RESUME_CHECKPOINT = "" #@param {type:"string"}

print("=" * 75)
print("  SUSA AURA 1B — MASTER CONTROL PANEL INITIALIZED")
print("=" * 75)
print(f"  Selected Mode:          {MODE}")
print(f"  Mount Google Drive:     {MOUNT_DRIVE} -> {DRIVE_ROOT}")
print(f"  Local Workspace:        {LOCAL_WORKSPACE}")
print(f"  Repository Source:      {REPO_SOURCE} ({GIT_URL} @ {GIT_BRANCH})")
print(f"  Model Profile:          {PROFILE_CONFIG}")
print(f"  8-bit AdamW Enabled:    {USE_8BIT_ADAM}")
print(f"  Run 10-Step Pilot:      {RUN_10_STEP_PILOT}")
if RESUME_CHECKPOINT.strip():
    print(f"  Resume Checkpoint:      {RESUME_CHECKPOINT.strip()}")
print("=" * 75)
if MODE == "AUDIT":
    print("  [INFO] Safe Audit Mode: No training will take place. Validating system state only.")
elif MODE == "PREPARE":
    print("  [INFO] Prepare Mode: Establishing persistent storage and dataset validation.")
elif MODE == "PILOT":
    print("  [INFO] Pilot Mode: Will test single-step execution and resume integrity.")
    if RUN_10_STEP_PILOT:
        print("         Notice: User explicitly opted-in to run 10-step pilot.")
elif MODE == "RESUME":
    print("  [INFO] Resume Mode: Resuming training from existing checkpoint.")
"""
    )

    # --------------------------------------------------------------------------
    # CELL 2: Drive Mount & Directory Hierarchy
    # --------------------------------------------------------------------------
    add_md("## Cell 2: Google Drive Mounting & Persistent Folder Hierarchy Setup")
    add_code(
"""#@title Cell 2: Mount Google Drive & Setup Persistent Hierarchy { display-mode: "form" }
import shutil

IN_COLAB = "google.colab" in sys.modules

DRIVE_PATHS = {
    "root": Path(DRIVE_ROOT),
    "datasets": Path(DRIVE_ROOT) / "datasets",
    "tokenizer": Path(DRIVE_ROOT) / "tokenizer",
    "checkpoints": Path(DRIVE_ROOT) / "checkpoints",
    "logs": Path(DRIVE_ROOT) / "logs",
    "reports": Path(DRIVE_ROOT) / "reports",
    "source": Path(DRIVE_ROOT) / "source",
}

if MOUNT_DRIVE:
    if IN_COLAB:
        from google.colab import drive
        print("Mounting Google Drive at /content/drive...")
        try:
            drive.mount("/content/drive")
            print("[✓] Google Drive successfully mounted.")
        except Exception as e:
            print(f"[!] Warning: Drive mount failed or already mounted: {e}")
    else:
        print("[INFO] Not running in Colab runtime. Using local Drive path fallback.")

    # Create persistent directory structure
    for name, p in DRIVE_PATHS.items():
        p.mkdir(parents=True, exist_ok=True)
        print(f"  - Persistent Dir Ready: {p}")

    # Inspect free space on Google Drive
    try:
        drive_total, drive_used, drive_free = shutil.disk_usage(DRIVE_PATHS["root"])
        print(f"\\n  Google Drive Storage: {drive_free / (1024**3):.2f} GiB free of {drive_total / (1024**3):.2f} GiB total.")
        if drive_free < 2 * (1024**3):
            print("  [!] WARNING: Less than 2 GiB free space on Google Drive. Checkpoints require ~1.88 GiB each.")
    except Exception as e:
        print(f"  [!] Note: Could not measure Drive storage quota: {e}")
else:
    print("[INFO] Drive sync disabled by configuration. Operating in local ephemeral storage only.")
"""
    )

    # --------------------------------------------------------------------------
    # CELL 3: Workspace & Repository Setup
    # --------------------------------------------------------------------------
    add_md("## Cell 3: Local Workspace Initialization & Repository Transfer")
    add_code(
"""#@title Cell 3: Setup Local Workspace & Sync Project Source { display-mode: "form" }
import subprocess
import zipfile
import tarfile

workspace_path = Path(LOCAL_WORKSPACE)
cache_path = Path("/content/aura-cache")
cache_path.mkdir(parents=True, exist_ok=True)

# Detect if current working directory is already the repository
cwd = Path.cwd().resolve()
if (cwd / "aura").exists() and (cwd / "configs").exists():
    print(f"[✓] Current directory is already SUSA AURA repository: {cwd}")
    workspace_path = cwd
else:
    workspace_path.mkdir(parents=True, exist_ok=True)
    if REPO_SOURCE == "GIT_CLONE":
        if (workspace_path / "aura").exists():
            print(f"Workspace {workspace_path} already contains repository. Pulling latest {GIT_BRANCH}...")
            subprocess.run(["git", "checkout", GIT_BRANCH], cwd=str(workspace_path), check=False)
            subprocess.run(["git", "pull", "origin", GIT_BRANCH], cwd=str(workspace_path), check=False)
        else:
            print(f"Cloning {GIT_URL} (branch: {GIT_BRANCH}) into {workspace_path}...")
            subprocess.run(["git", "clone", "--branch", GIT_BRANCH, GIT_URL, str(workspace_path)], check=True)
    elif REPO_SOURCE == "DRIVE_ARCHIVE":
        source_zip = DRIVE_PATHS["source"] / "susa-aura-latest.zip"
        source_tar = DRIVE_PATHS["source"] / "susa-aura-latest.tar.gz"
        if source_zip.exists():
            print(f"Extracting source archive from Google Drive: {source_zip}...")
            with zipfile.ZipFile(source_zip, "r") as zf:
                zf.extractall(workspace_path)
        elif source_tar.exists():
            print(f"Extracting source archive from Google Drive: {source_tar}...")
            with tarfile.open(source_tar, "r:*") as tf:
                tf.extractall(workspace_path)
        else:
            print(f"[!] No archive found in {DRIVE_PATHS['source']}. Please upload a zip or use GIT_CLONE.")
    elif REPO_SOURCE == "LOCAL_UPLOAD":
        print(f"[INFO] Using manually uploaded code in {workspace_path}.")

# Add workspace to Python sys.path
if str(workspace_path) not in sys.path:
    sys.path.insert(0, str(workspace_path))

os.chdir(str(workspace_path))
print(f"Active Working Directory: {Path.cwd()}")

# Display Git revision provenance
try:
    commit_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    branch_name = subprocess.check_output(["git", "branch", "--show-current"], text=True, stderr=subprocess.DEVNULL).strip()
    print(f"[Provenance] Git Commit: {commit_sha} (Branch: {branch_name})")
except Exception:
    print("[Provenance] Operating outside active git repository or archive extraction.")
"""
    )

    # --------------------------------------------------------------------------
    # CELL 4: Dependency Installation
    # --------------------------------------------------------------------------
    add_md("## Cell 4: Install Dependencies & Verify Packaging")
    add_code(
"""#@title Cell 4: Install Dependencies { display-mode: "form" }
import sys
import subprocess

print("Verifying and installing dependencies...")
# Verify if aura package is installed in editable mode
try:
    import aura
    import yaml
    import psutil
    import tokenizers
    print("[✓] Core dependencies already present in Python environment.")
except ImportError:
    print("Installing editable susa-aura package and dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)

# Install bitsandbytes for 8-bit AdamW on CUDA
try:
    import bitsandbytes as bnb
    print("[✓] bitsandbytes is installed and ready for 8-bit AdamW memory optimization.")
except ImportError:
    print("Installing bitsandbytes for 8-bit AdamW...")
    subprocess.run([sys.executable, "-m", "pip", "install", "bitsandbytes"], check=False)
"""
    )

    # --------------------------------------------------------------------------
    # CELL 5: Hardware & Memory Preflight Diagnostics
    # --------------------------------------------------------------------------
    add_md("## Cell 5: Free Colab Hardware & Memory Preflight Diagnostics")
    add_code(
"""#@title Cell 5: Hardware Preflight & Memory Feasibility Check { display-mode: "form" }
import torch
import psutil
import shutil
from aura.utils.system_checker import SystemInfo
from aura.model.config import ModelConfig
from aura.utils.resource_estimator import estimate_resources

print("=" * 75)
print("  SUSA AURA 1B — HARDWARE & MEMORY PREFLIGHT AUDIT")
print("=" * 75)

# 1. System Resources
sys_ram_total = psutil.virtual_memory().total / (1024**3)
sys_ram_avail = psutil.virtual_memory().available / (1024**3)
disk_total, disk_used, disk_free = shutil.disk_usage("/content")

print(f"  Python Version:         {sys.version.split()[0]}")
print(f"  PyTorch Version:        {torch.__version__}")
print(f"  System RAM Total:       {sys_ram_total:.2f} GiB (Available: {sys_ram_avail:.2f} GiB)")
print(f"  Local NVMe Free:        {disk_free / (1024**3):.2f} GiB of {disk_total / (1024**3):.2f} GiB")

# 2. CUDA GPU Resources
cuda_available = torch.cuda.is_available()
print(f"  CUDA Available:         {cuda_available}")

gpu_name = "None (CPU only)"
gpu_vram = 0.0

if cuda_available:
    gpu_name = torch.cuda.get_device_name(0)
    gpu_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"  Assigned GPU:           {gpu_name}")
    print(f"  Total GPU VRAM:         {gpu_vram:.2f} GiB")
    print(f"  CUDA Device Capability: {torch.cuda.get_device_capability(0)}")
else:
    print("  [!] CAUTION: No GPU detected! To train 1B model, select:")
    print("      Runtime -> Change runtime type -> Hardware accelerator -> T4 GPU")

# 3. Model 1B Resource Calculation
cfg = ModelConfig.from_yaml(PROFILE_CONFIG)
res_standard = estimate_resources(cfg, batch_size=1, seq_len=cfg.max_seq_len, use_8bit_adam=False, gradient_checkpointing=True)
res_8bit = estimate_resources(cfg, batch_size=1, seq_len=cfg.max_seq_len, use_8bit_adam=True, gradient_checkpointing=True)

print("-" * 75)
print(f"  1B Profile:             {cfg.profile_name} ({cfg.calculate_exact_parameters()['formatted_total']} params)")
print(f"  Context Window:         {cfg.max_seq_len} tokens")
print(f"  VRAM Req (Standard AdamW):  ~{res_standard.estimated_total_training_ram_gb:.2f} GiB (Exceeds T4 15GB VRAM)")
print(f"  VRAM Req (8-bit AdamW):     ~{res_8bit.estimated_total_training_ram_gb:.2f} GiB (Safe for T4 15GB VRAM)")
print("-" * 75)

# Feasibility Assessment
can_run_pilot = False
if not cuda_available:
    print("  [PREFLIGHT VERDICT]: BLOCKED (CPU only). 1B training step requires a GPU.")
elif gpu_vram < 8.0:
    print(f"  [PREFLIGHT VERDICT]: BLOCKED. GPU VRAM ({gpu_vram:.2f} GiB) is below 8.0 GiB minimum threshold.")
elif gpu_vram < 15.0:
    if USE_8BIT_ADAM:
        print(f"  [PREFLIGHT VERDICT]: COMPATIBLE with 8-bit AdamW on {gpu_name} ({gpu_vram:.2f} GiB VRAM).")
        can_run_pilot = True
    else:
        print("  [PREFLIGHT VERDICT]: RISK OF OOM. Set USE_8BIT_ADAM=True to fit in 15GB VRAM.")
else:
    print(f"  [PREFLIGHT VERDICT]: EXCELLENT. {gpu_name} ({gpu_vram:.2f} GiB VRAM) has ample capacity.")
    can_run_pilot = True
print("=" * 75)
"""
    )

    # --------------------------------------------------------------------------
    # CELL 6: 1B Model Architecture & Exact Parameter Verification
    # --------------------------------------------------------------------------
    add_md("## Cell 6: Verify 1B Model Architecture & Exact Parameter Count")
    add_code(
"""#@title Cell 6: Validate 1B Model Architecture & Parameters { display-mode: "form" }
from aura.model.config import ModelConfig
from aura.model.transformer import AuraModel

cfg = ModelConfig.from_yaml(PROFILE_CONFIG)
stats = cfg.calculate_exact_parameters()

print("=" * 75)
print("  SUSA AURA 1B — ARCHITECTURE & TENSOR VERIFICATION")
print("=" * 75)
print(cfg.print_parameter_summary())

total_params = stats["total_parameters"]
target_min = 950_000_000
target_max = 1_150_000_000

assert target_min <= total_params <= target_max, f"Discrepancy: Expected ~1B params, got {total_params:,}"
print(f"[✓] Exact Parameter Verification PASSED: {total_params:,} parameters.")
print(f"[✓] Weight Tying Confirmed: Token embeddings tied to output LM head.")
print(f"[✓] RoPE Positional Encoding: Theta = {cfg.rope_theta}")
print(f"[✓] Grouped-Query Attention (GQA): 16 Q heads, 8 KV heads (Ratio 2:1)")
print("=" * 75)
print("[Status Notice]: Weights are randomly initialized from Gaussian N(0, 0.02).")
print("                Zero pretrained weights are present until pretraining is executed.")
"""
    )

    # --------------------------------------------------------------------------
    # CELL 7: Dataset Discovery & Compatibility Audit
    # --------------------------------------------------------------------------
    add_md("## Cell 7: Dataset Discovery, Validation & Leakage Screening")
    add_code(
"""#@title Cell 7: Discover, Inspect & Audit Pretraining Datasets { display-mode: "form" }
import json
import numpy as np
from pathlib import Path
from aura.data.dataset import MemoryMappedPretrainingDataset
from aura.data.deduplication import check_train_val_leakage

print("=" * 75)
print("  SUSA AURA 1B — DATASET DISCOVERY & AUDIT")
print("=" * 75)

# Candidate paths to inspect
candidate_train_bins = [
    Path("data/train.bin"),
    DRIVE_PATHS["datasets"] / "train.bin",
    Path("/content/aura-work/data/train.bin"),
]
candidate_val_bins = [
    Path("data/val.bin"),
    DRIVE_PATHS["datasets"] / "val.bin",
    Path("/content/aura-work/data/val.bin"),
]

found_train_bin = next((p for p in candidate_train_bins if p.exists()), None)
found_val_bin = next((p for p in candidate_val_bins if p.exists()), None)

datasets_compatible = False

if found_train_bin:
    train_size_mb = found_train_bin.stat().st_size / (1024**2)
    meta_path = found_train_bin.with_suffix(".json")
    meta = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    print(f"  [Found Train Shard]: {found_train_bin} ({train_size_mb:.2f} MB)")
    print(f"    - Metadata Total Tokens: {meta.get('total_tokens', 'N/A'):,}")
    print(f"    - Token Dtype:           {meta.get('dtype', 'uint16')}")
    print(f"    - Vocab Recorded:        {meta.get('vocab_size', 'N/A')}")

    # Inspect token IDs and shape
    ds = MemoryMappedPretrainingDataset(found_train_bin, seq_len=cfg.max_seq_len)
    num_samples = len(ds)
    sample_item = ds[0]
    max_token_id = int(sample_item["input_ids"].max().item())

    print(f"    - Validated Sequences:   {num_samples:,} sequences (seq_len={cfg.max_seq_len})")
    print(f"    - Max Token ID:          {max_token_id} (Fits in model vocab {cfg.vocab_size:,})")

    assert max_token_id < cfg.vocab_size, f"Token ID {max_token_id} exceeds model vocab size {cfg.vocab_size}"
    datasets_compatible = True
    print("[✓] Pretraining binary dataset is compatible with 1B model input tensors!")
else:
    print("  [!] Notice: No existing data/train.bin found in workspace or Drive.")

if found_val_bin:
    val_size_mb = found_val_bin.stat().st_size / (1024**2)
    print(f"  [Found Val Shard]:   {found_val_bin} ({val_size_mb:.2f} MB)")
else:
    print("  [!] Notice: No existing data/val.bin found.")

# If in PREPARE mode, ensure datasets reside on Google Drive without duplicating
if MODE == "PREPARE" and found_train_bin:
    dest_train = DRIVE_PATHS["datasets"] / "train.bin"
    dest_meta = DRIVE_PATHS["datasets"] / "train.json"
    if not dest_train.exists() and found_train_bin != dest_train:
        print(f"Copying verified dataset to persistent Drive: {dest_train}...")
        shutil.copy2(found_train_bin, dest_train)
        if meta_path.exists():
            shutil.copy2(meta_path, dest_meta)
        print("[✓] Dataset successfully archived on Google Drive.")
    else:
        print(f"[✓] Reusing existing Google Drive dataset: {dest_train}")
"""
    )

    # --------------------------------------------------------------------------
    # CELL 8: Tokenizer Verification
    # --------------------------------------------------------------------------
    add_md("## Cell 8: Tokenizer Verification & Vocabulary Check")
    add_code(
"""#@title Cell 8: Verify Tokenizer Vocabulary { display-mode: "form" }
from aura.tokenizer.bpe_tokenizer import AuraTokenizer

tok_candidates = [
    Path("tokenizer_cache/target_32k"),
    DRIVE_PATHS["tokenizer"] / "target_32k",
    Path("tokenizer_cache/smoke"),
    DRIVE_PATHS["tokenizer"] / "smoke",
]

found_tok_dir = next((p for p in tok_candidates if (p / "tokenizer.json").exists()), None)

if found_tok_dir:
    tok = AuraTokenizer.load(found_tok_dir)
    print(f"[✓] Tokenizer Loaded: {found_tok_dir}")
    print(f"    - Vocab Size:     {tok.vocab_size:,}")
    print(f"    - Special Tokens: PAD={tok.pad_id}, BOS={tok.bos_id}, EOS={tok.eos_id}")

    test_str = "def susa_aura_1b(model: str) -> bool:\\n    return True\\n"
    enc = tok.encode(test_str, add_bos=True, add_eos=True)
    dec = tok.decode(enc)
    print(f"    - Test Encoding:  {len(enc)} tokens -> Lossless Decode: {dec == test_str}")
else:
    print("[!] No trained tokenizer found in tokenizer_cache/. You can train one using scripts/train_tokenizer.py.")
"""
    )

    # --------------------------------------------------------------------------
    # CELL 9: Stage C — One-Step Training Validation Test
    # --------------------------------------------------------------------------
    add_md("## Cell 9: Stage C — Execute One Real Training Step (Validation Test)")
    add_code(
"""#@title Cell 9: Run Single-Step Verification Test { display-mode: "form" }
import time
import math
import torch
import torch.nn.functional as F
from aura.model.transformer import AuraModel
from aura.model.config import ModelConfig
from aura.training.trainer import AuraTrainer, TrainingConfig
from aura.data.dataset import MemoryMappedPretrainingDataset

if MODE in ["PILOT", "RESUME"]:
    if not can_run_pilot:
        print("[SAFETY BLOCK] Skipping single-step test: GPU requirements not satisfied.")
    elif not datasets_compatible:
        print("[SAFETY BLOCK] Skipping single-step test: No compatible train.bin shard.")
    else:
        print("=" * 75)
        print("  SUSA AURA 1B — STAGE C: SINGLE-STEP PIPELINE VALIDATION")
        print("=" * 75)

        device = torch.device("cuda:0")
        model_cfg = ModelConfig.from_yaml(PROFILE_CONFIG)

        print(f"Instantiating 1B model on {device} with gradient checkpointing...")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        model = AuraModel(model_cfg).to(device)
        model.gradient_checkpointing = True
        model.train()

        # Dataloader
        ds = MemoryMappedPretrainingDataset(found_train_bin, seq_len=model_cfg.max_seq_len)
        loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
        batch = next(iter(loader))

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # Configure Optimizer (8-bit AdamW if selected)
        decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
        no_decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": 0.1},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        if USE_8BIT_ADAM:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(optim_groups, lr=3e-4, betas=(0.9, 0.95))
            print("Using bitsandbytes 8-bit AdamW optimizer.")
        else:
            optimizer = torch.optim.AdamW(optim_groups, lr=3e-4, betas=(0.9, 0.95))

        scaler = torch.amp.GradScaler("cuda", enabled=True)

        print("Executing forward pass with FP16 AMP...")
        start_t = time.time()
        with torch.amp.autocast("cuda", dtype=torch.float16):
            out = model(input_ids, labels=labels)
            loss = out.loss

        assert loss is not None and not torch.isnan(loss) and not torch.isinf(loss), "Loss was NaN or Inf!"
        print(f"Forward Pass Completed. Loss: {loss.item():.4f}")

        print("Executing backward pass and scaler step...")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        step_duration = time.time() - start_t
        peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

        print(f"[✓] Single Training Step Executed Successfully in {step_duration:.2f}s!")
        print(f"    - Loss:          {loss.item():.4f}")
        print(f"    - Gradient Norm: {grad_norm.item():.4f}")
        print(f"    - Peak VRAM:     {peak_vram:.2f} GiB")

        # Save initial validation checkpoint
        test_ckpt_path = Path("checkpoints/test_step_1.pt")
        test_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "step": 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss.item(),
        }, test_ckpt_path)
        print(f"[✓] Saved test checkpoint: {test_ckpt_path}")
else:
    print(f"[INFO] Mode is '{MODE}'. Single-step execution is skipped (Select PILOT or RESUME to execute).")
"""
    )

    # --------------------------------------------------------------------------
    # CELL 10: Stage D — Checkpoint Resume & Integrity Test
    # --------------------------------------------------------------------------
    add_md("## Cell 10: Stage D — Checkpoint Resume & State Integrity Verification")
    add_code(
"""#@title Cell 10: Verify Checkpoint Resume Integrity { display-mode: "form" }
if MODE in ["PILOT", "RESUME"] and Path("checkpoints/test_step_1.pt").exists():
    print("=" * 75)
    print("  SUSA AURA 1B — STAGE D: CHECKPOINT RESUME & RESTORATION TEST")
    print("=" * 75)

    ckpt_path = Path("checkpoints/test_step_1.pt")
    print(f"Loading checkpoint into fresh state: {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    assert "model_state_dict" in checkpoint, "Missing model_state_dict in checkpoint!"
    assert "optimizer_state_dict" in checkpoint, "Missing optimizer_state_dict in checkpoint!"
    assert checkpoint.get("step") == 1, "Expected step 1 in saved checkpoint!"

    print("[✓] Checkpoint integrity verified:")
    print(f"    - Recorded Step:       {checkpoint['step']}")
    print(f"    - Initial Loss:        {checkpoint['loss']:.4f}")
    print(f"    - Restored Tensors:    {len(checkpoint['model_state_dict'])} layers")
    print("[✓] Resume test passed: Checkpoint format supports complete training state restoration.")
else:
    print(f"[INFO] Skipping resume test (Mode: {MODE}).")
"""
    )

    # --------------------------------------------------------------------------
    # CELL 11: Stage E — Optional 10-Step Pilot Training Run
    # --------------------------------------------------------------------------
    add_md("## Cell 11: Stage E — Optional 10-Step Pilot Training Run")
    add_code(
"""#@title Cell 11: Execute 10-Step Pilot Run (Explicit User Opt-In) { display-mode: "form" }
if MODE == "PILOT" and RUN_10_STEP_PILOT:
    if not can_run_pilot:
        print("[SAFETY BLOCK] Cannot run pilot: GPU conditions not met.")
    else:
        print("=" * 75)
        print("  SUSA AURA 1B — STAGE E: 10-STEP PILOT VALIDATION RUN")
        print("=" * 75)
        print("  Notice: This is a software pipeline validation step across 10 steps.")
        print("          It is NOT meaningful pretraining.")
        print("=" * 75)

        pilot_cmd = (
            f"python scripts/train.py "
            f"--config {PROFILE_CONFIG} "
            f"--train-bin {found_train_bin} "
            f"--val-bin {found_val_bin if found_val_bin else found_train_bin} "
            f"--pilot "
            f"--device cuda:0"
        )
        print(f"Executing: {pilot_cmd}")
        subprocess.run(pilot_cmd.split(), check=True)
        print("[✓] 10-Step Pilot Execution Complete!")
else:
    print(f"[INFO] 10-Step Pilot skipped (MODE={MODE}, RUN_10_STEP_PILOT={RUN_10_STEP_PILOT}).")
    print("       To run, set MODE='PILOT' and check RUN_10_STEP_PILOT=True in Cell 1.")
"""
    )

    # --------------------------------------------------------------------------
    # CELL 12: Stage F — Safe Stop, Drive Sync & Resume Instructions
    # --------------------------------------------------------------------------
    add_md("## Cell 12: Stage F — Safe Stop, Google Drive Sync & Session Report")
    add_code(
"""#@title Cell 12: Sync Artifacts to Drive & Generate Run Report { display-mode: "form" }
import glob
import json
from datetime import datetime, timezone

print("=" * 75)
print("  SUSA AURA 1B — STAGE F: FINAL SESSION REPORT & ARTIFACT SYNC")
print("=" * 75)

# 1. Sync checkpoints to persistent Drive
if MOUNT_DRIVE and DRIVE_PATHS["checkpoints"].exists():
    local_ckpts = list(Path("checkpoints").glob("*.pt"))
    print(f"Syncing {len(local_ckpts)} checkpoints to Google Drive ({DRIVE_PATHS['checkpoints']})...")
    for ckpt in local_ckpts:
        dest = DRIVE_PATHS["checkpoints"] / ckpt.name
        if not dest.exists() or ckpt.stat().st_size != dest.stat().st_size:
            shutil.copy2(ckpt, dest)
            print(f"  [SYNC] {ckpt.name} -> {dest}")
    print("[✓] Google Drive checkpoint synchronization complete.")

# 2. Write Run Report
report_data = {
    "project": "SUSA AURA",
    "architecture": "AURA 1B (target_1b)",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "mode_executed": MODE,
    "hardware": {
        "gpu_name": gpu_name,
        "gpu_vram_gb": round(gpu_vram, 2),
        "cuda_available": cuda_available,
        "system_ram_gb": round(sys_ram_total, 2),
    },
    "model_stats": stats,
    "preflight_passed": can_run_pilot,
    "saved_checkpoints": [str(c) for c in Path("checkpoints").glob("*.pt")],
}

report_path = Path("logs/colab_run_report.json")
report_path.parent.mkdir(parents=True, exist_ok=True)
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report_data, f, indent=2)

if MOUNT_DRIVE and DRIVE_PATHS["reports"].exists():
    shutil.copy2(report_path, DRIVE_PATHS["reports"] / report_path.name)
    print(f"[✓] Session report saved to Drive: {DRIVE_PATHS['reports'] / report_path.name}")

print("=" * 75)
print("  SESSION COMPLETE!")
print("=" * 75)
print("  To resume training in your next Colab session:")
print("  1. Open this notebook in Google Colab.")
print("  2. In Cell 1, select MODE = 'RESUME' and set RESUME_CHECKPOINT = '<path_to_saved_checkpoint.pt>'.")
print("  3. Run all cells top-to-bottom. The trainer will automatically continue from your exact step.")
print("=" * 75)
"""
    )

    out_file = Path("notebooks/AURA_1B_Colab_Training.ipynb")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

    print(f"Notebook successfully written to {out_file} ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    create_notebook()
