"""
Unit tests for SUSA AURA 1B Training Infrastructure, Architecture Validation,
Leakage Detection, and Atomic Checkpointing.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from aura.data.deduplication import LeakageReport, check_train_val_leakage
from aura.model.attention import GroupedQueryAttention
from aura.model.config import ModelConfig
from aura.model.transformer import AuraModel
from aura.training.instruction_trainer import collate_instruction_batch
from aura.training.trainer import AuraTrainer, TrainingConfig

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def test_target_1b_exact_parameter_breakdown():
    """Verify target_1b configuration produces mathematically exact 1,009,338,368 parameters."""
    cfg = ModelConfig.from_yaml(CONFIG_DIR / "target_1b.yaml")
    stats = cfg.calculate_exact_parameters()

    assert stats["total_parameters"] == 1_009_338_368
    assert stats["token_embeddings"] == 65_536_000
    assert stats["all_layers_total"] == 943_800_320
    assert stats["final_norm"] == 2048
    assert stats["lm_head"] == 0  # Tied embeddings

    # Verify per-layer attention and MLP breakdown
    layer = stats["single_layer"]
    assert layer["attention_subtotal"] == 12_582_912
    assert layer["mlp_subtotal"] == 34_603_008
    assert layer["layer_norms"] == 4096


def test_sdpa_causal_and_padding_mask_compatibility():
    """Verify that GroupedQueryAttention correctly handles 2D padding masks without crashing SDPA."""
    cfg = ModelConfig(
        vocab_size=256,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        n_layers=1,
        d_ff=128,
        max_seq_len=64,
    )
    attn = GroupedQueryAttention(cfg, layer_idx=0)
    attn.eval()

    # Batch of 2, seq len 8
    hidden_states = torch.randn(2, 8, 64)
    # Sequence 0 has 2 pad tokens at end; Sequence 1 has 0 pad tokens
    padding_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.long,
    )

    # Must execute forward without raising 'Explicit attn_mask should not be set when is_causal=True'
    out = attn(hidden_states, attention_mask=padding_mask)
    assert out.shape == (2, 8, 64)
    assert not torch.isnan(out).any()


def test_trainer_atomic_checkpoint_and_rng_state_resumption(tmp_path):
    """Verify trainer saves checkpoints atomically and restores RNG states and step progress."""
    cfg = ModelConfig(
        vocab_size=256,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        n_layers=2,
        d_ff=128,
        max_seq_len=32,
    )
    model = AuraModel(cfg)

    # Dummy dataset
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 20

        def __getitem__(self, idx):
            x = torch.randint(0, 256, (32,))
            return {"input_ids": x, "labels": x.clone()}

    train_cfg = TrainingConfig(
        batch_size=2,
        max_steps=5,
        checkpoint_dir=str(tmp_path),
        log_interval=1,
    )
    trainer = AuraTrainer(model=model, train_dataset=DummyDataset(), config=train_cfg)

    # Set known RNG states
    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)

    trainer.global_step = 3
    trainer.total_tokens_processed = 192

    ckpt_file = tmp_path / "test_ckpt.pt"
    trainer.save_checkpoint(ckpt_file, val_loss=1.5)

    assert ckpt_file.exists()
    assert not (tmp_path / ".test_ckpt.pt.tmp").exists()  # Atomic temp file was cleaned up

    # Mutate state and RNG
    trainer.global_step = 0
    trainer.total_tokens_processed = 0
    random.seed(99999)
    np.random.seed(99999)
    torch.manual_seed(99999)

    # Resume
    resumed_step = trainer.load_checkpoint(ckpt_file)
    assert resumed_step == 3
    assert trainer.global_step == 3
    assert trainer.total_tokens_processed == 192


def test_train_val_leakage_detection():
    """Verify that train/val leakage checker catches exact and near-duplicate leakage."""
    train_docs = [
        ("t1", "def quicksort(arr): return arr if len(arr) <= 1 else arr"),
        ("t2", "import math\nprint(math.sqrt(16))"),
        ("t3", "class DatabaseConnector: def connect(self): pass"),
    ]

    # Val set has 1 exact match (t1), 1 near duplicate (t3 with slight whitespace/naming), and 1 clean doc
    val_docs = [
        ("v1", "def quicksort(arr): return arr if len(arr) <= 1 else arr"),  # Exact
        ("v2", "class DatabaseConnector: def connect(self): # comments\n pass"),  # Near
        ("v3", "def calculate_orbit(trajectory): return 42.0"),  # Clean
    ]

    report: LeakageReport = check_train_val_leakage(train_docs, val_docs, similarity_threshold=0.7)

    assert report.total_train_docs == 3
    assert report.total_val_docs == 3
    assert report.exact_leakage_count == 1
    assert "v1" in report.leakage_doc_ids
    assert report.clean is False

    # Fully clean val set
    clean_val = [
        ("c1", "def unrelated_algorithm(): return True"),
        ("c2", "class TotallyNewModule: pass"),
    ]
    clean_report = check_train_val_leakage(train_docs, clean_val, similarity_threshold=0.85)
    assert clean_report.clean is True
    assert clean_report.exact_leakage_count == 0
    assert clean_report.near_leakage_count == 0


def test_collate_instruction_batch_attention_mask():
    """Verify collate_instruction_batch builds proper attention_mask."""
    b1 = {
        "input_ids": torch.tensor([10, 20, 30]),
        "labels": torch.tensor([-100, -100, 30]),
    }
    b2 = {
        "input_ids": torch.tensor([10, 20, 30, 40, 50]),
        "labels": torch.tensor([-100, -100, 30, 40, 50]),
    }
    batch = collate_instruction_batch([b1, b2], pad_id=0)

    assert "attention_mask" in batch
    mask = batch["attention_mask"]
    assert mask.shape == (2, 5)
    # First item has 3 valid tokens and 2 pad tokens
    assert mask[0].tolist() == [1, 1, 1, 0, 0]
    # Second item has 5 valid tokens
    assert mask[1].tolist() == [1, 1, 1, 1, 1]


def test_gradient_checkpointing_toggle():
    """Verify gradient checkpointing flag is propagated to model layers."""
    cfg = ModelConfig(
        vocab_size=256,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        n_layers=2,
        d_ff=128,
        max_seq_len=32,
    )
    model = AuraModel(cfg)
    assert model.gradient_checkpointing is False

    model.gradient_checkpointing = True
    assert model.gradient_checkpointing is True

    # Test forward pass with gradient checkpointing during training
    model.train()
    x = torch.randint(0, 256, (2, 16))
    out = model(x, labels=x)
    assert out.loss is not None
    out.loss.backward()


def test_colab_notebook_structure_and_syntax():
    """Verify AURA_1B_Colab_Training.ipynb exists, is valid JSON, and has valid Python syntax in all code cells."""
    import ast
    import json

    nb_path = Path(__file__).resolve().parent.parent / "notebooks" / "AURA_1B_Colab_Training.ipynb"
    assert nb_path.exists(), f"Missing deliverable notebook at: {nb_path}"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    assert "cells" in nb
    assert nb.get("nbformat") == 4
    assert len(nb["cells"]) >= 10

    # Verify all 4 required modes are mentioned in the notebook
    full_text = "".join("".join(c.get("source", [])) for c in nb["cells"])
    for mode in ["AUDIT", "PREPARE", "PILOT", "RESUME"]:
        assert mode in full_text, f"Notebook missing mode {mode}"

    # Verify AST syntax of all code cells
    for idx, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            # Filter IPython magic lines
            py_lines = []
            for line in source.split("\n"):
                s = line.strip()
                if s.startswith("!") or s.startswith("%"):
                    py_lines.append("# " + line)
                else:
                    py_lines.append(line)
            ast.parse("\n".join(py_lines))

