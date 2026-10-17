"""
Unit tests for SUSA-1B CodeMind Pretraining Engine (Milestone 4).
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pytest
import torch

from aura.model.config import ModelConfig
from aura.model.transformer import AuraModel, SusaCodeMind
from aura.training.trainer import AuraTrainer, Trainer, TrainingConfig

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


class DummyTokenDataset(torch.utils.data.Dataset):
    """Synthetic dataset generating fixed predictable patterns for training tests."""

    def __init__(self, num_samples: int = 20, seq_len: int = 16, vocab_size: int = 1024) -> None:
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Simple repeating pattern: e.g. [1, 2, 3, 4, 1, 2, 3, 4...]
        pattern = torch.tensor([(i + idx) % 50 + 10 for i in range(self.seq_len)], dtype=torch.long)
        return {
            "input_ids": pattern,
            "labels": pattern.clone(),
        }


@pytest.fixture
def smoke_model() -> SusaCodeMind:
    cfg = ModelConfig.from_yaml(CONFIG_DIR / "smoke.yaml")
    return SusaCodeMind(cfg)


def test_lr_scheduler(smoke_model):
    """Verify linear warmup and cosine decay calculation."""
    cfg = TrainingConfig(lr=1e-3, min_lr=1e-4, warmup_steps=10, max_steps=100)
    trainer = Trainer(
        model=smoke_model,
        train_dataset=DummyTokenDataset(),
        config=cfg,
    )

    # At step 0, lr should be lr * 1 / 10
    lr_0 = trainer.get_lr(0)
    assert 0.0 < lr_0 < 1e-3

    # At warmup_steps - 1, lr should reach near max_lr
    lr_warmup = trainer.get_lr(9)
    assert math.isclose(lr_warmup, 1e-3, rel_tol=1e-2)

    # Midpoint step
    lr_mid = trainer.get_lr(55)
    assert 1e-4 < lr_mid < 1e-3

    # At max_steps, lr should reach min_lr
    lr_end = trainer.get_lr(100)
    assert math.isclose(lr_end, 1e-4, rel_tol=1e-2)


def test_optimizer_weight_decay_groups(smoke_model):
    """Verify 2D weights have decay while 1D norm weights have zero decay."""
    cfg = TrainingConfig(weight_decay=0.05)
    trainer = Trainer(
        model=smoke_model,
        train_dataset=DummyTokenDataset(),
        config=cfg,
    )

    groups = trainer.optimizer.param_groups
    assert len(groups) == 2
    assert groups[0]["weight_decay"] == 0.05  # 2D projection weights
    assert groups[1]["weight_decay"] == 0.0   # 1D norm weights


def test_training_loss_and_parameter_update(smoke_model):
    """Verify that a real training run decreases loss and modifies model weights."""
    train_ds = DummyTokenDataset(num_samples=20, seq_len=16, vocab_size=smoke_model.config.vocab_size)
    val_ds = DummyTokenDataset(num_samples=4, seq_len=16, vocab_size=smoke_model.config.vocab_size)

    initial_weight = smoke_model.tok_embeddings.weight.clone().detach()

    cfg = TrainingConfig(
        batch_size=2,
        grad_accum_steps=1,
        lr=0.01,
        min_lr=0.001,
        warmup_steps=2,
        max_steps=15,
        eval_interval=5,
        save_interval=10,
        log_interval=1,
    )

    trainer = Trainer(
        model=smoke_model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        config=cfg,
    )

    history = trainer.train()

    # 1. Assert training steps completed
    assert len(history) == 15
    assert trainer.global_step == 15

    # 2. Assert model parameters actually changed
    final_weight = smoke_model.tok_embeddings.weight.detach().cpu()
    diff = torch.norm(final_weight - initial_weight.cpu()).item()
    assert diff > 1e-4, f"Parameters did not update! Weight diff: {diff}"

    # 3. Assert initial loss vs final loss
    initial_loss = history[0]["train_loss"]
    final_loss = history[-1]["train_loss"]
    assert final_loss < initial_loss, f"Loss did not decrease! Initial: {initial_loss:.4f}, Final: {final_loss:.4f}"


def test_checkpoint_save_and_resumption(smoke_model):
    """Verify training checkpoint saving and resuming execution."""
    train_ds = DummyTokenDataset(num_samples=20, seq_len=16, vocab_size=smoke_model.config.vocab_size)

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_dir = Path(tmp_dir) / "checkpoints"
        cfg1 = TrainingConfig(
            batch_size=2,
            max_steps=5,
            checkpoint_dir=str(ckpt_dir),
            save_interval=5,
        )

        trainer1 = Trainer(
            model=smoke_model,
            train_dataset=train_ds,
            config=cfg1,
        )
        trainer1.train()

        ckpt_file = ckpt_dir / "checkpoint_step_5.pt"
        assert ckpt_file.exists()

        # Resume with second trainer
        model2 = SusaCodeMind(smoke_model.config)
        cfg2 = TrainingConfig(
            batch_size=2,
            max_steps=10,
            checkpoint_dir=str(ckpt_dir),
            save_interval=5,
        )
        trainer2 = Trainer(
            model=model2,
            train_dataset=train_ds,
            config=cfg2,
        )

        resumed_step = trainer2.load_checkpoint(ckpt_file)
        assert resumed_step == 5
        assert trainer2.global_step == 5

        # Complete steps 6 -> 10
        trainer2.train()
        assert trainer2.global_step == 10
