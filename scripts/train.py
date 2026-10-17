#!/usr/bin/env python3
"""
SUSA AURA — Pre-Training Execution Script.

Usage:
    python scripts/train.py --config configs/smoke.yaml --train-bin data/train.bin --val-bin data/val.bin
    python scripts/train.py --config configs/small.yaml --train-bin data/train.bin --val-bin data/val.bin --resume-from checkpoints/checkpoint_step_50.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from aura.data.dataset import MemoryMappedPretrainingDataset
from aura.model.config import ModelConfig
from aura.model.transformer import AuraModel, SusaCodeMind
from aura.training.trainer import AuraTrainer, Trainer, TrainingConfig
from aura.utils.resource_estimator import estimate_resources
from aura.utils.system_checker import SystemInfo


def main() -> int:
    parser = argparse.ArgumentParser(description="Pretrain SUSA AURA Transformer from scratch.")
    parser.add_argument("--config", type=str, default="configs/smoke.yaml", help="Path to YAML configuration file.")
    parser.add_argument("--train-bin", type=str, default="data/train.bin", help="Path to training .bin shard.")
    parser.add_argument("--val-bin", type=str, default="data/val.bin", help="Path to validation .bin shard.")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to checkpoint to resume training.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override maximum training steps.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override per-device batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")

    args = parser.parse_args()
    config_path = Path(args.config)

    print("=" * 70)
    print("  SUSA AURA — PRETRAINING LAUNCHER")
    print("=" * 70)

    # 1. Load Model and Training Configuration
    if not config_path.exists():
        print(f"  [ERROR] Configuration not found: {config_path}", file=sys.stderr)
        return 1

    with open(config_path, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)

    model_config = ModelConfig.from_dict(raw_cfg)
    train_config = TrainingConfig.from_dict(raw_cfg)

    # CLI Overrides
    if args.max_steps is not None:
        train_config.max_steps = args.max_steps
    if args.batch_size is not None:
        train_config.batch_size = args.batch_size
    if args.lr is not None:
        train_config.lr = args.lr

    print(model_config.print_parameter_summary())

    # 2. Resource & Safety Inspection
    sys_info = SystemInfo.inspect(target_dir=str(PROJECT_ROOT))
    res_est = estimate_resources(
        model_config,
        batch_size=train_config.batch_size,
        seq_len=model_config.max_seq_len,
    )

    if not sys_info.cuda_available and not res_est.feasible_on_local_laptop:
        print(f"\n  [SAFETY SHUTDOWN] Cannot train profile '{model_config.profile_name}' on local laptop CPU.")
        print(f"  Required Memory: ~{res_est.estimated_total_training_ram_gb:.2f} GiB.")
        print(f"  Available RAM:   {sys_info.ram_available_gb} GiB.")
        print("  Please use 'configs/smoke.yaml' locally or deploy on Google Colab GPU.")
        return 1

    # 3. Load Sharded Binary Datasets
    train_bin_path = Path(args.train_bin)
    val_bin_path = Path(args.val_bin)

    if not train_bin_path.exists():
        print(f"  [ERROR] Training dataset shard not found: {train_bin_path}", file=sys.stderr)
        print("  Please run `python scripts/prepare_dataset.py` first.", file=sys.stderr)
        return 1

    print("\n  [INFO] Initializing memory-mapped pretraining datasets...")
    train_dataset = MemoryMappedPretrainingDataset(train_bin_path, seq_len=model_config.max_seq_len)
    val_dataset = (
        MemoryMappedPretrainingDataset(val_bin_path, seq_len=model_config.max_seq_len)
        if val_bin_path.exists()
        else None
    )

    print(f"  - Train dataset: {len(train_dataset):,} samples ({train_dataset.total_tokens:,} tokens)")
    if val_dataset:
        print(f"  - Val dataset:   {len(val_dataset):,} samples ({val_dataset.total_tokens:,} tokens)")

    # 4. Instantiate Model & Trainer
    print("\n  [INFO] Instantiating model from randomly initialized weights...")
    model = AuraModel(model_config)
    trainer = AuraTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=train_config,
    )

    # 5. Checkpoint Resumption
    if args.resume_from:
        print(f"\n  [INFO] Resuming training from checkpoint: {args.resume_from}")
        resumed_step = trainer.load_checkpoint(args.resume_from)
        print(f"  Resumed from global step: {resumed_step}")

    # 6. Execute Pretraining
    trainer.train()
    return 0


if __name__ == "__main__":
    sys.exit(main())
