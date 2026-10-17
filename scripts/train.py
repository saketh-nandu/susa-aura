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
from aura.training.checkpoint_manager import find_best_checkpoint, validate_checkpoint_file
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
    parser.add_argument("--pilot", action="store_true", help="Run short pilot run (10 steps) to validate forward/backward and checkpointing.")
    parser.add_argument("--dry-run", action="store_true", help="Perform architecture validation and resource checking without training.")
    parser.add_argument("--device", type=str, default=None, help="Explicit device string (e.g. 'cuda', 'cpu').")
    parser.add_argument("--additional-steps", type=int, default=None, help="Number of additional steps to train after checkpoint resumption.")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Directory for local checkpoints.")
    parser.add_argument("--drive-checkpoint-dir", type=str, default=None, help="Persistent Google Drive directory for immediate checkpoint synchronization.")
    parser.add_argument("--keep-last", type=int, default=None, help="Number of recent checkpoints to retain.")

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
    if args.checkpoint_dir is not None:
        train_config.checkpoint_dir = args.checkpoint_dir
    if args.drive_checkpoint_dir is not None:
        train_config.drive_checkpoint_dir = args.drive_checkpoint_dir
    if args.keep_last is not None:
        train_config.keep_last_checkpoints = args.keep_last

    if args.pilot:
        print("\n  [INFO] Pilot validation mode enabled (10 steps, verbose logging and checkpointing).")
        train_config.max_steps = 10
        train_config.save_interval = 5
        train_config.eval_interval = 5
        train_config.log_interval = 1

    print(model_config.print_parameter_summary())

    # 2. Resource & Safety Inspection
    sys_info = SystemInfo.inspect(target_dir=str(PROJECT_ROOT))
    res_est = estimate_resources(
        model_config,
        batch_size=train_config.batch_size,
        seq_len=model_config.max_seq_len,
    )

    if args.dry_run:
        print(res_est.summary())
        print("\n  [INFO] Dry-run complete. Architecture, parameter calculation, and resource constraints verified.")
        return 0

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
    import torch
    device = torch.device(args.device) if args.device else None
    model = AuraModel(model_config)
    trainer = AuraTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=train_config,
        device=device,
    )

    # 5. Checkpoint Resumption
    if args.resume_from:
        search_dirs = [Path(train_config.checkpoint_dir)]
        if train_config.drive_checkpoint_dir:
            search_dirs.append(Path(train_config.drive_checkpoint_dir))

        explicit_p = None if args.resume_from.lower() == "auto" else args.resume_from
        print(f"\n  [INFO] Locating resumption checkpoint (Target: {args.resume_from})...")
        candidate = find_best_checkpoint(
            search_dirs=search_dirs,
            explicit_path=explicit_p,
            expected_vocab_size=model_config.vocab_size,
            expected_d_model=model_config.d_model,
        )

        if candidate is None or not candidate.is_valid:
            err = candidate.error_message if candidate else "No valid checkpoint found in search directories."
            print(f"\n  [RESUME HALT] Cannot resume training:\n  {err}", file=sys.stderr)
            print("  Please check that the checkpoint exists, or run with --pilot to train from step 1.", file=sys.stderr)
            return 1

        print(f"  [✓] Verified Checkpoint Selected:")
        print(f"      - Path:           {candidate.path}")
        print(f"      - Step:           {candidate.step}")
        print(f"      - Size:           {candidate.size_bytes / (1024**2):.2f} MB")
        if candidate.metadata and "val_loss" in candidate.metadata and candidate.metadata["val_loss"] is not None:
            print(f"      - Recorded Loss:  {candidate.metadata['val_loss']:.4f}")

        resumed_step = trainer.load_checkpoint(candidate.path)
        print(f"  Resumed from global step: {resumed_step}")
        if args.additional_steps is not None:
            train_config.max_steps = resumed_step + args.additional_steps
            print(f"  [INFO] Training for {args.additional_steps} additional steps (New target step: {train_config.max_steps}).")
        elif train_config.max_steps <= resumed_step:
            train_config.max_steps = resumed_step + 10
            print(f"  [INFO] Target max_steps was <= resumed step. Auto-extended target to step {train_config.max_steps} (+10 steps).")
    trainer.train()
    return 0


if __name__ == "__main__":
    sys.exit(main())
