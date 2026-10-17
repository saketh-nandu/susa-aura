#!/usr/bin/env python3
"""
SUSA AURA — Supervised Instruction Fine-Tuning CLI.

Fine-tunes a pretrained base checkpoint on coding and reasoning instructions with prompt loss masking.

Usage:
    python scripts/train_instruct.py --base-checkpoint checkpoints/checkpoint_step_8.pt --tokenizer-dir tokenizer_cache/smoke --use-synthetic --max-steps 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from aura.model.config import ModelConfig
from aura.model.transformer import AuraModel, SusaCodeMind
from aura.tokenizer.bpe_tokenizer import AuraTokenizer, SusaTokenizer
from aura.training.instruction_trainer import (
    InstructionDataset,
    generate_sample_instructions,
    load_instruction_jsonl,
)
from aura.training.trainer import AuraTrainer, Trainer, TrainingConfig


def main() -> int:
    parser = argparse.ArgumentParser(description="Instruction-tune SUSA AURA on instruction-following and coding tasks.")
    parser.add_argument("--base-checkpoint", type=str, required=True, help="Path to base pretrained checkpoint (.pt).")
    parser.add_argument("--tokenizer-dir", type=str, required=True, help="Path to tokenizer directory.")
    parser.add_argument("--data-file", type=str, default=None, help="Path to JSONL file containing instruction pairs.")
    parser.add_argument("--use-synthetic", action="store_true", help="Use synthetic coding instructions for testing.")
    parser.add_argument("--output-dir", type=str, default="checkpoints/instruct", help="Directory for fine-tuned checkpoints.")
    parser.add_argument("--max-steps", type=int, default=15, help="Maximum fine-tuning steps.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.0001, help="Fine-tuning learning rate (typically lower than pretraining).")

    args = parser.parse_args()

    print("=" * 70)
    print("  SUSA AURA — SUPERVISED INSTRUCTION FINE-TUNING")
    print("=" * 70)

    # 1. Load Tokenizer
    tok_path = Path(args.tokenizer_dir)
    tokenizer = AuraTokenizer.load(tok_path)
    print(f"  Loaded Tokenizer: {tok_path} (Vocab size: {tokenizer.vocab_size:,})")

    # 2. Load Instruction Samples
    if args.data_file and Path(args.data_file).exists():
        print(f"  Loading instructions from: {args.data_file}...")
        samples = load_instruction_jsonl(args.data_file)
    else:
        print("  Generating synthetic coding instructions for pipeline verification...")
        samples = generate_sample_instructions(count=50)

    print(f"  Total instruction pairs: {len(samples)}")

    # 3. Create Dataset with Prompt Loss Masking
    print("  Applying prompt loss masking (-100 on user prompt tokens)...")
    dataset = InstructionDataset(samples=samples, tokenizer=tokenizer, max_seq_len=256)

    # 4. Load Base Model Checkpoint
    ckpt_path = Path(args.base_checkpoint)
    print(f"  Loading base model checkpoint: {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model_cfg = ModelConfig.from_dict(checkpoint["model_config"])
    model = AuraModel(model_cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  Base weights successfully loaded ({model.count_parameters():,} parameters).")

    # 5. Configure Fine-Tuning Hyperparameters
    train_cfg = TrainingConfig(
        batch_size=args.batch_size,
        lr=args.lr,
        min_lr=args.lr * 0.1,
        warmup_steps=2,
        max_steps=args.max_steps,
        checkpoint_dir=args.output_dir,
        save_interval=args.max_steps,
        log_interval=1,
    )

    trainer = AuraTrainer(
        model=model,
        train_dataset=dataset,
        config=train_cfg,
    )

    # 6. Run Instruction Tuning
    trainer.train()

    # 7. Save Final Instruction-Tuned Checkpoint
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_ckpt = out_dir / "checkpoint_instruct_final.pt"
    trainer.save_checkpoint(final_ckpt)
    print(f"\n  [SUCCESS] Instruction-tuned model checkpoint saved to: {final_ckpt}")
    print("=" * 70 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
