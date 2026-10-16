"""
SUSA-1B CodeMind — Pre-Training & Optimization Engine.

Implements real autoregressive next-token prediction training with mixed precision,
AdamW with weight decay separation, cosine learning rate schedule with linear warmup,
gradient accumulation, gradient clipping, tokens/sec throughput tracking, memory tracking,
periodic checkpointing, and resume-from-checkpoint support.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import psutil
import torch
from torch.utils.data import DataLoader, Dataset

from susacodemind.model.transformer import SusaCodeMind


@dataclass
class TrainingConfig:
    """Configuration hyperparameters for pretraining runs."""

    batch_size: int = 2
    grad_accum_steps: int = 1
    lr: float = 1e-3
    min_lr: float = 1e-4
    warmup_steps: int = 5
    max_steps: int = 20
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    mixed_precision: str = "no"  # "no", "fp16", or "bf16"
    seed: int = 42
    eval_interval: int = 5
    eval_steps: int = 2
    save_interval: int = 10
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 1
    gradient_checkpointing: bool = False
    use_8bit_adam: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingConfig:
        train_data = data.get("training", data)
        valid_keys = {
            "batch_size",
            "grad_accum_steps",
            "lr",
            "min_lr",
            "warmup_steps",
            "max_steps",
            "weight_decay",
            "grad_clip",
            "mixed_precision",
            "seed",
            "eval_interval",
            "eval_steps",
            "save_interval",
            "checkpoint_dir",
            "log_interval",
            "gradient_checkpointing",
            "use_8bit_adam",
        }
        filtered = {k: v for k, v in train_data.items() if k in valid_keys}
        return cls(**filtered)


class Trainer:
    """High-performance PyTorch training loop for SUSA-1B CodeMind."""

    def __init__(
        self,
        model: SusaCodeMind,
        train_dataset: Dataset,
        val_dataset: Dataset | None = None,
        config: TrainingConfig | None = None,
        device: torch.device | None = None,
        collate_fn: Any | None = None,
    ) -> None:
        self.config = config or TrainingConfig()
        torch.manual_seed(self.config.seed)

        # Device selection: CUDA -> CPU
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)
        if getattr(self.config, "gradient_checkpointing", False):
            self.model.gradient_checkpointing = True

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # Dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
        )

        if self.val_dataset is not None and len(self.val_dataset) > 0:
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                drop_last=False,
                collate_fn=collate_fn,
            )
        else:
            self.val_loader = None

        # Optimizer setup with weight decay separation
        self.optimizer = self._configure_optimizers()

        # Mixed precision and scaler
        self.use_amp = self.config.mixed_precision in ["fp16", "bf16"] and self.device.type == "cuda"
        self.amp_dtype = (
            torch.bfloat16
            if self.config.mixed_precision == "bf16"
            else torch.float16
        )
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(self.config.mixed_precision == "fp16" and self.device.type == "cuda"),
        )

        self.global_step = 0
        self.history: list[dict[str, Any]] = []

    def _configure_optimizers(self) -> torch.optim.Optimizer:
        """Configures AdamW with weight decay separation.

        Decays 2D weight matrices (projections, embeddings).
        Does NOT decay 1D parameters (RMSNorm scales, biases).
        """
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() >= 2:
                decay_params.append(param)
            else:
                no_decay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        use_8bit = False
        if getattr(self.config, "use_8bit_adam", False) or (self.model.count_parameters() > 300_000_000 and self.device.type == "cuda"):
            try:
                import bitsandbytes as bnb
                if self.device.type == "cuda":
                    use_8bit = True
            except ImportError:
                use_8bit = False

        if use_8bit:
            import bitsandbytes as bnb
            print("  [Memory Engine] Utilizing bitsandbytes 8-bit AdamW (reduces optimizer memory by 75% for 1B scaling).")
            optimizer = bnb.optim.AdamW8bit(
                optim_groups,
                lr=self.config.lr,
                betas=(0.9, 0.95),
                eps=1e-8,
            )
        else:
            # Standard modern AdamW: beta1=0.9, beta2=0.95, eps=1e-8
            optimizer = torch.optim.AdamW(
                optim_groups,
                lr=self.config.lr,
                betas=(0.9, 0.95),
                eps=1e-8,
            )
        return optimizer

    def get_lr(self, step: int) -> float:
        """Cosine learning rate schedule with linear warmup."""
        # 1. Linear warmup
        if step < self.config.warmup_steps:
            return self.config.lr * (step + 1) / (self.config.warmup_steps + 1e-8)

        # 2. If beyond max steps, return min_lr
        if step > self.config.max_steps:
            return self.config.min_lr

        # 3. Cosine decay
        decay_ratio = (step - self.config.warmup_steps) / max(
            1, (self.config.max_steps - self.config.warmup_steps)
        )
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.config.min_lr + coeff * (self.config.lr - self.config.min_lr)

    def _update_lr(self, lr: float) -> None:
        """Sets learning rate for all optimizer parameter groups."""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Runs evaluation loop over validation dataset."""
        if self.val_loader is None:
            return {"val_loss": 0.0, "perplexity": 0.0}

        self.model.eval()
        total_loss = 0.0
        steps = 0

        for batch in self.val_loader:
            if steps >= self.config.eval_steps:
                break

            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            if self.use_amp:
                with torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                    out = self.model(input_ids, labels=labels)
            else:
                out = self.model(input_ids, labels=labels)

            if out.loss is not None:
                total_loss += out.loss.item()
                steps += 1

        self.model.train()
        avg_loss = total_loss / max(1, steps)
        perplexity = math.exp(min(avg_loss, 20.0))  # Cap to prevent overflow
        return {"val_loss": avg_loss, "perplexity": perplexity}

    def save_checkpoint(self, filepath: str | Path, val_loss: float | None = None) -> Path:
        """Saves complete training checkpoint."""
        save_path = Path(filepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if hasattr(self.scaler, "state_dict") else None,
            "model_config": self.model.config.to_dict(),
            "training_config": asdict(self.config),
            "val_loss": val_loss,
        }
        torch.save(state, save_path)
        return save_path

    def load_checkpoint(self, filepath: str | Path) -> int:
        """Loads model weights, optimizer, and training step to resume training."""
        ckpt_path = Path(filepath)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"] is not None:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.global_step = checkpoint.get("step", 0)
        return self.global_step

    def train(self) -> list[dict[str, Any]]:
        """Executes main training loop."""
        print("\n" + "=" * 70)
        print("  SUSA-1B CODEMIND — PRETRAINING ENGINE STARTED")
        print("=" * 70)
        print(f"  Device:            {self.device} (AMP: {self.config.mixed_precision})")
        print(f"  Batch size:        {self.config.batch_size} (Grad Accum: {self.config.grad_accum_steps})")
        print(f"  Target steps:      {self.config.max_steps}")
        print(f"  Learning rate:     {self.config.lr} -> {self.config.min_lr}")
        print("=" * 70)

        self.model.train()
        train_iter = iter(self.train_loader)
        accum_loss = 0.0
        start_time = time.time()
        tokens_processed = 0

        ckpt_dir = Path(self.config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        try:
            while self.global_step < self.config.max_steps:
                # 1. Update learning rate
                lr = self.get_lr(self.global_step)
                self._update_lr(lr)

                self.optimizer.zero_grad()
                step_loss = 0.0

                # 2. Gradient Accumulation Loop
                for _ in range(self.config.grad_accum_steps):
                    try:
                        batch = next(train_iter)
                    except StopIteration:
                        train_iter = iter(self.train_loader)
                        batch = next(train_iter)

                    input_ids = batch["input_ids"].to(self.device)
                    labels = batch["labels"].to(self.device)
                    tokens_in_batch = input_ids.numel()

                    if self.use_amp:
                        with torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                            out = self.model(input_ids, labels=labels)
                            loss = out.loss / self.config.grad_accum_steps
                        self.scaler.scale(loss).backward()
                    else:
                        out = self.model(input_ids, labels=labels)
                        loss = out.loss / self.config.grad_accum_steps
                        loss.backward()

                    step_loss += loss.item()
                    tokens_processed += tokens_in_batch

                # 3. Clip gradients and optimizer step
                if self.use_amp and self.config.mixed_precision == "fp16":
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                    self.optimizer.step()

                self.global_step += 1
                accum_loss += step_loss

                # 4. Periodic Logging
                if self.global_step % self.config.log_interval == 0:
                    elapsed = time.time() - start_time
                    tok_per_sec = tokens_processed / max(1e-5, elapsed)

                    # Memory tracking
                    if self.device.type == "cuda":
                        mem_info = f"VRAM: {torch.cuda.memory_allocated() / (1024**2):.1f}MB"
                    else:
                        mem_info = f"RAM: {psutil.Process(os.getpid()).memory_info().rss / (1024**2):.1f}MB"

                    avg_train_loss = accum_loss / self.config.log_interval
                    accum_loss = 0.0

                    log_entry = {
                        "step": self.global_step,
                        "train_loss": avg_train_loss,
                        "lr": lr,
                        "tok_per_sec": tok_per_sec,
                    }

                    # Periodic validation
                    if self.global_step % self.config.eval_interval == 0:
                        eval_metrics = self.evaluate()
                        log_entry.update(eval_metrics)
                        val_str = f"| Val Loss: {eval_metrics['val_loss']:.4f} | PPL: {eval_metrics['perplexity']:.2f}"
                    else:
                        val_str = ""

                    print(
                        f"  Step {self.global_step:>5}/{self.config.max_steps} "
                        f"| Loss: {avg_train_loss:.4f} "
                        f"| LR: {lr:.6f} "
                        f"| {tok_per_sec:.1f} tok/s "
                        f"| {mem_info} "
                        f"{val_str}"
                    )

                    self.history.append(log_entry)
                    # Reset window timer
                    start_time = time.time()
                    tokens_processed = 0

                # 5. Periodic Checkpoint Saving
                if self.global_step % self.config.save_interval == 0 or self.global_step == self.config.max_steps:
                    val_loss_to_save = self.history[-1].get("val_loss") if self.history else None
                    ckpt_file = ckpt_dir / f"checkpoint_step_{self.global_step}.pt"
                    self.save_checkpoint(ckpt_file, val_loss=val_loss_to_save)
                    print(f"  [CKPT] Saved checkpoint to: {ckpt_file}")

        except KeyboardInterrupt:
            print("\n  [!] Training interrupted by user. Saving emergency recovery checkpoint...")
            emergency_path = ckpt_dir / f"checkpoint_step_{self.global_step}_interrupted.pt"
            self.save_checkpoint(emergency_path)
            print(f"  [CKPT] Emergency recovery checkpoint saved to: {emergency_path}")

        print("\n" + "=" * 70)
        print("  STATUS: PRETRAINING RUN COMPLETE!")
        print("=" * 70 + "\n")
        return self.history
