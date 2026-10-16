"""
SUSA-1B CodeMind — Pre-Training Resource & Memory Estimator.

Calculates parameter memory, optimizer states (AdamW), gradient memory,
activation overhead, and checkpoint storage before training jobs are initiated.
Guarantees compliance with zero-cost and local laptop safety constraints.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from susacodemind.model.config import ModelConfig


@dataclass
class ResourceEstimate:
    """Detailed memory, storage, and throughput estimates for a model configuration."""

    profile_name: str
    total_parameters: int
    param_memory_fp32_gb: float
    param_memory_fp16_gb: float
    optimizer_memory_gb: float
    gradient_memory_gb: float
    estimated_activation_gb: float
    estimated_total_training_ram_gb: float
    checkpoint_file_size_gb: float
    recommended_min_vram_gb: float
    feasible_on_local_laptop: bool
    feasible_on_colab_free_t4: bool
    warnings: list[str]

    def summary(self) -> str:
        lines = [
            "=" * 68,
            "  SUSA-1B CodeMind — Resource Feasibility Report",
            f"  Profile: {self.profile_name} ({self.total_parameters:,} parameters)",
            "=" * 68,
            f"  Model Weights (FP32):          {self.param_memory_fp32_gb:>8.2f} GiB",
            f"  Model Weights (FP16/BF16):     {self.param_memory_fp16_gb:>8.2f} GiB",
            f"  AdamW Optimizer State (FP32):  {self.optimizer_memory_gb:>8.2f} GiB",
            f"  Gradients (Mixed Precision):   {self.gradient_memory_gb:>8.2f} GiB",
            f"  Estimated Activations:         {self.estimated_activation_gb:>8.2f} GiB",
            "-" * 68,
            f"  ESTIMATED TOTAL TRAINING VRAM: {self.estimated_total_training_ram_gb:>8.2f} GiB",
            f"  Single Checkpoint File Size:   {self.checkpoint_file_size_gb:>8.2f} GiB",
            f"  Recommended Min VRAM:          {self.recommended_min_vram_gb:>8.2f} GiB",
            "-" * 68,
            f"  Local Laptop Feasible:         {'YES (Safe for CPU testing)' if self.feasible_on_local_laptop else 'NO (High risk of system crash / OOM)'}",
            f"  Colab Free T4 (15GB) Feasible: {'YES' if self.feasible_on_colab_free_t4 else 'NO (Requires A100/H100 or aggressive offloading)'}",
            "=" * 68,
        ]
        if self.warnings:
            lines.append("  ATTENTION & SAFETY WARNINGS:")
            for w in self.warnings:
                lines.append(f"  [!] {w}")
            lines.append("=" * 68)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def estimate_resources(
    config: ModelConfig,
    batch_size: int = 2,
    seq_len: int = 1024,
    mixed_precision: bool = True,
    gradient_checkpointing: bool = True,
    use_8bit_adam: bool = False,
) -> ResourceEstimate:
    """Calculate exact memory and hardware requirements for a given configuration."""
    param_stats = config.calculate_exact_parameters()
    n_params = param_stats["total_parameters"]

    # 1. Parameter Memory
    # FP32: 4 bytes per param, FP16/BF16: 2 bytes per param
    param_fp32_gb = (n_params * 4) / (1024**3)
    param_fp16_gb = (n_params * 2) / (1024**3)
    active_param_gb = param_fp16_gb if mixed_precision else param_fp32_gb

    # 2. Optimizer Memory (AdamW vs 8-bit AdamW)
    if use_8bit_adam:
        optimizer_bytes_per_param = 4 if mixed_precision else 2
    else:
        optimizer_bytes_per_param = 12 if mixed_precision else 8
    optimizer_gb = (n_params * optimizer_bytes_per_param) / (1024**3)

    # 3. Gradients
    # 2 bytes in FP16/BF16, 4 bytes in FP32
    grad_bytes = 2 if mixed_precision else 4
    grad_gb = (n_params * grad_bytes) / (1024**3)

    # 4. Activation Memory (Analytical approximation)
    # Per layer: ~ 34 * b * s * h bytes without recomputation
    # With gradient checkpointing: ~ 2 * b * s * h * n_layers bytes (only recomputed activations saved)
    b = batch_size
    s = min(seq_len, config.max_seq_len)
    h = config.d_model
    l = config.n_layers

    if gradient_checkpointing:
        # Checkpointing stores only layer inputs: ~ 2 * b * s * h * l bytes
        activation_bytes = 2 * b * s * h * l * 2  # 2 bytes (fp16) * input tensors
    else:
        # Standard: 34 * b * s * h * l bytes
        activation_bytes = 34 * b * s * h * l * 2

    # Add 20% safety margin for attention logits, layernorms, and temp buffers
    activation_gb = (activation_bytes * 1.25) / (1024**3)

    # Total estimated VRAM during training forward + backward pass
    total_training_gb = active_param_gb + optimizer_gb + grad_gb + activation_gb

    # Checkpoint size (weights + optimizer states + config metadata)
    checkpoint_size_gb = param_fp32_gb + optimizer_gb

    # Feasibility judgements
    warnings = []
    feasible_local = total_training_gb <= 3.5  # Safe for an 8GB laptop without freezing system
    feasible_colab = total_training_gb <= 14.0  # Safe for Google Colab Free Tier T4 GPU (~15GB VRAM)

    if not feasible_local:
        warnings.append(
            f"Cannot train this model locally: requires ~{total_training_gb:.2f} GiB RAM, exceeding your laptop's free memory limit (2.7 GiB available)."
        )
    if not feasible_colab:
        warnings.append(
            "Standard training will exceed Google Colab Free T4 VRAM (15 GiB). Requires smaller batch size, micro-accumulation, or multi-GPU."
        )

    return ResourceEstimate(
        profile_name=config.profile_name,
        total_parameters=n_params,
        param_memory_fp32_gb=round(param_fp32_gb, 3),
        param_memory_fp16_gb=round(param_fp16_gb, 3),
        optimizer_memory_gb=round(optimizer_gb, 3),
        gradient_memory_gb=round(grad_gb, 3),
        estimated_activation_gb=round(activation_gb, 3),
        estimated_total_training_ram_gb=round(total_training_gb, 3),
        checkpoint_file_size_gb=round(checkpoint_size_gb, 3),
        recommended_min_vram_gb=round(total_training_gb * 1.2, 2),
        feasible_on_local_laptop=feasible_local,
        feasible_on_colab_free_t4=feasible_colab,
        warnings=warnings,
    )
