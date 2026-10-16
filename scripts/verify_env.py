#!/usr/bin/env python3
"""
SUSA-1B CodeMind — Environment & Configuration Verification Script.

Inspects hardware, validates all configuration profiles, checks mathematical
parameter counts, and prints a comprehensive health report.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from susacodemind.model.config import ModelConfig
from susacodemind.utils.resource_estimator import estimate_resources
from susacodemind.utils.system_checker import SystemInfo


def main() -> int:
    print("\n" + "=" * 70)
    print("  SUSA-1B CODEMIND — ENVIRONMENT & CONFIGURATION VERIFICATION")
    print("=" * 70)

    # 1. System Diagnostics
    sys_info = SystemInfo.inspect(target_dir=str(PROJECT_ROOT))
    print(sys_info.summary())

    # 2. Config Profiles Verification
    config_dir = PROJECT_ROOT / "configs"
    profiles = ["smoke.yaml", "small.yaml", "medium.yaml", "target_1b.yaml"]

    print("\n" + "=" * 70)
    print("  ARCHITECTURAL PROFILES & PARAMETER VERIFICATION")
    print("=" * 70)

    for p in profiles:
        cfg_path = config_dir / p
        if not cfg_path.exists():
            print(f"  [ERROR] Missing configuration file: {cfg_path}")
            return 1

        try:
            cfg = ModelConfig.from_yaml(cfg_path)
            stats = cfg.calculate_exact_parameters()
            est = estimate_resources(cfg, batch_size=2, seq_len=cfg.max_seq_len)

            print(f"\n  Profile: {cfg.profile_name.upper()} ({p})")
            print(f"    - Layers: {cfg.n_layers}, Heads: {cfg.n_heads}, KV Heads: {cfg.n_kv_heads}, Dim: {cfg.d_model}, FFN: {cfg.d_ff}")
            print(f"    - Exact Parameter Count: {stats['formatted_total']} ({stats['millions']}M / {stats['billions']}B)")
            print(f"    - Model Weights (FP16):  {est.param_memory_fp16_gb:.2f} GiB")
            print(f"    - Est. Total Training VRAM: {est.estimated_total_training_ram_gb:.2f} GiB")
            print(f"    - Safe on Laptop CPU:    {est.feasible_on_local_laptop}")
            print(f"    - Safe on Colab T4 (15G): {est.feasible_on_colab_free_t4}")

        except Exception as e:
            print(f"  [FAIL] Failed parsing {p}: {e}")
            return 1

    print("\n" + "=" * 70)
    print("  STATUS: ALL SYSTEM & ARCHITECTURAL CHECKS PASSED SUCCESSFULLY.")
    print("=" * 70 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
