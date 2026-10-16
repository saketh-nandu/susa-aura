#!/usr/bin/env python3
"""
SUSA-1B CodeMind — Parameter Counter & Resource Estimator CLI.

Usage:
    python scripts/count_params.py configs/smoke.yaml
    python scripts/count_params.py configs/target_1b.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from susacodemind.model.config import ModelConfig
from susacodemind.utils.resource_estimator import estimate_resources


def main() -> int:
    parser = argparse.ArgumentParser(description="Calculate exact parameter counts and memory estimates.")
    parser.add_argument("config_file", nargs="?", default=None, type=str, help="Path to the model YAML configuration.")
    parser.add_argument("--config", "-c", type=str, default=None, help="Path to the model YAML configuration.")
    parser.add_argument("--batch-size", type=int, default=2, help="Hypothetical batch size for activation memory.")
    parser.add_argument("--seq-len", type=int, default=1024, help="Sequence length for memory calculation.")

    args = parser.parse_args()
    config_target = args.config or args.config_file
    if not config_target:
        parser.print_help()
        return 1

    config_path = Path(config_target)
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}", file=sys.stderr)
        return 1

    cfg = ModelConfig.from_yaml(config_path)
    print(cfg.print_parameter_summary())

    est = estimate_resources(cfg, batch_size=args.batch_size, seq_len=args.seq_len)
    print(est.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
