"""
Unit tests for SUSA-1B CodeMind ModelConfig and resource calculations.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from susacodemind.model.config import ModelConfig
from susacodemind.utils.resource_estimator import estimate_resources

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def test_smoke_config_loading():
    """Verify smoke test profile loads and satisfies dimension constraints."""
    cfg = ModelConfig.from_yaml(CONFIG_DIR / "smoke.yaml")
    assert cfg.vocab_size == 1024
    assert cfg.d_model == 64
    assert cfg.n_heads == 4
    assert cfg.n_kv_heads == 2
    assert cfg.n_layers == 2
    assert cfg.head_dim == 16
    assert cfg.d_ff == 176

    stats = cfg.calculate_exact_parameters()
    assert stats["total_parameters"] > 0
    assert stats["total_parameters"] < 1_000_000  # < 1M parameters for smoke test


def test_small_config_loading():
    """Verify small experiment profile loads."""
    cfg = ModelConfig.from_yaml(CONFIG_DIR / "small.yaml")
    stats = cfg.calculate_exact_parameters()
    # Should be ~34M parameters
    assert 20_000_000 < stats["total_parameters"] < 50_000_000


def test_medium_config_loading():
    """Verify medium experiment profile loads."""
    cfg = ModelConfig.from_yaml(CONFIG_DIR / "medium.yaml")
    stats = cfg.calculate_exact_parameters()
    # Should be ~221M parameters
    assert 150_000_000 < stats["total_parameters"] < 350_000_000


def test_target_1b_config_loading():
    """Verify target 1B profile is approximately 1.0 Billion parameters."""
    cfg = ModelConfig.from_yaml(CONFIG_DIR / "target_1b.yaml")
    stats = cfg.calculate_exact_parameters()
    total_params = stats["total_parameters"]

    # Target: ~1.009B parameters (between 0.95B and 1.15B)
    assert 950_000_000 <= total_params <= 1_150_000_000, f"Expected ~1B parameters, got {total_params:,}"


def test_config_dimension_validation():
    """Verify invalid dimension combinations raise clear exceptions."""
    # d_model not divisible by n_heads
    with pytest.raises(ValueError, match="divisible by n_heads"):
        ModelConfig(d_model=65, n_heads=4)

    # n_heads not divisible by n_kv_heads
    with pytest.raises(ValueError, match="divisible by n_kv_heads"):
        ModelConfig(d_model=64, n_heads=4, n_kv_heads=3)

    # non-positive values
    with pytest.raises(ValueError, match="vocab_size must be positive"):
        ModelConfig(vocab_size=0)


def test_resource_estimation():
    """Verify resource estimator detects laptop safety boundaries."""
    smoke_cfg = ModelConfig.from_yaml(CONFIG_DIR / "smoke.yaml")
    smoke_est = estimate_resources(smoke_cfg, batch_size=2, seq_len=128)
    assert smoke_est.feasible_on_local_laptop is True

    target_cfg = ModelConfig.from_yaml(CONFIG_DIR / "target_1b.yaml")
    target_est = estimate_resources(target_cfg, batch_size=2, seq_len=2048)
    # Target 1B model must NOT be marked feasible for training on a 8GB local laptop
    assert target_est.feasible_on_local_laptop is False
    assert len(target_est.warnings) > 0
