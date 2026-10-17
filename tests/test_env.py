"""
Unit tests for SUSA-1B CodeMind environment and hardware diagnostics.
"""

from __future__ import annotations

import sys

import torch

from aura.utils.system_checker import SystemInfo


def test_python_and_pytorch_versions():
    """Verify minimum required Python and PyTorch environments."""
    major, minor = sys.version_info[:2]
    assert (major, minor) >= (3, 10), f"Python version must be >= 3.10, got {major}.{minor}"

    torch_version = torch.__version__
    assert torch_version is not None
    assert len(torch_version) > 0


def test_system_info_inspection():
    """Verify system diagnostics gather valid hardware metrics."""
    info = SystemInfo.inspect()
    assert info.os_name in ["Linux", "Darwin", "Windows"]
    assert info.cpu_count_logical >= 1
    assert info.ram_total_gb > 0
    assert info.ram_available_gb > 0
    assert info.disk_total_gb > 0
    assert info.disk_available_gb > 0

    # Summary string formatting check
    summary_text = info.summary()
    assert "SUSA AURA" in summary_text
    assert "Operating System:" in summary_text
    assert "RAM Total" in summary_text


def test_tensor_cpu_operations():
    """Verify basic PyTorch CPU operations function correctly on Fedora without CUDA."""
    x = torch.ones((2, 4), dtype=torch.float32)
    y = torch.ones((4, 2), dtype=torch.float32) * 2.0
    z = torch.matmul(x, y)
    assert z.shape == (2, 2)
    assert torch.allclose(z, torch.full((2, 2), 8.0))
