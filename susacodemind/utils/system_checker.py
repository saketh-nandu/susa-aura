"""
SUSA-1B CodeMind — System Diagnostics & Hardware Verification.

Inspects local and cloud environments (Fedora, Google Colab, etc.) to ensure
safe memory boundaries and transparent resource reporting before execution.
"""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import psutil
import torch


@dataclass
class SystemInfo:
    """System hardware and runtime diagnostics."""

    os_name: str
    os_release: str
    machine_arch: str
    python_version: str
    pytorch_version: str
    cpu_count_logical: int
    cpu_count_physical: int
    ram_total_gb: float
    ram_available_gb: float
    ram_used_gb: float
    disk_total_gb: float
    disk_available_gb: float
    cuda_available: bool
    gpu_devices: List[Dict[str, Any]]

    @classmethod
    def inspect(cls, target_dir: str = ".") -> SystemInfo:
        """Collect current machine runtime and hardware metrics."""
        vm = psutil.virtual_memory()
        disk = psutil.disk_usage(os.path.abspath(target_dir))

        gpu_devices = []
        cuda_avail = torch.cuda.is_available()
        if cuda_avail:
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total_vram_gb = round(props.total_memory / (1024**3), 2)
                gpu_devices.append(
                    {
                        "index": i,
                        "name": props.name,
                        "total_vram_gb": total_vram_gb,
                        "major": props.major,
                        "minor": props.minor,
                    }
                )

        return cls(
            os_name=platform.system(),
            os_release=platform.release(),
            machine_arch=platform.machine(),
            python_version=sys.version.split()[0],
            pytorch_version=torch.__version__,
            cpu_count_logical=psutil.cpu_count(logical=True) or 1,
            cpu_count_physical=psutil.cpu_count(logical=False) or 1,
            ram_total_gb=round(vm.total / (1024**3), 2),
            ram_available_gb=round(vm.available / (1024**3), 2),
            ram_used_gb=round(vm.used / (1024**3), 2),
            disk_total_gb=round(disk.total / (1024**3), 2),
            disk_available_gb=round(disk.free / (1024**3), 2),
            cuda_available=cuda_avail,
            gpu_devices=gpu_devices,
        )

    def summary(self) -> str:
        """Produce a human-readable diagnostic report."""
        lines = [
            "=" * 64,
            "  SUSA-1B CodeMind — Hardware & Environment Diagnostics",
            "=" * 64,
            f"  Operating System:      {self.os_name} ({self.os_release}, {self.machine_arch})",
            f"  Python Version:        {self.python_version}",
            f"  PyTorch Version:       {self.pytorch_version}",
            f"  CPU Threads:           {self.cpu_count_logical} (Cores: {self.cpu_count_physical})",
            f"  RAM Total / Avail:     {self.ram_total_gb} GiB / {self.ram_available_gb} GiB",
            f"  Disk Total / Free:     {self.disk_total_gb} GiB / {self.disk_available_gb} GiB free",
            "-" * 64,
            f"  CUDA Available:        {self.cuda_available}",
        ]
        if self.cuda_available:
            for g in self.gpu_devices:
                lines.append(
                    f"  GPU #{g['index']}:              {g['name']} ({g['total_vram_gb']} GiB VRAM)"
                )
        else:
            lines.append("  Acceleration:          CPU only (Safe for testing, smoke tests, tokenizer training)")
            lines.append("  Laptop Safety Policy:  Full 1B training forbidden on local CPU.")

        lines.append("=" * 64)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
