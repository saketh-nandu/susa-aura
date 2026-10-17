"""
SUSA AURA — Dataset Ingestion & Preprocessing Subsystem.
"""

from aura.data.cleaner import DataCleaner, FilterResult
from aura.data.deduplication import Deduplicator, LeakageReport, check_train_val_leakage, compute_sha256
from aura.data.dataset import MemoryMappedPretrainingDataset, build_binary_dataset
from aura.data.synthetic import SyntheticSample, generate_synthetic_samples

__all__ = [
    "DataCleaner",
    "FilterResult",
    "Deduplicator",
    "LeakageReport",
    "check_train_val_leakage",
    "compute_sha256",
    "MemoryMappedPretrainingDataset",
    "build_binary_dataset",
    "SyntheticSample",
    "generate_synthetic_samples",
]
