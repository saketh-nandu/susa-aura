"""
Unit tests for SUSA-1B CodeMind Memory-Mapped Binary Dataset Pipeline.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import pytest
import torch
from torch.utils.data import DataLoader

from susacodemind.data.dataset import MemoryMappedPretrainingDataset, build_binary_dataset
from susacodemind.data.synthetic import generate_synthetic_samples
from susacodemind.tokenizer.bpe_tokenizer import SusaTokenizer


@pytest.fixture(scope="module")
def sample_dataset_path() -> tuple[Path, SusaTokenizer]:
    """Generates synthetic data, trains tokenizer, and writes binary dataset shard."""
    tmp_dir = tempfile.mkdtemp()
    shard_path = Path(tmp_dir) / "test_shard.bin"

    samples = generate_synthetic_samples(count=30)
    corpus = [s.content for s in samples]

    tokenizer = SusaTokenizer.create_untrained()
    tokenizer.train_from_iterator(corpus, vocab_size=512, min_frequency=2)

    build_binary_dataset(corpus, tokenizer, shard_path, chunk_flush_size=1000)
    return shard_path, tokenizer


def test_binary_dataset_creation(sample_dataset_path):
    """Verify binary shard and json metadata creation."""
    shard_path, _ = sample_dataset_path
    assert shard_path.exists()
    meta_path = shard_path.with_suffix(".json")
    assert meta_path.exists()


def test_memory_mapped_dataloader(sample_dataset_path):
    """Verify MemoryMappedPretrainingDataset loads without RAM bloat and feeds DataLoader."""
    shard_path, _ = sample_dataset_path
    seq_len = 64
    batch_size = 4

    ds = MemoryMappedPretrainingDataset(bin_path=shard_path, seq_len=seq_len)
    assert len(ds) > 0

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    batch = next(iter(loader))

    assert "input_ids" in batch
    assert "labels" in batch
    assert batch["input_ids"].shape == (batch_size, seq_len)
    assert batch["labels"].shape == (batch_size, seq_len)
    assert batch["input_ids"].dtype == torch.int64
