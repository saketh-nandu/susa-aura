"""
SUSA AURA — Memory-Mapped Binary Dataset Pipeline.

Implements zero-RAM chunked streaming tokenization and memory-mapped binary dataset
shards for high-throughput training on commodity laptops and cloud GPUs without memory exhaustion.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset

from aura.tokenizer.bpe_tokenizer import AuraTokenizer, SusaTokenizer


class MemoryMappedPretrainingDataset(Dataset):
    """Memory-mapped binary dataset for autoregressive next-token prediction."""

    def __init__(self, bin_path: Union[str, Path], seq_len: int = 128) -> None:
        self.bin_path = Path(bin_path)
        self.seq_len = seq_len

        meta_path = self.bin_path.with_suffix(".json")
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing metadata file for dataset: {meta_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        raw_dtype = str(self.metadata.get("dtype", "uint16"))
        if "uint16" in raw_dtype:
            self.dtype = np.uint16
        elif "uint32" in raw_dtype:
            self.dtype = np.uint32
        elif "int64" in raw_dtype:
            self.dtype = np.int64
        else:
            self.dtype = np.dtype(raw_dtype)

        self.total_tokens = self.metadata["total_tokens"]

        # Memory map the binary file (read-only)
        self.data = np.memmap(self.bin_path, dtype=self.dtype, mode="r", shape=(self.total_tokens,))

        # Number of non-overlapping sequence windows
        # Each sample requires seq_len tokens for input and next-token prediction
        self.num_samples = max(0, self.total_tokens // self.seq_len)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = idx * self.seq_len
        end = start + self.seq_len

        # Read window from memory map
        chunk = self.data[start:end].astype(np.int64)
        tokens = torch.from_numpy(chunk)

        # In AuraModel, shift is handled internally when both input_ids and labels are passed
        return {
            "input_ids": tokens,
            "labels": tokens.clone(),
        }


def build_binary_dataset(
    texts: Iterable[str],
    tokenizer: SusaTokenizer,
    output_bin: Union[str, Path],
    max_tokens: Optional[int] = None,
    dtype: np.dtype = np.uint16,
    chunk_flush_size: int = 50_000,
) -> Dict[str, Any]:
    """Tokenizes text stream and writes directly to disk in chunks to minimize RAM usage.

    Args:
        texts: Iterator yielding raw text/code documents
        tokenizer: Trained SusaTokenizer instance
        output_bin: Output binary path (e.g. data/train.bin)
        max_tokens: Optional cap on total tokens written
        dtype: Data type (np.uint16 supports vocabs up to 65,535; np.uint32 for larger)
        chunk_flush_size: Number of tokens buffered in RAM before flushing to disk
    """
    out_path = Path(output_bin)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_tokens = 0
    buffer: list[int] = []

    with open(out_path, "wb") as f_out:
        for doc in texts:
            if max_tokens is not None and total_tokens >= max_tokens:
                break

            # Encode document with EOS token suffix
            doc_tokens = tokenizer.encode(doc, add_eos=True)
            buffer.extend(doc_tokens)

            # Flush to disk in chunks
            if len(buffer) >= chunk_flush_size:
                to_write = buffer[:chunk_flush_size]
                buffer = buffer[chunk_flush_size:]
                arr = np.array(to_write, dtype=dtype)
                arr.tofile(f_out)
                total_tokens += len(to_write)

        # Flush remaining tokens
        if buffer:
            arr = np.array(buffer, dtype=dtype)
            arr.tofile(f_out)
            total_tokens += len(buffer)

    # Save metadata JSON file next to binary file
    meta_path = out_path.with_suffix(".json")
    metadata = {
        "bin_file": out_path.name,
        "total_tokens": total_tokens,
        "dtype": np.dtype(dtype).name,
        "vocab_size": tokenizer.vocab_size,
        "chunk_flush_size": chunk_flush_size,
    }
    with open(meta_path, "w", encoding="utf-8") as f_meta:
        json.dump(metadata, f_meta, indent=2)

    return metadata
