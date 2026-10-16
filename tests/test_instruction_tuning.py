"""
Unit tests for SUSA-1B CodeMind Instruction Tuning and Prompt Loss Masking.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from susacodemind.tokenizer.bpe_tokenizer import SusaTokenizer
from susacodemind.training.instruction_trainer import (
    InstructionDataset,
    InstructionSample,
    collate_instruction_batch,
)

TOKENIZER_DIR = Path(__file__).resolve().parent.parent / "tokenizer_cache" / "smoke"


def get_or_create_smoke_tokenizer() -> SusaTokenizer:
    if (TOKENIZER_DIR / "tokenizer.json").exists():
        return SusaTokenizer.load(TOKENIZER_DIR)
    from susacodemind.data.synthetic import generate_synthetic_samples

    tok = SusaTokenizer.create_untrained()
    samples = generate_synthetic_samples(count=10)
    tok.train_from_iterator([s.content for s in samples], vocab_size=256, min_frequency=1)
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    tok.save(TOKENIZER_DIR)
    return tok


@pytest.fixture
def tokenizer() -> SusaTokenizer:
    return get_or_create_smoke_tokenizer()


def test_instruction_loss_masking(tokenizer):
    """Verify user prompt tokens are masked with -100 while response tokens have true target IDs."""
    samples = [
        InstructionSample(
            instruction="Write a function that returns true.",
            response="def is_true():\n    return True\n",
        )
    ]

    dataset = InstructionDataset(samples=samples, tokenizer=tokenizer, max_seq_len=256)
    item = dataset[0]

    input_ids = item["input_ids"]
    labels = item["labels"]

    assert input_ids.shape == labels.shape
    # Ensure -100 masking is present on prompt tokens
    assert -100 in labels.tolist()

    # Find where labels transition from -100 to actual response tokens
    masked_count = sum(1 for x in labels.tolist() if x == -100)
    assert masked_count > 0
    assert masked_count < len(labels)

    # Check that there are unmasked response tokens and they match input_ids
    unmasked_tokens = [
        (inp, lbl) for inp, lbl in zip(input_ids.tolist(), labels.tolist()) if lbl != -100
    ]
    assert len(unmasked_tokens) > 0
    for inp, lbl in unmasked_tokens:
        assert inp == lbl


def test_collate_instruction_batch(tokenizer):
    """Verify dynamic padding stacks batches with varying lengths."""
    samples = [
        InstructionSample(instruction="Short task", response="pass"),
        InstructionSample(instruction="A much longer instruction task for coding", response="def foo():\n    return 42\n"),
    ]

    dataset = InstructionDataset(samples=samples, tokenizer=tokenizer, max_seq_len=64)
    batch = collate_instruction_batch([dataset[0], dataset[1]], pad_id=tokenizer.pad_id)

    assert "input_ids" in batch
    assert "labels" in batch
    assert batch["input_ids"].dim() == 2
    assert batch["input_ids"].size(0) == 2
