"""
Unit tests for SUSA-1B CodeMind Inference & Code Generation.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from susacodemind.inference.generator import CodeGenerator
from susacodemind.model.config import ModelConfig
from susacodemind.model.transformer import SusaCodeMind
from susacodemind.tokenizer.bpe_tokenizer import SusaTokenizer

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
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
def smoke_generator() -> CodeGenerator:
    cfg = ModelConfig.from_yaml(CONFIG_DIR / "smoke.yaml")
    model = SusaCodeMind(cfg)
    tokenizer = get_or_create_smoke_tokenizer()
    return CodeGenerator(model=model, tokenizer=tokenizer, device=torch.device("cpu"))


def test_generator_output(smoke_generator):
    """Verify CodeGenerator synthesizes text tokens."""
    prompt = "def add(x, y):"
    res = smoke_generator.generate(
        prompt,
        max_new_tokens=10,
        temperature=0.8,
        top_k=20,
        top_p=0.9,
    )
    assert len(res) > len(prompt)
    assert prompt in res


def test_generator_stop_string(smoke_generator):
    """Verify stop_strings truncate output at requested marker."""
    prompt = "def func():\n    return 1\n"
    res = smoke_generator.generate(
        prompt,
        max_new_tokens=20,
        stop_strings=["\n"],
    )
    assert "\n" in res
