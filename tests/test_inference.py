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


@pytest.fixture
def smoke_generator() -> CodeGenerator:
    cfg = ModelConfig.from_yaml(CONFIG_DIR / "smoke.yaml")
    model = SusaCodeMind(cfg)
    tokenizer = SusaTokenizer.load(TOKENIZER_DIR)
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
