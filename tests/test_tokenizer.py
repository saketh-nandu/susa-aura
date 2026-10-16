"""
Unit tests for SUSA-1B CodeMind Byte-Level BPE Tokenizer.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from susacodemind.data.synthetic import generate_synthetic_samples
from susacodemind.tokenizer.bpe_tokenizer import SusaTokenizer


@pytest.fixture(scope="module")
def trained_tokenizer() -> SusaTokenizer:
    """Trains a compact Byte-Level BPE tokenizer on synthetic code."""
    samples = generate_synthetic_samples(count=50)
    corpus = [s.content for s in samples]

    tokenizer = SusaTokenizer.create_untrained()
    tokenizer.train_from_iterator(corpus, vocab_size=512, min_frequency=2)
    return tokenizer


def test_tokenizer_special_tokens(trained_tokenizer):
    """Verify standard special token assignments."""
    assert trained_tokenizer.pad_id == 0
    assert trained_tokenizer.bos_id == 1
    assert trained_tokenizer.eos_id == 2
    assert trained_tokenizer.unk_id == 3
    assert trained_tokenizer.vocab_size >= 256  # Initial byte alphabet + special tokens


def test_tokenizer_roundtrip_lossless(trained_tokenizer):
    """Verify encode -> decode produces an exact byte-for-byte match."""
    code_snippets = [
        "def hello_world():\n    print('Hello SUSA!')\n",
        "const add = (a, b) => a + b;\n",
        "SELECT id, name FROM users WHERE active = 1;\n",
        "// Unicode comments: 🚀 💻 π ≈ 3.14159\n",
    ]

    for snippet in code_snippets:
        encoded = trained_tokenizer.encode(snippet)
        assert len(encoded) > 0
        decoded = trained_tokenizer.decode(encoded, skip_special_tokens=True)
        assert snippet == decoded, f"Failed roundtrip: expected {snippet!r}, got {decoded!r}"


def test_tokenizer_bos_eos_wrapping(trained_tokenizer):
    """Verify add_bos and add_eos properly place token markers."""
    text = "x = 42"
    encoded = trained_tokenizer.encode(text, add_bos=True, add_eos=True)
    assert encoded[0] == trained_tokenizer.bos_id
    assert encoded[-1] == trained_tokenizer.eos_id


def test_tokenizer_save_and_load(trained_tokenizer):
    """Verify tokenizer serialization and deserialization."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "test_tok"
        trained_tokenizer.save(save_path)

        assert (save_path / "tokenizer.json").exists()
        assert (save_path / "tokenizer_config.json").exists()

        loaded_tok = SusaTokenizer.load(save_path)
        assert loaded_tok.vocab_size == trained_tokenizer.vocab_size

        test_text = "def calculate(): return 100"
        orig_enc = trained_tokenizer.encode(test_text)
        loaded_enc = loaded_tok.encode(test_text)
        assert orig_enc == loaded_enc


def test_tokenizer_train_from_files_resilient_utf8():
    """Verify train_from_files safely handles non-UTF-8 bytes and corrupted files without crashing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        # 1. Clean valid Python file
        f_clean = tmp_path / "clean.py"
        f_clean.write_text("def add(a: int, b: int) -> int:\n    return a + b\n", encoding="utf-8")

        # 2. File containing non-UTF-8 raw binary bytes (e.g. latin-1 / invalid bytes)
        f_invalid = tmp_path / "corrupted.py"
        f_invalid.write_bytes(b"# Test comment \xff\xfe\xfa\nclass Foo:\n    pass\n")

        tok = SusaTokenizer.create_untrained()
        # Must train without raising "stream did not contain valid UTF-8"
        tok.train_from_files([str(f_clean), str(f_invalid)], vocab_size=512, min_frequency=1)
        assert tok.vocab_size >= 256
        encoded = tok.encode("def add(x, y): return x + y")
        assert len(encoded) > 0
