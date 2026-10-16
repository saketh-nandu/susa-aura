"""
SUSA-1B CodeMind — Byte-Level Byte-Pair Encoding (BPE) Tokenizer.

Trains and runs a Byte-Level BPE tokenizer with full code coverage, zero out-of-vocabulary
failures on raw UTF-8, Fill-In-the-Middle (FIM) special tokens, and version tracking.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from tokenizers.processors import ByteLevel as ByteLevelProcessor

SPECIAL_TOKENS = [
    "<|pad|>",          # ID: 0 (Padding)
    "<|bos|>",          # ID: 1 (Beginning of Sequence)
    "<|eos|>",          # ID: 2 (End of Sequence)
    "<|unk|>",          # ID: 3 (Unknown / Fallback)
    "<|code_prefix|>",  # ID: 4 (FIM Prefix)
    "<|code_middle|>",  # ID: 5 (FIM Middle)
    "<|code_suffix|>",  # ID: 6 (FIM Suffix)
]


class SusaTokenizer:
    """Byte-Level BPE Tokenizer for SUSA-1B CodeMind."""

    def __init__(self, tokenizer: Tokenizer | None = None) -> None:
        self.tokenizer = tokenizer
        self.version = "1.0.0"

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def bos_id(self) -> int:
        return 1

    @property
    def eos_id(self) -> int:
        return 2

    @property
    def unk_id(self) -> int:
        return 3

    @property
    def vocab_size(self) -> int:
        if self.tokenizer is None:
            return len(SPECIAL_TOKENS)
        return self.tokenizer.get_vocab_size()

    @classmethod
    def create_untrained(cls) -> SusaTokenizer:
        """Constructs an empty Byte-Level BPE tokenizer scaffold."""
        bpe_model = models.BPE(unk_token="<|unk|>")
        tokenizer = Tokenizer(bpe_model)

        # Byte-level pre-tokenization allows raw UTF-8 bytes to be mapped seamlessly
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=True)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = ByteLevelProcessor(trim_offsets=False)
        return cls(tokenizer)

    def train_from_iterator(
        self,
        text_iterator: Iterable[str],
        vocab_size: int = 32000,
        min_frequency: int = 2,
    ) -> None:
        """Trains the BPE tokenizer from an iterator of source text/code strings."""
        instance = self.create_untrained()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )

        instance.tokenizer.train_from_iterator(text_iterator, trainer=trainer)
        self.tokenizer = instance.tokenizer

    def train_from_files(
        self,
        files: list[str],
        vocab_size: int = 32000,
        min_frequency: int = 2,
    ) -> None:
        """Trains the BPE tokenizer directly from a list of file paths."""
        instance = self.create_untrained()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )

        instance.tokenizer.train(files, trainer=trainer)
        self.tokenizer = instance.tokenizer

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """Encodes text to a list of token IDs, optionally wrapping with BOS / EOS."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer has not been trained or loaded yet.")

        encoding = self.tokenizer.encode(text)
        token_ids = list(encoding.ids)

        if add_bos:
            token_ids = [self.bos_id] + token_ids
        if add_eos:
            token_ids = token_ids + [self.eos_id]

        return token_ids

    def decode(
        self,
        token_ids: list[int] | torch.Tensor,
        skip_special_tokens: bool = True,
    ) -> str:
        """Decodes token IDs back into string."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer has not been trained or loaded yet.")

        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().tolist()
            # If 2D tensor [1, S], flatten to 1D
            if token_ids and isinstance(token_ids[0], list):
                token_ids = token_ids[0]

        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def save(self, output_dir: str | Path) -> None:
        """Saves tokenizer model and version metadata to directory."""
        if self.tokenizer is None:
            raise RuntimeError("Cannot save an uninitialized tokenizer.")

        save_path = Path(output_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        tokenizer_file = save_path / "tokenizer.json"
        self.tokenizer.save(str(tokenizer_file))

        config_data = {
            "version": self.version,
            "vocab_size": self.vocab_size,
            "special_tokens": SPECIAL_TOKENS,
            "pad_id": self.pad_id,
            "bos_id": self.bos_id,
            "eos_id": self.eos_id,
            "unk_id": self.unk_id,
        }
        config_file = save_path / "tokenizer_config.json"
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)

    @classmethod
    def load(cls, model_dir: str | Path) -> SusaTokenizer:
        """Loads a saved tokenizer from directory."""
        dir_path = Path(model_dir)
        tokenizer_file = dir_path / "tokenizer.json"
        if not tokenizer_file.exists():
            raise FileNotFoundError(f"Missing tokenizer file: {tokenizer_file}")

        tokenizer = Tokenizer.from_file(str(tokenizer_file))
        instance = cls(tokenizer)

        config_file = dir_path / "tokenizer_config.json"
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                instance.version = data.get("version", "1.0.0")

        return instance
