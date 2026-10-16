"""
SUSA-1B CodeMind — Instruction Tuning Subsystem.

Implements supervised fine-tuning (SFT) for coding instructions with loss masking
applied to prompt tokens (so gradient updates only optimize assistant code responses).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import Dataset

from susacodemind.model.transformer import SusaCodeMind
from susacodemind.tokenizer.bpe_tokenizer import SusaTokenizer
from susacodemind.training.trainer import Trainer, TrainingConfig


@dataclass
class InstructionSample:
    """A coding instruction prompt and desired expert response."""

    instruction: str
    response: str
    system_prompt: str = "You are SUSA CodeMind, an expert software engineer and AI assistant."


class InstructionDataset(Dataset):
    """Dataset with prompt loss masking for Supervised Fine-Tuning (SFT)."""

    def __init__(
        self,
        samples: List[InstructionSample],
        tokenizer: SusaTokenizer,
        max_seq_len: int = 512,
    ) -> None:
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.encoded_data: List[Dict[str, torch.Tensor]] = []
        self._prepare_dataset()

    def _format_prompt(self, sample: InstructionSample) -> Tuple[str, str]:
        """Formats conversational template with explicit user and assistant tags."""
        prompt_text = (
            f"<|bos|>[SYSTEM]: {sample.system_prompt}\n"
            f"[USER]: {sample.instruction}\n"
            f"[ASSISTANT]: "
        )
        response_text = f"{sample.response}<|eos|>"
        return prompt_text, response_text

    def _prepare_dataset(self) -> None:
        """Encodes samples and masks prompt tokens with -100."""
        for s in self.samples:
            prompt_str, response_str = self._format_prompt(s)

            # Tokenize prompt and response separately to pinpoint prompt boundary
            prompt_ids = self.tokenizer.encode(prompt_str, add_bos=False)
            response_ids = self.tokenizer.encode(response_str, add_bos=False)

            full_ids = prompt_ids + response_ids
            # Truncate to max_seq_len
            full_ids = full_ids[: self.max_seq_len]

            # Build label tensor with -100 masking on prompt tokens
            labels = full_ids.copy()
            prompt_len = min(len(prompt_ids), len(full_ids))
            for i in range(prompt_len):
                labels[i] = -100  # Mask out user prompt from loss calculation

            input_tensor = torch.tensor(full_ids, dtype=torch.long)
            label_tensor = torch.tensor(labels, dtype=torch.long)

            self.encoded_data.append(
                {
                    "input_ids": input_tensor,
                    "labels": label_tensor,
                }
            )

    def __len__(self) -> int:
        return len(self.encoded_data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.encoded_data[idx]


def collate_instruction_batch(
    batch: List[Dict[str, torch.Tensor]],
    pad_id: int = 0,
) -> Dict[str, torch.Tensor]:
    """Dynamically pads variable-length instruction sequences in a batch."""
    max_len = max(item["input_ids"].size(0) for item in batch)

    batch_input_ids = []
    batch_labels = []

    for item in batch:
        inp = item["input_ids"]
        lbl = item["labels"]
        pad_len = max_len - inp.size(0)

        if pad_len > 0:
            padded_inp = torch.cat([inp, torch.full((pad_len,), pad_id, dtype=torch.long)])
            padded_lbl = torch.cat([lbl, torch.full((pad_len,), -100, dtype=torch.long)])
        else:
            padded_inp = inp
            padded_lbl = lbl

        batch_input_ids.append(padded_inp)
        batch_labels.append(padded_lbl)

    return {
        "input_ids": torch.stack(batch_input_ids),
        "labels": torch.stack(batch_labels),
    }
