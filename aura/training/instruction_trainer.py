"""
SUSA AURA — Instruction Tuning Subsystem.

Implements supervised fine-tuning (SFT) for coding and reasoning instructions with loss masking
applied to prompt tokens (so gradient updates only optimize assistant responses).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from aura.tokenizer.bpe_tokenizer import AuraTokenizer, SusaTokenizer


@dataclass
class InstructionSample:
    """An instruction prompt and desired expert response."""

    instruction: str
    response: str
    system_prompt: str = "You are SUSA AURA, an Adaptive Universal Reasoning Assistant."


class InstructionDataset(Dataset):
    """Dataset with prompt loss masking for Supervised Fine-Tuning (SFT)."""

    def __init__(
        self,
        samples: list[InstructionSample],
        tokenizer: AuraTokenizer | SusaTokenizer,
        max_seq_len: int = 512,
    ) -> None:
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.encoded_data: list[dict[str, torch.Tensor]] = []
        self._prepare_dataset()

    def _format_prompt(self, sample: InstructionSample) -> tuple[str, str]:
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

            # Pad up to max_seq_len so all batch samples have identical tensor shape
            pad_len = self.max_seq_len - len(full_ids)
            if pad_len > 0:
                full_ids = full_ids + [self.tokenizer.pad_id] * pad_len
                labels = labels + [-100] * pad_len

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

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.encoded_data[idx]


def collate_instruction_batch(
    batch: list[dict[str, torch.Tensor]],
    pad_id: int = 0,
) -> dict[str, torch.Tensor]:
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

    stacked_inputs = torch.stack(batch_input_ids)
    stacked_labels = torch.stack(batch_labels)
    attention_mask = (stacked_inputs != pad_id).long()

    return {
        "input_ids": stacked_inputs,
        "labels": stacked_labels,
        "attention_mask": attention_mask,
    }


def load_instruction_jsonl(filepath: str | Path) -> list[InstructionSample]:
    """Loads instruction samples from a JSONL file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Instruction dataset not found: {path}")

    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            inst = item.get("instruction") or item.get("prompt") or ""
            resp = item.get("response") or item.get("completion") or item.get("output") or ""
            sys_prompt = item.get("system_prompt", "You are SUSA AURA, an Adaptive Universal Reasoning Assistant.")
            if inst and resp:
                samples.append(
                    InstructionSample(
                        instruction=inst,
                        response=resp,
                        system_prompt=sys_prompt,
                    )
                )
    return samples


def generate_sample_instructions(count: int = 30) -> list[InstructionSample]:
    """Generates synthetic coding instruction samples for testing fine-tuning pipelines."""
    tasks = [
        ("Write a Python function to check if a string is a palindrome.",
         "def is_palindrome(s: str) -> bool:\n    clean = ''.join(c.lower() for c in s if c.isalnum())\n    return clean == clean[::-1]"),
        ("Implement a binary search function in Python.",
         "def binary_search(arr: list[int], target: int) -> int:\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid + 1\n        else:\n            high = mid - 1\n    return -1"),
        ("Write a function to compute the factorial of a number using recursion.",
         "def factorial(n: int) -> int:\n    if n < 0:\n        raise ValueError('Negative numbers have no factorial')\n    return 1 if n <= 1 else n * factorial(n - 1)"),
        ("How do I safely parse a JSON string in Python?",
         "import json\n\ndef safe_parse_json(raw_text: str) -> dict:\n    try:\n        return json.loads(raw_text)\n    except json.JSONDecodeError as err:\n        print(f'JSON decode failure: {err}')\n        return {}"),
        ("Write a function to remove duplicates from a list while preserving order.",
         "def remove_duplicates(items: list) -> list:\n    seen = set()\n    result = []\n    for item in items:\n        if item not in seen:\n            seen.add(item)\n            result.append(item)\n    return result"),
    ]

    samples = []
    for i in range(count):
        inst_template, resp_template = tasks[i % len(tasks)]
        samples.append(
            InstructionSample(
                instruction=f"{inst_template} (Task variant {i})",
                response=f"{resp_template}\n# Tested for variant {i}",
            )
        )
    return samples
