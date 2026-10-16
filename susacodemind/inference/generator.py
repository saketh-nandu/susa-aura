"""
SUSA-1B CodeMind — Text Generation & Inference Subsystem.

Provides high-efficiency autoregressive text and code generation with KV-caching,
temperature, top-k, nucleus (top-p) sampling, and code-block stopping criteria.
"""

from __future__ import annotations

from pathlib import Path

import torch

from susacodemind.model.config import ModelConfig
from susacodemind.model.transformer import SusaCodeMind
from susacodemind.tokenizer.bpe_tokenizer import SusaTokenizer


class CodeGenerator:
    """High-level interface for code synthesis and completion with SusaCodeMind."""

    def __init__(
        self,
        model: SusaCodeMind,
        tokenizer: SusaTokenizer,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.tokenizer = tokenizer

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        tokenizer_path: str | Path,
        config_path: str | Path | None = None,
        device: torch.device | None = None,
    ) -> CodeGenerator:
        """Loads model and tokenizer directly from saved checkpoint files."""
        ckpt_file = Path(checkpoint_path)
        if not ckpt_file.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(ckpt_file, map_location=dev, weights_only=False)

        # Reconstruct ModelConfig from checkpoint or explicit file
        if "model_config" in checkpoint:
            cfg = ModelConfig.from_dict(checkpoint["model_config"])
        elif config_path is not None:
            cfg = ModelConfig.from_yaml(config_path)
        else:
            raise ValueError("No model_config found in checkpoint; please provide config_path.")

        model = SusaCodeMind(cfg)
        model.load_state_dict(checkpoint["model_state_dict"])

        tokenizer = SusaTokenizer.load(tokenizer_path)
        return cls(model=model, tokenizer=tokenizer, device=dev)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        add_bos: bool = True,
        stop_strings: list[str] | None = None,
    ) -> str:
        """Synthesizes text/code following the given prompt."""
        input_ids = self.tokenizer.encode(prompt, add_bos=add_bos)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        out_ids = self.model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=self.tokenizer.eos_id,
            use_cache=True,
        )

        decoded = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)

        # Handle custom stop strings if provided
        if stop_strings:
            # Strip prompt prefix from output if present
            generated_text = decoded.removeprefix(prompt)
            for stop in stop_strings:
                if stop in generated_text:
                    generated_text = generated_text[: generated_text.index(stop)]
            return prompt + generated_text

        return decoded
