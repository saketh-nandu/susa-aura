#!/usr/bin/env python3
"""
SUSA-1B CodeMind — Interactive Code Generation CLI.

Usage:
    python scripts/generate.py --checkpoint checkpoints/checkpoint_step_8.pt --tokenizer-dir tokenizer_cache/smoke --prompt "def quicksort(arr):"
    python scripts/generate.py --checkpoint checkpoints/checkpoint_step_8.pt --tokenizer-dir tokenizer_cache/smoke --interactive
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from susacodemind.inference.generator import CodeGenerator


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate code with SUSA-1B CodeMind.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt).")
    parser.add_argument("--tokenizer-dir", type=str, required=True, help="Path to tokenizer directory.")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt string to complete.")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Maximum new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (0.0 for greedy).")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling threshold.")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k filtering threshold.")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive multi-turn REPL.")

    args = parser.parse_args()

    print("=" * 70)
    print("  SUSA-1B CODEMIND — INFERENCE ENGINE")
    print("=" * 70)
    print(f"  Loading checkpoint: {args.checkpoint}...")

    generator = CodeGenerator.from_checkpoint(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer_dir,
    )
    print(f"  Model loaded: {generator.model.count_parameters():,} parameters on {generator.device}.")

    if args.interactive:
        print("\n  Entering Interactive Coding REPL. Type 'exit' or 'quit' to quit.\n")
        while True:
            try:
                user_prompt = input("CodeMind> ")
                if user_prompt.strip().lower() in ["exit", "quit"]:
                    break
                if not user_prompt.strip():
                    continue

                output = generator.generate(
                    user_prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                )
                print("-" * 50)
                print(output)
                print("-" * 50 + "\n")
            except (KeyboardInterrupt, EOFError):
                break
    else:
        prompt = args.prompt or "def fibonacci(n: int) -> int:\n"
        print(f"\n  [PROMPT]:\n{prompt}\n")
        output = generator.generate(
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        print("=" * 70)
        print("  [GENERATED OUTPUT]:")
        print("=" * 70)
        print(output)
        print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
