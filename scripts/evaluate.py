#!/usr/bin/env python3
"""
SUSA AURA — Model Evaluation & Benchmarking CLI.

Evaluates trained model checkpoints for:
1. Validation Loss and Perplexity on held-out test/validation binary shards.
2. Code Syntax Validity via Python AST analysis.
3. Functional correctness (Pass@1) in an isolated execution sandbox.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/instruct/checkpoint_instruct_final.pt --tokenizer-dir tokenizer_cache/smoke --val-bin data/val.bin
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from aura.data.dataset import MemoryMappedPretrainingDataset
from aura.evaluation import (
    CodeEvaluator,
    CodingBenchmarkTask,
    IsolatedCodeRunner,
)
from aura.inference.generator import AuraGenerator, CodeGenerator

BENCHMARK_TASKS = [
    CodingBenchmarkTask(
        task_id="add_numbers",
        prompt="def add(a: int, b: int) -> int:\n    \"\"\"Return the sum of a and b.\"\"\"\n",
        test_assertions="assert add(2, 3) == 5\nassert add(-1, 1) == 0\nassert add(0, 0) == 0",
        entry_point="add",
    ),
    CodingBenchmarkTask(
        task_id="is_even",
        prompt="def is_even(n: int) -> bool:\n    \"\"\"Return True if n is even, False otherwise.\"\"\"\n",
        test_assertions="assert is_even(4) is True\nassert is_even(7) is False\nassert is_even(0) is True",
        entry_point="is_even",
    ),
    CodingBenchmarkTask(
        task_id="reverse_string",
        prompt="def reverse_string(s: str) -> str:\n    \"\"\"Return the reversed string.\"\"\"\n",
        test_assertions="assert reverse_string('susa') == 'asus'\nassert reverse_string('') == ''",
        entry_point="reverse_string",
    ),
    CodingBenchmarkTask(
        task_id="find_maximum",
        prompt="def find_max(numbers: list[int]) -> int:\n    \"\"\"Return the maximum number in list.\"\"\"\n",
        test_assertions="assert find_max([1, 9, 3]) == 9\nassert find_max([-5, -1]) == -1",
        entry_point="find_max",
    ),
]


def evaluate_perplexity(generator: AuraGenerator | CodeGenerator, val_bin_path: Path) -> dict:
    """Calculates cross-entropy validation loss and perplexity on binary dataset shard."""
    if not val_bin_path.exists():
        return {}

    dataset = MemoryMappedPretrainingDataset(val_bin_path, seq_len=generator.model.config.max_seq_len)
    if len(dataset) == 0:
        return {}

    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
    generator.model.eval()

    total_loss = 0.0
    steps = 0
    with torch.no_grad():
        for batch in loader:
            inp = batch["input_ids"].to(generator.device)
            lbl = batch["labels"].to(generator.device)
            out = generator.model(inp, labels=lbl)
            if out.loss is not None:
                total_loss += out.loss.item()
                steps += 1
            if steps >= 20:  # Bound steps for speed
                break

    avg_loss = total_loss / max(1, steps)
    ppl = math.exp(min(avg_loss, 20.0))
    return {"val_loss": avg_loss, "perplexity": ppl}


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate SUSA AURA on validation shards and coding tasks.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pt).")
    parser.add_argument("--tokenizer-dir", type=str, required=True, help="Path to tokenizer directory.")
    parser.add_argument("--val-bin", type=str, default=None, help="Optional path to validation .bin shard.")
    parser.add_argument("--timeout", type=float, default=3.0, help="Sandbox execution timeout per task.")

    args = parser.parse_args()

    print("=" * 70)
    print("  SUSA AURA — EVALUATION & BENCHMARKING SUITE")
    print("=" * 70)

    generator = AuraGenerator.from_checkpoint(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer_dir,
    )
    print(f"  Loaded model: {generator.model.count_parameters():,} parameters on {generator.device}.")

    # 1. Perplexity Evaluation
    if args.val_bin:
        val_path = Path(args.val_bin)
        print(f"\n  [1/2] Computing validation perplexity on: {val_path}...")
        metrics = evaluate_perplexity(generator, val_path)
        if metrics:
            print(f"        - Validation Loss: {metrics['val_loss']:.4f}")
            print(f"        - Perplexity (PPL):{metrics['perplexity']:.2f}")

    # 2. Functional Coding Tasks Benchmark
    print(f"\n  [2/2] Evaluating on {len(BENCHMARK_TASKS)} standard coding benchmark tasks...")
    evaluator = CodeEvaluator(sandbox=IsolatedCodeRunner(default_timeout_sec=args.timeout))

    task_solutions = {}
    for t in BENCHMARK_TASKS:
        # Generate code solution
        gen_code = generator.generate(
            t.prompt,
            max_new_tokens=48,
            temperature=0.2,
            stop_strings=["\n\n\n", "# Task", "assert "],
        )
        task_solutions[t.task_id] = gen_code

    bench_results = evaluator.run_benchmark(task_solutions, BENCHMARK_TASKS)

    print("\n" + "=" * 70)
    print("  EVALUATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Total Coding Tasks:   {bench_results['total_tasks']}")
    print(f"  Syntax Valid Rate:    {bench_results['syntax_accuracy'] * 100:.1f}%")
    print(f"  Pass@1 Benchmark:     {bench_results['pass_at_1'] * 100:.1f}%")
    print("=" * 70 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
