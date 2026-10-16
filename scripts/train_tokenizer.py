#!/usr/bin/env python3
"""
SUSA-1B CodeMind — Tokenizer Training CLI.

Usage:
    python scripts/train_tokenizer.py --use-synthetic --vocab-size 1024 --output-dir tokenizer_cache/smoke
    python scripts/train_tokenizer.py --input-dir /path/to/code --vocab-size 32000 --output-dir tokenizer_cache/target_32k
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from susacodemind.data.synthetic import generate_synthetic_samples
from susacodemind.tokenizer.bpe_tokenizer import SusaTokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description="Train SUSA-1B CodeMind BPE Tokenizer from scratch.")
    parser.add_argument("--vocab-size", type=int, default=1024, help="Vocabulary size (e.g. 1024 for smoke, 32000 for target).")
    parser.add_argument("--output-dir", type=str, default="tokenizer_cache/smoke", help="Directory to save trained tokenizer.")
    parser.add_argument("--input-dir", type=str, default=None, help="Directory containing raw training files.")
    parser.add_argument("--use-synthetic", action="store_true", help="Use synthetic code corpus (strictly for pipeline tests).")

    args = parser.parse_args()
    output_path = Path(args.output_dir)

    print("=" * 70)
    print("  SUSA-1B CODEMIND — TOKENIZER TRAINING")
    print("=" * 70)
    print(f"  Target Vocab Size: {args.vocab_size:,}")
    print(f"  Output Directory:  {output_path}")

    tokenizer = SusaTokenizer.create_untrained()

    if args.use_synthetic or args.input_dir is None:
        print("\n  [INFO] Generating clean synthetic corpus for pipeline validation...")
        samples = generate_synthetic_samples(count=100)
        corpus = [s.content for s in samples]
        print(f"  Generated {len(corpus)} synthetic documents across Python, JS, and Bash.")
        print("  Training Byte-Level BPE from memory stream...")
        tokenizer.train_from_iterator(corpus, vocab_size=args.vocab_size, min_frequency=2)
    else:
        in_path = Path(args.input_dir)
        if not in_path.exists():
            print(f"  [ERROR] Input path does not exist: {in_path}", file=sys.stderr)
            return 1
        files = [str(p) for p in in_path.glob("**/*") if p.is_file() and not p.name.startswith(".")]
        print(f"  Found {len(files)} files in {in_path}. Training BPE...")
        tokenizer.train_from_files(files, vocab_size=args.vocab_size, min_frequency=2)

    # Save tokenizer
    tokenizer.save(output_path)
    print("\n  [SUCCESS] Tokenizer trained successfully!")
    print(f"  - Vocab Size Achieved: {tokenizer.vocab_size:,}")
    print(f"  - Saved to:            {output_path.resolve()}")

    # Round-trip verification
    test_code = "def add(x: int, y: int) -> int:\n    return x + y\n"
    encoded = tokenizer.encode(test_code, add_bos=True, add_eos=True)
    decoded = tokenizer.decode(encoded, skip_special_tokens=True)
    print("\n  Round-Trip Fidelity Test:")
    print(f"  - Input String:        {test_code.strip()!r}")
    print(f"  - Token IDs ({len(encoded)}):      {encoded}")
    print(f"  - Decoded String:      {decoded.strip()!r}")
    assert test_code == decoded, "Decoded text does not match input exactly!"
    print("  - Fidelity:            EXACT MATCH (100% loss-free roundtrip)")
    print("=" * 70 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
