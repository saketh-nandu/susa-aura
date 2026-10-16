#!/usr/bin/env python3
"""
SUSA-1B CodeMind — Dataset Ingestion, Sanitization & Binary Tokenization CLI.

Usage:
    python scripts/prepare_dataset.py --tokenizer-dir tokenizer_cache/smoke --use-synthetic
    python scripts/prepare_dataset.py --tokenizer-dir tokenizer_cache/target_32k --input-dir /path/to/code
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from susacodemind.data import (
    DataCleaner,
    Deduplicator,
    build_binary_dataset,
    generate_synthetic_samples,
)
from susacodemind.tokenizer import SusaTokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean, deduplicate, filter secrets, and prepare binary pretraining shards.")
    parser.add_argument("--tokenizer-dir", type=str, default="tokenizer_cache/smoke", help="Path to saved tokenizer.")
    parser.add_argument("--output-dir", type=str, default="data", help="Directory to save .bin and .json shards.")
    parser.add_argument("--input-dir", type=str, default=None, help="Directory containing source code files.")
    parser.add_argument("--use-synthetic", action="store_true", help="Generate synthetic samples for pipeline verification.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split proportion.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible train/val splits.")

    args = parser.parse_args()
    random.seed(args.seed)

    print("=" * 70)
    print("  SUSA-1B CODEMIND — DATASET INGESTION & SHARD PREPARATION")
    print("=" * 70)

    # 1. Load Tokenizer
    tok_path = Path(args.tokenizer_dir)
    if not tok_path.exists():
        print(f"  [ERROR] Tokenizer directory not found at: {tok_path}", file=sys.stderr)
        print("  Please run `python scripts/train_tokenizer.py` first.", file=sys.stderr)
        return 1

    tokenizer = SusaTokenizer.load(tok_path)
    print(f"  Loaded Tokenizer: {tok_path} (Vocab size: {tokenizer.vocab_size:,})")

    # 2. Gather raw documents
    cleaner = DataCleaner()
    dedup = Deduplicator()
    valid_documents: list[str] = []
    total_redacted_secrets = 0
    duplicate_count = 0
    rejected_count = 0

    if args.use_synthetic or args.input_dir is None:
        print("\n  [1/4] Ingesting synthetic code samples for pipeline test...")
        raw_samples = [s.content for s in generate_synthetic_samples(count=150)]
    else:
        in_path = Path(args.input_dir)
        print(f"\n  [1/4] Scanning files from: {in_path}...")
        binary_exts = {
            ".png", ".jpg", ".jpeg", ".gif", ".ico", ".webp", ".pdf",
            ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".bin", ".pt",
            ".pyc", ".pyd", ".so", ".dylib", ".dll", ".exe", ".whl",
            ".woff", ".woff2", ".ttf", ".eot", ".svg", ".idx", ".pack",
            ".rev", ".o", ".a",
        }
        raw_samples = []
        for p in in_path.glob("**/*"):
            if (
                p.is_file()
                and not any(part.startswith(".") for part in p.parts)
                and p.suffix.lower() not in binary_exts
            ):
                try:
                    raw_samples.append(p.read_text(encoding="utf-8", errors="ignore"))
                except Exception:
                    pass

    print(f"        Total initial files ingested: {len(raw_samples)}")

    # 3. Clean, scrub secrets, and deduplicate
    print("\n  [2/4] Sanitizing, scrubbing secrets, and deduplicating...")
    for idx, doc in enumerate(raw_samples):
        # Cleaning & Secret scrubbing
        res = cleaner.clean_document(doc)
        if not res.is_valid:
            rejected_count += 1
            continue

        total_redacted_secrets += res.secrets_redacted_count

        # Exact and Near Deduplication
        if dedup.is_exact_duplicate(res.text) or dedup.is_near_duplicate(f"doc_{idx}", res.text):
            duplicate_count += 1
            continue

        valid_documents.append(res.text)

    print(f"        - Clean valid documents kept: {len(valid_documents)}")
    print(f"        - Duplicates removed:         {duplicate_count}")
    print(f"        - Malformed/empty rejected:   {rejected_count}")
    print(f"        - Secrets detected & redacted:{total_redacted_secrets}")

    # 4. Split into train and validation sets
    print("\n  [3/4] Creating Train/Validation split...")
    random.shuffle(valid_documents)
    n_val = max(1, int(len(valid_documents) * args.val_ratio))
    val_docs = valid_documents[:n_val]
    train_docs = valid_documents[n_val:]
    print(f"        - Training documents:   {len(train_docs)}")
    print(f"        - Validation documents: {len(val_docs)}")

    # 5. Tokenize into memory-mapped binary shards
    print("\n  [4/4] Writing zero-RAM streaming binary shards...")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_bin = out_dir / "train.bin"
    val_bin = out_dir / "val.bin"

    train_meta = build_binary_dataset(train_docs, tokenizer, train_bin)
    val_meta = build_binary_dataset(val_docs, tokenizer, val_bin)

    print(f"        - Train shard: {train_bin} ({train_meta['total_tokens']:,} tokens)")
    print(f"        - Val shard:   {val_bin} ({val_meta['total_tokens']:,} tokens)")

    print("\n" + "=" * 70)
    print("  STATUS: DATASET PREPARATION COMPLETED SUCCESSFULLY!")
    print("=" * 70 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
