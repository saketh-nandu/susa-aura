"""
SUSA-1B CodeMind — Exact and Near-Deduplication Engine.

Removes identical files and near-duplicate source files using SHA-256 and
k-shingle MinHash approximations to prevent benchmark leakage and overfitting.
"""

from __future__ import annotations

import hashlib
import re
from typing import Dict, Iterable, List, Set, Tuple


def compute_sha256(text: str) -> str:
    """Computes SHA-256 hash of text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Deduplicator:
    """Combines exact hash filtering with shingle-based near-deduplication."""

    def __init__(self, k_shingle_size: int = 5, minhash_num_perm: int = 32, similarity_threshold: float = 0.85) -> None:
        self.k_shingle_size = k_shingle_size
        self.minhash_num_perm = minhash_num_perm
        self.similarity_threshold = similarity_threshold

        self.seen_exact_hashes: Set[str] = set()
        self.seen_signatures: List[Tuple[str, List[int]]] = []

    def is_exact_duplicate(self, text: str) -> bool:
        """Checks if text has been seen before via exact SHA-256."""
        h = compute_sha256(text)
        if h in self.seen_exact_hashes:
            return True
        self.seen_exact_hashes.add(h)
        return False

    def get_shingles(self, text: str) -> Set[str]:
        """Extracts whitespace-normalized character or token k-shingles."""
        tokens = re.findall(r"\w+|[^\w\s]", text)
        if len(tokens) < self.k_shingle_size:
            return set(tokens)
        return {
            " ".join(tokens[i : i + self.k_shingle_size])
            for i in range(len(tokens) - self.k_shingle_size + 1)
        }

    def compute_minhash_signature(self, shingles: Set[str]) -> List[int]:
        """Computes a compact MinHash signature across num_perm hash functions."""
        if not shingles:
            return [0] * self.minhash_num_perm

        signature = [float("inf")] * self.minhash_num_perm
        for shingle in shingles:
            shingle_bytes = shingle.encode("utf-8")
            # Generate deterministic hashes for each permutation
            for i in range(self.minhash_num_perm):
                # Hash combining permutation index and shingle
                h_val = int(hashlib.md5(f"{i}_{shingle}".encode("utf-8")).hexdigest()[:8], 16)
                if h_val < signature[i]:
                    signature[i] = h_val

        return [int(s) if s != float("inf") else 0 for s in signature]

    def estimate_jaccard_similarity(self, sig1: List[int], sig2: List[int]) -> float:
        """Estimates Jaccard similarity as fraction of matching signature hashes."""
        if len(sig1) != len(sig2) or len(sig1) == 0:
            return 0.0
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / len(sig1)

    def is_near_duplicate(self, doc_id: str, text: str) -> bool:
        """Checks if text is a near-duplicate of any previously seen document."""
        shingles = self.get_shingles(text)
        sig = self.compute_minhash_signature(shingles)

        for existing_id, existing_sig in self.seen_signatures:
            sim = self.estimate_jaccard_similarity(sig, existing_sig)
            if sim >= self.similarity_threshold:
                return True

        self.seen_signatures.append((doc_id, sig))
        return False

    def reset(self) -> None:
        """Clear all deduplication state."""
        self.seen_exact_hashes.clear()
        self.seen_signatures.clear()
