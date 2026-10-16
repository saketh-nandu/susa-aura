"""
SUSA-1B CodeMind — Exact and Near-Deduplication Engine.

Removes identical files and near-duplicate source files using SHA-256 and
k-shingle MinHash approximations to prevent benchmark leakage and overfitting.
"""

from __future__ import annotations

import hashlib
import random
import re
from typing import Dict, Iterable, List, Set, Tuple


def compute_sha256(text: str) -> str:
    """Computes SHA-256 hash of text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Deduplicator:
    """Combines exact hash filtering with high-speed MinHash and LSH indexing."""

    _MERSENNE_PRIME = (1 << 61) - 1

    def __init__(
        self,
        k_shingle_size: int = 5,
        minhash_num_perm: int = 32,
        similarity_threshold: float = 0.85,
        num_bands: int = 8,
    ) -> None:
        self.k_shingle_size = k_shingle_size
        self.minhash_num_perm = minhash_num_perm
        self.similarity_threshold = similarity_threshold
        self.num_bands = num_bands
        self.rows_per_band = max(1, minhash_num_perm // num_bands)

        # Precompute deterministic universal hash parameters (a_i, b_i)
        rng = random.Random(42)
        self.perms: list[tuple[int, int]] = [
            (
                rng.randint(1, self._MERSENNE_PRIME - 1),
                rng.randint(0, self._MERSENNE_PRIME - 1),
            )
            for _ in range(minhash_num_perm)
        ]

        self.seen_exact_hashes: set[str] = set()
        self.seen_signatures: dict[str, list[int]] = {}
        # LSH Buckets: band_idx -> (bucket_hash -> list of doc_ids)
        self.lsh_buckets: list[dict[int, list[str]]] = [{} for _ in range(self.num_bands)]

    def is_exact_duplicate(self, text: str) -> bool:
        """Checks if text has been seen before via exact SHA-256."""
        h = compute_sha256(text)
        if h in self.seen_exact_hashes:
            return True
        self.seen_exact_hashes.add(h)
        return False

    def get_shingles(self, text: str, max_shingles: int = 1500) -> set[str]:
        """Extracts whitespace-normalized character or token k-shingles."""
        tokens = re.findall(r"\w+|[^\w\s]", text)
        if len(tokens) < self.k_shingle_size:
            return set(tokens)

        total_shingles = len(tokens) - self.k_shingle_size + 1
        stride = max(1, total_shingles // max_shingles) if total_shingles > max_shingles else 1

        return {
            " ".join(tokens[i : i + self.k_shingle_size])
            for i in range(0, total_shingles, stride)
        }

    def compute_minhash_signature(self, shingles: set[str]) -> list[int]:
        """Computes MinHash signature using universal hashing over shingle integer hashes."""
        if not shingles:
            return [0] * self.minhash_num_perm

        # Hash each unique shingle once into 64-bit integer
        h_vals = [
            int.from_bytes(hashlib.md5(s.encode("utf-8")).digest()[:8], "little")
            for s in shingles
        ]

        sig: list[int] = []
        p = self._MERSENNE_PRIME
        for a, b in self.perms:
            min_val = min((a * h + b) % p for h in h_vals)
            sig.append(min_val)

        return sig

    def estimate_jaccard_similarity(self, sig1: list[int], sig2: list[int]) -> float:
        """Estimates Jaccard similarity as fraction of matching signature hashes."""
        if len(sig1) != len(sig2) or len(sig1) == 0:
            return 0.0
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / len(sig1)

    def is_near_duplicate(self, doc_id: str, text: str) -> bool:
        """Checks if text is a near-duplicate of any previously seen document via LSH indexing."""
        shingles = self.get_shingles(text)
        sig = self.compute_minhash_signature(shingles)

        # Fast LSH Candidate Retrieval
        candidate_ids: set[str] = set()
        band_hashes: list[int] = []
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            band_chunk = tuple(sig[start : start + self.rows_per_band])
            b_hash = hash(band_chunk)
            band_hashes.append(b_hash)
            bucket = self.lsh_buckets[band_idx]
            if b_hash in bucket:
                candidate_ids.update(bucket[b_hash])

        # If no LSH candidates found, check fallback if few docs seen, otherwise test candidates
        if candidate_ids:
            for cand_id in candidate_ids:
                existing_sig = self.seen_signatures[cand_id]
                sim = self.estimate_jaccard_similarity(sig, existing_sig)
                if sim >= self.similarity_threshold:
                    return True
        elif len(self.seen_signatures) < 100:
            # Fallback direct comparison when collection is small
            for existing_sig in self.seen_signatures.values():
                sim = self.estimate_jaccard_similarity(sig, existing_sig)
                if sim >= self.similarity_threshold:
                    return True

        # Index the new document
        self.seen_signatures[doc_id] = sig
        for band_idx, b_hash in enumerate(band_hashes):
            self.lsh_buckets[band_idx].setdefault(b_hash, []).append(doc_id)

        return False

    def reset(self) -> None:
        """Clear all deduplication state."""
        self.seen_exact_hashes.clear()
        self.seen_signatures.clear()
        for bucket in self.lsh_buckets:
            bucket.clear()
