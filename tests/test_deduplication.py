"""
Unit tests for SUSA-1B CodeMind Deduplicator.
"""

from __future__ import annotations

import pytest

from susacodemind.data.deduplication import Deduplicator


@pytest.fixture
def dedup() -> Deduplicator:
    return Deduplicator(k_shingle_size=4, minhash_num_perm=32, similarity_threshold=0.85)


def test_exact_deduplication(dedup):
    """Verify exact duplicate detection flags identical texts."""
    doc1 = "def compute_total(a, b):\n    return a + b\n"
    doc2 = "def compute_total(a, b):\n    return a + b\n"
    doc3 = "def compute_difference(a, b):\n    return a - b\n"

    assert dedup.is_exact_duplicate(doc1) is False
    assert dedup.is_exact_duplicate(doc2) is True  # Duplicate
    assert dedup.is_exact_duplicate(doc3) is False  # New document


def test_near_deduplication(dedup):
    """Verify MinHash detects nearly identical code with minor variable changes."""
    base_code = """
def process_customer_order(order_id, user_id, amount):
    if amount <= 0:
        raise ValueError("Invalid amount")
    print(f"Processing order {order_id} for user {user_id}")
    return {"status": "success", "order_id": order_id, "amount": amount}
"""
    # Clone with minor 1-word comment difference
    cloned_code = base_code + "# Slight variation added here\n"

    assert dedup.is_near_duplicate("doc1", base_code) is False
    assert dedup.is_near_duplicate("doc2", cloned_code) is True  # Near duplicate

    # Completely different code should not trigger duplicate detection
    different_code = """
class MatrixMultiplier:
    def multiply(self, a, b):
        return [[sum(x * y for x, y in zip(row, col)) for col in zip(*b)] for row in a]
"""
    assert dedup.is_near_duplicate("doc3", different_code) is False
