"""Unit tests for the pure retrieval metrics."""
from evaluation.metrics import (
    hit_at_k, recall_at_k, precision_at_k, reciprocal_rank, mean,
)


def test_hit_at_k_found_and_not_found():
    assert hit_at_k(["a", "b", "c"], {"c"}, 3) == 1.0
    assert hit_at_k(["a", "b", "c"], {"c"}, 2) == 0.0  # c is at rank 3
    assert hit_at_k(["a", "b"], {"z"}, 5) == 0.0


def test_recall_at_k_partial():
    # 1 of 2 relevant docs in top 3
    assert recall_at_k(["a", "b", "c"], {"a", "z"}, 3) == 0.5
    assert recall_at_k(["a", "b", "c"], {"a", "b"}, 3) == 1.0


def test_precision_at_k_dedups_docs():
    # same doc repeated (multiple chunks) counts once
    assert precision_at_k(["a", "a", "b"], {"a"}, 3) == 0.5  # dedup -> [a, b]


def test_reciprocal_rank():
    assert reciprocal_rank(["x", "y", "a"], {"a"}) == 1.0 / 3
    assert reciprocal_rank(["a", "b"], {"a"}) == 1.0
    assert reciprocal_rank(["x", "y"], {"a"}) == 0.0


def test_empty_relevant_is_zero_not_crash():
    assert recall_at_k(["a"], set(), 5) == 0.0
    assert hit_at_k(["a"], [], 5) == 0.0
    assert reciprocal_rank(["a"], set()) == 0.0


def test_mean_handles_empty():
    assert mean([]) == 0.0
    assert mean([1.0, 0.0]) == 0.5
