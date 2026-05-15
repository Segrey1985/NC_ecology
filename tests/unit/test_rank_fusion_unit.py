from __future__ import annotations

import pytest

from src.retrieval.rank_fusion import fuse_ranked_lists


def test_fuse_ranked_lists_max_score():
    per_query = [
        ("q1", [("a", 0.5), ("b", 0.9)]),
        ("q2", [("a", 0.8), ("b", 0.4)]),
    ]
    fused = fuse_ranked_lists(per_query, strategy="max_score")
    by_key = {item.key: item.score for item in fused}
    assert by_key["a"] == 0.8
    assert by_key["b"] == 0.9


def test_fuse_ranked_lists_sum_score():
    per_query = [
        ("q1", [("a", 0.5)]),
        ("q2", [("a", 0.3)]),
    ]
    fused = fuse_ranked_lists(per_query, strategy="sum_score")
    assert len(fused) == 1
    assert fused[0].key == "a"
    assert fused[0].score == pytest.approx(0.8)
    assert set(fused[0].source_queries) == {"q1", "q2"}


def test_fuse_ranked_lists_rrf_dedupes_within_query():
    per_query = [
        ("q1", [("a", 0.1), ("a", 0.9), ("b", 0.5)]),
    ]
    fused = fuse_ranked_lists(per_query, strategy="rrf", rrf_k=60)
    by_key = {item.key: item.score for item in fused}
    assert by_key["a"] == pytest.approx(1.0 / (60 + 1))
    assert by_key["b"] == pytest.approx(1.0 / (60 + 3))


def test_fuse_ranked_lists_rrf_merges_queries():
    per_query = [
        ("q1", [("a", 0.0)]),
        ("q2", [("a", 0.0)]),
    ]
    fused = fuse_ranked_lists(per_query, strategy="rrf", rrf_k=0)
    assert len(fused) == 1
    assert fused[0].key == "a"
    assert fused[0].score == pytest.approx(2.0)
    assert set(fused[0].source_queries) == {"q1", "q2"}


def test_fuse_ranked_lists_empty():
    assert fuse_ranked_lists([]) == []
