from __future__ import annotations

from types import SimpleNamespace

import pytest


def _point(point_id: str | int, score: float, text: str):
    return SimpleNamespace(id=point_id, score=score, payload={"text": text})


def test_merge_retrieval_results_max_score():
    from src.retrieval.retrieval_expansion import merge_retrieval_results

    per_query = [
        ("q1", [_point("p1", 0.5, "chunk A"), _point("p2", 0.9, "chunk B")]),
        ("q2", [_point("p1", 0.8, "chunk A"), _point("p2", 0.4, "chunk B")]),
    ]
    merged = merge_retrieval_results(per_query, strategy="max_score")
    by_id = {c.point_id: c.score for c in merged}
    assert by_id["p1"] == 0.8
    assert by_id["p2"] == 0.9
    assert {c.text for c in merged} == {"chunk A", "chunk B"}


def test_merge_retrieval_results_skips_points_without_text():
    from src.retrieval.retrieval_expansion import merge_retrieval_results

    per_query = [
        ("q1", [_point("p1", 0.5, ""), _point("p2", 0.9, "ok")]),
    ]
    merged = merge_retrieval_results(per_query, strategy="max_score")
    assert len(merged) == 1
    assert merged[0].point_id == "p2"


def test_merge_retrieval_results_empty_query_list():
    from src.retrieval.retrieval_expansion import merge_retrieval_results

    assert merge_retrieval_results([("q1", [])]) == []


def test_search_by_multi_rag_queries_empty():
    from src.retrieval.retrieval_expansion import search_by_multi_rag_queries

    assert search_by_multi_rag_queries([], object(), "col") == []


def test_search_by_multi_rag_queries_preserves_order_on_error(
    monkeypatch: pytest.MonkeyPatch,
):
    import src.retrieval.retrieval_expansion as mod

    class FakeQdrant:
        def run_query(self, query, **kwargs):
            if query == "bad":
                raise RuntimeError("search failed")
            return [_point("p1", 0.7, f"text-{query}")]

    result = mod.search_by_multi_rag_queries(
        ["ok", "bad", "also-ok"],
        FakeQdrant(),
        "col",
        max_workers=2,
    )
    assert [q for q, _ in result] == ["ok", "bad", "also-ok"]
    assert result[0][1][0].payload["text"] == "text-ok"
    assert result[1][1] == []
    assert result[2][1][0].payload["text"] == "text-also-ok"
