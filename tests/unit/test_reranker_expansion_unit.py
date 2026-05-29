from __future__ import annotations

import pytest


def test_merge_rerank_results_max_score():
    from src.retrieval.reranker_expansion import merge_rerank_results

    per_query = [
        (
            "q1",
            [
                {"index": 0, "text": "chunk A", "score": 0.5},
                {"index": 1, "text": "chunk B", "score": 0.9},
            ],
        ),
        (
            "q2",
            [
                {"index": 0, "text": "chunk A", "score": 0.8},
                {"index": 1, "text": "chunk B", "score": 0.4},
            ],
        ),
    ]
    merged = merge_rerank_results(per_query, strategy="max_score")
    by_text = {hit["text"]: hit["score"] for hit in merged}
    assert by_text["chunk A"] == 0.8
    assert by_text["chunk B"] == 0.9


def test_rerank_with_expanded_queries(monkeypatch: pytest.MonkeyPatch):
    import src.retrieval.reranker_expansion as mod

    calls: list[str] = []

    def fake_rerank(query: str, chunks: list[str], reranker_model: str, *, top_n: int = 5):
        calls.append(query)
        return [
            {"index": 0, "text": chunks[0], "score": 1.0},
            {"index": 1, "text": chunks[1], "score": 0.5},
        ]

    monkeypatch.setattr(mod, "rerank_chunks", fake_rerank)

    result = mod.rerank_with_expanded_queries(
        ["q1", "q2"],
        ["t1", "t2"],
        'model',
        top_n=2,
        merge_strategy="max_score",
        max_workers=1,
    )
    assert calls == ["q1", "q2"]
    assert len(result) == 2
    assert result[0]["index"] == 0
    assert result[0]["text"] == "t1"
