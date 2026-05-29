import pytest


CHUNKS = ["chunk one", "chunk two", "chunk three"]
QUERY = "Наименование объекта строительства"
FAKE_RESULTS = [
    {"index": 0, "text": CHUNKS[0], "score": 0.9},
    {"index": 1, "text": CHUNKS[1], "score": 0.7},
    {"index": 2, "text": CHUNKS[2], "score": 0.5},
]


def test_rerank_chunks(monkeypatch: pytest.MonkeyPatch):
    import src.retrieval.reranker as reranker_mod
    from config.config_file import cfg

    monkeypatch.setattr(reranker_mod, "rerank_with_api", lambda **kwargs: FAKE_RESULTS)

    reranked = reranker_mod.rerank_chunks(QUERY, CHUNKS, cfg.RERANKER_MODEL, top_n=3)
    assert len(reranked) == 3
    assert reranked[0]["text"] == CHUNKS[0]


def test_local_rerank_chunks(monkeypatch: pytest.MonkeyPatch):
    import src.retrieval.reranker as reranker_mod
    from numpy import array

    class FakeCrossEncoder:
        def predict(self, pairs, batch_size=32):
            return array([0.1, 0.9, 0.5])

    monkeypatch.setattr(reranker_mod, "_get_local_reranker", lambda _name: FakeCrossEncoder())

    reranked = reranker_mod.rerank_with_local_reranker(
        model_name="qilowoq/bge-reranker-v2-m3-en-ru",
        query=QUERY,
        chunks=CHUNKS,
        top_n=3,
    )
    assert len(reranked) == 3
    assert reranked[0]["index"] == 1


def test_api_rerank_chunks(monkeypatch: pytest.MonkeyPatch):
    import src.retrieval.reranker as reranker_mod

    class FakeResponse:
        status_code = 200

        def json(self):
            return {
                "results": [
                    {
                        "index": i,
                        "document": {"text": text},
                        "relevance_score": score,
                    }
                    for i, (text, score) in enumerate(
                        zip(CHUNKS, [0.9, 0.7, 0.5], strict=True)
                    )
                ]
            }

    monkeypatch.setattr(reranker_mod.requests, "post", lambda *args, **kwargs: FakeResponse())

    reranked = reranker_mod.rerank_with_api(
        model_name="rerank-v3.5", query=QUERY, chunks=CHUNKS, top_n=3
    )
    assert len(reranked) == 3
    assert reranked[0]["text"] == CHUNKS[0]
