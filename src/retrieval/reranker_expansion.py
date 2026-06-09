"""Rerank одного набора чанков несколькими запросами.

`rerank_chunks()` уже возвращает достаточный формат: `index`, `text`, `score`.
Здесь мы только запускаем rerank для нескольких запросов и объединяем ранги
по исходному `index` чанка.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TypedDict

from src.retrieval.rank_fusion import MergeStrategy, fuse_ranked_lists
from src.retrieval.reranker import rerank_chunks, TOP_N
from src.utils.logger import logger

DEFAULT_MERGE_STRATEGY: MergeStrategy = "rrf"


class RerankHit(TypedDict):
    """Публичный формат результата reranker."""

    index: int
    text: str
    score: float


def _valid_hit(raw_hit: dict, chunks: list[str]) -> RerankHit | None:
    """Приводит результат reranker к ожидаемому формату или отбрасывает его."""
    try:
        index = int(raw_hit["index"])
    except (KeyError, TypeError, ValueError):
        logger.warning("[reranker_expansion] skip hit without valid index: %r", raw_hit)
        return None

    if index < 0 or index >= len(chunks):
        logger.warning("[reranker_expansion] skip hit with out-of-range index: %r", raw_hit)
        return None

    text = str(raw_hit.get("text") or chunks[index])
    score = float(raw_hit.get("score", 0.0) or 0.0)
    return {"index": index, "text": text, "score": score}


def multiple_rerank(
    queries: list[str],
    chunks: list[str],
    reranker_model: str,
    *,
    max_workers: int | None = 4,
    score_all: bool = True,
) -> list[tuple[str, list[RerankHit]]]:
    """Параллельно запускает rerank для каждого запроса и сохраняет порядок запросов."""
    clean_queries = [query.strip() for query in queries if query and query.strip()]
    if not clean_queries or not chunks:
        return []

    top_n = len(chunks) if score_all else TOP_N
    results: list[tuple[str, list[RerankHit]]] = [(query, []) for query in clean_queries]
    workers = max_workers or min(8, len(clean_queries))

    def _rerank(query: str) -> list[RerankHit]:
        raw_hits = rerank_chunks(
            query,
            chunks,
            reranker_model=reranker_model,
            top_n=top_n,
        )
        return [
            hit
            for raw_hit in raw_hits
            if (hit := _valid_hit(raw_hit, chunks)) is not None
        ]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_idx = {
            executor.submit(_rerank, q): i
            for i, q in enumerate(clean_queries)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            query = clean_queries[idx]
            try:
                ranked = future.result()
            except Exception:
                logger.exception(f"[reranker_expansion] rerank failed for {query=}")
                ranked = []
            results[idx] = (query, ranked)

    return results


def _texts_by_index(per_query_results: list[tuple[str, list[RerankHit]]]) -> dict[int, str]:
    texts: dict[int, str] = {}
    for _query, hits in per_query_results:
        for hit in hits:
            texts[hit["index"]] = hit["text"]
    return texts


def merge_rerank_results(
    per_query_results: list[tuple[str, list[RerankHit]]],
    *,
    strategy: MergeStrategy = DEFAULT_MERGE_STRATEGY,
    rrf_k: int = 60,
) -> list[RerankHit]:
    """Сливает rerank-результаты по исходному индексу чанка."""
    if not per_query_results:
        return []

    text_by_index = _texts_by_index(per_query_results)

    fused = fuse_ranked_lists(
        per_query_results,
        key=lambda hit: hit["index"],
        score=lambda hit: hit["score"],
        strategy=strategy,
        rrf_k=rrf_k,
    )

    merged: list[RerankHit] = []
    for item in fused:
        index = int(item.key)
        score = item.score
        text = text_by_index.get(index)
        if text is None:
            continue
        merged.append({"index": index, "text": text, "score": score})
    return merged


def rerank_with_expanded_queries(
    queries: list[str],
    chunks: list[str],
    reranker_model: str,
    *,
    top_n: int = TOP_N,
    merge_strategy: MergeStrategy = DEFAULT_MERGE_STRATEGY,
    rrf_k: int = 60,
    max_workers: int | None = 4,
) -> list[RerankHit]:
    """Выполнить rerank по нескольким запросам, объединить результаты и вернуть top_n."""
    clean_queries = [query.strip() for query in queries if query and query.strip()]
    if not clean_queries or not chunks:
        return []

    per_query = multiple_rerank(
        clean_queries,
        chunks,
        reranker_model,
        max_workers=max_workers,
        score_all=True,
    )
    merged = merge_rerank_results(
        per_query,
        strategy=merge_strategy,
        rrf_k=rrf_k,
    )
    return merged[:top_n]
