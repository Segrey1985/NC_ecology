"""
Query expansion для cross-encoder reranker: multiple rerank → merge.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Literal

from src.retrieval.rank_fusion import (
    FusedRankItem,
    MergeStrategy,
    fuse_ranked_lists,
    normalize_text,
)
from src.retrieval.reranker import rerank_chunks
from src.utils.logger import logger

DEFAULT_MERGE_STRATEGY: MergeStrategy = "rrf"


@dataclass(frozen=True, slots=True)
class RerankedChunk:
    """Чанк после merge нескольких rerank-запросов."""

    text: str
    score: float
    source_queries: tuple[str, ...] = ()


def multiple_rerank(
    queries: list[str],
    chunks: list[str],
    reranker_model: str,
    *,
    max_workers: int | None = 4,
    score_all: bool = True,
) -> list[tuple[str, list[tuple[str, float]]]]:
    """
    Параллельный rerank по каждому expanded-запросу на одном наборе чанков.

    Returns:
        Список пар (query, [(text, score), ...]) в порядке `queries`.
    """
    if not queries or not chunks:
        return []

    top_n = len(chunks) if score_all else 5
    results: list[tuple[str, list[tuple[str, float]]]] = [(q, []) for q in queries]
    workers = max_workers or min(8, max(len(queries), 1))

    def _rerank(query: str) -> list[tuple[str, float]]:
        return rerank_chunks(query, chunks, reranker_model=reranker_model, top_n=top_n)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_idx = {
            executor.submit(_rerank, q): i
            for i, q in enumerate(queries)
            if q.strip()
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            query = queries[idx]
            try:
                ranked = future.result()
            except Exception:
                logger.exception(
                    "[reranker_expansion] rerank failed for query=%r",
                    query,
                )
                ranked = []
            results[idx] = (query, ranked)

    return results


def merge_rerank_results(
    per_query_results: list[tuple[str, list[tuple[str, float]]]],
    *,
    strategy: MergeStrategy = DEFAULT_MERGE_STRATEGY,
    rrf_k: int = 60,
) -> list[RerankedChunk]:
    """Сливает rerank-результаты по нормализованному тексту чанка."""
    if not per_query_results:
        return []

    text_by_key: dict[str, str] = {}
    ranked_by_key: list[tuple[str, list[tuple[str, float]]]] = []

    for query, ranked in per_query_results:
        keyed: list[tuple[str, float]] = []
        for text, score in ranked:
            if not text:
                continue
            key = normalize_text(text)
            text_by_key.setdefault(key, text)
            keyed.append((key, float(score)))
        ranked_by_key.append((query, keyed))

    fused: list[FusedRankItem] = fuse_ranked_lists(
        ranked_by_key,
        strategy=strategy,
        rrf_k=rrf_k,
    )

    return [
        RerankedChunk(
            text=text_by_key[item.key],
            score=item.score,
            source_queries=item.source_queries,
        )
        for item in fused
        if item.key in text_by_key
    ]


def rerank_with_expanded_queries(
    queries: list[str],
    chunks: list[str],
    reranker_model: str,
    *,
    top_n: int = 5,
    merge_strategy: MergeStrategy = DEFAULT_MERGE_STRATEGY,
    rrf_k: int = 60,
    max_workers: int | None = 4,
) -> list[tuple[str, float]]:
    """
    Multiple rerank → merge → top_n.
    """
    clean_queries = [q.strip() for q in queries if q and q.strip()]
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
    return [(c.text, c.score) for c in merged[:top_n]]
