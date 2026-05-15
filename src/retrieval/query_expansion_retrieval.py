"""
RAG-пайплайн с query expansion: multiple retrieval → merge.

Ожидаемый порядок вызовов (остальные этапы — на стороне вызывающего кода):

    expanded_queries = ...          # Query Expansion
    per_query = multiple_retrieval(...)
    merged = merge_retrieval_results(per_query)
    texts = chunks_to_texts(merged) # → reranker → context assembly → LLM
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.retrieval.rank_fusion import MergeStrategy, fuse_ranked_lists
from src.utils.logger import logger

if TYPE_CHECKING:
    from qdrant_client.http.models.models import ScoredPoint

    from src.retrieval.qdrant import QdrantService


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """Единый элемент после merge — готов к reranking."""

    text: str
    score: float
    point_id: str | int | None = None
    payload: dict = field(default_factory=dict)
    source_queries: tuple[str, ...] = ()


def multiple_retrieval(
    queries: list[str],
    qdrant_service: QdrantService,
    collection_name: str,
    *,
    limit: int = 50,
    part_names: list[str] | None = None,
    max_workers: int | None = 4,
) -> list[tuple[str, list[ScoredPoint]]]:
    if not queries:
        return []

    def _search(query: str) -> list[ScoredPoint]:
        return qdrant_service.run_query(
            query,
            collection_name=collection_name,
            limit=limit,
            part_names=part_names,
        )

    workers = max_workers or min(8, len(queries))
    results: list[tuple[str, list[ScoredPoint]]] = [(q, []) for q in queries]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_idx = {
            executor.submit(_search, q): i
            for i, q in enumerate(queries)
            if q.strip()
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = (queries[idx], future.result())
            except Exception:
                logger.exception(
                    "[query_expansion_retrieval] retrieval failed for query=%r",
                    queries[idx],
                )

    return results


def merge_retrieval_results(
    per_query_results: list[tuple[str, list[ScoredPoint]]],
    *,
    strategy: MergeStrategy = "rrf",
    rrf_k: int = 60,
) -> list[RetrievedChunk]:
    """
    Объединяет результаты нескольких запросов в один ранжированный список.

    Дубликаты по point.id схлопываются на этапе merge.
    Стратегии: max_score, sum_score, rrf (см. rank_fusion.fuse_ranked_lists).
    """
    if not per_query_results:
        return []

    meta: dict[str | int, dict] = {}
    ranked_by_key: list[tuple[str, list[tuple[str | int, float]]]] = []

    for query, points in per_query_results:
        ranked: list[tuple[str | int, float]] = []
        for point in points:
            pid = point.id
            if pid is None:
                continue
            text = (point.payload or {}).get("text", "")
            if not text:
                continue
            ranked.append((pid, float(point.score or 0.0)))
            if pid not in meta:
                meta[pid] = {
                    "text": text,
                    "payload": dict(point.payload or {}),
                    "point_id": pid,
                }
        ranked_by_key.append((query, ranked))

    fused = fuse_ranked_lists(
        ranked_by_key,
        strategy=strategy,
        rrf_k=rrf_k,
    )

    return [
        RetrievedChunk(
            text=meta[item.key]["text"],
            score=item.score,
            point_id=meta[item.key]["point_id"],
            payload=meta[item.key]["payload"],
            source_queries=item.source_queries,
        )
        for item in fused
        if item.key in meta
    ]


def chunks_to_texts(chunks: list[RetrievedChunk]) -> list[str]:
    """Тексты чанков для reranker (совместимо с rerank_chunks)."""
    return [c.text for c in chunks if c.text]
