"""Retrieval по нескольким RAG-запросам.

Qdrant ищет по коротким child-текстам (`text`), а LLM должен получать широкий
контекст (`parent_text`). Этот модуль сохраняет оба текста рядом, чтобы дальше
не приходилось восстанавливать связь между ними.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from src.retrieval.rank_fusion import MergeStrategy, fuse_ranked_lists
from src.utils.logger import logger

if TYPE_CHECKING:
    from qdrant_client.http.models.models import ScoredPoint

    from src.retrieval.qdrant import QdrantService


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """Результат retrieval после слияния нескольких запросов."""

    text: str
    parent_text: str
    score: float
    point_id: str | int
    source_queries: tuple[str, ...] = ()


def search_by_multi_rag_queries(
    rag_queries: list[str],
    qdrant_service: QdrantService,
    collection_name: str,
    *,
    limit: int = 50,
    part_names: list[str] | None = None,
    max_workers: int | None = 4,
) -> list[tuple[str, list[ScoredPoint]]]:
    """Запускает Qdrant-поиск для каждого непустого запроса.

    Возвращает пары ``(query, points)`` в порядке входных непустых запросов.
    При ошибке отдельного запроса на его месте остаётся пустой список.
    """
    queries = [query.strip() for query in rag_queries if query and query.strip()]
    if not queries:
        return []

    def _search(query: str) -> list[ScoredPoint]:
        return qdrant_service.run_query(
            query,
            collection_name=collection_name,
            limit=limit,
            part_names=part_names,
        )

    workers = max_workers or min(4, len(queries))
    results: list[tuple[str, list[ScoredPoint]]] = [(query, []) for query in queries]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_idx = {
            executor.submit(_search, query): i
            for i, query in enumerate(queries)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            query = queries[idx]
            try:
                points = future.result()
            except Exception:
                logger.exception(
                    "[retrieval_expansion] retrieval failed for query=%r",
                    query,
                )
                points = []
            results[idx] = (query, points)

    return results


def merge_retrieval_results(
    per_query_results: list[tuple[str, list[ScoredPoint]]],
    *,
    strategy: MergeStrategy = "rrf",
    rrf_k: int = 60,
) -> list[RetrievedChunk]:
    """Объединяет результаты нескольких retrieval-запросов в один ранжированный список."""
    if not per_query_results:
        return []

    chunks_by_id: dict[str | int, RetrievedChunk] = {}
    chunks_by_query: list[tuple[str, list[RetrievedChunk]]] = []

    for query, points in per_query_results:
        chunks: list[RetrievedChunk] = []
        for point in points:
            if point.id is None:
                continue

            payload = point.payload or {}
            text = str(payload.get("text") or "").strip()
            if not text:
                continue

            point_id = point.id
            chunk = RetrievedChunk(
                text=text,
                parent_text=str(payload.get("parent_text") or text).strip(),
                score=float(point.score or 0.0),
                point_id=point_id,
            )
            chunks.append(chunk)
            chunks_by_id.setdefault(point_id, chunk)
        chunks_by_query.append((query, chunks))

    fused = fuse_ranked_lists(
        chunks_by_query,
        key=lambda chunk: chunk.point_id,
        score=lambda chunk: chunk.score,
        strategy=strategy,
        rrf_k=rrf_k,
    )

    merged: list[RetrievedChunk] = []
    for item in fused:
        chunk = chunks_by_id.get(item.key)
        if chunk is None:
            continue
        merged.append(
            replace(chunk, score=item.score, source_queries=item.source_queries)
        )
    return merged


def chunks_to_texts(chunks: list[RetrievedChunk]) -> list[str]:
    """Вернуть child-тексты для reranker."""
    return [c.text for c in chunks if c.text]
