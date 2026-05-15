"""
RAG-пайплайн с query expansion: multiple retrieval → merge → deduplication.

Ожидаемый порядок вызовов (остальные этапы — на стороне вызывающего кода):

    expanded_queries = ...          # Query Expansion
    per_query = multiple_retrieval(...)
    merged = merge_retrieval_results(per_query)
    unique = deduplicate_chunks(merged)
    texts = chunks_to_texts(unique) # → reranker → context assembly → LLM
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Literal, TYPE_CHECKING

from src.utils.logger import logger

if TYPE_CHECKING:
    from qdrant_client.http.models.models import ScoredPoint

    from src.retrieval.qdrant import QdrantService

MergeStrategy = Literal["max_score", "sum_score", "rrf"]
DedupStrategy = Literal["point_id", "text"]


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """Единый элемент после merge/dedup — готов к reranking."""

    text: str  # текст чанка (payload['text'])
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

    Стратегии:
        max_score — для одной точки берётся максимальный score из Qdrant;
        sum_score — сумма score по всем запросам, где точка встретилась;
        rrf       — reciprocal rank fusion: sum 1/(k + rank).
    """
    if not per_query_results:
        return []

    if strategy == "rrf":
        return _merge_rrf(per_query_results, k=rrf_k)

    accum: dict[str | int, dict] = {}

    for query, points in per_query_results:
        for point in points:
            pid = point.id
            if pid is None:
                continue
            text = (point.payload or {}).get("text", "")
            raw_score = float(point.score or 0.0)

            if pid not in accum:
                accum[pid] = {
                    "text": text,
                    "score": raw_score,
                    "payload": dict(point.payload or {}),
                    "source_queries": {query},
                    "point_id": pid,
                }
                continue

            entry = accum[pid]
            entry["source_queries"].add(query)
            if strategy == "max_score":
                if raw_score > entry["score"]:
                    entry["score"] = raw_score
            elif strategy == "sum_score":
                entry["score"] += raw_score
            else:
                raise ValueError(f"Unknown merge strategy: {strategy!r}")

    merged = [
        RetrievedChunk(
            text=e["text"],
            score=e["score"],
            point_id=e["point_id"],
            payload=e["payload"],
            source_queries=tuple(sorted(e["source_queries"])),
        )
        for e in accum.values()
        if e["text"]
    ]
    merged.sort(key=lambda c: c.score, reverse=True)
    return merged


def _merge_rrf(
    per_query_results: list[tuple[str, list[ScoredPoint]]],
    *,
    k: int,
) -> list[RetrievedChunk]:
    rrf_scores: dict[str | int, float] = {}
    meta: dict[str | int, dict] = {}

    
    for query, points in per_query_results:
        seen_in_query = set()
        for rank, point in enumerate(points, start=1):
            pid = point.id
            if pid is None or pid in seen_in_query:
                continue
            seen_in_query.add(pid)
            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (k + rank)
            if pid not in meta:
                meta[pid] = {
                    "text": (point.payload or {}).get("text", ""),
                    "payload": dict(point.payload or {}),
                    "source_queries": {query},
                    "point_id": pid,
                }
            else:
                meta[pid]["source_queries"].add(query)

    merged = [
        RetrievedChunk(
            text=m["text"],
            score=rrf_scores[pid],
            point_id=m["point_id"],
            payload=m["payload"],
            source_queries=tuple(sorted(m["source_queries"])),
        )
        for pid, m in meta.items()
        if m["text"]
    ]
    merged.sort(key=lambda c: c.score, reverse=True)
    return merged


def deduplicate_chunks(
    chunks: list[RetrievedChunk],
    *,
    by: DedupStrategy = "point_id",
) -> list[RetrievedChunk]:
    """
    Удаляет дубликаты, сохраняя элемент с наибольшим score.

    by=point_id — по id точки Qdrant (рекомендуется после merge);
    by=text     — по нормализованному тексту чанка.
    """
    if not chunks:
        return []

    best: dict[str | int, RetrievedChunk] = {}

    for chunk in chunks:
        key: str | int
        if by == "point_id":
            if chunk.point_id is None:
                key = _normalize_text(chunk.text)
            else:
                key = chunk.point_id
        elif by == "text":
            key = _normalize_text(chunk.text)
        else:
            raise ValueError(f"Unknown dedup strategy: {by!r}")

        existing = best.get(key)
        if existing is None or chunk.score > existing.score:
            best[key] = chunk

    deduped = list(best.values())
    deduped.sort(key=lambda c: c.score, reverse=True)
    return deduped


def chunks_to_texts(chunks: list[RetrievedChunk]) -> list[str]:
    """Тексты чанков для reranker (совместимо с rerank_chunks)."""
    return [c.text for c in chunks if c.text]


def _normalize_text(text: str) -> str:
    return " ".join(text.split())
