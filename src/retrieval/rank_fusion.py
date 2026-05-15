"""
Общие стратегии слияния ранжированных списков (retrieval, reranker expansion).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Hashable, Literal, TypeVar

MergeStrategy = Literal["max_score", "sum_score", "rrf"]

K = TypeVar("K", bound=Hashable)


@dataclass(frozen=True, slots=True)
class FusedRankItem(Generic[K]):
    key: K
    score: float
    source_queries: tuple[str, ...] = ()


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def fuse_ranked_lists(
    per_query_results: list[tuple[str, list[tuple[K, float]]]],
    *,
    strategy: MergeStrategy = "rrf",
    rrf_k: int = 60,
    dedupe_within_query: bool = True,
) -> list[FusedRankItem[K]]:
    """
    Объединяет несколько ранжированных списков по ключу.

    Для RRF порядок элементов в каждом списке = rank (1-based).
    Для max_score / sum_score используются числовые score из пар (key, score).
    """
    if not per_query_results:
        return []

    if strategy == "rrf":
        return _fuse_rrf(per_query_results, k=rrf_k, dedupe_within_query=dedupe_within_query)

    accum: dict[K, dict] = {}
    for query, ranked in per_query_results:
        for key, raw_score in ranked:
            score = float(raw_score)
            if key not in accum:
                accum[key] = {"score": score, "source_queries": {query}}
                continue
            entry = accum[key]
            entry["source_queries"].add(query)
            if strategy == "max_score":
                if score > entry["score"]:
                    entry["score"] = score
            elif strategy == "sum_score":
                entry["score"] += score
            else:
                raise ValueError(f"Unknown merge strategy: {strategy!r}")

    fused = [
        FusedRankItem(
            key=key,
            score=entry["score"],
            source_queries=tuple(sorted(entry["source_queries"])),
        )
        for key, entry in accum.items()
    ]
    fused.sort(key=lambda item: item.score, reverse=True)
    return fused


def _fuse_rrf(
    per_query_results: list[tuple[str, list[tuple[K, float]]]],
    *,
    k: int,
    dedupe_within_query: bool,
) -> list[FusedRankItem[K]]:
    rrf_scores: dict[K, float] = {}
    sources: dict[K, set[str]] = {}

    for query, ranked in per_query_results:
        seen: set[K] = set()
        for rank, (key, _score) in enumerate(ranked, start=1):
            if dedupe_within_query and key in seen:
                continue
            seen.add(key)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            sources.setdefault(key, set()).add(query)

    fused = [
        FusedRankItem(
            key=key,
            score=rrf_scores[key],
            source_queries=tuple(sorted(sources[key])),
        )
        for key in rrf_scores
    ]
    fused.sort(key=lambda item: item.score, reverse=True)
    return fused
