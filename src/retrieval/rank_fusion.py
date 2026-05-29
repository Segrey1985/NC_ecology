"""
Общие стратегии слияния ранжированных списков (retrieval, reranker expansion).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Sequence, TypeVar

MergeStrategy = Literal["max_score", "sum_score", "rrf"]
RankKey = str | int
RankedItem = TypeVar("RankedItem")


@dataclass(frozen=True, slots=True)
class FusedRankItem:
    key: RankKey
    score: float
    source_queries: tuple[str, ...] = ()


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def fuse_ranked_lists(
    per_query_results: Sequence[tuple[str, Sequence[RankedItem]]],
    *,
    key: Callable[[RankedItem], RankKey],
    score: Callable[[RankedItem], float],
    strategy: MergeStrategy = "rrf",
    rrf_k: int = 60,
    dedupe_within_query: bool = True,
) -> list[FusedRankItem]:
    """
    Объединяет несколько ранжированных списков.

    Вызывающий код передает исходные элементы как есть, а `key` и `score` объясняют,
    как из элемента получить ключ дедупликации и численный score.
    Для RRF важен порядок элементов внутри каждого списка.
    """
    if not per_query_results:
        return []

    if strategy == "rrf":
        return _fuse_rrf(
            per_query_results,
            key=key,
            k=rrf_k,
            dedupe_within_query=dedupe_within_query,
        )

    accum: dict[RankKey, dict] = {}
    for query, items in per_query_results:
        for item in items:
            item_key = key(item)
            item_score = float(score(item))
            if item_key not in accum:
                accum[item_key] = {"score": item_score, "source_queries": {query}}
                continue
            entry = accum[item_key]
            entry["source_queries"].add(query)
            if strategy == "max_score":
                if item_score > entry["score"]:
                    entry["score"] = item_score
            elif strategy == "sum_score":
                entry["score"] += item_score
            else:
                raise ValueError(f"Unknown merge strategy: {strategy!r}")

    fused = [
        FusedRankItem(
            key=item_key,
            score=entry["score"],
            source_queries=tuple(sorted(entry["source_queries"])),
        )
        for item_key, entry in accum.items()
    ]
    fused.sort(key=lambda item: item.score, reverse=True)
    return fused


def _fuse_rrf(
    per_query_results: Sequence[tuple[str, Sequence[RankedItem]]],
    *,
    key: Callable[[RankedItem], RankKey],
    k: int,
    dedupe_within_query: bool,
) -> list[FusedRankItem]:
    rrf_scores: dict[RankKey, float] = {}
    sources: dict[RankKey, set[str]] = {}

    for query, items in per_query_results:
        seen: set[RankKey] = set()
        for rank, item in enumerate(items, start=1):
            item_key = key(item)
            if dedupe_within_query and item_key in seen:
                continue
            seen.add(item_key)
            rrf_scores[item_key] = rrf_scores.get(item_key, 0.0) + 1.0 / (k + rank)
            sources.setdefault(item_key, set()).add(query)

    fused = [
        FusedRankItem(
            key=item_key,
            score=rrf_scores[item_key],
            source_queries=tuple(sorted(sources[item_key])),
        )
        for item_key in rrf_scores
    ]
    fused.sort(key=lambda item: item.score, reverse=True)
    return fused
