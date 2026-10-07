"""Reciprocal Rank Fusion (Cormack, Clarke & Büttcher, SIGIR 2009).

    RRF(d) = Σ over retrievers r of  1 / (k + rank_r(d))      ranks start at 1

A document a retriever didn't return contributes nothing for that retriever.

Why ranks and not scores: a cosine similarity (roughly 0.3 to 0.9 here) and a BM25 score
(0 to 30 or more, unbounded, depends on query length) are on unrelated scales. Adding them
needs a normalisation that is itself a tuning choice. RRF only uses positions, so it needs
no calibration. k = 60 damps the advantage of rank 1 over rank 2. It is the value from the
original paper and the common default (D-022).
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import TypeVar

RRF_K = 60

T = TypeVar("T", bound=Hashable)


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[T]], k: int = RRF_K
) -> list[tuple[T, float]]:
    """Fuse ranked lists (best first) into one, best first. Ties break on first appearance."""
    scores: dict[T, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    # dicts keep insertion order and sorted() is stable, so ties are deterministic.
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
