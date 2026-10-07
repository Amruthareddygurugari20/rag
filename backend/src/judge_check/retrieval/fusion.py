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


# --- Weighted fusion (D-028) -----------------------------------------------------------------
#
# Equal-weight RRF assumes both retrievers are equally trustworthy. When one is much weaker on
# a corpus, it drags the fused list towards its mistakes. Two weighted alternatives:
#
#   weighted RRF:  score(d) = Σ_r w_r / (k + rank_r(d))
#   linear:        score(d) = Σ_r w_r · minmax_r(d),   minmax_r(d) = (s - min_r)/(max_r - min_r)
#
# In the linear form a document missing from retriever r's candidate list gets 0 for r: below
# everything r did return. Min-max is computed per query, over that query's candidates.


def weighted_rrf(
    rankings: Sequence[Sequence[T]], weights: Sequence[float], k: int = RRF_K
) -> list[tuple[T, float]]:
    """RRF with a weight per ranking. Equal weights give the same order as plain RRF."""
    if len(rankings) != len(weights):
        raise ValueError("one weight per ranking")
    scores: dict[T, float] = {}
    for ranking, w in zip(rankings, weights, strict=True):
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + w / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def minmax(scored: Sequence[tuple[T, float]]) -> dict[T, float]:
    """Scale one retriever's scores for one query to [0, 1]. All-equal scores map to 1."""
    if not scored:
        return {}
    values = [s for _, s in scored]
    lo, hi = min(values), max(values)
    if hi == lo:
        return {item: 1.0 for item, _ in scored}
    return {item: (s - lo) / (hi - lo) for item, s in scored}


def linear_fusion(
    scored_lists: Sequence[Sequence[tuple[T, float]]], weights: Sequence[float]
) -> list[tuple[T, float]]:
    """Weighted sum of per-query min-max-normalised scores. Ties keep first-seen order."""
    if len(scored_lists) != len(weights):
        raise ValueError("one weight per list")
    normalised = [minmax(s) for s in scored_lists]
    order: dict[T, None] = {}
    for s in scored_lists:
        for item, _ in s:
            order.setdefault(item, None)
    fused = {
        item: sum(w * n.get(item, 0.0) for w, n in zip(weights, normalised, strict=True))
        for item in order
    }
    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
