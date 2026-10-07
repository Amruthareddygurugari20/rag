"""Sweep the dense/BM25 fusion weight, honestly (D-028).

Retrieve each question's dense and BM25 candidate lists once, then fuse them in memory for
every weight α (weight on dense; BM25 gets 1 − α):
  α = 1    dense only
  α = 0.5  equal weights (plain RRF's order)
  α = 0    BM25 only

Picking the best α on the same questions you report on is tuning on the test set, and the
reported number would be optimistic. So the claim-bearing number is **cross-fitted**: split
the questions into two fixed halves, choose α on one half, score that α on the *other*
half, swap, and pool the held-out scores. The full curve is reported too, as description,
not as a result.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from judge_check.retrieval.evaluate import QueryScores, per_query_scores
from judge_check.retrieval.fusion import linear_fusion, weighted_rrf

Method = Literal["rrf", "linear"]
ALPHAS: tuple[float, ...] = tuple(round(0.1 * i, 1) for i in range(11))


@dataclass(frozen=True)
class Candidates:
    dense: list[tuple[int, float]]  # (chunk id, cosine), best first
    bm25: list[tuple[int, float]]  # (chunk id, BM25 score), best first


def fuse(c: Candidates, method: Method, alpha: float) -> list[int]:
    weights = (alpha, 1.0 - alpha)
    if method == "rrf":
        fused = weighted_rrf([[i for i, _ in c.dense], [i for i, _ in c.bm25]], weights)
    elif method == "linear":
        fused = linear_fusion([c.dense, c.bm25], weights)
    else:
        raise ValueError(f"unknown method {method!r}")
    return [i for i, _ in fused]


def scores_at(
    cands: dict[int, Candidates],
    gold: dict[int, set[int]],
    method: Method,
    alpha: float,
    k: int,
) -> dict[int, QueryScores]:
    # Truncate to k before scoring, so MRR is MRR@k like eval-retrieval reports.
    rankings = {q: fuse(c, method, alpha)[:k] for q, c in cands.items()}
    return per_query_scores(rankings, gold, k)


def mean(scores: dict[int, QueryScores], metric: str, qids: Sequence[int]) -> float:
    return sum(getattr(scores[q], metric) for q in qids) / len(qids)


def two_folds(qids: Sequence[int]) -> tuple[list[int], list[int]]:
    """Fixed split: alternate questions in id order. No randomness, nothing to re-roll."""
    ordered = sorted(qids)
    return ordered[0::2], ordered[1::2]


@dataclass(frozen=True)
class CrossFit:
    alpha_chosen_on_a: float  # chosen on fold A, scored on fold B
    alpha_chosen_on_b: float
    held_out: dict[int, QueryScores]  # every question scored by an α chosen without it


def crossfit(
    cands: dict[int, Candidates],
    gold: dict[int, set[int]],
    method: Method,
    k: int,
    metric: str = "rr",
    alphas: Sequence[float] = ALPHAS,
) -> CrossFit:
    fold_a, fold_b = two_folds(list(cands))
    by_alpha = {a: scores_at(cands, gold, method, a, k) for a in alphas}

    def choose(train: list[int]) -> float:
        # Highest mean metric on the training fold. Among tied α, the one closest to 0.5
        # (equal weights, the untuned default): the data must earn any departure from equal
        # weighting. Remaining ties (0.4 vs 0.6) go to the higher α.
        return max(alphas, key=lambda a: (mean(by_alpha[a], metric, train), -abs(a - 0.5), a))

    a_on_a, a_on_b = choose(fold_a), choose(fold_b)
    held_out = {q: by_alpha[a_on_a][q] for q in fold_b} | {q: by_alpha[a_on_b][q] for q in fold_a}
    return CrossFit(a_on_a, a_on_b, held_out)
