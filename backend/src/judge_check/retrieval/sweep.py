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


def _choose(by_alpha: dict, metric: str, train: Sequence[int], alphas: Sequence[float]) -> float:
    # Highest mean metric on the training fold. Among tied α, the one closest to 0.5
    # (equal weights, the untuned default): the data must earn any departure from equal
    # weighting. Remaining ties (0.4 vs 0.6) go to the higher α.
    return max(alphas, key=lambda a: (mean(by_alpha[a], metric, train), -abs(a - 0.5), a))


def _crossfit_split(
    by_alpha: dict, fold_a: list[int], fold_b: list[int], metric: str, alphas: Sequence[float]
) -> CrossFit:
    a_on_a = _choose(by_alpha, metric, fold_a, alphas)
    a_on_b = _choose(by_alpha, metric, fold_b, alphas)
    held_out = {q: by_alpha[a_on_a][q] for q in fold_b} | {q: by_alpha[a_on_b][q] for q in fold_a}
    return CrossFit(a_on_a, a_on_b, held_out)


def crossfit(
    cands: dict[int, Candidates],
    gold: dict[int, set[int]],
    method: Method,
    k: int,
    metric: str = "rr",
    alphas: Sequence[float] = ALPHAS,
) -> CrossFit:
    """One fixed 2-fold split (alternating question ids)."""
    by_alpha = {a: scores_at(cands, gold, method, a, k) for a in alphas}
    fold_a, fold_b = two_folds(list(cands))
    return _crossfit_split(by_alpha, fold_a, fold_b, metric, alphas)


@dataclass(frozen=True)
class RepeatedCrossFit:
    """Cross-fitting over many random 2-fold partitions.

    The spread of `delta_vs_dense` across partitions measures how much the *choice of α*
    depends on which 100 questions it was chosen on. It is NOT a confidence interval for
    the population: every partition reuses the same 200 questions. Use the paired SE for
    sampling uncertainty, and this spread to see whether the selection itself is stable.
    """

    alphas_chosen: list[float]  # two per partition
    held_out_mean: list[float]  # cross-fitted mean metric, one per partition
    delta_vs_dense: list[float]  # held-out mean minus dense mean, one per partition


def repeated_crossfit(
    cands: dict[int, Candidates],
    gold: dict[int, set[int]],
    method: Method,
    k: int,
    n_repeats: int = 200,
    seed: int = 20261007,
    metric: str = "rr",
    alphas: Sequence[float] = ALPHAS,
) -> RepeatedCrossFit:
    # Imported here to keep one implementation of the version-stable seeded shuffle.
    from judge_check.datasets.hotpotqa import seeded_permutation

    by_alpha = {a: scores_at(cands, gold, method, a, k) for a in alphas}
    dense = scores_at(cands, gold, method, 1.0, k)
    qids = sorted(cands)
    dense_mean = mean(dense, metric, qids)
    chosen, held, delta = [], [], []
    for r in range(n_repeats):
        perm = seeded_permutation(qids, seed + r)
        half = len(perm) // 2
        cf = _crossfit_split(by_alpha, perm[:half], perm[half:], metric, alphas)
        m = mean(cf.held_out, metric, qids)
        chosen += [cf.alpha_chosen_on_a, cf.alpha_chosen_on_b]
        held.append(m)
        delta.append(m - dense_mean)
    return RepeatedCrossFit(chosen, held, delta)


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile, q in [0, 100] (no numpy needed)."""
    xs = sorted(values)
    if not xs:
        raise ValueError("empty")
    pos = (len(xs) - 1) * q / 100
    lo, hi = int(pos), min(int(pos) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)
