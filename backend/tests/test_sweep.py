"""Fusion sweep and cross-fitting on synthetic candidates (no DB, no model)."""

import pytest

from judge_check.retrieval.sweep import (
    Candidates,
    crossfit,
    fuse,
    percentile,
    repeated_crossfit,
    scores_at,
    two_folds,
)


def test_alpha_extremes_reproduce_each_retriever() -> None:
    c = Candidates(dense=[(1, 0.9), (2, 0.8)], bm25=[(3, 20.0), (1, 5.0)])
    for method in ("rrf", "linear"):
        assert fuse(c, method, 1.0)[:2] == [1, 2]
        assert fuse(c, method, 0.0)[:2] == [3, 1]


def test_two_folds_partition_deterministically() -> None:
    a, b = two_folds([5, 1, 4, 2, 3])
    assert (a, b) == ([1, 3, 5], [2, 4])


def test_crossfit_scores_each_fold_with_alpha_chosen_on_the_other() -> None:
    # Odd-indexed questions (fold A: ids 0,2,4..) are answered only by dense, fold B
    # (1,3,5..) only by BM25. Fold A therefore picks a dense-heavy alpha and fold B a
    # BM25-heavy one, and each is scored on the *other* fold, where it is wrong.
    cands, gold = {}, {}
    for q in range(10):
        good, bad = 100 + q, 200 + q
        if q % 2 == 0:  # fold A: dense right, BM25 wrong
            cands[q] = Candidates(dense=[(good, 0.9), (bad, 0.1)], bm25=[(bad, 9.0), (good, 1.0)])
        else:  # fold B: the opposite
            cands[q] = Candidates(dense=[(bad, 0.9), (good, 0.1)], bm25=[(good, 9.0), (bad, 1.0)])
        gold[q] = {good}
    cf = crossfit(cands, gold, "linear", k=1)
    # Fold A needs a dense-leaning alpha, fold B a BM25-leaning one.
    assert cf.alpha_chosen_on_a >= 0.5 > cf.alpha_chosen_on_b
    # Held-out: every question is scored with the wrong alpha, so MRR@1 is 0. The in-sample
    # best would have looked perfect for each fold. This is the optimism cross-fitting removes.
    assert all(s.rr == 0.0 for s in cf.held_out.values())
    assert len(cf.held_out) == 10
    in_sample = scores_at({q: cands[q] for q in (0, 2, 4, 6, 8)}, gold, "linear", 1.0, 1)
    assert all(s.rr == 1.0 for s in in_sample.values())


def test_tied_alphas_resolve_towards_equal_weights() -> None:
    # Every alpha gives a perfect score: the choice must be 0.5, not an arbitrary extreme.
    cands = {q: Candidates(dense=[(q, 0.9)], bm25=[(q, 5.0)]) for q in range(6)}
    gold = {q: {q} for q in range(6)}
    cf = crossfit(cands, gold, "rrf", k=1)
    assert cf.alpha_chosen_on_a == cf.alpha_chosen_on_b == 0.5


def test_crossfit_finds_dense_when_dense_is_always_right() -> None:
    cands = {
        q: Candidates(dense=[(q, 0.9), (50 + q, 0.2)], bm25=[(50 + q, 30.0), (q, 2.0)])
        for q in range(8)
    }
    gold = {q: {q} for q in range(8)}
    cf = crossfit(cands, gold, "rrf", k=1)
    assert cf.alpha_chosen_on_a >= 0.5 and cf.alpha_chosen_on_b >= 0.5
    assert sum(s.rr for s in cf.held_out.values()) == pytest.approx(8.0)


def test_repeated_crossfit_is_reproducible_and_sized() -> None:
    cands = {
        q: Candidates(dense=[(q, 0.9), (50 + q, 0.2)], bm25=[(50 + q, 30.0), (q, 2.0)])
        for q in range(12)
    }
    gold = {q: {q} for q in range(12)}
    a = repeated_crossfit(cands, gold, "rrf", k=1, n_repeats=7, seed=3)
    b = repeated_crossfit(cands, gold, "rrf", k=1, n_repeats=7, seed=3)
    assert a == b
    assert len(a.alphas_chosen) == 14 and len(a.delta_vs_dense) == 7
    # Dense is always right here, so every partition matches dense exactly.
    assert all(d == pytest.approx(0.0) for d in a.delta_vs_dense)


def test_percentile() -> None:
    assert percentile([3, 1, 2, 4], 50) == pytest.approx(2.5)
    assert percentile([5], 97.5) == 5
    assert percentile(list(range(101)), 2.5) == pytest.approx(2.5)
