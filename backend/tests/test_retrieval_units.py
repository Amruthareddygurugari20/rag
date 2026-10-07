"""Pure-logic retrieval pieces: no database, no model."""

import numpy as np
import pytest

from judge_check.embeddings import check_unit_norm
from judge_check.retrieval.evaluate import score_rankings
from judge_check.retrieval.fusion import reciprocal_rank_fusion
from judge_check.retrieval.tokenize import tokenize


def test_tokenize() -> None:
    assert tokenize("The U.S. Army's 1st Division, in Zürich!") == [
        "the", "u", "s", "army", "s", "1st", "division", "in", "zürich",
    ]  # fmt: skip


def test_rrf_matches_hand_computation() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["c", "a"]], k=60)
    expected = {
        "a": 1 / 61 + 1 / 62,
        "b": 1 / 62,
        "c": 1 / 63 + 1 / 61,
    }
    assert [x for x, _ in fused] == ["a", "c", "b"]
    for item, score in fused:
        assert score == pytest.approx(expected[item])


def test_rrf_rewards_agreement_over_a_single_top_rank() -> None:
    # "x" is first in one list only; "y" is second in both. Agreement wins.
    fused = reciprocal_rank_fusion([["x", "y"], ["z", "y"]])
    assert fused[0][0] == "y"


def test_rrf_ties_are_deterministic() -> None:
    assert reciprocal_rank_fusion([["a"], ["b"]]) == reciprocal_rank_fusion([["a"], ["b"]])
    assert [x for x, _ in reciprocal_rank_fusion([["a"], ["b"]])] == ["a", "b"]


def test_score_rankings() -> None:
    gold = {1: {10, 11}, 2: {20}}
    rankings = {1: [10, 99, 11], 2: [98, 97, 20]}
    m = score_rankings(rankings, gold, k=2)
    assert m.recall == pytest.approx((1 / 2 + 0) / 2)
    assert m.complete == 0.0
    assert m.mrr == pytest.approx((1 / 1 + 1 / 3) / 2)
    assert score_rankings(rankings, gold, k=3).complete == 1.0


def test_score_rankings_rejects_question_without_gold() -> None:
    with pytest.raises(ValueError, match="no gold"):
        score_rankings({1: [1]}, {}, k=1)


def test_check_unit_norm() -> None:
    v = np.array([[3.0, 4.0]]) / 5.0
    assert check_unit_norm(v) is v
    with pytest.raises(ValueError, match="not unit-normalised"):
        check_unit_norm(np.array([[3.0, 4.0]]))
