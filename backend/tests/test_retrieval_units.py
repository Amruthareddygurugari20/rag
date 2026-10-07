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


def test_paired_difference() -> None:
    from judge_check.retrieval.evaluate import paired_difference

    a = [0.5, 1.0, 0.0, 1.0]
    b = [1.0, 1.0, 0.5, 1.0]  # b - a = [0.5, 0, 0.5, 0]
    d, se = paired_difference(a, b)
    assert d == pytest.approx(0.25)
    # sample SD of [0.5, 0, 0.5, 0] is 0.2887; SE = 0.2887 / 2
    assert se == pytest.approx(0.288675 / 2, rel=1e-5)
    with pytest.raises(ValueError):
        paired_difference([1.0], [1.0])


def test_weighted_rrf_equal_weights_match_plain_rrf() -> None:
    from judge_check.retrieval.fusion import weighted_rrf

    lists = [["a", "b", "c"], ["c", "d", "a"]]
    plain = [x for x, _ in reciprocal_rank_fusion(lists)]
    assert [x for x, _ in weighted_rrf(lists, [0.5, 0.5])] == plain
    # All weight on the first list reproduces it (items only in the other list trail at 0).
    assert [x for x, _ in weighted_rrf(lists, [1.0, 0.0])][:3] == ["a", "b", "c"]


def test_minmax_and_linear_fusion() -> None:
    from judge_check.retrieval.fusion import linear_fusion, minmax

    assert minmax([("a", 0.9), ("b", 0.5), ("c", 0.7)]) == pytest.approx(
        {"a": 1.0, "b": 0.0, "c": 0.5}
    )
    assert minmax([("a", 3.0), ("b", 3.0)]) == {"a": 1.0, "b": 1.0}
    dense = [("a", 0.9), ("b", 0.5)]  # normalised a=1, b=0
    lexical = [("b", 40.0), ("c", 10.0)]  # normalised b=1, c=0
    fused = dict(linear_fusion([dense, lexical], [0.7, 0.3]))
    assert fused == pytest.approx({"a": 0.7, "b": 0.3, "c": 0.0})
    assert [x for x, _ in linear_fusion([dense, lexical], [1.0, 0.0])][:2] == ["a", "b"]
