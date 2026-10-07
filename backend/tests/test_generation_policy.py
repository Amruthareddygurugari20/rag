"""The eval-eligibility gate is measured plausibility, pre-registered (D-032 A)."""

import pytest

from judge_check.generation_policy import (
    MIN_REVIEWED,
    PLAUSIBLE_LB_THRESHOLD,
    is_eval_eligible,
    min_plausible_to_pass,
    wilson_lower_bound,
)


def test_preregistered_gate_values() -> None:
    # Changing these is a decision, not a refactor: it must go through DECISIONS.md.
    assert (PLAUSIBLE_LB_THRESHOLD, MIN_REVIEWED) == (0.80, 40)


def quadratic_oracle(k: int, n: int, z: float = 1.959963984540054) -> float:
    """Independent derivation: the Wilson bound is the lower root of
    (p_hat - p)^2 = z^2 p (1 - p) / n. Different algebra from the implementation."""
    import math

    ph = k / n
    a, b, c = 1 + z * z / n, -(2 * ph + z * z / n), ph * ph
    return (-b - math.sqrt(b * b - 4 * a * c)) / (2 * a)


@pytest.mark.parametrize(("k", "n"), [(37, 40), (36, 40), (88, 100), (0, 10), (10, 10), (1, 1)])
def test_wilson_lower_bound_matches_independent_derivation(k: int, n: int) -> None:
    assert wilson_lower_bound(k, n) == pytest.approx(quadratic_oracle(k, n), abs=1e-12)


def test_wilson_lower_bound_matches_scipy() -> None:
    stats = pytest.importorskip("scipy.stats")
    for k, n in [(37, 40), (88, 100), (5, 9)]:
        expected = stats.binomtest(k, n).proportion_ci(method="wilson").low
        assert wilson_lower_bound(k, n) == pytest.approx(expected, abs=1e-12)


def test_ten_out_of_ten_is_not_proof() -> None:
    # A perfect small sample is still compatible with a mediocre generator.
    assert wilson_lower_bound(10, 10) < PLAUSIBLE_LB_THRESHOLD


def test_gate_boundary() -> None:
    assert min_plausible_to_pass(40) == 37
    assert min_plausible_to_pass(100) == 88
    assert is_eval_eligible(37, 40) and not is_eval_eligible(36, 40)


def test_too_few_reviews_is_never_eligible() -> None:
    assert min_plausible_to_pass(MIN_REVIEWED - 1) is None
    assert not is_eval_eligible(39, 39)  # perfect, but not enough evidence


def test_no_model_is_special_cased() -> None:
    # The gate takes counts only: a model's name can neither admit nor exclude it.
    import inspect

    assert list(inspect.signature(is_eval_eligible).parameters) == ["plausible", "reviewed"]


def test_invalid_counts_raise() -> None:
    with pytest.raises(ValueError):
        wilson_lower_bound(5, 4)
