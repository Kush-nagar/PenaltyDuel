import math

import pytest

from src.features.shrinkage import (
    beta_binomial_mean,
    dirichlet_multinomial_mean,
    entropy,
)


def test_beta_binomial_mean_shrinks_perfect_small_sample_to_prior():
    result = beta_binomial_mean(successes=2, trials=2, alpha=7.5, beta=2.5)

    assert result == pytest.approx(9.5 / 12.0)
    assert result < 1.0


def test_beta_binomial_mean_returns_prior_for_cold_start():
    result = beta_binomial_mean(successes=0, trials=0, alpha=7.5, beta=2.5)

    assert result == pytest.approx(0.75)


def test_beta_binomial_mean_rejects_impossible_counts():
    with pytest.raises(ValueError, match="successes"):
        beta_binomial_mean(successes=3, trials=2, alpha=7.5, beta=2.5)


def test_dirichlet_multinomial_mean_returns_complete_probability_vector():
    categories = ["low_left", "low_center", "low_right"]

    result = dirichlet_multinomial_mean(
        counts={"low_left": 2},
        prior_alpha={"low_left": 1.0, "low_center": 1.0, "low_right": 1.0},
        categories=categories,
    )

    assert result == {
        "low_left": pytest.approx(3 / 5),
        "low_center": pytest.approx(1 / 5),
        "low_right": pytest.approx(1 / 5),
    }
    assert sum(result.values()) == pytest.approx(1.0)


def test_dirichlet_multinomial_mean_returns_prior_for_cold_start():
    result = dirichlet_multinomial_mean(
        counts={},
        prior_alpha={"left": 2.0, "right": 1.0},
        categories=["left", "right"],
    )

    assert result == {"left": pytest.approx(2 / 3), "right": pytest.approx(1 / 3)}


def test_entropy_is_zero_for_certain_distribution_and_one_for_uniform():
    assert entropy({"left": 1.0, "right": 0.0}) == pytest.approx(0.0)
    assert entropy({"left": 0.5, "right": 0.5}) == pytest.approx(1.0)
    assert entropy({"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}) == pytest.approx(1.0)
    assert math.isfinite(entropy({"a": 0.2, "b": 0.8}))
