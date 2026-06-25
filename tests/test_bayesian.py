import numpy as np
import pandas as pd
import pytest

from src.features.shrinkage import beta_binomial_mean, beta_prior_from_rate
from src.models.bayesian import (
    BetaPosterior,
    beta_binomial_posterior,
    build_keeper_profiles,
    build_shooter_profiles,
    dirichlet_zone_posterior,
)

PRIOR_RATE = 0.75
PRIOR_STRENGTH = 10.0


def test_posterior_mean_matches_shrinkage_helper():
    alpha0, beta0 = beta_prior_from_rate(PRIOR_RATE, PRIOR_STRENGTH)
    post = beta_binomial_posterior(
        successes=6, trials=8, prior_rate=PRIOR_RATE, prior_strength=PRIOR_STRENGTH
    )
    expected = beta_binomial_mean(successes=6, trials=8, alpha=alpha0, beta=beta0)

    assert isinstance(post, BetaPosterior)
    assert post.mean == pytest.approx(expected)


def test_credible_interval_within_unit_and_brackets_mean():
    post = beta_binomial_posterior(
        successes=6, trials=8, prior_rate=PRIOR_RATE, prior_strength=PRIOR_STRENGTH
    )
    low, high = post.credible_interval(mass=0.9)

    assert 0.0 <= low < post.mean < high <= 1.0


def test_more_data_narrows_interval():
    sparse = beta_binomial_posterior(
        successes=1, trials=1, prior_rate=PRIOR_RATE, prior_strength=PRIOR_STRENGTH
    )
    rich = beta_binomial_posterior(
        successes=30, trials=40, prior_rate=PRIOR_RATE, prior_strength=PRIOR_STRENGTH
    )
    sparse_low, sparse_high = sparse.credible_interval(mass=0.9)
    rich_low, rich_high = rich.credible_interval(mass=0.9)

    assert (rich_high - rich_low) < (sparse_high - sparse_low)


def test_dirichlet_zone_posterior_intervals_sum_close_and_bounded():
    counts = {z: c for z, c in zip(
        ["low_left", "low_center", "low_right", "high_left", "high_center", "high_right"],
        [5, 1, 3, 0, 0, 1],
    )}
    means, intervals = dirichlet_zone_posterior(counts, prior_strength=12.0)

    assert pytest.approx(sum(means.values()), rel=1e-6) == 1.0
    for zone, (low, high) in intervals.items():
        assert 0.0 <= low <= means[zone] <= high <= 1.0


def _toy_penalties() -> pd.DataFrame:
    rows = []
    # shooter A: many kicks, mostly goals. shooter B: one kick.
    for i in range(20):
        rows.append({"shooter_id": "A", "keeper_id": "K1", "outcome_bin": 1 if i < 17 else 0,
                     "shot_zone": "low_left" if i % 2 == 0 else "low_right"})
    rows.append({"shooter_id": "B", "keeper_id": "K2", "outcome_bin": 0, "shot_zone": "high_center"})
    return pd.DataFrame(rows)


def test_build_shooter_profiles_flags_low_data_and_widens_interval():
    profiles = build_shooter_profiles(_toy_penalties())

    by_id = profiles.set_index("shooter_id")
    assert bool(by_id.loc["B", "is_imputed"]) is False  # B has 1 real kick
    assert by_id.loc["A", "n_pens"] == 20
    width_a = by_id.loc["A", "conv_ci_high"] - by_id.loc["A", "conv_ci_low"]
    width_b = by_id.loc["B", "conv_ci_high"] - by_id.loc["B", "conv_ci_low"]
    assert width_b > width_a


def test_build_keeper_profiles_basic_columns():
    profiles = build_keeper_profiles(_toy_penalties())

    assert {"keeper_id", "n_faced", "save_rate_mean", "save_ci_low", "save_ci_high"} <= set(
        profiles.columns
    )
    assert (profiles["save_ci_low"] <= profiles["save_rate_mean"]).all()
    assert (profiles["save_rate_mean"] <= profiles["save_ci_high"]).all()
