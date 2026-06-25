"""
Lightweight Bayesian player/keeper profiles (Step 7, rung 2 - conjugate form).

Provides posterior point estimates *and credible intervals* for shooter
conversion, keeper save rate, and shot-zone placement, using closed-form
conjugate updates (Beta-Binomial, Dirichlet-multinomial). This is the cheap,
no-MCMC path to the uncertainty the explanation cards need: the interval widens
visibly when a player has little data.

A full hierarchical sampler (PyMC/NumPyro) with partial pooling across
foot x position x league is intentionally NOT built here - on data this sparse it
adds heavy dependencies and runtime for negligible accuracy gain. The conjugate
priors below already shrink toward the global rate and quantify uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

from src.features.as_of import SHOT_ZONES
from src.features.build import (
    DEFAULT_GOAL_RATE_PRIOR,
    RATE_PRIOR_STRENGTH,
    ZONE_PRIOR_STRENGTH,
)
from src.features.shrinkage import beta_prior_from_rate, entropy, symmetric_dirichlet_prior


@dataclass
class BetaPosterior:
    """A Beta(alpha, beta) posterior over a rate."""

    alpha: float
    beta: float

    @property
    def mean(self) -> float:
        return float(self.alpha / (self.alpha + self.beta))

    def credible_interval(self, *, mass: float = 0.9) -> tuple[float, float]:
        if not 0 < mass < 1:
            raise ValueError("mass must be between 0 and 1")
        tail = (1 - mass) / 2
        low = float(beta_dist.ppf(tail, self.alpha, self.beta))
        high = float(beta_dist.ppf(1 - tail, self.alpha, self.beta))
        return low, high


def beta_binomial_posterior(
    *,
    successes: float,
    trials: float,
    prior_rate: float,
    prior_strength: float,
) -> BetaPosterior:
    """Conjugate Beta posterior for a Bernoulli rate."""
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("require 0 <= successes <= trials")
    alpha0, beta0 = beta_prior_from_rate(prior_rate, prior_strength)
    return BetaPosterior(alpha=alpha0 + successes, beta=beta0 + (trials - successes))


def dirichlet_zone_posterior(
    counts: dict[str, float],
    *,
    prior_strength: float = ZONE_PRIOR_STRENGTH,
    mass: float = 0.9,
) -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
    """Posterior zone means + per-zone marginal Beta credible intervals."""
    prior = symmetric_dirichlet_prior(SHOT_ZONES, total_strength=prior_strength)
    alphas = {z: prior[z] + float(counts.get(z, 0.0)) for z in SHOT_ZONES}
    total = sum(alphas.values())

    means = {z: alphas[z] / total for z in SHOT_ZONES}
    tail = (1 - mass) / 2
    intervals: dict[str, tuple[float, float]] = {}
    for z in SHOT_ZONES:
        a = alphas[z]
        b = total - alphas[z]
        intervals[z] = (
            float(beta_dist.ppf(tail, a, b)),
            float(beta_dist.ppf(1 - tail, a, b)),
        )
    return means, intervals


def _confidence_label(n: int) -> str:
    if n >= 11:
        return "high"
    if n >= 4:
        return "medium"
    if n >= 1:
        return "low"
    return "none"


def build_shooter_profiles(
    penalties: pd.DataFrame,
    *,
    prior_rate: float = DEFAULT_GOAL_RATE_PRIOR,
    prior_strength: float = RATE_PRIOR_STRENGTH,
    mass: float = 0.9,
) -> pd.DataFrame:
    """Career Bayesian profile (conversion + placement + uncertainty) per shooter."""
    rows: list[dict[str, object]] = []
    for shooter_id, group in penalties.groupby("shooter_id"):
        n = int(len(group))
        goals = float(group["outcome_bin"].sum())
        post = beta_binomial_posterior(
            successes=goals, trials=n, prior_rate=prior_rate, prior_strength=prior_strength
        )
        low, high = post.credible_interval(mass=mass)

        zone_counts = {z: float((group["shot_zone"] == z).sum()) for z in SHOT_ZONES}
        zone_means, _ = dirichlet_zone_posterior(zone_counts)
        favored_zone = max(zone_means, key=zone_means.get)

        rows.append(
            {
                "shooter_id": shooter_id,
                "n_pens": n,
                "goals": goals,
                "conv_rate_raw": goals / n if n > 0 else np.nan,
                "conv_rate_mean": post.mean,
                "conv_ci_low": low,
                "conv_ci_high": high,
                "favored_zone": favored_zone,
                "favored_zone_prob": zone_means[favored_zone],
                "zone_entropy": entropy(zone_means),
                "is_imputed": n == 0,
                "confidence": _confidence_label(n),
            }
        )
    return pd.DataFrame(rows)


def build_keeper_profiles(
    penalties: pd.DataFrame,
    *,
    prior_rate: float = DEFAULT_GOAL_RATE_PRIOR,
    prior_strength: float = RATE_PRIOR_STRENGTH,
    mass: float = 0.9,
) -> pd.DataFrame:
    """Career Bayesian save-rate profile (with uncertainty) per keeper."""
    faced = penalties.loc[penalties["keeper_id"].notna()]
    save_prior_rate = 1 - prior_rate
    rows: list[dict[str, object]] = []
    for keeper_id, group in faced.groupby("keeper_id"):
        n = int(len(group))
        saves = float((group["outcome_bin"] == 0).sum())
        post = beta_binomial_posterior(
            successes=saves,
            trials=n,
            prior_rate=save_prior_rate,
            prior_strength=prior_strength,
        )
        low, high = post.credible_interval(mass=mass)
        rows.append(
            {
                "keeper_id": keeper_id,
                "n_faced": n,
                "saves": saves,
                "save_rate_raw": saves / n if n > 0 else np.nan,
                "save_rate_mean": post.mean,
                "save_ci_low": low,
                "save_ci_high": high,
                "is_imputed": n == 0,
                "confidence": _confidence_label(n),
            }
        )
    return pd.DataFrame(rows)
