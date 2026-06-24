"""
Shrinkage utilities for sparse penalty histories.

These helpers are intentionally small and deterministic. They provide the
empirical-Bayes building blocks used by the as-of feature builder.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def beta_binomial_mean(
    *,
    successes: int | float,
    trials: int | float,
    alpha: float,
    beta: float,
) -> float:
    """Posterior mean for a Bernoulli rate with a Beta(alpha, beta) prior."""
    if alpha <= 0 or beta <= 0:
        raise ValueError("alpha and beta must be positive")
    if trials < 0:
        raise ValueError("trials must be non-negative")
    if successes < 0 or successes > trials:
        raise ValueError("successes must be between 0 and trials")
    return (alpha + successes) / (alpha + beta + trials)


def dirichlet_multinomial_mean(
    *,
    counts: Mapping[str, int | float],
    prior_alpha: Mapping[str, float],
    categories: Sequence[str],
) -> dict[str, float]:
    """Posterior category probabilities for multinomial counts."""
    if not categories:
        raise ValueError("categories must not be empty")

    posterior: dict[str, float] = {}
    for category in categories:
        alpha = float(prior_alpha.get(category, 0.0))
        count = float(counts.get(category, 0.0))
        if alpha <= 0:
            raise ValueError("every category must have a positive prior alpha")
        if count < 0:
            raise ValueError("counts must be non-negative")
        posterior[category] = alpha + count

    total = sum(posterior.values())
    if total <= 0:
        raise ValueError("posterior total must be positive")
    return {category: value / total for category, value in posterior.items()}


def entropy(probabilities: Mapping[str, float]) -> float:
    """
    Normalized Shannon entropy in [0, 1].

    A certain distribution has entropy 0. A uniform distribution over the
    non-zero support has entropy 1.
    """
    values = [float(value) for value in probabilities.values() if float(value) > 0]
    if len(values) <= 1:
        return 0.0

    total = sum(values)
    if total <= 0:
        return 0.0
    normalized = [value / total for value in values]
    raw_entropy = -sum(value * math.log(value) for value in normalized)
    return raw_entropy / math.log(len(normalized))


def beta_prior_from_rate(rate: float, strength: float) -> tuple[float, float]:
    """Convert a prior mean and pseudo-count strength into alpha/beta."""
    if not 0 < rate < 1:
        raise ValueError("rate must be strictly between 0 and 1")
    if strength <= 0:
        raise ValueError("strength must be positive")
    return rate * strength, (1 - rate) * strength


def symmetric_dirichlet_prior(
    categories: Sequence[str],
    *,
    total_strength: float,
) -> dict[str, float]:
    """Build a symmetric Dirichlet prior over the provided categories."""
    if not categories:
        raise ValueError("categories must not be empty")
    if total_strength <= 0:
        raise ValueError("total_strength must be positive")
    alpha = total_strength / len(categories)
    return {category: alpha for category in categories}
