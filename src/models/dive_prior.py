"""
Dirichlet-shrunk keeper dive distribution estimator.

Estimates P(dive=d | keeper) for d in {left, center, right} using
Dirichlet-multinomial shrinkage. Population prior is built from aggregate
counts (typically Dataset 2 WC kicks). Per-keeper counts update the prior.
"""

from __future__ import annotations

import pandas as pd

from src.features.shrinkage import dirichlet_multinomial_mean, entropy, symmetric_dirichlet_prior

DIVE_DIRECTIONS = ["left", "center", "right"]

# Default prior strength when no external population counts are provided.
# Symmetric over 3 directions = 4.0/3 ≈ 1.33 per cell → weak but non-zero.
DEFAULT_PRIOR_STRENGTH = 4.0


def build_population_prior(
    df: pd.DataFrame,
    *,
    prior_strength: float = DEFAULT_PRIOR_STRENGTH,
) -> dict[str, float]:
    """
    Build a Dirichlet prior from aggregate dive counts across all keepers.

    If df is empty or has no dive labels, falls back to a symmetric prior.
    """
    if df.empty or "keeper_dive_direction" not in df.columns:
        return symmetric_dirichlet_prior(DIVE_DIRECTIONS, total_strength=prior_strength)

    labeled = df[df["keeper_dive_direction"].isin(DIVE_DIRECTIONS)]
    if labeled.empty:
        return symmetric_dirichlet_prior(DIVE_DIRECTIONS, total_strength=prior_strength)

    total = len(labeled)
    counts = labeled["keeper_dive_direction"].value_counts()
    # Scale raw counts to prior_strength so the prior doesn't dwarf per-keeper data
    scale = prior_strength / total
    return {d: float(counts.get(d, 0) * scale + 1e-6) for d in DIVE_DIRECTIONS}


def estimate_keeper_dive_distributions(
    statsbomb_df: pd.DataFrame,
    kaggle_df: pd.DataFrame | None,
    *,
    population_prior: dict[str, float] | None = None,
) -> dict[str, dict[str, float]]:
    """
    Estimate P(dive | keeper) for every keeper_id in statsbomb_df.

    Returns a mapping {keeper_id: {"left": p, "center": p, "right": p}}.
    Keepers with no observed dive data receive the shrunk population prior.
    """
    if population_prior is None:
        pop_df = kaggle_df if kaggle_df is not None else pd.DataFrame()
        population_prior = build_population_prior(pop_df)

    # Collect per-keeper dive counts from all labeled sources
    keeper_counts: dict[str, dict[str, float]] = {}

    def _accumulate(df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return
        need = {"keeper_id", "keeper_dive_direction"}
        if not need.issubset(df.columns):
            return
        labeled = df[df["keeper_dive_direction"].isin(DIVE_DIRECTIONS)]
        for kid, grp in labeled.groupby("keeper_id"):
            vc = grp["keeper_dive_direction"].value_counts()
            if kid not in keeper_counts:
                keeper_counts[kid] = {d: 0.0 for d in DIVE_DIRECTIONS}
            for d in DIVE_DIRECTIONS:
                keeper_counts[kid][d] += float(vc.get(d, 0))

    _accumulate(statsbomb_df)
    if kaggle_df is not None:
        _accumulate(kaggle_df)

    # Estimate distribution for every keeper seen in statsbomb_df
    all_keepers = statsbomb_df["keeper_id"].dropna().unique()
    result: dict[str, dict[str, float]] = {}
    for kid in all_keepers:
        counts = keeper_counts.get(str(kid), {d: 0.0 for d in DIVE_DIRECTIONS})
        dist = dirichlet_multinomial_mean(
            counts=counts,
            prior_alpha=population_prior,
            categories=DIVE_DIRECTIONS,
        )
        result[str(kid)] = dist

    return result


def dive_entropy(dist: dict[str, float]) -> float:
    """Normalized Shannon entropy of a dive distribution."""
    return entropy(dist)


def attach_dive_features(
    df: pd.DataFrame,
    keeper_dive_distributions: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """
    Add keeper_dive_{left,center,right}_prob and keeper_dive_entropy columns.

    Keepers not in the distributions dict receive population-mean values
    (all columns still non-null — every row gets a value).
    """
    out = df.copy()
    lefts, centers, rights, entropies = [], [], [], []

    for _, row in df.iterrows():
        kid = str(row.get("keeper_id", "")) if pd.notna(row.get("keeper_id")) else ""
        dist = keeper_dive_distributions.get(kid)
        if dist is None:
            # fallback: uniform — will be replaced by population prior in practice
            dist = {d: 1.0 / 3 for d in DIVE_DIRECTIONS}
        lefts.append(dist["left"])
        centers.append(dist["center"])
        rights.append(dist["right"])
        entropies.append(dive_entropy(dist))

    out["keeper_dive_left_prob"] = lefts
    out["keeper_dive_center_prob"] = centers
    out["keeper_dive_right_prob"] = rights
    out["keeper_dive_entropy"] = entropies
    return out
