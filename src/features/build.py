"""
Assemble the Step 5 / Step 8.5 leak-safe modeling table.
"""

from __future__ import annotations

import pandas as pd

from src.features.as_of import SHOT_ZONES, build_as_of_features
from src.features.shrinkage import (
    beta_binomial_mean,
    beta_prior_from_rate,
    dirichlet_multinomial_mean,
    entropy,
    symmetric_dirichlet_prior,
)


RATE_PRIOR_STRENGTH = 10.0
ZONE_PRIOR_STRENGTH = 12.0
DEFAULT_GOAL_RATE_PRIOR = 0.75


def _safe_rate(successes: float, trials: float, default: float) -> float:
    if trials <= 0:
        return default
    return successes / trials


def _pressure_proxy(row: pd.Series) -> str:
    if not bool(row.get("is_shootout", False)):
        return "in_game"
    kick_index = row.get("shootout_kick_index")
    if pd.isna(kick_index):
        return "shootout_unknown"
    if float(kick_index) > 5:
        return "shootout_sudden_death"
    return "shootout_kicks_1_5"


def _default_priors() -> tuple[tuple[float, float], tuple[float, float], dict[str, float]]:
    """Leak-safe priors that do not inspect the current dataset outcomes."""
    shooter_alpha, shooter_beta = beta_prior_from_rate(
        DEFAULT_GOAL_RATE_PRIOR,
        RATE_PRIOR_STRENGTH,
    )

    keeper_alpha, keeper_beta = beta_prior_from_rate(
        1 - DEFAULT_GOAL_RATE_PRIOR,
        RATE_PRIOR_STRENGTH,
    )

    zone_prior = symmetric_dirichlet_prior(
        SHOT_ZONES,
        total_strength=ZONE_PRIOR_STRENGTH,
    )
    return (shooter_alpha, shooter_beta), (keeper_alpha, keeper_beta), zone_prior


def build_modeling_table(
    df: pd.DataFrame,
    *,
    keeper_dive_distributions: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    """
    Build Step 5 / Step 8.5 features from enriched penalties.

    keeper_dive_distributions: optional mapping {keeper_id → {"left": p, ...}}
        produced by src.models.dive_prior. When provided, adds
        keeper_dive_{left,center,right}_prob and keeper_dive_entropy columns.
        When absent, those columns are omitted (zone-only resolution path used).
    """
    features = build_as_of_features(df)
    shooter_prior, keeper_prior, zone_prior = _default_priors()
    shooter_alpha, shooter_beta = shooter_prior
    keeper_alpha, keeper_beta = keeper_prior

    features["shooter_conv_rate_shrunk"] = features.apply(
        lambda row: beta_binomial_mean(
            successes=row["shooter_goals_before"],
            trials=row["shooter_n_pens_before"],
            alpha=shooter_alpha,
            beta=shooter_beta,
        ),
        axis=1,
    )
    features["shooter_conv_rate_is_imputed"] = features["shooter_n_pens_before"] == 0

    keeper_saves_before = (
        features["keeper_n_faced_before"] - features["keeper_goals_allowed_before"]
    )
    features["keeper_save_rate_shrunk"] = [
        beta_binomial_mean(
            successes=saves,
            trials=faced,
            alpha=keeper_alpha,
            beta=keeper_beta,
        )
        for saves, faced in zip(keeper_saves_before, features["keeper_n_faced_before"])
    ]
    features["keeper_save_rate_is_imputed"] = features["keeper_n_faced_before"] == 0

    zone_prob_rows: list[dict[str, float]] = []
    for _, row in features.iterrows():
        counts = {
            zone: row[f"shooter_zone_count_{zone}_before"]
            for zone in SHOT_ZONES
        }
        probs = dirichlet_multinomial_mean(
            counts=counts,
            prior_alpha=zone_prior,
            categories=SHOT_ZONES,
        )
        zone_prob_rows.append(probs)

    for zone in SHOT_ZONES:
        features[f"shooter_zone_prob_{zone}"] = [
            probs[zone] for probs in zone_prob_rows
        ]
    features["shooter_zone_entropy"] = [entropy(probs) for probs in zone_prob_rows]
    features["shooter_zone_probs_is_imputed"] = features["shooter_n_pens_before"] == 0

    features["expected_guess_correct"] = 0.0
    features["side_conflict_score"] = 0.0
    features["stakes"] = features.apply(_pressure_proxy, axis=1)
    features["shooter_foot"] = features.get("shot_body_part_name")
    features["shooter_position"] = features.get("shooter_position_name")

    if "competition_stage_name" in features.columns:
        features["stage"] = features["competition_stage_name"]

    if "season_name" in features.columns:
        features["era"] = features["season_name"]

    features["shooter_conv_rate_raw_before"] = [
        _safe_rate(goals, attempts, float("nan"))
        for goals, attempts in zip(
            features["shooter_goals_before"],
            features["shooter_n_pens_before"],
        )
    ]

    if "shooter_preferred_foot" in features.columns:
        features["shooter_foot_is_right"] = (
            features["shooter_preferred_foot"].fillna("") == "right"
        ).astype(float)
    else:
        features["shooter_foot_is_right"] = 0.0

    if "keeper_preferred_foot" in features.columns:
        features["keeper_foot_is_right"] = (
            features["keeper_preferred_foot"].fillna("") == "right"
        ).astype(float)
    else:
        features["keeper_foot_is_right"] = 0.0

    if keeper_dive_distributions is not None:
        from src.models.dive_prior import attach_dive_features
        features = attach_dive_features(features, keeper_dive_distributions)

    return features
