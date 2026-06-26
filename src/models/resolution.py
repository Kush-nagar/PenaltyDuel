"""
Composed placement -> goal resolution path (Step 7 / Step 8.5).

Zone-only formula (when dive data absent):
    P(goal) = sum_z  P(zone=z | shooter) * P(goal | zone=z)

Full formula (when keeper dive distribution available):
    P(goal) = sum_z sum_d  P(zone=z | shooter) * P(dive=d | keeper) * P(goal | z, d)

Both are leak-safe when resolution tables are fit on training rows only and
shooter zone distributions are as-of (prior-only) Step 5 features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RESOLUTION_ZONES = [
    "low_left",
    "low_center",
    "low_right",
    "high_left",
    "high_center",
    "high_right",
]

DIVE_DIRECTIONS = ["left", "center", "right"]

# Minimum fraction of rows with dive labels to fit the zone×dive table.
MIN_DIVE_COVERAGE = 0.05

EPS = 1e-6


def fit_resolution_table(
    train_df: pd.DataFrame,
    *,
    smoothing: float = 5.0,
) -> dict[str, float]:
    """Empirical P(goal | zone), Laplace-smoothed toward the train global rate."""
    if "shot_zone" not in train_df.columns or "outcome_bin" not in train_df.columns:
        raise KeyError("train_df must contain shot_zone and outcome_bin")
    if smoothing <= 0:
        raise ValueError("smoothing must be positive")

    global_rate = float(train_df["outcome_bin"].mean())
    global_rate = float(np.clip(global_rate, EPS, 1 - EPS))

    table: dict[str, float] = {}
    for zone in RESOLUTION_ZONES:
        rows = train_df[train_df["shot_zone"] == zone]
        n = len(rows)
        goals = float(rows["outcome_bin"].sum())
        value = (goals + smoothing * global_rate) / (n + smoothing)
        table[zone] = float(np.clip(value, EPS, 1 - EPS))
    return table


def fit_dive_resolution_table(
    train_df: pd.DataFrame,
    *,
    smoothing: float = 3.0,
) -> dict[tuple[str, str], float] | None:
    """
    Empirical P(goal | zone, dive), Laplace-smoothed.

    Returns None when dive coverage is below MIN_DIVE_COVERAGE — caller falls
    back to the zone-only resolution table.
    """
    if "shot_zone" not in train_df.columns or "outcome_bin" not in train_df.columns:
        raise KeyError("train_df must contain shot_zone and outcome_bin")
    if "keeper_dive_direction" not in train_df.columns:
        return None

    labeled = train_df[train_df["keeper_dive_direction"].isin(DIVE_DIRECTIONS)]
    coverage = len(labeled) / max(len(train_df), 1)
    if coverage < MIN_DIVE_COVERAGE:
        return None

    global_rate = float(train_df["outcome_bin"].mean())
    global_rate = float(np.clip(global_rate, EPS, 1 - EPS))

    table: dict[tuple[str, str], float] = {}
    for zone in RESOLUTION_ZONES:
        for dive in DIVE_DIRECTIONS:
            rows = labeled[
                (labeled["shot_zone"] == zone)
                & (labeled["keeper_dive_direction"] == dive)
            ]
            n = len(rows)
            goals = float(rows["outcome_bin"].sum()) if n > 0 else 0.0
            value = (goals + smoothing * global_rate) / (n + smoothing)
            table[(zone, dive)] = float(np.clip(value, EPS, 1 - EPS))
    return table


def compose_goal_prob(
    df: pd.DataFrame,
    table: dict[str, float],
    *,
    dive_table: dict[tuple[str, str], float] | None = None,
) -> np.ndarray:
    """
    Compose P(goal) from shooter zone distribution and resolution table(s).

    When dive_table is provided AND the df contains keeper_dive_{left,center,right}_prob
    columns, uses the full zone×dive formula. Otherwise falls back to zone-only.
    """
    missing = [f"shooter_zone_prob_{z}" for z in RESOLUTION_ZONES
               if f"shooter_zone_prob_{z}" not in df.columns]
    if missing:
        raise KeyError(f"missing zone probability columns: {missing}")

    dive_cols = [f"keeper_dive_{d}_prob" for d in DIVE_DIRECTIONS]
    use_dive = (
        dive_table is not None
        and all(c in df.columns for c in dive_cols)
    )

    if use_dive:
        weighted = np.zeros(len(df), dtype=float)
        for zone in RESOLUTION_ZONES:
            zone_prob = df[f"shooter_zone_prob_{zone}"].to_numpy(dtype=float)
            for dive in DIVE_DIRECTIONS:
                dive_prob = df[f"keeper_dive_{dive}_prob"].to_numpy(dtype=float)
                p_goal = dive_table.get((zone, dive), table.get(zone, 0.5))
                weighted += zone_prob * dive_prob * p_goal
        return np.clip(weighted, EPS, 1 - EPS)

    # Zone-only fallback
    weighted = np.zeros(len(df), dtype=float)
    weight_total = np.zeros(len(df), dtype=float)
    for zone in RESOLUTION_ZONES:
        zone_prob = df[f"shooter_zone_prob_{zone}"].to_numpy(dtype=float)
        weighted += zone_prob * table[zone]
        weight_total += zone_prob

    with np.errstate(divide="ignore", invalid="ignore"):
        composed = np.where(weight_total > 0, weighted / weight_total, weighted)
    return np.clip(composed, EPS, 1 - EPS)
