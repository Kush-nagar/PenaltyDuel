"""
Composed placement -> goal resolution path (Step 7).

The resolution table holds the empirical, smoothed probability of scoring given
the shot zone. The composed goal probability is the shooter's as-of zone
distribution dotted with that table:

    P(goal) = sum_z  P(zone=z | shooter) * P(goal | zone=z)

This is leak-safe as long as the resolution table is fit on training rows only
and the shooter zone distribution is the as-of (prior-only) Step 5 feature.

Dive direction is intentionally absent: `keeper_dive_direction` has 0% coverage
in the StatsBomb-only dataset, so the full zone x dive resolution is deferred.
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


def compose_goal_prob(df: pd.DataFrame, table: dict[str, float]) -> np.ndarray:
    """Compose P(goal) from the shooter zone distribution and resolution table."""
    missing = [f"shooter_zone_prob_{z}" for z in RESOLUTION_ZONES
               if f"shooter_zone_prob_{z}" not in df.columns]
    if missing:
        raise KeyError(f"missing zone probability columns: {missing}")

    weighted = np.zeros(len(df), dtype=float)
    weight_total = np.zeros(len(df), dtype=float)
    for zone in RESOLUTION_ZONES:
        zone_prob = df[f"shooter_zone_prob_{zone}"].to_numpy(dtype=float)
        weighted += zone_prob * table[zone]
        weight_total += zone_prob

    # Normalize in case zone probabilities do not sum exactly to 1.
    with np.errstate(divide="ignore", invalid="ignore"):
        composed = np.where(weight_total > 0, weighted / weight_total, weighted)
    return np.clip(composed, EPS, 1 - EPS)
