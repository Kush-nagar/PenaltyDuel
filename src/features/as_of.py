"""
Leak-safe expanding-window feature construction.

Every history field in this module is computed using rows strictly before the
current penalty in chronological event order.
"""

from __future__ import annotations

import pandas as pd


SHOT_ZONES = [
    "low_left",
    "low_center",
    "low_right",
    "high_left",
    "high_center",
    "high_right",
]

ORDER_COLUMNS = ["match_date", "match_id", "index_in_match"]


def _ordered_with_position(df: pd.DataFrame) -> pd.DataFrame:
    ordered = df.copy()
    ordered["_input_order"] = range(len(ordered))
    ordered["_match_date_ts"] = pd.to_datetime(ordered["match_date"], errors="coerce")
    return ordered.sort_values(
        ["_match_date_ts", "match_id", "index_in_match", "_input_order"],
        kind="mergesort",
    )


def build_as_of_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return penalty rows plus prior-only shooter and keeper history features.

    The returned DataFrame preserves the input row order.
    """
    ordered = _ordered_with_position(df)

    ordered["shooter_n_pens_before"] = ordered.groupby("shooter_id").cumcount()
    ordered["shooter_goals_before"] = (
        ordered.groupby("shooter_id")["outcome_bin"].cumsum() - ordered["outcome_bin"]
    )
    ordered["shooter_conv_rate_raw_before"] = (
        ordered["shooter_goals_before"] / ordered["shooter_n_pens_before"]
    )

    for zone in SHOT_ZONES:
        current_zone = (ordered["shot_zone"] == zone).astype(int)
        ordered[f"shooter_zone_count_{zone}_before"] = (
            current_zone.groupby(ordered["shooter_id"]).cumsum() - current_zone
        )

    ordered["keeper_n_faced_before"] = 0
    ordered["keeper_goals_allowed_before"] = 0
    keeper_resolved = ordered["keeper_id"].notna()
    keeper_rows = ordered.loc[keeper_resolved]
    keeper_key = keeper_rows["keeper_id"]
    ordered.loc[keeper_resolved, "keeper_n_faced_before"] = (
        keeper_rows.groupby(keeper_key).cumcount()
    )
    ordered.loc[keeper_resolved, "keeper_goals_allowed_before"] = (
        keeper_rows.groupby(keeper_key)["outcome_bin"].cumsum() - keeper_rows["outcome_bin"]
    )
    ordered["keeper_save_rate_raw_before"] = (
        (ordered["keeper_n_faced_before"] - ordered["keeper_goals_allowed_before"])
        / ordered["keeper_n_faced_before"]
    )

    result = ordered.sort_values("_input_order", kind="mergesort").drop(
        columns=["_input_order", "_match_date_ts"]
    )
    return result.reset_index(drop=True)
