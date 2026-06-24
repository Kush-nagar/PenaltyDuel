"""
Safe data splitters for penalty model evaluation.

The primary split is temporal. Grouped splits are provided for later
generalization studies, but they intentionally do not randomize by default.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def temporal_train_test_split(
    df: pd.DataFrame,
    *,
    cutoff: str | pd.Timestamp,
    date_col: str = "match_date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows into train < cutoff and test >= cutoff."""
    if date_col not in df.columns:
        raise KeyError(date_col)

    dates = pd.to_datetime(df[date_col], errors="coerce")
    cutoff_ts = pd.Timestamp(cutoff)
    train = df.loc[dates < cutoff_ts].copy()
    test = df.loc[dates >= cutoff_ts].copy()
    return train.reset_index(drop=True), test.reset_index(drop=True)


def grouped_holdout_split(
    df: pd.DataFrame,
    *,
    group_col: str,
    holdout_groups: Iterable[object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out every row whose group value is in `holdout_groups`."""
    if group_col not in df.columns:
        raise KeyError(group_col)

    holdout = set(holdout_groups)
    test_mask = df[group_col].isin(holdout)
    train = df.loc[~test_mask].copy()
    test = df.loc[test_mask].copy()
    return train.reset_index(drop=True), test.reset_index(drop=True)


def temporal_cutoff_by_fraction(
    df: pd.DataFrame,
    *,
    train_fraction: float = 0.8,
    date_col: str = "match_date",
) -> pd.Timestamp:
    """
    Choose a chronological cutoff that leaves roughly `train_fraction` in train.

    The returned cutoff is a date value present at the selected boundary; rows on
    that date go to the test side to preserve the train < cutoff invariant.
    """
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if date_col not in df.columns:
        raise KeyError(date_col)

    dates = pd.to_datetime(df[date_col], errors="coerce").sort_values()
    if dates.empty:
        raise ValueError("cannot choose cutoff for an empty DataFrame")
    cutoff_index = min(max(int(len(dates) * train_fraction), 1), len(dates) - 1)
    return pd.Timestamp(dates.iloc[cutoff_index])
