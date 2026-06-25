"""
Generalization studies for Step 8.

Leave-players-out and leave-competition-out grouped CV, comparing the full
combined model against a priors-only forecast (the train global rate) on groups
never seen in training. Outcome metrics are also bucketed by `shooter_n_pens_before`
to show *where* personal signal actually helps.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    brier_score,
    expected_calibration_error,
    kendall_tau,
    log_loss,
    spearman_corr,
)
from src.models.combo import predict_combined_proba, train_combined_model

EPS = 1e-6

_BUCKETS = [(0, 0, "0"), (1, 3, "1-3"), (4, 10, "4-10"), (11, np.inf, "11+")]
BUCKET_ORDER = ["0", "1-3", "4-10", "11+"]


def n_pens_bucket(n: float) -> str:
    """Map a prior-penalty count to a coarse support bucket."""
    value = float(n)
    for low, high, label in _BUCKETS:
        if low <= value <= high:
            return label
    return "11+"


def group_kfold(
    df: pd.DataFrame,
    *,
    group_col: str,
    n_splits: int = 5,
    seed: int = 0,
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Yield (train, test) folds where each fold holds out a disjoint set of groups."""
    if group_col not in df.columns:
        raise KeyError(group_col)
    groups = df[group_col].dropna().unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    folds = np.array_split(groups, n_splits)
    for fold_groups in folds:
        if len(fold_groups) == 0:
            continue
        test_mask = df[group_col].isin(set(fold_groups))
        train = df.loc[~test_mask].reset_index(drop=True)
        test = df.loc[test_mask].reset_index(drop=True)
        if train.empty or test.empty:
            continue
        yield train, test


def evaluate_generalization(
    modeling_table: pd.DataFrame,
    *,
    group_col: str,
    n_splits: int = 5,
    combo_C: float = 1.0,
    seed: int = 0,
) -> dict[str, object]:
    """Out-of-fold grouped evaluation: full combined model vs priors-only floor."""
    full_pred = np.full(len(modeling_table), np.nan)
    priors_pred = np.full(len(modeling_table), np.nan)
    index_pos = {pid: i for i, pid in enumerate(modeling_table["penalty_id"])}

    for train, test in group_kfold(
        modeling_table, group_col=group_col, n_splits=n_splits, seed=seed
    ):
        if train["outcome_bin"].nunique() < 2:
            continue
        global_rate = float(np.clip(train["outcome_bin"].mean(), EPS, 1 - EPS))
        model = train_combined_model(train, C=combo_C, global_rate=global_rate)
        preds = predict_combined_proba(model, test)
        for pid, p in zip(test["penalty_id"], preds):
            pos = index_pos[pid]
            full_pred[pos] = p
            priors_pred[pos] = global_rate

    mask = ~np.isnan(full_pred)
    y = modeling_table["outcome_bin"].to_numpy(dtype=float)[mask]
    full = full_pred[mask]
    priors = priors_pred[mask]
    n_pens = modeling_table["shooter_n_pens_before"].to_numpy(dtype=float)[mask]

    overall = {
        "n": int(mask.sum()),
        "full": {
            "log_loss": log_loss(y, full),
            "brier": brier_score(y, full),
            "ece": expected_calibration_error(y, full, n_bins=8),
        },
        "priors_only": {
            "log_loss": log_loss(y, priors),
            "brier": brier_score(y, priors),
            "ece": expected_calibration_error(y, priors, n_bins=8),
        },
    }

    by_bucket = []
    buckets = np.array([n_pens_bucket(v) for v in n_pens])
    for label in BUCKET_ORDER:
        sel = buckets == label
        if sel.sum() == 0:
            continue
        by_bucket.append(
            {
                "bucket": label,
                "n": int(sel.sum()),
                "full_log_loss": log_loss(y[sel], full[sel]),
                "priors_log_loss": log_loss(y[sel], priors[sel]),
                "full_brier": brier_score(y[sel], full[sel]),
                "priors_brier": brier_score(y[sel], priors[sel]),
            }
        )

    return {"group_col": group_col, "overall": overall, "by_n_pens": by_bucket}


def conversion_ranking_correlation(
    df: pd.DataFrame,
    *,
    pred_col: str,
    group_col: str = "shooter_id",
    min_kicks: int = 3,
) -> dict[str, float]:
    """Rank-correlate predicted vs realized conversion per group (ranking sanity)."""
    grouped = df.groupby(group_col).agg(
        pred_mean=(pred_col, "mean"),
        realized=("outcome_bin", "mean"),
        kicks=("outcome_bin", "size"),
    )
    eligible = grouped[grouped["kicks"] >= min_kicks]
    if len(eligible) < 2:
        return {"spearman": float("nan"), "kendall": float("nan"), "n_shooters": int(len(eligible))}
    return {
        "spearman": spearman_corr(eligible["pred_mean"], eligible["realized"]),
        "kendall": kendall_tau(eligible["pred_mean"], eligible["realized"]),
        "n_shooters": int(len(eligible)),
    }
