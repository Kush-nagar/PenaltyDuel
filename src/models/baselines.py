"""
Baseline probability models for Step 6.

These baselines are deliberately simple and leak-safe when fed the Step 5
modeling table: raw and shrunk player rates come from as-of history features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.metrics import binary_classification_metrics


BASELINE_COLUMNS = {
    "global": "pred_global",
    "shooter_raw": "pred_shooter_raw",
    "shooter_shrunk": "pred_shooter_shrunk",
    "keeper_raw": "pred_keeper_raw",
    "keeper_shrunk": "pred_keeper_shrunk",
}


def add_baseline_predictions(
    modeling_table: pd.DataFrame,
    *,
    global_rate: float | None = None,
) -> pd.DataFrame:
    """Add global, shooter-only, and keeper-only baseline probabilities."""
    table = modeling_table.copy()
    if global_rate is None:
        global_rate = float(table["outcome_bin"].mean())
    global_rate = float(np.clip(global_rate, 1e-6, 1 - 1e-6))

    table["pred_global"] = global_rate
    table["pred_shooter_raw"] = table["shooter_conv_rate_raw_before"]
    table["pred_shooter_shrunk"] = table["shooter_conv_rate_shrunk"]
    table["pred_keeper_raw"] = 1 - table["keeper_save_rate_raw_before"]
    table["pred_keeper_shrunk"] = 1 - table["keeper_save_rate_shrunk"]
    return table


def _prediction_for_evaluation(table: pd.DataFrame, pred_col: str) -> pd.Series:
    """Use global fallback for raw baselines when no player history exists."""
    return table[pred_col].fillna(table["pred_global"]).clip(1e-6, 1 - 1e-6)


def evaluate_baselines(
    table_with_predictions: pd.DataFrame,
    *,
    baseline_for_skill: str = "pred_global",
    n_bins: int = 10,
) -> pd.DataFrame:
    """Return one metric row per Step 6 baseline."""
    rows: list[dict[str, float | str]] = []
    y_true = table_with_predictions["outcome_bin"]
    skill_baseline = _prediction_for_evaluation(
        table_with_predictions,
        baseline_for_skill,
    )

    for baseline_name, pred_col in BASELINE_COLUMNS.items():
        y_prob = _prediction_for_evaluation(table_with_predictions, pred_col)
        metrics = binary_classification_metrics(
            y_true=y_true,
            y_prob=y_prob,
            baseline_prob=skill_baseline,
            n_bins=n_bins,
        )
        rows.append({"baseline": baseline_name, **metrics})

    return pd.DataFrame(rows)
