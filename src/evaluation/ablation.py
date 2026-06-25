"""
Ablation suite for Step 8.

Toggle one feature group at a time in the combined log-odds model and report the
change in log loss / Brier / ECE under rolling temporal CV. A "shrinkage off"
variant swaps the shrunk rates for raw as-of rates to show shrinkage earns its
keep on sparse players.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    brier_score,
    expected_calibration_error,
    log_loss,
)
from src.models.combo import predict_combined_proba, train_combined_model
from src.models.training import rolling_temporal_folds

EPS = 1e-6

_SHOOTER_FEATURES = ["shooter_logodds_offset", "shooter_zone_entropy", "log1p_shooter_n_pens"]
_KEEPER_FEATURES = ["keeper_concede_logodds_offset", "log1p_keeper_n_faced"]
_PRESSURE_FEATURES = ["is_shootout"]
_ALL = _SHOOTER_FEATURES + _KEEPER_FEATURES + _PRESSURE_FEATURES

ABLATION_VARIANTS: dict[str, list[str]] = {
    "full": _ALL,
    "minus_keeper": _SHOOTER_FEATURES + _PRESSURE_FEATURES,
    "minus_shooter": _KEEPER_FEATURES + _PRESSURE_FEATURES,
    "minus_pressure": _SHOOTER_FEATURES + _KEEPER_FEATURES,
}


def _cv_metrics(
    modeling_table: pd.DataFrame,
    feature_names: list[str],
    *,
    n_folds: int,
    test_fraction: float,
    combo_C: float,
) -> dict[str, float]:
    losses, briers, eces = [], [], []
    for train, test in rolling_temporal_folds(
        modeling_table, n_folds=n_folds, test_fraction=test_fraction
    ):
        if train["outcome_bin"].nunique() < 2:
            continue
        global_rate = float(np.clip(train["outcome_bin"].mean(), EPS, 1 - EPS))
        model = train_combined_model(
            train, C=combo_C, global_rate=global_rate, feature_names=feature_names
        )
        preds = predict_combined_proba(model, test)
        y = test["outcome_bin"].to_numpy(dtype=float)
        losses.append(log_loss(y, preds))
        briers.append(brier_score(y, preds))
        eces.append(expected_calibration_error(y, preds, n_bins=8))
    return {
        "log_loss": float(np.mean(losses)),
        "brier": float(np.mean(briers)),
        "ece": float(np.mean(eces)),
    }


def _shrinkage_off_table(modeling_table: pd.DataFrame) -> pd.DataFrame:
    """Swap shrunk rates for raw as-of rates (global fallback when no history)."""
    table = modeling_table.copy()
    global_rate = float(table["outcome_bin"].mean())
    table["shooter_conv_rate_shrunk"] = table["shooter_conv_rate_raw_before"].fillna(
        global_rate
    )
    table["keeper_save_rate_shrunk"] = table["keeper_save_rate_raw_before"].fillna(
        1 - global_rate
    )
    return table


def run_ablation(
    modeling_table: pd.DataFrame,
    *,
    n_folds: int = 5,
    test_fraction: float = 0.5,
    combo_C: float = 1.0,
) -> pd.DataFrame:
    """Run every ablation variant and return a delta table vs the full model."""
    rows: list[dict[str, float | str]] = []

    variant_metrics: dict[str, dict[str, float]] = {}
    for variant, features in ABLATION_VARIANTS.items():
        variant_metrics[variant] = _cv_metrics(
            modeling_table,
            features,
            n_folds=n_folds,
            test_fraction=test_fraction,
            combo_C=combo_C,
        )

    variant_metrics["shrinkage_off"] = _cv_metrics(
        _shrinkage_off_table(modeling_table),
        ABLATION_VARIANTS["full"],
        n_folds=n_folds,
        test_fraction=test_fraction,
        combo_C=combo_C,
    )

    full = variant_metrics["full"]
    order = ["full", "minus_keeper", "minus_shooter", "minus_pressure", "shrinkage_off"]
    for variant in order:
        m = variant_metrics[variant]
        rows.append(
            {
                "variant": variant,
                "log_loss": m["log_loss"],
                "brier": m["brier"],
                "ece": m["ece"],
                "d_log_loss": m["log_loss"] - full["log_loss"],
                "d_brier": m["brier"] - full["brier"],
                "d_ece": m["ece"] - full["ece"],
            }
        )
    return pd.DataFrame(rows)
