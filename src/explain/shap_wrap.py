"""SHAP explanation wrapper for the CombinedOutcomeModel (LogisticRegression)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from src.models.combo import CombinedOutcomeModel, build_combo_features
from src.serving.predict import KeeperProfile, ShooterProfile, build_feature_row

FEATURE_LABELS: dict[str, str] = {
    "shooter_logodds_offset": "Shooter conversion record",
    "keeper_concede_logodds_offset": "Keeper vulnerability",
    "shooter_zone_entropy": "Shooter placement variety",
    "is_shootout": "Shootout context",
    "log1p_shooter_n_pens": "Shooter sample size",
    "log1p_keeper_n_faced": "Keeper sample size",
}


def build_explainer(
    model: CombinedOutcomeModel,
    modeling_table: pd.DataFrame,
) -> shap.LinearExplainer:
    """Build a LinearExplainer using modeling_table rows as background distribution."""
    X = build_combo_features(
        modeling_table,
        global_rate=model.global_rate,
        features=model.feature_names,
    )
    return shap.LinearExplainer(model.estimator, X)


def explain_matchup(
    explainer: shap.LinearExplainer,
    shooter: ShooterProfile,
    keeper: KeeperProfile,
    pipeline,
) -> list[dict]:
    """
    Compute SHAP factors for a shooter×keeper matchup.

    Returns list of factor dicts sorted by abs(shap_value) descending.
    SHAP values are in probability space: +0.04 means +4 percentage points to P(goal).
    """
    model: CombinedOutcomeModel = pipeline.combined_model
    feature_row = build_feature_row(shooter, keeper, pipeline)
    X = build_combo_features(
        feature_row,
        global_rate=model.global_rate,
        features=model.feature_names,
    )

    explanation = explainer(X)
    shap_vals = explanation.values[0]  # shape (n_features,) for binary classification

    total_abs = float(np.sum(np.abs(shap_vals)))
    factors: list[dict] = []
    for feat_name, sv in zip(model.feature_names, shap_vals):
        sv = float(sv)
        factors.append({
            "feature": feat_name,
            "label": FEATURE_LABELS.get(feat_name, feat_name),
            "shap_value": round(sv, 4),
            "direction": "positive" if sv > 0.001 else "negative" if sv < -0.001 else "neutral",
            "pct_impact": round(abs(sv) / total_abs * 100, 1) if total_abs > 0 else 0.0,
        })

    factors.sort(key=lambda f: abs(f["shap_value"]), reverse=True)
    return factors
