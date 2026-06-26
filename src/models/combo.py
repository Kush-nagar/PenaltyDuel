"""
Combined log-odds outcome model (Step 7 primary headline model).

This model combines the shooter and keeper shrunk rates in log-odds space, the
principled way to merge two marginal probabilities (a learned Bradley-Terry /
"log5" style combination). On sparse, noisy penalty data this smooth combination
is the model that consistently beats the shrunk-marginal baseline, while the
gradient-boosted tree (see `outcome_lgbm.py`) is kept as an overfitting
cross-check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

EPS = 1e-6

COMBO_FEATURE_NAMES = [
    "shooter_logodds_offset",
    "keeper_concede_logodds_offset",
    "shooter_zone_entropy",
    "is_shootout",
    "log1p_shooter_n_pens",
    "log1p_keeper_n_faced",
]


def _logit(p: np.ndarray | pd.Series) -> np.ndarray:
    arr = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(arr / (1 - arr))


def build_combo_features(
    df: pd.DataFrame,
    *,
    global_rate: float,
    features: list[str] | None = None,
) -> np.ndarray:
    """Construct the log-odds feature matrix, optionally restricted to a subset."""
    required = [
        "shooter_conv_rate_shrunk",
        "keeper_save_rate_shrunk",
        "shooter_zone_entropy",
        "is_shootout",
        "shooter_n_pens_before",
        "keeper_n_faced_before",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"missing combo feature columns: {missing}")

    def _col(name: str, default: float = 0.0) -> np.ndarray:
        if name in df.columns:
            return df[name].fillna(default).to_numpy(dtype=float)
        return np.full(len(df), default, dtype=float)

    global_logit = _logit(np.array([global_rate]))[0]
    columns = {
        "shooter_logodds_offset": _logit(df["shooter_conv_rate_shrunk"]) - global_logit,
        "keeper_concede_logodds_offset": _logit(1 - df["keeper_save_rate_shrunk"].to_numpy())
        - global_logit,
        "shooter_zone_entropy": df["shooter_zone_entropy"].fillna(0).to_numpy(dtype=float),
        "is_shootout": df["is_shootout"].astype(float).to_numpy(),
        "log1p_shooter_n_pens": np.log1p(
            df["shooter_n_pens_before"].fillna(0).to_numpy(dtype=float)
        ),
        "log1p_keeper_n_faced": np.log1p(
            df["keeper_n_faced_before"].fillna(0).to_numpy(dtype=float)
        ),
        "shooter_foot_is_right": _col("shooter_foot_is_right"),
        "shooter_foot_missing": _col("shooter_preferred_foot_is_missing"),
        "shooter_height_cm": _col("shooter_height_cm"),
        "keeper_foot_is_right": _col("keeper_foot_is_right"),
        "keeper_foot_missing": _col("keeper_preferred_foot_is_missing"),
        "keeper_height_cm": _col("keeper_height_cm"),
    }
    names = features if features is not None else COMBO_FEATURE_NAMES
    unknown = [name for name in names if name not in columns]
    if unknown:
        raise KeyError(f"unknown combo features: {unknown}")
    return np.column_stack([columns[name] for name in names])


@dataclass
class CombinedOutcomeModel:
    estimator: LogisticRegression
    global_rate: float
    feature_names: list[str]
    backend: str = "logodds_logistic"


def train_combined_model(
    train_df: pd.DataFrame,
    *,
    C: float = 1.0,
    global_rate: float | None = None,
    feature_names: list[str] | None = None,
) -> CombinedOutcomeModel:
    """Fit the regularized log-odds logistic combination."""
    if "outcome_bin" not in train_df.columns:
        raise KeyError("outcome_bin")
    if global_rate is None:
        global_rate = float(train_df["outcome_bin"].mean())
    global_rate = float(np.clip(global_rate, EPS, 1 - EPS))

    y = train_df["outcome_bin"].astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        raise ValueError("training data must contain both outcome classes")

    names = list(feature_names) if feature_names is not None else list(COMBO_FEATURE_NAMES)
    if not names:
        raise ValueError("feature_names must not be empty")
    X = build_combo_features(train_df, global_rate=global_rate, features=names)
    estimator = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
    estimator.fit(X, y)
    return CombinedOutcomeModel(
        estimator=estimator,
        global_rate=global_rate,
        feature_names=names,
    )


def predict_combined_proba(model: CombinedOutcomeModel, df: pd.DataFrame) -> np.ndarray:
    """Predict goal probability from the combined model."""
    X = build_combo_features(df, global_rate=model.global_rate, features=model.feature_names)
    probs = model.estimator.predict_proba(X)[:, 1]
    return np.clip(np.asarray(probs, dtype=float), EPS, 1 - EPS)
