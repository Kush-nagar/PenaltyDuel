"""
Direct Step 7 outcome model.

LightGBM is the primary backend. A regularized logistic fallback keeps tests and
development usable if LightGBM is unavailable in a future environment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


OUTCOME_FEATURE_COLUMNS = [
    "pred_keeper_shrunk",
    "pred_shooter_shrunk",
    "shooter_conv_rate_shrunk",
    "shooter_n_pens_before",
    "keeper_save_rate_shrunk",
    "keeper_n_faced_before",
    "shooter_zone_entropy",
    "shooter_zone_prob_low_left",
    "shooter_zone_prob_low_center",
    "shooter_zone_prob_low_right",
    "shooter_zone_prob_high_left",
    "shooter_zone_prob_high_center",
    "shooter_zone_prob_high_right",
    "expected_guess_correct",
    "is_shootout",
    "shootout_kick_index",
    "shooter_foot_is_right",
    "shooter_preferred_foot_is_missing",
    "shooter_height_cm",
    "keeper_foot_is_right",
    "keeper_preferred_foot_is_missing",
    "keeper_height_cm",
    "keeper_dive_left_prob",
    "keeper_dive_center_prob",
    "keeper_dive_right_prob",
    "keeper_dive_entropy",
]


@dataclass
class OutcomeModel:
    estimator: object
    feature_columns: list[str]
    backend: str
    best_iteration: int | None = None


def select_outcome_features(
    modeling_table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return numeric feature matrix and binary target."""
    missing = [col for col in OUTCOME_FEATURE_COLUMNS if col not in modeling_table.columns]
    if missing:
        raise KeyError(f"missing outcome feature columns: {missing}")
    if "outcome_bin" not in modeling_table.columns:
        raise KeyError("outcome_bin")

    X = modeling_table[OUTCOME_FEATURE_COLUMNS].copy()
    X["is_shootout"] = X["is_shootout"].astype(float)
    X["shootout_kick_index"] = X["shootout_kick_index"].fillna(0).astype(float)
    X = X.fillna(0).astype(float)
    y = modeling_table["outcome_bin"].astype(int)
    return X, y


def monotone_constraints_for_features(feature_columns: list[str]) -> list[int]:
    """LightGBM monotonic constraints aligned with feature semantics."""
    constraints = []
    for feature in feature_columns:
        if feature == "shooter_conv_rate_shrunk":
            constraints.append(1)
        elif feature == "keeper_save_rate_shrunk":
            constraints.append(-1)
        else:
            constraints.append(0)
    return constraints


def _train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame | None,
    y_valid: pd.Series | None,
) -> OutcomeModel:
    import lightgbm as lgb

    estimator = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        num_leaves=7,
        min_child_samples=30,
        reg_alpha=0.1,
        reg_lambda=1.0,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        verbosity=-1,
        monotone_constraints=monotone_constraints_for_features(list(X_train.columns)),
    )
    fit_kwargs = {}
    if X_valid is not None and y_valid is not None and len(X_valid) > 0:
        fit_kwargs["eval_set"] = [(X_valid, y_valid)]
        fit_kwargs["eval_metric"] = "binary_logloss"
        fit_kwargs["callbacks"] = [
            lgb.early_stopping(stopping_rounds=20, verbose=False),
        ]
    estimator.fit(X_train, y_train, **fit_kwargs)
    best_iteration = getattr(estimator, "best_iteration_", None)
    return OutcomeModel(
        estimator=estimator,
        feature_columns=list(X_train.columns),
        backend="lightgbm",
        best_iteration=best_iteration,
    )


def _train_logistic(X_train: pd.DataFrame, y_train: pd.Series) -> OutcomeModel:
    estimator = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    estimator.fit(X_train, y_train)
    return OutcomeModel(
        estimator=estimator,
        feature_columns=list(X_train.columns),
        backend="logistic",
    )


def train_outcome_model(
    train_df: pd.DataFrame,
    *,
    valid_df: pd.DataFrame | None = None,
) -> OutcomeModel:
    """Train the direct outcome model on leak-safe Step 5 features."""
    X_train, y_train = select_outcome_features(train_df)
    X_valid = y_valid = None
    if valid_df is not None and len(valid_df) > 0:
        X_valid, y_valid = select_outcome_features(valid_df)

    if y_train.nunique() < 2:
        raise ValueError("training data must contain both outcome classes")

    try:
        return _train_lightgbm(X_train, y_train, X_valid, y_valid)
    except ImportError:
        return _train_logistic(X_train, y_train)


def predict_outcome_proba(model: OutcomeModel, modeling_table: pd.DataFrame) -> np.ndarray:
    """Predict goal probabilities for a Step 5 modeling table."""
    X, _ = select_outcome_features(modeling_table)
    X = X[model.feature_columns]
    probs = model.estimator.predict_proba(X)[:, 1]
    return np.clip(np.asarray(probs, dtype=float), 1e-6, 1 - 1e-6)
