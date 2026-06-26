"""
Step 7 training/evaluation orchestration.

Provides:
- the legacy direct-model evaluation (kept for the LightGBM cross-check report),
- a full outcome pipeline (combined log-odds model + composed placement path +
  meta-blender + calibration),
- rolling temporal cross-validation with an explicit baseline gate.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.evaluation.metrics import brier_score, log_loss
from src.evaluation.metrics import binary_classification_metrics
from src.evaluation.splitters import temporal_train_test_split
from src.models.blender import MetaBlender, predict_blend, train_meta_blender
from src.models.calibration import ProbabilityCalibrator, select_calibration_method
from src.models.combo import (
    CombinedOutcomeModel,
    predict_combined_proba,
    train_combined_model,
)
from src.models.outcome_lgbm import predict_outcome_proba, train_outcome_model
from src.models.resolution import compose_goal_prob, fit_dive_resolution_table, fit_resolution_table

EPS = 1e-6


def split_train_calibration(
    train_and_calibration: pd.DataFrame,
    *,
    calibration_fraction: float = 0.2,
    date_col: str = "match_date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split pre-cutoff rows into earlier train and later calibration rows."""
    if not 0 < calibration_fraction < 1:
        raise ValueError("calibration_fraction must be between 0 and 1")
    ordered = train_and_calibration.copy()
    ordered["_date"] = pd.to_datetime(ordered[date_col], errors="coerce")
    ordered = ordered.sort_values(["_date", "penalty_id"], kind="mergesort")
    calibration_size = max(1, int(round(len(ordered) * calibration_fraction)))
    calibration_size = min(calibration_size, len(ordered) - 1)
    train = ordered.iloc[:-calibration_size].drop(columns=["_date"]).copy()
    calibration = ordered.iloc[-calibration_size:].drop(columns=["_date"]).copy()
    return train.reset_index(drop=True), calibration.reset_index(drop=True)


# ── Full outcome pipeline ───────────────────────────────────────────────────


def _keeper_concede_prob(df: pd.DataFrame) -> np.ndarray:
    return np.clip(1 - df["keeper_save_rate_shrunk"].to_numpy(dtype=float), EPS, 1 - EPS)


@dataclass
class OutcomePipeline:
    combined_model: CombinedOutcomeModel
    resolution_table: dict[str, float]
    dive_table: dict[tuple[str, str], float] | None
    blender: MetaBlender | None
    calibrator: ProbabilityCalibrator
    global_rate: float
    calibration_method: str


def _fit_diagnostic_blender(
    train_df: pd.DataFrame,
    global_rate: float,
    *,
    blend_C: float,
    resolution_smoothing: float,
    calibration_fraction: float,
) -> MetaBlender | None:
    """Fit a stacker on a held-out slice for the report's blend diagnostic only."""
    core, calib = split_train_calibration(
        train_df, calibration_fraction=calibration_fraction
    )
    calib_y = calib["outcome_bin"].to_numpy(dtype=int)
    if core["outcome_bin"].nunique() < 2 or len(np.unique(calib_y)) < 2:
        return None
    core_model = train_combined_model(core, C=1.0, global_rate=global_rate)
    core_table = fit_resolution_table(core, smoothing=resolution_smoothing)
    components = {
        "combo": predict_combined_proba(core_model, calib),
        "composed": compose_goal_prob(calib, core_table),
        "keeper": _keeper_concede_prob(calib),
    }
    return train_meta_blender(components, calib_y, C=blend_C)


def fit_pipeline(
    train_df: pd.DataFrame,
    *,
    combo_C: float = 1.0,
    blend_C: float = 1.0,
    resolution_smoothing: float = 5.0,
    calibration_fraction: float = 0.25,
    calibration_methods: tuple[str, ...] = ("none", "sigmoid"),
) -> OutcomePipeline:
    """
    Fit the headline calibrated combined log-odds model on the full train set.

    The resolution table and a diagnostic blender are also fit so the report can
    show the composed placement path and the (honest) blend comparison, but the
    headline depends only on the combined model + calibration.
    """
    global_rate = float(np.clip(train_df["outcome_bin"].mean(), EPS, 1 - EPS))

    combined_model = train_combined_model(train_df, C=combo_C, global_rate=global_rate)
    resolution_table = fit_resolution_table(train_df, smoothing=resolution_smoothing)
    dive_table = fit_dive_resolution_table(train_df, smoothing=resolution_smoothing)

    combo_train = predict_combined_proba(combined_model, train_df)
    y_train = train_df["outcome_bin"].to_numpy(dtype=int)
    method = select_calibration_method(combo_train, y_train, methods=calibration_methods)
    calibrator = ProbabilityCalibrator(method=method).fit(combo_train, y_train)

    blender = _fit_diagnostic_blender(
        train_df,
        global_rate,
        blend_C=blend_C,
        resolution_smoothing=resolution_smoothing,
        calibration_fraction=calibration_fraction,
    )

    return OutcomePipeline(
        combined_model=combined_model,
        resolution_table=resolution_table,
        dive_table=dive_table,
        blender=blender,
        calibrator=calibrator,
        global_rate=global_rate,
        calibration_method=method,
    )


def predict_pipeline(pipeline: OutcomePipeline, df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Predict every component plus the calibrated headline probability."""
    combo = predict_combined_proba(pipeline.combined_model, df)
    composed = compose_goal_prob(df, pipeline.resolution_table, dive_table=pipeline.dive_table)
    keeper = _keeper_concede_prob(df)
    combo_calibrated = pipeline.calibrator.predict(combo)

    if pipeline.blender is None:
        blend = combo
    else:
        blend = predict_blend(
            pipeline.blender,
            {"combo": combo, "composed": composed, "keeper": keeper},
        )

    return {
        "combo": combo,
        "composed": composed,
        "keeper_shrunk": keeper,
        "blend": blend,
        "combo_calibrated": combo_calibrated,
    }


# ── Rolling temporal cross-validation ───────────────────────────────────────


def rolling_temporal_folds(
    df: pd.DataFrame,
    *,
    n_folds: int = 5,
    test_fraction: float = 0.5,
    date_col: str = "match_date",
    min_train: int = 50,
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Yield expanding-window (train, test) folds over the most recent rows."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    ordered = df.copy()
    ordered["_date"] = pd.to_datetime(ordered[date_col], errors="coerce")
    ordered = ordered.sort_values(["_date", "penalty_id"], kind="mergesort").reset_index(
        drop=True
    )
    n = len(ordered)
    start = int(n * (1 - test_fraction))
    bounds = np.linspace(start, n, n_folds + 1).astype(int)
    for i in range(n_folds):
        lo, hi = int(bounds[i]), int(bounds[i + 1])
        if lo < min_train or hi - lo < 1:
            continue
        train = ordered.iloc[:lo].drop(columns=["_date"]).reset_index(drop=True)
        test = ordered.iloc[lo:hi].drop(columns=["_date"]).reset_index(drop=True)
        yield train, test


def _summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}


def evaluate_outcome_models(
    modeling_table: pd.DataFrame,
    *,
    n_folds: int = 5,
    test_fraction: float = 0.5,
    headline_model: str = "combo_calibrated",
    combo_C: float = 1.0,
) -> dict[str, object]:
    """Run rolling temporal CV and check the headline model against the baseline."""
    model_names = ["global", "keeper_shrunk", "composed", "combo", "blend", "combo_calibrated"]
    fold_logloss: dict[str, list[float]] = {name: [] for name in model_names}
    fold_brier: dict[str, list[float]] = {name: [] for name in model_names}
    headline_wins = 0
    n_used = 0

    for train, test in rolling_temporal_folds(
        modeling_table,
        n_folds=n_folds,
        test_fraction=test_fraction,
    ):
        if train["outcome_bin"].nunique() < 2:
            continue
        pipeline = fit_pipeline(train, combo_C=combo_C)
        preds = predict_pipeline(pipeline, test)
        preds["global"] = np.repeat(pipeline.global_rate, len(test))
        y_test = test["outcome_bin"].to_numpy(dtype=int)

        for name in model_names:
            fold_logloss[name].append(log_loss(y_test, preds[name]))
            fold_brier[name].append(brier_score(y_test, preds[name]))

        if (
            log_loss(y_test, preds[headline_model])
            < log_loss(y_test, preds["keeper_shrunk"])
        ):
            headline_wins += 1
        n_used += 1

    cv_summary: dict[str, dict[str, float]] = {}
    for name in model_names:
        ll = _summarize(fold_logloss[name])
        br = _summarize(fold_brier[name])
        cv_summary[name] = {
            "log_loss_mean": ll["mean"],
            "log_loss_std": ll["std"],
            "brier_mean": br["mean"],
            "brier_std": br["std"],
        }

    baseline_ll = cv_summary["keeper_shrunk"]["log_loss_mean"]
    baseline_br = cv_summary["keeper_shrunk"]["brier_mean"]
    headline_ll = cv_summary[headline_model]["log_loss_mean"]
    headline_br = cv_summary[headline_model]["brier_mean"]
    brier_skill = 1 - (headline_br / baseline_br) if baseline_br > 0 else float("nan")

    gate = {
        "beats_keeper_shrunk": bool(
            headline_ll < baseline_ll and headline_br < baseline_br
        ),
        "log_loss_improvement": float(baseline_ll - headline_ll),
        "brier_improvement": float(baseline_br - headline_br),
        "brier_skill_score": float(brier_skill),
        "folds_won": headline_wins,
        "folds_total": n_used,
    }

    return {
        "headline_model": headline_model,
        "n_folds": n_used,
        "cv_summary": cv_summary,
        "per_fold": {"log_loss": fold_logloss, "brier": fold_brier},
        "gate": gate,
    }


# ── Legacy direct-model evaluation (LightGBM cross-check) ────────────────────


def evaluate_direct_outcome_model(
    modeling_table: pd.DataFrame,
    *,
    cutoff: str | pd.Timestamp,
    calibration_fraction: float = 0.2,
    calibration_method: str = "sigmoid",
) -> dict[str, object]:
    """Train direct LightGBM model, calibrate on a held-out slice, evaluate on test."""
    pre_cutoff, test = temporal_train_test_split(modeling_table, cutoff=cutoff)
    train, calibration = split_train_calibration(
        pre_cutoff,
        calibration_fraction=calibration_fraction,
    )

    model = train_outcome_model(train, valid_df=calibration)
    calibration_raw = predict_outcome_proba(model, calibration)
    calibrator = ProbabilityCalibrator(method=calibration_method).fit(
        calibration_raw,
        calibration["outcome_bin"].to_numpy(),
    )

    test_raw = predict_outcome_proba(model, test)
    test_calibrated = calibrator.predict(test_raw)

    baseline_prob = (
        test["pred_keeper_shrunk"]
        if "pred_keeper_shrunk" in test.columns
        else pd.Series([test["outcome_bin"].mean()] * len(test), index=test.index)
    )
    y_test = test["outcome_bin"]

    return {
        "split": {
            "train_rows": len(train),
            "calibration_rows": len(calibration),
            "test_rows": len(test),
            "cutoff": str(pd.Timestamp(cutoff).date()),
            "calibration_method": calibration_method,
        },
        "model": {
            "backend": model.backend,
            "features": model.feature_columns,
            "best_iteration": model.best_iteration,
        },
        "metrics": {
            "keeper_shrunk_baseline": binary_classification_metrics(
                y_true=y_test,
                y_prob=baseline_prob,
                baseline_prob=baseline_prob,
            ),
            "direct": binary_classification_metrics(
                y_true=y_test,
                y_prob=test_raw,
                baseline_prob=baseline_prob,
            ),
            "direct_calibrated": binary_classification_metrics(
                y_true=y_test,
                y_prob=test_calibrated,
                baseline_prob=baseline_prob,
            ),
        },
    }
