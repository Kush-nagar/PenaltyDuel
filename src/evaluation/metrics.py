"""
Probability metrics for binary penalty outcome evaluation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd


EPSILON = 1e-15


def _arrays(
    y_true: Sequence[int | float],
    y_prob: Sequence[int | float],
) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true, dtype=float)
    prob = np.asarray(y_prob, dtype=float)
    if true.shape != prob.shape:
        raise ValueError("y_true and y_prob must have the same shape")
    mask = ~np.isnan(true) & ~np.isnan(prob)
    true = true[mask]
    prob = prob[mask]
    if true.size == 0:
        raise ValueError("no valid observations")
    return true, np.clip(prob, EPSILON, 1 - EPSILON)


def log_loss(y_true: Sequence[int | float], y_prob: Sequence[int | float]) -> float:
    """Binary log loss with probability clipping."""
    true, prob = _arrays(y_true, y_prob)
    losses = true * np.log(prob) + (1 - true) * np.log(1 - prob)
    return float(-np.mean(losses))


def brier_score(y_true: Sequence[int | float], y_prob: Sequence[int | float]) -> float:
    """Mean squared error of predicted probabilities."""
    true, prob = _arrays(y_true, y_prob)
    return float(np.mean((prob - true) ** 2))


def brier_skill_score(
    y_true: Sequence[int | float],
    y_prob: Sequence[int | float],
    baseline_prob: Sequence[int | float],
) -> float:
    """Brier Skill Score relative to a baseline forecast."""
    model_brier = brier_score(y_true, y_prob)
    baseline_brier = brier_score(y_true, baseline_prob)
    if baseline_brier == 0:
        return float("nan")
    return float(1 - model_brier / baseline_brier)


def roc_auc(y_true: Sequence[int | float], y_prob: Sequence[int | float]) -> float:
    """ROC-AUC via average positive-vs-negative ranking."""
    true, prob = _arrays(y_true, y_prob)
    positives = prob[true == 1]
    negatives = prob[true == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return float("nan")

    wins = 0.0
    for pos_score in positives:
        wins += float(np.sum(pos_score > negatives))
        wins += 0.5 * float(np.sum(pos_score == negatives))
    return float(wins / (len(positives) * len(negatives)))


def pr_auc(y_true: Sequence[int | float], y_prob: Sequence[int | float]) -> float:
    """Area under the precision-recall curve using trapezoids over recall."""
    true, prob = _arrays(y_true, y_prob)
    if np.sum(true == 1) == 0:
        return float("nan")

    order = np.argsort(-prob, kind="mergesort")
    sorted_true = true[order]
    positives = float(np.sum(sorted_true == 1))

    recalls = [0.0]
    precisions = [1.0]
    tp = 0.0
    fp = 0.0
    for label in sorted_true:
        if label == 1:
            tp += 1
        else:
            fp += 1
        recalls.append(tp / positives)
        precisions.append(tp / (tp + fp))

    return float(np.trapezoid(precisions, recalls))


def reliability_table(
    y_true: Sequence[int | float],
    y_prob: Sequence[int | float],
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Calibration bins with mean prediction and observed conversion rate."""
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    true, prob = _arrays(y_true, y_prob)
    raw_bins = np.floor(prob * n_bins).astype(int)
    bins = np.clip(raw_bins, 0, n_bins - 1)

    rows = []
    for bin_idx in range(n_bins):
        mask = bins == bin_idx
        if not np.any(mask):
            continue
        rows.append(
            {
                "bin": bin_idx,
                "bin_low": bin_idx / n_bins,
                "bin_high": (bin_idx + 1) / n_bins,
                "count": int(np.sum(mask)),
                "mean_predicted": float(np.mean(prob[mask])),
                "observed_rate": float(np.mean(true[mask])),
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(
    y_true: Sequence[int | float],
    y_prob: Sequence[int | float],
    *,
    n_bins: int = 10,
) -> float:
    """Weighted mean absolute calibration gap."""
    table = reliability_table(y_true, y_prob, n_bins=n_bins)
    total = table["count"].sum()
    if total == 0:
        return float("nan")
    weighted_gap = (
        table["count"]
        * (table["mean_predicted"] - table["observed_rate"]).abs()
    ).sum()
    return float(weighted_gap / total)


def accuracy_at_threshold(
    y_true: Sequence[int | float],
    y_prob: Sequence[int | float],
    *,
    threshold: float = 0.5,
) -> float:
    """Accuracy footnote for completeness; not a primary metric."""
    true, prob = _arrays(y_true, y_prob)
    predicted = prob >= threshold
    return float(np.mean(predicted == true))


def binary_classification_metrics(
    *,
    y_true: Sequence[int | float],
    y_prob: Sequence[int | float],
    baseline_prob: Sequence[int | float] | None = None,
    n_bins: int = 10,
) -> dict[str, float]:
    """Step 6 metric bundle for binary penalty outcome probabilities."""
    true, prob = _arrays(y_true, y_prob)
    if baseline_prob is None:
        baseline = np.repeat(float(np.mean(true)), len(true))
    else:
        _, baseline = _arrays(y_true, baseline_prob)

    return {
        "log_loss": log_loss(true, prob),
        "brier": brier_score(true, prob),
        "brier_skill_score": brier_skill_score(true, prob, baseline),
        "roc_auc": roc_auc(true, prob),
        "pr_auc": pr_auc(true, prob),
        "ece": expected_calibration_error(true, prob, n_bins=n_bins),
        "accuracy_footnote": accuracy_at_threshold(true, prob),
    }


def format_metric(value: float, digits: int = 4) -> str:
    """Format metric values for markdown reports."""
    if value is None or math.isnan(float(value)):
        return "n/a"
    return f"{float(value):.{digits}f}"
