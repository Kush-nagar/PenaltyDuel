"""
Probability calibration for Step 7 outcome predictions.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

EPS = 1e-6


class ProbabilityCalibrator:
    """Calibrate raw probabilities with identity, Platt/sigmoid, or isotonic mapping."""

    def __init__(self, method: str = "sigmoid") -> None:
        if method not in {"none", "sigmoid", "isotonic"}:
            raise ValueError("method must be 'none', 'sigmoid', or 'isotonic'")
        self.method = method
        self._model: LogisticRegression | IsotonicRegression | None = None
        self._constant: float | None = None

    def fit(self, raw_prob: np.ndarray, y_true: np.ndarray) -> "ProbabilityCalibrator":
        raw = np.asarray(raw_prob, dtype=float).reshape(-1)
        target = np.asarray(y_true, dtype=int).reshape(-1)
        if raw.shape != target.shape:
            raise ValueError("raw_prob and y_true must have the same shape")
        if raw.size == 0:
            raise ValueError("cannot calibrate an empty array")

        raw = np.clip(raw, EPS, 1 - EPS)
        if self.method == "none":
            return self

        unique_targets = np.unique(target)
        if len(unique_targets) == 1:
            self._constant = float(unique_targets[0])
            return self

        if self.method == "sigmoid":
            model = LogisticRegression(C=1.0, solver="lbfgs")
            model.fit(raw.reshape(-1, 1), target)
            self._model = model
        else:
            model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            model.fit(raw, target)
            self._model = model
        return self

    def predict(self, raw_prob: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw_prob, dtype=float).reshape(-1)
        raw = np.clip(raw, EPS, 1 - EPS)
        if self.method == "none":
            return raw
        if self._constant is not None:
            return np.clip(np.repeat(self._constant, len(raw)), EPS, 1 - EPS)
        if self._model is None:
            raise RuntimeError("calibrator must be fitted before predict")

        if self.method == "sigmoid":
            calibrated = self._model.predict_proba(raw.reshape(-1, 1))[:, 1]
        else:
            calibrated = self._model.predict(raw)
        return np.clip(np.asarray(calibrated, dtype=float), EPS, 1 - EPS)


def select_calibration_method(
    raw_prob: np.ndarray,
    y_true: np.ndarray,
    *,
    methods: tuple[str, ...] = ("none", "sigmoid"),
    n_splits: int = 5,
) -> str:
    """
    Pick the calibration method with the lowest out-of-fold log loss.

    Scoring is done with internal stratified K-fold CV, not in-sample, because
    in-sample scoring always favours isotonic (it can memorise the slice) and
    then generalises terribly on small calibration sets. "none" guards against
    calibration that would hurt an already well-calibrated model.
    """
    from sklearn.model_selection import StratifiedKFold

    from src.evaluation.metrics import log_loss

    raw = np.asarray(raw_prob, dtype=float).reshape(-1)
    target = np.asarray(y_true, dtype=int).reshape(-1)

    class_counts = np.bincount(target, minlength=2)
    usable_splits = int(min(n_splits, class_counts.min()))
    if usable_splits < 2:
        return "none"

    splitter = StratifiedKFold(n_splits=usable_splits, shuffle=True, random_state=0)
    best_method = "none"
    best_loss = float("inf")
    for method in methods:
        fold_losses: list[float] = []
        for train_idx, test_idx in splitter.split(raw, target):
            if len(np.unique(target[train_idx])) < 2:
                continue
            calibrator = ProbabilityCalibrator(method=method).fit(
                raw[train_idx], target[train_idx]
            )
            fold_losses.append(
                log_loss(target[test_idx], calibrator.predict(raw[test_idx]))
            )
        if not fold_losses:
            continue
        loss = float(np.mean(fold_losses))
        if loss < best_loss - 1e-9:
            best_loss = loss
            best_method = method
    return best_method
