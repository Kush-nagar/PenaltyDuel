"""
Meta-blender for Step 7.

A small logistic regression over component probabilities (combined model,
composed placement model, keeper marginal) in log-odds space. Components are
always stacked in a fixed, name-sorted column order so prediction is invariant
to dict insertion order.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    arr = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(arr / (1 - arr))


def _stack(component_probs: Mapping[str, np.ndarray], names: list[str]) -> np.ndarray:
    columns = [_logit(component_probs[name]) for name in names]
    return np.column_stack(columns)


@dataclass
class MetaBlender:
    estimator: LogisticRegression
    component_names: list[str]


def train_meta_blender(
    component_probs: Mapping[str, np.ndarray],
    y: np.ndarray,
    *,
    C: float = 1.0,
) -> MetaBlender:
    """Fit a logistic stacker over the supplied component probabilities."""
    if not component_probs:
        raise ValueError("component_probs must not be empty")
    names = sorted(component_probs)
    X = _stack(component_probs, names)
    target = np.asarray(y, dtype=int)
    if len(np.unique(target)) < 2:
        raise ValueError("blender target must contain both classes")

    estimator = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
    estimator.fit(X, target)
    return MetaBlender(estimator=estimator, component_names=names)


def predict_blend(
    blender: MetaBlender,
    component_probs: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Predict the blended probability for the supplied components."""
    missing = [name for name in blender.component_names if name not in component_probs]
    if missing:
        raise KeyError(f"missing blend components: {missing}")
    X = _stack(component_probs, blender.component_names)
    probs = blender.estimator.predict_proba(X)[:, 1]
    return np.clip(np.asarray(probs, dtype=float), EPS, 1 - EPS)
