"""
Placement (shot-zone) evaluation for Step 8.

Treats the as-of shooter zone distribution (Step 5 Dirichlet-shrunk feature) as a
6-class placement forecast and scores it against the realized `shot_zone` on the
temporal test split. Compared with a uniform forecast and the train marginal zone
frequency. Dive placement stays out of scope (0% `keeper_dive_direction` coverage).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.metrics import macro_f1, multiclass_log_loss, top_k_accuracy
from src.evaluation.splitters import temporal_train_test_split
from src.features.as_of import SHOT_ZONES

EPS = 1e-6


def _zone_matrix(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [df[f"shooter_zone_prob_{zone}"].to_numpy(dtype=float) for zone in SHOT_ZONES]
    )


def _placement_metrics(y_true, probs: np.ndarray) -> dict[str, float]:
    pred_labels = [SHOT_ZONES[i] for i in np.argmax(probs, axis=1)]
    return {
        "log_loss": multiclass_log_loss(y_true, probs, SHOT_ZONES),
        "top1": top_k_accuracy(y_true, probs, SHOT_ZONES, k=1),
        "top2": top_k_accuracy(y_true, probs, SHOT_ZONES, k=2),
        "macro_f1": macro_f1(y_true, pred_labels, SHOT_ZONES),
    }


def evaluate_placement(
    modeling_table: pd.DataFrame,
    *,
    cutoff,
) -> dict[str, object]:
    """Score the as-of placement forecast against uniform and marginal baselines."""
    train, test = temporal_train_test_split(modeling_table, cutoff=cutoff)

    if "shot_zone_is_missing" in test.columns:
        test = test.loc[~test["shot_zone_is_missing"].astype(bool)]
    test = test.loc[test["shot_zone"].isin(SHOT_ZONES)].reset_index(drop=True)
    if test.empty:
        raise ValueError("no test rows with a known shot zone")

    y_true = test["shot_zone"].tolist()
    n = len(test)

    model_probs = _zone_matrix(test)

    uniform_probs = np.full((n, len(SHOT_ZONES)), 1 / len(SHOT_ZONES))

    train_known = train.loc[train["shot_zone"].isin(SHOT_ZONES)]
    counts = train_known["shot_zone"].value_counts()
    marginal = np.array([counts.get(zone, 0) + 1.0 for zone in SHOT_ZONES])
    marginal = marginal / marginal.sum()
    marginal_probs = np.tile(marginal, (n, 1))

    return {
        "n_test": n,
        "model": _placement_metrics(y_true, model_probs),
        "uniform": _placement_metrics(y_true, uniform_probs),
        "marginal": _placement_metrics(y_true, marginal_probs),
    }
