import pandas as pd
import pytest

from src.models.outcome_lgbm import (
    OUTCOME_FEATURE_COLUMNS,
    monotone_constraints_for_features,
    predict_outcome_proba,
    select_outcome_features,
    train_outcome_model,
)


def modeling_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "outcome_bin": [0, 0, 1, 1, 0, 1],
            "pred_keeper_shrunk": [0.3, 0.4, 0.8, 0.9, 0.35, 0.75],
            "pred_shooter_shrunk": [0.2, 0.3, 0.7, 0.8, 0.35, 0.75],
            "shooter_conv_rate_shrunk": [0.2, 0.3, 0.7, 0.8, 0.35, 0.75],
            "shooter_n_pens_before": [4, 5, 6, 7, 3, 8],
            "keeper_save_rate_shrunk": [0.7, 0.6, 0.2, 0.1, 0.65, 0.25],
            "keeper_n_faced_before": [5, 4, 7, 8, 3, 6],
            "shooter_zone_entropy": [0.8, 0.7, 0.4, 0.3, 0.75, 0.35],
            "shooter_zone_prob_low_left": [0.2, 0.2, 0.4, 0.5, 0.2, 0.45],
            "shooter_zone_prob_low_center": [0.1, 0.1, 0.2, 0.1, 0.1, 0.2],
            "shooter_zone_prob_low_right": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
            "shooter_zone_prob_high_left": [0.2, 0.2, 0.1, 0.1, 0.2, 0.1],
            "shooter_zone_prob_high_center": [0.1, 0.1, 0.05, 0.05, 0.1, 0.03],
            "shooter_zone_prob_high_right": [0.2, 0.2, 0.05, 0.05, 0.2, 0.02],
            "expected_guess_correct": [0.5, 0.4, 0.2, 0.1, 0.45, 0.15],
            "is_shootout": [False, True, False, True, False, True],
            "shootout_kick_index": [None, 2, None, 6, None, 7],
        }
    )


def test_select_outcome_features_returns_numeric_matrix_and_target():
    X, y = select_outcome_features(modeling_rows())

    assert list(X.columns) == OUTCOME_FEATURE_COLUMNS
    assert y.tolist() == [0, 0, 1, 1, 0, 1]
    assert X.isna().sum().sum() == 0


def test_monotone_constraints_match_feature_semantics():
    constraints = monotone_constraints_for_features(OUTCOME_FEATURE_COLUMNS)

    assert constraints[OUTCOME_FEATURE_COLUMNS.index("shooter_conv_rate_shrunk")] == 1
    assert constraints[OUTCOME_FEATURE_COLUMNS.index("keeper_save_rate_shrunk")] == -1
    assert len(constraints) == len(OUTCOME_FEATURE_COLUMNS)


def test_train_outcome_model_predicts_probabilities():
    rows = modeling_rows()

    model = train_outcome_model(rows.iloc[:4], valid_df=rows.iloc[4:])
    probs = predict_outcome_proba(model, rows)

    assert len(probs) == len(rows)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0
