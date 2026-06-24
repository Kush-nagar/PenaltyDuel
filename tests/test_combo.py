import numpy as np
import pandas as pd

from src.models.combo import (
    COMBO_FEATURE_NAMES,
    build_combo_features,
    predict_combined_proba,
    train_combined_model,
)


def synthetic_rows(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        strong = i % 2 == 0
        shooter_rate = 0.85 if strong else 0.45
        keeper_save = 0.15 if strong else 0.45
        outcome = 1 if strong else 0
        rows.append(
            {
                "outcome_bin": outcome,
                "shooter_conv_rate_shrunk": shooter_rate,
                "keeper_save_rate_shrunk": keeper_save,
                "shooter_zone_entropy": 0.4 if strong else 0.8,
                "is_shootout": bool(i % 3 == 0),
                "shooter_n_pens_before": i % 7,
                "keeper_n_faced_before": i % 5,
            }
        )
    return pd.DataFrame(rows)


def test_build_combo_features_returns_named_matrix():
    X = build_combo_features(synthetic_rows(), global_rate=0.74)

    assert X.shape == (60, len(COMBO_FEATURE_NAMES))
    assert not np.isnan(X).any()


def test_combined_model_predicts_unit_interval():
    df = synthetic_rows()

    model = train_combined_model(df)
    probs = predict_combined_proba(model, df)

    assert probs.shape == (len(df),)
    assert np.all(probs >= 0) and np.all(probs <= 1)


def test_combined_model_ranks_strong_matchup_above_weak():
    df = synthetic_rows()
    model = train_combined_model(df)

    strong = pd.DataFrame(
        [
            {
                "shooter_conv_rate_shrunk": 0.9,
                "keeper_save_rate_shrunk": 0.1,
                "shooter_zone_entropy": 0.4,
                "is_shootout": False,
                "shooter_n_pens_before": 10,
                "keeper_n_faced_before": 10,
            }
        ]
    )
    weak = strong.copy()
    weak["shooter_conv_rate_shrunk"] = 0.3
    weak["keeper_save_rate_shrunk"] = 0.5

    assert predict_combined_proba(model, strong)[0] > predict_combined_proba(model, weak)[0]
