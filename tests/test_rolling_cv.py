import numpy as np
import pandas as pd

from src.models.baselines import add_baseline_predictions
from src.models.training import evaluate_outcome_models, rolling_temporal_folds


def synthetic_modeling_table(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    base_date = pd.Timestamp("2019-01-01")
    for i in range(n):
        shooter_rate = rng.uniform(0.4, 0.9)
        keeper_save = rng.uniform(0.1, 0.5)
        # outcome driven by both shooter and keeper signal plus noise
        logit = (shooter_rate - 0.6) * 4 + (0.3 - keeper_save) * 4
        prob = 1 / (1 + np.exp(-logit))
        outcome = int(rng.uniform() < prob)
        zone_probs = rng.dirichlet(np.ones(6))
        row = {
            "penalty_id": f"p{i}",
            "match_date": (base_date + pd.Timedelta(days=i)).strftime("%Y-%m-%d"),
            "outcome_bin": outcome,
            "shot_zone": rng.choice(
                ["low_left", "low_center", "low_right", "high_left", "high_center", "high_right"]
            ),
            "shooter_conv_rate_shrunk": shooter_rate,
            "shooter_conv_rate_raw_before": shooter_rate,
            "keeper_save_rate_shrunk": keeper_save,
            "keeper_save_rate_raw_before": keeper_save,
            "shooter_zone_entropy": rng.uniform(0.3, 0.9),
            "is_shootout": bool(i % 4 == 0),
            "shootout_kick_index": float(i % 6 + 1) if i % 4 == 0 else np.nan,
            "shooter_n_pens_before": i % 9,
            "keeper_n_faced_before": i % 7,
        }
        for z, p in zip(
            ["low_left", "low_center", "low_right", "high_left", "high_center", "high_right"],
            zone_probs,
        ):
            row[f"shooter_zone_prob_{z}"] = p
        rows.append(row)
    table = pd.DataFrame(rows)
    return add_baseline_predictions(table, global_rate=float(table["outcome_bin"].mean()))


def test_rolling_temporal_folds_have_disjoint_increasing_test_blocks():
    table = synthetic_modeling_table()

    folds = list(rolling_temporal_folds(table, n_folds=4, test_fraction=0.5))

    assert len(folds) == 4
    seen = set()
    last_max_train = -1
    for train, test in folds:
        assert len(train) > 0 and len(test) > 0
        # train precedes test in time
        assert pd.to_datetime(train["match_date"]).max() <= pd.to_datetime(test["match_date"]).min()
        ids = set(test["penalty_id"])
        assert ids.isdisjoint(seen)
        seen |= ids
        assert len(train) >= last_max_train
        last_max_train = len(train)


def test_evaluate_outcome_models_returns_gate_and_headline_metrics():
    table = synthetic_modeling_table()

    result = evaluate_outcome_models(table, n_folds=4, test_fraction=0.5)

    assert "gate" in result
    assert "headline_model" in result
    assert "cv_summary" in result
    headline = result["headline_model"]
    assert headline in result["cv_summary"]
    summary = result["cv_summary"][headline]
    assert summary["log_loss_mean"] >= 0
    assert summary["brier_mean"] >= 0
    # gate keys present and boolean
    assert isinstance(result["gate"]["beats_keeper_shrunk"], bool)
    assert "keeper_shrunk" in result["cv_summary"]
