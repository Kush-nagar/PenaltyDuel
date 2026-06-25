import numpy as np
import pandas as pd

from src.evaluation.ablation import ABLATION_VARIANTS, run_ablation
from src.models.baselines import add_baseline_predictions


def ablation_table(n: int = 220) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    rows = []
    for i in range(n):
        shooter_rate = rng.uniform(0.4, 0.95)
        keeper_save = rng.uniform(0.1, 0.45)
        prob = 1 / (1 + np.exp(-((shooter_rate - 0.6) * 5 + (0.3 - keeper_save) * 5)))
        outcome = int(rng.uniform() < prob)
        rows.append(
            {
                "penalty_id": f"p{i}",
                "match_date": (pd.Timestamp("2018-01-01") + pd.Timedelta(days=i)).strftime("%Y-%m-%d"),
                "outcome_bin": outcome,
                "shooter_conv_rate_shrunk": shooter_rate,
                "shooter_conv_rate_raw_before": shooter_rate,
                "keeper_save_rate_shrunk": keeper_save,
                "keeper_save_rate_raw_before": keeper_save,
                "shooter_zone_entropy": rng.uniform(0.3, 0.9),
                "is_shootout": bool(i % 4 == 0),
                "shooter_n_pens_before": i % 12,
                "keeper_n_faced_before": i % 8,
            }
        )
    table = pd.DataFrame(rows)
    return add_baseline_predictions(table, global_rate=float(table["outcome_bin"].mean()))


def test_run_ablation_returns_one_row_per_variant():
    table = ablation_table()
    result = run_ablation(table, n_folds=4, test_fraction=0.5)

    assert set(result["variant"]) >= set(ABLATION_VARIANTS) | {"shrinkage_off"}
    for col in ["log_loss", "brier", "ece", "d_log_loss", "d_brier"]:
        assert col in result.columns


def test_full_variant_has_zero_delta():
    table = ablation_table()
    result = run_ablation(table, n_folds=4, test_fraction=0.5)

    full_row = result[result["variant"] == "full"].iloc[0]
    assert full_row["d_log_loss"] == 0.0
    assert full_row["d_brier"] == 0.0
