import numpy as np
import pandas as pd

from src.evaluation.generalization import (
    conversion_ranking_correlation,
    evaluate_generalization,
    group_kfold,
    n_pens_bucket,
)
from src.models.baselines import add_baseline_predictions


def signal_table(n: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    rows = []
    for i in range(n):
        shooter_id = f"s{i % 40}"
        shooter_rate = rng.uniform(0.4, 0.95)
        keeper_save = rng.uniform(0.1, 0.45)
        prob = 1 / (1 + np.exp(-((shooter_rate - 0.6) * 5 + (0.3 - keeper_save) * 5)))
        outcome = int(rng.uniform() < prob)
        zone_probs = rng.dirichlet(np.ones(6))
        row = {
            "penalty_id": f"p{i}",
            "match_date": (pd.Timestamp("2018-01-01") + pd.Timedelta(days=i)).strftime("%Y-%m-%d"),
            "shooter_id": shooter_id,
            "competition_id": i % 5,
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
            "shooter_n_pens_before": i % 14,
            "keeper_n_faced_before": i % 8,
        }
        for z, p in zip(
            ["low_left", "low_center", "low_right", "high_left", "high_center", "high_right"],
            zone_probs,
        ):
            row[f"shooter_zone_prob_{z}"] = p
        rows.append(row)
    table = pd.DataFrame(rows)
    return add_baseline_predictions(table, global_rate=float(table["outcome_bin"].mean()))


def test_n_pens_bucket_boundaries():
    assert n_pens_bucket(0) == "0"
    assert n_pens_bucket(1) == "1-3"
    assert n_pens_bucket(3) == "1-3"
    assert n_pens_bucket(4) == "4-10"
    assert n_pens_bucket(10) == "4-10"
    assert n_pens_bucket(11) == "11+"


def test_group_kfold_partitions_groups_disjointly():
    table = signal_table()
    folds = list(group_kfold(table, group_col="shooter_id", n_splits=4, seed=0))

    assert len(folds) == 4
    all_test_groups = set()
    for train, test in folds:
        train_groups = set(train["shooter_id"])
        test_groups = set(test["shooter_id"])
        assert train_groups.isdisjoint(test_groups)
        assert test_groups.isdisjoint(all_test_groups)
        all_test_groups |= test_groups


def test_evaluate_generalization_reports_buckets_and_beats_priors():
    table = signal_table()
    result = evaluate_generalization(table, group_col="shooter_id", n_splits=4)

    assert "overall" in result and "by_n_pens" in result
    assert result["overall"]["full"]["log_loss"] <= result["overall"]["priors_only"]["log_loss"]
    assert len(result["by_n_pens"]) >= 1
    for row in result["by_n_pens"]:
        assert "bucket" in row and "n" in row and "full_log_loss" in row


def test_conversion_ranking_correlation_positive_with_signal():
    table = signal_table()
    table["pred"] = table["shooter_conv_rate_shrunk"]
    out = conversion_ranking_correlation(
        table, pred_col="pred", group_col="shooter_id", min_kicks=3
    )
    assert "spearman" in out and "kendall" in out and "n_shooters" in out
    assert out["spearman"] > 0
