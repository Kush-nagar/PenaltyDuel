import numpy as np
import pandas as pd

from src.evaluation.placement import evaluate_placement

ZONES = ["low_left", "low_center", "low_right", "high_left", "high_center", "high_right"]


def placement_table(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(9)
    rows = []
    for i in range(n):
        true_zone = ZONES[i % 6]
        # informative distribution: mass concentrated on the true zone
        probs = np.full(6, 0.04)
        probs[i % 6] = 1 - 0.04 * 5
        row = {
            "penalty_id": f"p{i}",
            "match_date": (pd.Timestamp("2018-01-01") + pd.Timedelta(days=i)).strftime("%Y-%m-%d"),
            "shot_zone": true_zone,
            "shot_zone_is_missing": False,
        }
        for z, p in zip(ZONES, probs):
            row[f"shooter_zone_prob_{z}"] = p
        rows.append(row)
    return pd.DataFrame(rows)


def test_evaluate_placement_model_beats_uniform():
    table = placement_table()
    cutoff = pd.Timestamp("2018-03-01")

    result = evaluate_placement(table, cutoff=cutoff)

    assert "model" in result and "uniform" in result and "marginal" in result
    assert result["model"]["log_loss"] < result["uniform"]["log_loss"]
    assert 0.0 <= result["model"]["top1"] <= 1.0
    assert result["model"]["top2"] >= result["model"]["top1"]
    assert "macro_f1" in result["model"]
