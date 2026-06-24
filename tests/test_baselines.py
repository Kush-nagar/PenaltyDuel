import pandas as pd
import pytest

from src.features.build import build_modeling_table
from src.models.baselines import (
    add_baseline_predictions,
    evaluate_baselines,
)


def toy_penalties() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "penalty_id": "p1",
                "match_date": "2020-01-01",
                "match_id": 1,
                "index_in_match": 1,
                "shooter_id": 10,
                "keeper_id": 100,
                "outcome_bin": 1,
                "shot_zone": "low_left",
                "shot_body_part_name": "Right Foot",
                "shooter_position_name": "Forward",
                "competition_name": "League A",
                "competition_stage_name": "Regular Season",
                "is_shootout": False,
                "shootout_kick_index": None,
            },
            {
                "penalty_id": "p2",
                "match_date": "2020-02-01",
                "match_id": 2,
                "index_in_match": 1,
                "shooter_id": 10,
                "keeper_id": 100,
                "outcome_bin": 0,
                "shot_zone": "low_right",
                "shot_body_part_name": "Right Foot",
                "shooter_position_name": "Forward",
                "competition_name": "League A",
                "competition_stage_name": "Regular Season",
                "is_shootout": False,
                "shootout_kick_index": None,
            },
            {
                "penalty_id": "p3",
                "match_date": "2020-03-01",
                "match_id": 3,
                "index_in_match": 1,
                "shooter_id": 20,
                "keeper_id": 200,
                "outcome_bin": 1,
                "shot_zone": "high_center",
                "shot_body_part_name": "Left Foot",
                "shooter_position_name": "Midfield",
                "competition_name": "League B",
                "competition_stage_name": "Final",
                "is_shootout": True,
                "shootout_kick_index": 6,
            },
        ]
    )


def test_add_baseline_predictions_exposes_global_shooter_and_keeper_probabilities():
    table = build_modeling_table(toy_penalties())

    result = add_baseline_predictions(table, global_rate=0.8)

    assert result["pred_global"].tolist() == [0.8, 0.8, 0.8]
    assert result.loc[1, "pred_shooter_shrunk"] == pytest.approx(
        result.loc[1, "shooter_conv_rate_shrunk"]
    )
    assert result.loc[1, "pred_keeper_shrunk"] == pytest.approx(
        1 - result.loc[1, "keeper_save_rate_shrunk"]
    )
    assert result.loc[0, "pred_shooter_raw"] != result.loc[0, "pred_shooter_raw"]
    assert result.loc[0, "pred_keeper_raw"] != result.loc[0, "pred_keeper_raw"]


def test_evaluate_baselines_returns_one_row_per_baseline():
    table = build_modeling_table(toy_penalties())
    table = add_baseline_predictions(table, global_rate=0.75)

    result = evaluate_baselines(table)

    assert result["baseline"].tolist() == [
        "global",
        "shooter_raw",
        "shooter_shrunk",
        "keeper_raw",
        "keeper_shrunk",
    ]
    assert set(["log_loss", "brier", "ece", "accuracy_footnote"]).issubset(result.columns)
