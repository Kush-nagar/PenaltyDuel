import pandas as pd
import pytest

from src.features.as_of import build_as_of_features
from src.features.build import build_modeling_table


def toy_penalties() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "penalty_id": "later_same_shooter",
                "match_date": "2020-03-01",
                "match_id": 3,
                "index_in_match": 5,
                "shooter_id": 10,
                "keeper_id": 100,
                "outcome_bin": 0,
                "shot_zone": "low_right",
                "shot_body_part_name": "Right Foot",
                "shooter_position_name": "Forward",
                "competition_name": "League A",
                "competition_stage_name": "Final",
                "is_shootout": True,
                "shootout_kick_index": 6,
            },
            {
                "penalty_id": "first_same_shooter",
                "match_date": "2020-01-01",
                "match_id": 1,
                "index_in_match": 10,
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
                "penalty_id": "middle_same_shooter",
                "match_date": "2020-02-01",
                "match_id": 2,
                "index_in_match": 7,
                "shooter_id": 10,
                "keeper_id": 200,
                "outcome_bin": 1,
                "shot_zone": "low_left",
                "shot_body_part_name": "Right Foot",
                "shooter_position_name": "Forward",
                "competition_name": "League A",
                "competition_stage_name": "Quarter-finals",
                "is_shootout": False,
                "shootout_kick_index": None,
            },
            {
                "penalty_id": "other_shooter_same_keeper",
                "match_date": "2020-02-15",
                "match_id": 2,
                "index_in_match": 8,
                "shooter_id": 20,
                "keeper_id": 100,
                "outcome_bin": 0,
                "shot_zone": "high_center",
                "shot_body_part_name": "Left Foot",
                "shooter_position_name": "Midfield",
                "competition_name": "League B",
                "competition_stage_name": "Semi-finals",
                "is_shootout": False,
                "shootout_kick_index": None,
            },
        ]
    )


def test_as_of_features_use_only_prior_penalties_and_restore_input_order():
    df = toy_penalties()

    features = build_as_of_features(df)

    assert features["penalty_id"].tolist() == df["penalty_id"].tolist()

    first = features.set_index("penalty_id").loc["first_same_shooter"]
    middle = features.set_index("penalty_id").loc["middle_same_shooter"]
    later = features.set_index("penalty_id").loc["later_same_shooter"]

    assert first["shooter_n_pens_before"] == 0
    assert first["shooter_goals_before"] == 0
    assert first["shooter_conv_rate_raw_before"] != first["shooter_conv_rate_raw_before"]

    assert middle["shooter_n_pens_before"] == 1
    assert middle["shooter_goals_before"] == 1
    assert middle["shooter_zone_count_low_left_before"] == 1

    assert later["shooter_n_pens_before"] == 2
    assert later["shooter_goals_before"] == 2
    assert later["keeper_n_faced_before"] == 2
    assert later["keeper_goals_allowed_before"] == 1


def test_modeling_table_features_do_not_change_when_current_outcome_changes():
    original = toy_penalties()
    mutated = original.copy()
    mutated.loc[mutated["penalty_id"] == "later_same_shooter", "outcome_bin"] = 1

    original_features = build_modeling_table(original).set_index("penalty_id")
    mutated_features = build_modeling_table(mutated).set_index("penalty_id")

    feature_cols = [
        "shooter_n_pens_before",
        "shooter_goals_before",
        "shooter_conv_rate_shrunk",
        "keeper_n_faced_before",
        "keeper_goals_allowed_before",
        "keeper_save_rate_shrunk",
        "shooter_zone_prob_low_left",
        "shooter_zone_prob_low_right",
        "shooter_zone_entropy",
    ]

    pd.testing.assert_series_equal(
        original_features.loc["later_same_shooter", feature_cols],
        mutated_features.loc["later_same_shooter", feature_cols],
        check_names=False,
    )


def test_modeling_table_shrinks_zone_probs_and_keeps_context_columns():
    features = build_modeling_table(toy_penalties()).set_index("penalty_id")
    later = features.loc["later_same_shooter"]

    assert later["shooter_zone_prob_low_left"] > later["shooter_zone_prob_high_center"]
    assert later["shooter_zone_prob_low_left"] < 1.0
    assert later["shooter_zone_entropy"] > 0
    assert later["is_shootout"] is True or later["is_shootout"] == pytest.approx(1.0)
    assert later["stakes"] == "shootout_sudden_death"


def test_missing_keeper_rows_do_not_share_synthetic_keeper_history():
    df = toy_penalties()
    df.loc[df["penalty_id"].isin(["first_same_shooter", "middle_same_shooter"]), "keeper_id"] = None

    features = build_modeling_table(df).set_index("penalty_id")

    assert features.loc["first_same_shooter", "keeper_n_faced_before"] == 0
    assert features.loc["middle_same_shooter", "keeper_n_faced_before"] == 0
    assert bool(features.loc["first_same_shooter", "keeper_save_rate_is_imputed"]) is True
    assert bool(features.loc["middle_same_shooter", "keeper_save_rate_is_imputed"]) is True
