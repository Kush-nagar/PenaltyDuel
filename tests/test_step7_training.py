import pandas as pd

from src.models.training import evaluate_direct_outcome_model


def toy_modeling_table() -> pd.DataFrame:
    rows = []
    for idx in range(24):
        high_signal = idx % 4 in (2, 3)
        rows.append(
            {
                "penalty_id": f"p{idx}",
                "match_date": f"2020-01-{idx + 1:02d}",
                "outcome_bin": int(high_signal),
                "shooter_conv_rate_shrunk": 0.8 if high_signal else 0.25,
                "shooter_n_pens_before": idx % 6,
                "keeper_save_rate_shrunk": 0.2 if high_signal else 0.75,
                "keeper_n_faced_before": idx % 5,
                "shooter_zone_entropy": 0.4 if high_signal else 0.8,
                "shooter_zone_prob_low_left": 0.4 if high_signal else 0.15,
                "shooter_zone_prob_low_center": 0.2 if high_signal else 0.1,
                "shooter_zone_prob_low_right": 0.2,
                "shooter_zone_prob_high_left": 0.1,
                "shooter_zone_prob_high_center": 0.05 if high_signal else 0.2,
                "shooter_zone_prob_high_right": 0.05 if high_signal else 0.25,
                "expected_guess_correct": 0.2 if high_signal else 0.6,
                "is_shootout": idx % 3 == 0,
                "shootout_kick_index": 6 if idx % 3 == 0 else None,
                "pred_keeper_shrunk": 0.25 if not high_signal else 0.8,
                "pred_shooter_shrunk": 0.25 if not high_signal else 0.8,
            }
        )
    return pd.DataFrame(rows)


def test_evaluate_direct_outcome_model_returns_uncalibrated_and_calibrated_metrics():
    result = evaluate_direct_outcome_model(
        toy_modeling_table(),
        cutoff="2020-01-17",
        calibration_fraction=0.25,
    )

    assert result["split"]["train_rows"] == 12
    assert result["split"]["calibration_rows"] == 4
    assert result["split"]["test_rows"] == 8
    assert result["metrics"]["direct"]["log_loss"] >= 0
    assert result["metrics"]["direct_calibrated"]["brier"] >= 0
    assert result["metrics"]["keeper_shrunk_baseline"]["brier"] >= 0
    assert "backend" in result["model"]
