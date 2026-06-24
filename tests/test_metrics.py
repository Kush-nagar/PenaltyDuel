import pytest

from src.evaluation.metrics import (
    binary_classification_metrics,
    brier_score,
    brier_skill_score,
    expected_calibration_error,
    log_loss,
    pr_auc,
    reliability_table,
    roc_auc,
)


def test_log_loss_clips_probabilities_and_matches_known_value():
    y_true = [1, 0]
    y_prob = [0.8, 0.2]

    assert log_loss(y_true, y_prob) == pytest.approx(0.22314355)
    assert log_loss([1, 0], [1.0, 0.0]) < 1e-6


def test_brier_and_skill_score():
    y_true = [1, 0, 1, 0]
    model = [0.9, 0.1, 0.8, 0.2]
    baseline = [0.5, 0.5, 0.5, 0.5]

    assert brier_score(y_true, model) == pytest.approx(0.025)
    assert brier_skill_score(y_true, model, baseline) == pytest.approx(0.9)


def test_auc_metrics_handle_tied_scores():
    y_true = [0, 0, 1, 1]
    y_prob = [0.1, 0.4, 0.35, 0.8]

    assert roc_auc(y_true, y_prob) == pytest.approx(0.75)
    assert pr_auc(y_true, y_prob) == pytest.approx(0.7916666667)


def test_reliability_table_and_ece_are_weighted_by_bin_size():
    y_true = [0, 1, 1, 1]
    y_prob = [0.1, 0.2, 0.8, 0.9]

    table = reliability_table(y_true, y_prob, n_bins=2)

    assert table["count"].tolist() == [2, 2]
    assert table["mean_predicted"].tolist() == pytest.approx([0.15, 0.85])
    assert table["observed_rate"].tolist() == pytest.approx([0.5, 1.0])
    assert expected_calibration_error(y_true, y_prob, n_bins=2) == pytest.approx(0.25)


def test_binary_classification_metrics_returns_step6_fields():
    metrics = binary_classification_metrics(
        y_true=[0, 1, 1, 0],
        y_prob=[0.1, 0.7, 0.8, 0.4],
        baseline_prob=[0.5, 0.5, 0.5, 0.5],
    )

    assert set(metrics) == {
        "log_loss",
        "brier",
        "brier_skill_score",
        "roc_auc",
        "pr_auc",
        "ece",
        "accuracy_footnote",
    }
    assert metrics["brier_skill_score"] > 0
