import numpy as np
import pytest

from src.evaluation.metrics import (
    kendall_tau,
    macro_f1,
    multiclass_log_loss,
    spearman_corr,
    top_k_accuracy,
)

CLASSES = ["a", "b", "c"]


def test_multiclass_log_loss_rewards_confident_correct():
    y_true = ["a", "b", "c"]
    confident = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    unsure = np.full((3, 3), 1 / 3)

    assert multiclass_log_loss(y_true, confident, CLASSES) < multiclass_log_loss(
        y_true, unsure, CLASSES
    )
    assert multiclass_log_loss(y_true, unsure, CLASSES) == pytest.approx(
        np.log(3), rel=1e-6
    )


def test_top_k_accuracy_counts_true_in_top_k():
    y_true = ["a", "b", "c"]
    probs = np.array([[0.6, 0.3, 0.1], [0.5, 0.4, 0.1], [0.2, 0.3, 0.5]])

    assert top_k_accuracy(y_true, probs, CLASSES, k=1) == pytest.approx(2 / 3)
    assert top_k_accuracy(y_true, probs, CLASSES, k=2) == pytest.approx(1.0)


def test_macro_f1_perfect_and_imperfect():
    y_true = ["a", "a", "b", "b"]
    perfect = ["a", "a", "b", "b"]
    assert macro_f1(y_true, perfect, ["a", "b"]) == pytest.approx(1.0)

    wrong = ["a", "a", "a", "a"]
    # class b has zero recall -> F1 0; class a precision 0.5 recall 1 -> F1 2/3
    assert macro_f1(y_true, wrong, ["a", "b"]) == pytest.approx((2 / 3) / 2)


def test_spearman_and_kendall_monotonic():
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]
    assert spearman_corr(x, y) == pytest.approx(1.0)
    assert kendall_tau(x, y) == pytest.approx(1.0)

    y_rev = [10, 8, 6, 4, 2]
    assert spearman_corr(x, y_rev) == pytest.approx(-1.0)
    assert kendall_tau(x, y_rev) == pytest.approx(-1.0)
