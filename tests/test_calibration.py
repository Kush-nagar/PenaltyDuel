import numpy as np
import pytest

from src.models.calibration import ProbabilityCalibrator, select_calibration_method


def test_sigmoid_calibrator_returns_probabilities_in_unit_interval():
    raw_prob = np.array([0.1, 0.2, 0.8, 0.9])
    y_true = np.array([0, 0, 1, 1])

    calibrator = ProbabilityCalibrator(method="sigmoid").fit(raw_prob, y_true)
    calibrated = calibrator.predict(raw_prob)

    assert calibrated.shape == raw_prob.shape
    assert np.all(calibrated >= 0)
    assert np.all(calibrated <= 1)
    assert calibrated[0] < calibrated[-1]


def test_isotonic_calibrator_is_monotone_and_clips_out_of_bounds():
    raw_prob = np.array([0.2, 0.4, 0.6, 0.8])
    y_true = np.array([0, 0, 1, 1])

    calibrator = ProbabilityCalibrator(method="isotonic").fit(raw_prob, y_true)
    calibrated = calibrator.predict(np.array([-0.5, 0.3, 0.7, 1.5]))

    assert calibrated.tolist() == sorted(calibrated.tolist())
    assert np.all(calibrated >= 0)
    assert np.all(calibrated <= 1)


def test_calibrator_rejects_unknown_method():
    with pytest.raises(ValueError, match="method"):
        ProbabilityCalibrator(method="unknown")


def test_none_calibrator_returns_clipped_identity():
    raw_prob = np.array([0.0, 0.3, 0.7, 1.0])

    calibrator = ProbabilityCalibrator(method="none").fit(raw_prob, np.array([0, 0, 1, 1]))
    calibrated = calibrator.predict(raw_prob)

    assert np.all(calibrated > 0) and np.all(calibrated < 1)
    assert calibrated[1] == pytest.approx(0.3)
    assert calibrated[2] == pytest.approx(0.7)


def test_select_calibration_method_returns_valid_choice():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, size=200)
    raw = np.clip(0.5 + 0.25 * (y - 0.5) + rng.normal(0, 0.1, 200), 0.01, 0.99)

    method = select_calibration_method(raw, y)

    assert method in {"none", "sigmoid", "isotonic"}
