"""Tests for imbalance weights and temperature-scaling calibration."""
import numpy as np

from src.train import compute_class_weights, fit_temperature, softmax_with_temperature


def test_class_weights_are_higher_for_rare_classes():
    y = np.array([0] * 90 + [1] * 10)  # class 1 is rare
    weights = compute_class_weights(y)
    assert weights[1] > weights[0]


def test_softmax_with_temperature_rows_sum_to_one():
    logits = np.array([[2.0, 1.0, 0.0], [0.5, 0.5, 3.0]])
    probs = softmax_with_temperature(logits, temperature=2.0)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_fit_temperature_softens_overconfident_logits():
    # Large logits with some mistakes are overconfident -> optimal T should be > 1.
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, size=200)
    logits = np.full((200, 3), -5.0)
    logits[np.arange(200), y] = 5.0
    # flip 25% of the "correct" peaks to a wrong class to create miscalibration
    wrong = rng.random(200) < 0.25
    logits[wrong] = np.roll(logits[wrong], 1, axis=1)
    temperature = fit_temperature(logits, y)
    assert temperature > 1.0
