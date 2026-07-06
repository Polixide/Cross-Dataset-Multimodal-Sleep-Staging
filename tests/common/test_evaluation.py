"""Tests for the evaluation and calibration metrics."""
import numpy as np

from src.common.evaluation import (
    brier_score_multiclass,
    compute_metrics,
    expected_calibration_error,
    probabilistic_metrics,
    reliability_curve,
    summarize_folds,
)

LABELS = [0, 1, 2, 3, 4]


def test_perfect_prediction_scores_one():
    y = np.array([0, 1, 2, 3, 4, 0, 1, 2])
    m = compute_metrics(y, y, labels=LABELS)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["cohen_kappa"] == 1.0


def test_confusion_matrix_has_full_class_shape():
    y_true = np.array([0, 1, 2, 3, 4])
    y_pred = np.array([0, 1, 2, 3, 4])
    m = compute_metrics(y_true, y_pred, labels=LABELS)
    assert np.array(m["confusion_matrix"]).shape == (5, 5)


def test_per_class_lists_cover_all_labels():
    y_true = np.array([0, 1, 2, 3, 4])
    y_pred = np.array([0, 1, 2, 3, 0])
    m = compute_metrics(y_true, y_pred, labels=LABELS)
    assert len(m["per_class_f1"]) == 5


def test_ece_and_brier_are_zero_for_confident_correct():
    y_true = np.array([0, 1])
    y_prob = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert expected_calibration_error(y_true, y_prob) == 0.0
    assert brier_score_multiclass(y_true, y_prob) == 0.0


def test_probabilistic_metrics_perfect_probabilities():
    y_true = np.array([0, 1, 2])
    y_prob = np.eye(3)  # perfectly confident and correct
    m = probabilistic_metrics(y_true, y_prob)
    assert m["macro_auprc"] == 1.0
    assert m["macro_roc_auc"] == 1.0
    assert m["ece"] == 0.0
    assert m["brier"] == 0.0


def test_reliability_curve_returns_matching_arrays():
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.4, 0.6]])
    confidence, accuracy = reliability_curve(y_true, y_prob, n_bins=5)
    assert len(confidence) == len(accuracy)
    assert np.all((accuracy >= 0) & (accuracy <= 1))


def test_summarize_folds_reports_mean_and_std():
    folds = [
        {"macro_f1": 0.8, "cohen_kappa": 0.7, "balanced_accuracy": 0.75},
        {"macro_f1": 0.6, "cohen_kappa": 0.5, "balanced_accuracy": 0.65},
    ]
    summary = summarize_folds(folds)
    assert summary["macro_f1"]["mean"] == 0.7
    assert summary["macro_f1"]["std"] > 0
