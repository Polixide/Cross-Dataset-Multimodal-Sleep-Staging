"""Evaluation metrics and reporting (master plan: Evaluation metrics + Calibration).

Accuracy alone is insufficient under class imbalance, so the headline metric is
macro-F1, reported with Cohen's kappa, balanced accuracy, per-class scores and
the confusion matrix. Probability-based metrics add one-vs-rest AUPRC / ROC-AUC
and calibration (ECE, Brier). Plotting lives in scripts/make_figures.py.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred, labels=None):
    """Return the core classification metrics as a plain, JSON-serializable dict."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "per_class_f1": f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0).tolist(),
        "per_class_recall": recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0).tolist(),
        "per_class_precision": precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0).tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def _one_hot(y_true, n_classes):
    one_hot = np.zeros((len(y_true), n_classes), dtype=float)
    one_hot[np.arange(len(y_true)), y_true] = 1.0
    return one_hot


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Multi-class ECE using top-1 confidence versus top-1 correctness."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    confidences = y_prob.max(axis=1)
    predictions = y_prob.argmax(axis=1)
    correct = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for low, high in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (confidences > low) & (confidences <= high)
        if in_bin.any():
            gap = abs(correct[in_bin].mean() - confidences[in_bin].mean())
            ece += in_bin.mean() * gap
    return float(ece)


def brier_score_multiclass(y_true, y_prob):
    """Mean squared error between one-hot labels and predicted probabilities."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    one_hot = _one_hot(y_true, y_prob.shape[1])
    return float(np.mean(np.sum((y_prob - one_hot) ** 2, axis=1)))


def probabilistic_metrics(y_true, y_prob):
    """One-vs-rest AUPRC and ROC-AUC plus ECE and Brier score.

    Per-class scores are NaN for classes that are absent (or the only class) in
    y_true; the macro scores ignore those with nanmean.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n_classes = y_prob.shape[1]
    one_hot = _one_hot(y_true, n_classes)

    per_class_auprc, per_class_roc_auc = [], []
    for c in range(n_classes):
        column = one_hot[:, c]
        if column.sum() == 0 or column.sum() == len(column):
            per_class_auprc.append(float("nan"))
            per_class_roc_auc.append(float("nan"))
        else:
            per_class_auprc.append(float(average_precision_score(column, y_prob[:, c])))
            per_class_roc_auc.append(float(roc_auc_score(column, y_prob[:, c])))

    return {
        "macro_auprc": float(np.nanmean(per_class_auprc)),
        "macro_roc_auc": float(np.nanmean(per_class_roc_auc)),
        "per_class_auprc": per_class_auprc,
        "per_class_roc_auc": per_class_roc_auc,
        "ece": expected_calibration_error(y_true, y_prob),
        "brier": brier_score_multiclass(y_true, y_prob),
    }


def reliability_curve(y_true, y_prob, n_bins=10):
    """Return (mean_confidence, accuracy) per bin for a reliability diagram."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    confidences = y_prob.max(axis=1)
    correct = (y_prob.argmax(axis=1) == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    mean_confidence, accuracy = [], []
    for low, high in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (confidences > low) & (confidences <= high)
        if in_bin.any():
            mean_confidence.append(float(confidences[in_bin].mean()))
            accuracy.append(float(correct[in_bin].mean()))
    return np.array(mean_confidence), np.array(accuracy)


def summarize_folds(fold_metrics, keys=("macro_f1", "cohen_kappa", "balanced_accuracy")):
    """Aggregate per-fold scalar metrics into mean and std for a compact report."""
    summary = {}
    for key in keys:
        values = np.array([metrics[key] for metrics in fold_metrics], dtype=float)
        summary[key] = {"mean": float(values.mean()), "std": float(values.std())}
    return summary
