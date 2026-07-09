"""Generate presentation figures from saved results.

Plots whatever inputs are provided:
- confusion matrix + per-class F1 (from a metrics JSON),
- ROC and PR curves (one-vs-rest) + reliability diagram (from a probs .npz),
- learning curves (from a DL metrics JSON with a training history),
- class distribution (from a processed .npz),
- internal-vs-external macro-F1 (from a second metrics JSON).

Usage:
    python scripts/shared/make_figures.py --metrics results/logs/ml_metrics_rf.json \
        --probs results/logs/ml_test_probs_rf.npz --data data/processed/sleep_edf.npz
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, auc, precision_recall_curve, roc_curve

from src.common.data import load_processed_dataset
from src.common.evaluation import reliability_curve
from src.common.utils import STAGE_NAMES, ensure_dir, get_logger, load_json

logger = get_logger("make_figures")


def save_figure(out_path):
    """Save the active figure, explicitly replacing an older file."""
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    out_path.unlink(missing_ok=True)
    plt.savefig(out_path, dpi=150)


def confusion_and_f1(metrics):
    """Extract (confusion_matrix, per_class_f1) from a metrics JSON (test or CV)."""
    if "test_metrics" in metrics:
        block = metrics["test_metrics"]
        return np.array(block["confusion_matrix"]), np.array(block["per_class_f1"])
    if "metrics" in metrics and "confusion_matrix" in metrics["metrics"]:
        block = metrics["metrics"]
        return np.array(block["confusion_matrix"]), np.array(block["per_class_f1"])
    if "loso_metrics" in metrics:
        block = metrics["loso_metrics"]
        return np.array(block["confusion_matrix"]), np.array(block["per_class_f1"])
    if "folds" in metrics:  # cross-validation output
        cm = np.sum([fold["confusion_matrix"] for fold in metrics["folds"]], axis=0)
        f1 = np.mean([fold["per_class_f1"] for fold in metrics["folds"]], axis=0)
        return cm, f1
    return None, None


def internal_macro_f1(metrics):
    if "test_metrics" in metrics:
        return metrics["test_metrics"]["macro_f1"]
    if "cv_summary" in metrics:
        return metrics["cv_summary"]["macro_f1"]["mean"]
    if "loso_metrics" in metrics:
        return metrics["loso_metrics"]["macro_f1"]
    return None


def plot_confusion_matrix(confusion, out_path):
    ConfusionMatrixDisplay(np.array(confusion), display_labels=STAGE_NAMES).plot(cmap="Blues", colorbar=False)
    plt.title("Confusion matrix")
    plt.tight_layout()
    save_figure(out_path)
    plt.close()


def plot_per_class_f1(per_class_f1, out_path):
    plt.figure(figsize=(6, 4))
    plt.bar(STAGE_NAMES, per_class_f1)
    plt.ylim(0, 1)
    plt.ylabel("F1 score")
    plt.title("Per-class F1")
    plt.tight_layout()
    save_figure(out_path)
    plt.close()


def plot_roc_curves(y_true, y_prob, out_path):
    plt.figure(figsize=(6, 5))
    for c, name in enumerate(STAGE_NAMES):
        binary = (y_true == c).astype(int)
        if binary.sum() in (0, len(binary)):
            continue
        fpr, tpr, _ = roc_curve(binary, y_prob[:, c])
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.2f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curves (one-vs-rest)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    save_figure(out_path)
    plt.close()


def plot_pr_curves(y_true, y_prob, out_path):
    plt.figure(figsize=(6, 5))
    for c, name in enumerate(STAGE_NAMES):
        binary = (y_true == c).astype(int)
        if binary.sum() in (0, len(binary)):
            continue
        precision, recall, _ = precision_recall_curve(binary, y_prob[:, c])
        plt.plot(recall, precision, label=name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-recall curves (one-vs-rest)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    save_figure(out_path)
    plt.close()


def plot_reliability(y_true, y_prob, out_path):
    confidence, accuracy = reliability_curve(y_true, y_prob)
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Perfect calibration")
    plt.plot(confidence, accuracy, "o-", label="Model")
    plt.xlabel("Mean predicted confidence")
    plt.ylabel("Empirical accuracy")
    plt.title("Reliability diagram")
    plt.legend(fontsize=8)
    plt.tight_layout()
    save_figure(out_path)
    plt.close()


def plot_learning_curves(history, out_path):
    epochs = [step["epoch"] for step in history]
    fig, ax_loss = plt.subplots(figsize=(6, 4))
    ax_loss.plot(epochs, [step["train_loss"] for step in history], "b-", label="Train loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Train loss", color="b")
    ax_f1 = ax_loss.twinx()
    ax_f1.plot(epochs, [step["val_macro_f1"] for step in history], "g-", label="Val macro-F1")
    ax_f1.set_ylabel("Val macro-F1", color="g")
    plt.title("Learning curves")
    plt.tight_layout()
    save_figure(out_path)
    plt.close()


def plot_class_distribution(y, out_path):
    counts = np.bincount(np.asarray(y), minlength=len(STAGE_NAMES))
    plt.figure(figsize=(6, 4))
    plt.bar(STAGE_NAMES, counts)
    plt.ylabel("Epoch count")
    plt.title("Class distribution")
    plt.tight_layout()
    save_figure(out_path)
    plt.close()


def plot_internal_vs_external(internal_f1, external_f1, out_path):
    plt.figure(figsize=(5, 4))
    plt.bar(["Internal", "External"], [internal_f1, external_f1])
    plt.ylim(0, 1)
    plt.ylabel("Macro-F1")
    plt.title("Internal vs external generalization")
    plt.tight_layout()
    save_figure(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Build result figures from saved outputs.")
    parser.add_argument("--metrics", default="results/logs/ml_metrics_rf.json")
    parser.add_argument("--probs", help="Test-set probabilities .npz (y_true, y_prob).")
    parser.add_argument("--data", help="Processed .npz for the class-distribution plot.")
    parser.add_argument("--external", help="External metrics JSON for the comparison plot.")
    parser.add_argument("--out-dir", default="results/figures")
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)

    if Path(args.metrics).exists():
        metrics = load_json(args.metrics)
        confusion, per_class_f1 = confusion_and_f1(metrics)
        if confusion is not None:
            plot_confusion_matrix(confusion, out_dir / "confusion_matrix.png")
            plot_per_class_f1(per_class_f1, out_dir / "per_class_f1.png")
            logger.info("Saved confusion matrix and per-class F1.")
        if "history" in metrics:
            plot_learning_curves(metrics["history"], out_dir / "learning_curves.png")
            logger.info("Saved learning curves.")
        if args.external and Path(args.external).exists():
            internal = internal_macro_f1(metrics)
            external = load_json(args.external)["metrics"]["macro_f1"]
            if internal is not None:
                plot_internal_vs_external(internal, external, out_dir / "internal_vs_external.png")
                logger.info("Saved internal-vs-external comparison.")
    else:
        logger.warning("Metrics file %s not found; skipping metric-based figures.", args.metrics)

    if args.probs and Path(args.probs).exists():
        probs = np.load(args.probs)
        y_true, y_prob = probs["y_true"], probs["y_prob"]
        plot_roc_curves(y_true, y_prob, out_dir / "roc_curves.png")
        plot_pr_curves(y_true, y_prob, out_dir / "pr_curves.png")
        plot_reliability(y_true, y_prob, out_dir / "reliability_diagram.png")
        logger.info("Saved ROC, PR and reliability figures.")

    if args.data and Path(args.data).exists():
        plot_class_distribution(load_processed_dataset(args.data).y, out_dir / "class_distribution.png")
        logger.info("Saved class distribution.")


if __name__ == "__main__":
    main()
