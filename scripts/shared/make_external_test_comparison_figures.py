"""Build direct external-test comparison figures between two datasets."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.utils import STAGE_NAMES, ensure_dir, get_logger

logger = get_logger("make_external_test_comparison_figures")

RUN_ORDER = [
    ("logreg_none", "class_weight", "LogReg none"),
    ("logreg_sigmoid", "class_weight", "LogReg sigmoid"),
    ("rf_none", "class_weight", "RF none"),
    ("rf_sigmoid", "class_weight", "RF sigmoid"),
    ("xgb_none", "class_weight", "XGB none"),
    ("xgb_sigmoid", "class_weight", "XGB sigmoid"),
    ("xgb_smote_half_sigmoid", "smote_half", "XGB half-SMOTE sigmoid"),
]

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced accuracy",
    "macro_f1": "Macro-F1",
    "weighted_f1": "Weighted F1",
    "cohen_kappa": "Cohen's kappa",
    "ece": "ECE",
    "brier": "Brier score",
}


def save_figure(path):
    path = Path(path)
    ensure_dir(path.parent)
    path.unlink(missing_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def keyed_frame(path):
    table = pd.read_csv(path)
    table["run_key"] = table["model"].astype(str) + "__" + table["balance"].astype(str)
    table = table.drop_duplicates("run_key", keep="last")
    return table.set_index("run_key", drop=False)


def ordered_runs(left, right):
    runs = []
    for model, balance, label in RUN_ORDER:
        key = f"{model}__{balance}"
        if key in left.index and key in right.index:
            runs.append((key, label))
    return runs


def plot_metric(left, right, runs, metric, left_label, right_label, out_path):
    labels = [label for _, label in runs]
    x = np.arange(len(runs))
    width = 0.36
    left_values = [float(left.loc[key, metric]) for key, _ in runs]
    right_values = [float(right.loc[key, metric]) for key, _ in runs]
    ymax = 1 if metric != "brier" else max(left_values + right_values) * 1.15

    plt.figure(figsize=(10, 4.8))
    plt.bar(x - width / 2, left_values, width, label=left_label, color="#4C78A8")
    plt.bar(x + width / 2, right_values, width, label=right_label, color="#E15759")
    plt.ylim(0, ymax)
    plt.ylabel(METRIC_LABELS[metric])
    plt.title(f"{left_label} vs {right_label}: {METRIC_LABELS[metric]}")
    plt.xticks(x, labels, rotation=24, ha="right")
    plt.legend()
    save_figure(out_path)


def plot_per_class_f1(left, right, runs, left_label, right_label, out_path):
    cols = 2
    rows = int(np.ceil(len(runs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.6 * rows), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    x = np.arange(len(STAGE_NAMES))
    width = 0.36

    for axis, (key, label) in zip(axes, runs):
        left_values = [float(left.loc[key, f"f1_{stage}"]) for stage in STAGE_NAMES]
        right_values = [float(right.loc[key, f"f1_{stage}"]) for stage in STAGE_NAMES]
        axis.bar(x - width / 2, left_values, width, label=left_label, color="#4C78A8")
        axis.bar(x + width / 2, right_values, width, label=right_label, color="#E15759")
        axis.set_title(label)
        axis.set_xticks(x, STAGE_NAMES)
        axis.set_ylim(0, 1)
        axis.set_ylabel("F1 score")

    for axis in axes[len(runs):]:
        axis.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle(f"Per-class F1: {left_label} vs {right_label}", y=1.02)
    save_figure(out_path)


def main():
    parser = argparse.ArgumentParser(description="Build external test-vs-test comparison figures.")
    parser.add_argument("--left", default="results/tables/ml_test_hmc.csv")
    parser.add_argument("--right", default="results/tables/ml_test_isruc.csv")
    parser.add_argument("--left-label", default="HMC")
    parser.add_argument("--right-label", default="ISRUC")
    parser.add_argument("--out-dir", default="results/figures/test_hmc_vs_isruc")
    args = parser.parse_args()

    left = keyed_frame(args.left)
    right = keyed_frame(args.right)
    runs = ordered_runs(left, right)
    if not runs:
        raise RuntimeError("No matching model/balance runs found between the two test tables.")

    out_dir = ensure_dir(args.out_dir)
    slug = f"{args.left_label.lower()}_vs_{args.right_label.lower()}"
    for metric in ("macro_f1", "balanced_accuracy", "accuracy", "weighted_f1", "cohen_kappa", "ece", "brier"):
        plot_metric(
            left,
            right,
            runs,
            metric,
            args.left_label,
            args.right_label,
            out_dir / f"{slug}_{metric}.png",
        )
    plot_per_class_f1(
        left,
        right,
        runs,
        args.left_label,
        args.right_label,
        out_dir / f"{slug}_per_class_f1.png",
    )
    logger.info("Saved %d HMC-vs-ISRUC comparison runs to %s", len(runs), out_dir)


if __name__ == "__main__":
    main()
