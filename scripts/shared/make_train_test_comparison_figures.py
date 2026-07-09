"""Build train Sleep vs external-dataset comparison figures from summary CSVs."""
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

logger = get_logger("make_train_test_comparison_figures")

RUN_ORDER = [
    ("logreg_none_class-weight", "LogReg none"),
    ("logreg_sigmoid_class-weight", "LogReg sigmoid"),
    ("rf_class-weight", "RF"),
    ("xgb_class-weight", "XGB"),
    ("xgb_half-smote", "XGB half-SMOTE"),
]

METRIC_LABELS = {
    "macro_f1": "Macro-F1",
    "balanced_accuracy": "Balanced accuracy",
    "accuracy": "Accuracy",
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


def train_run_key(row):
    model = str(row["model"])
    balance = str(row["balance"])
    calibration = str(row.get("calibration", ""))
    loso = str(row.get("loso", "False")).lower() == "true"
    if loso:
        return None
    if model == "logreg" and calibration == "none":
        return "logreg_none_class-weight"
    if model == "logreg" and calibration == "sigmoid":
        return "logreg_sigmoid_class-weight"
    if model == "rf" and balance == "class_weight":
        return "rf_class-weight"
    if model == "xgb" and balance == "class_weight":
        return "xgb_class-weight"
    if model == "xgb" and balance == "smote_half":
        return "xgb_half-smote"
    return None


def test_run_key(row):
    model = str(row["model"])
    balance = str(row["balance"])
    if model in {"logreg_raw", "logreg_none"}:
        return "logreg_none_class-weight"
    if model == "logreg_sigmoid":
        return "logreg_sigmoid_class-weight"
    if model in {"rf", "rf_none", "rf_sigmoid"} and balance == "class_weight":
        return "rf_class-weight"
    if model in {"xgb", "xgb_none", "xgb_sigmoid"} and balance == "class_weight":
        return "xgb_class-weight"
    if model in {"xgb_smote_half", "xgb_smote_half_sigmoid"}:
        return "xgb_half-smote"
    return None


def keyed_frame(frame, key_func):
    out = frame.copy()
    out["run_key"] = out.apply(key_func, axis=1)
    out = out.dropna(subset=["run_key"])
    out["_priority"] = out.apply(run_priority, axis=1)
    out = out.sort_values("_priority").drop_duplicates("run_key", keep="last")
    out = out.drop(columns="_priority")
    return out.set_index("run_key", drop=False)


def run_priority(row):
    """Choose one canonical row when old/new labels map to the same run."""
    model = str(row.get("model", ""))
    balance = str(row.get("balance", ""))
    calibration = str(row.get("calibration", ""))
    if model in {"logreg", "logreg_raw", "logreg_none"} and calibration in {"none", ""}:
        return 30
    if model in {"rf_sigmoid", "xgb_sigmoid", "xgb_smote_half_sigmoid"}:
        return 30
    if model in {"rf", "xgb", "xgb_smote_half"}:
        return 20
    if calibration == "sigmoid":
        return 20
    if balance == "smote_half":
        return 15
    return 10


def ordered_keys(train, test):
    return [key for key, _ in RUN_ORDER if key in train.index and key in test.index]


def plot_test_model_comparison(test, keys, metric, dataset_name, out_path):
    labels = [label for key, label in RUN_ORDER if key in keys]
    values = [test.loc[key, metric] for key in keys]
    plt.figure(figsize=(7, 4))
    plt.bar(labels, values, color="#4C78A8")
    plt.ylim(0, 1 if metric not in {"brier"} else max(values) * 1.2)
    plt.ylabel(METRIC_LABELS[metric])
    plt.title(f"{dataset_name} test {METRIC_LABELS[metric]}")
    plt.xticks(rotation=20, ha="right")
    save_figure(out_path)


def plot_train_vs_test(train, test, keys, metric, dataset_name, out_path):
    labels = [label for key, label in RUN_ORDER if key in keys]
    x = np.arange(len(keys))
    width = 0.36
    train_values = [train.loc[key, metric] for key in keys]
    test_values = [test.loc[key, metric] for key in keys]
    ymax = 1 if metric not in {"brier"} else max(train_values + test_values) * 1.2

    plt.figure(figsize=(8, 4.5))
    plt.bar(x - width / 2, train_values, width, label="Sleep train/internal", color="#59A14F")
    plt.bar(x + width / 2, test_values, width, label=f"{dataset_name} test", color="#E15759")
    plt.ylim(0, ymax)
    plt.ylabel(METRIC_LABELS[metric])
    plt.title(f"Sleep vs {dataset_name} {METRIC_LABELS[metric]}")
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.legend()
    save_figure(out_path)


def plot_per_class_f1(train, test, keys, dataset_name, out_path):
    cols = 2
    rows = int(np.ceil(len(keys) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11, 3.6 * rows), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    x = np.arange(len(STAGE_NAMES))
    width = 0.36
    for axis, key in zip(axes, keys):
        label = dict(RUN_ORDER)[key]
        train_values = [train.loc[key, f"f1_{stage}"] for stage in STAGE_NAMES]
        test_values = [test.loc[key, f"f1_{stage}"] for stage in STAGE_NAMES]
        axis.bar(x - width / 2, train_values, width, label="Sleep train/internal", color="#59A14F")
        axis.bar(x + width / 2, test_values, width, label=f"{dataset_name} test", color="#E15759")
        axis.set_title(label)
        axis.set_xticks(x, STAGE_NAMES)
        axis.set_ylim(0, 1)
        axis.set_ylabel("F1 score")
    for axis in axes[len(keys):]:
        axis.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle(f"Per-class F1: Sleep vs {dataset_name}", y=1.02)
    save_figure(out_path)


def main():
    parser = argparse.ArgumentParser(description="Build Sleep-vs-external summary figures.")
    parser.add_argument("--train", default="results/tables/ml_train_sleep.csv")
    parser.add_argument("--test", default="results/tables/ml_test_hmc.csv")
    parser.add_argument("--dataset-name", default="HMC")
    parser.add_argument("--out-dir", default="results/figures/test/summary")
    args = parser.parse_args()

    train = keyed_frame(pd.read_csv(args.train), train_run_key)
    test = keyed_frame(pd.read_csv(args.test), test_run_key)
    keys = ordered_keys(train, test)
    out_dir = ensure_dir(args.out_dir)
    dataset_slug = args.dataset_name.lower()

    for metric in ("macro_f1", "balanced_accuracy"):
        plot_test_model_comparison(
            test, keys, metric, args.dataset_name,
            out_dir / f"{dataset_slug}_model_comparison_{metric}.png",
        )
        plot_train_vs_test(
            train, test, keys, metric, args.dataset_name,
            out_dir / f"sleep_vs_{dataset_slug}_{metric}.png",
        )

    for metric in ("ece", "brier"):
        plot_train_vs_test(
            train, test, keys, metric, args.dataset_name,
            out_dir / f"sleep_vs_{dataset_slug}_{metric}.png",
        )

    plot_per_class_f1(
        train, test, keys, args.dataset_name,
        out_dir / f"sleep_vs_{dataset_slug}_per_class_f1.png",
    )
    logger.info("Saved %d Sleep-vs-%s comparison figure groups to %s", len(keys), args.dataset_name, out_dir)


if __name__ == "__main__":
    main()
