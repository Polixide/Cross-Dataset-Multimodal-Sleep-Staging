"""Build internal Sleep-EDF GroupKFold-vs-test comparison figures."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.utils import ensure_dir, get_logger

logger = get_logger("make_internal_cv_test_figures")

METRICS = [
    ("macro_f1", "cv_macro_f1_mean", "Macro-F1"),
    ("cohen_kappa", "cv_cohen_kappa_mean", "Cohen's kappa"),
]


def save_figure(path):
    path = Path(path)
    ensure_dir(path.parent)
    path.unlink(missing_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def run_label(row):
    model = row["model"]
    balance = row["balance"].replace("_", "-")
    calibration = row["calibration"] if pd.notna(row["calibration"]) else "none"
    return f"{model} {balance} {calibration}"


def load_internal_runs(path):
    table = pd.read_csv(path)
    table = table[table["loso"].astype(str).str.lower() == "false"].copy()
    table = table.dropna(subset=["cv_macro_f1_mean", "cv_cohen_kappa_mean"])
    table["run"] = table.apply(run_label, axis=1)
    return table


def write_summary(table, out_path):
    rows = []
    for _, row in table.iterrows():
        rows.append({
            "run": row["run"],
            "model": row["model"],
            "balance": row["balance"],
            "calibration": row["calibration"],
            "cv_macro_f1_mean": row["cv_macro_f1_mean"],
            "internal_test_macro_f1": row["macro_f1"],
            "macro_f1_gap_test_minus_cv": row["macro_f1"] - row["cv_macro_f1_mean"],
            "cv_cohen_kappa_mean": row["cv_cohen_kappa_mean"],
            "internal_test_cohen_kappa": row["cohen_kappa"],
            "cohen_kappa_gap_test_minus_cv": row["cohen_kappa"] - row["cv_cohen_kappa_mean"],
            "cv_train_macro_f1_mean": row.get("cv_train_macro_f1_mean", np.nan),
            "overfitting_gap": row.get("overfitting_gap", np.nan),
            "overfitting_status": row.get("overfitting_status", ""),
        })
    summary = pd.DataFrame(rows)
    ensure_dir(Path(out_path).parent)
    summary.to_csv(out_path, index=False)
    return summary


def plot_cv_vs_test(table, test_col, cv_col, label, out_path):
    x = np.arange(len(table))
    width = 0.36
    plt.figure(figsize=(9, 4.8))
    plt.bar(x - width / 2, table[cv_col], width, label="GroupKFold CV (train subjects)", color="#4C78A8")
    plt.bar(x + width / 2, table[test_col], width, label="Internal test Sleep-EDF", color="#F58518")
    plt.ylim(0, 1)
    plt.ylabel(label)
    plt.title(f"GroupKFold vs internal test: {label}")
    plt.xticks(x, table["run"], rotation=22, ha="right")
    plt.legend()
    save_figure(out_path)


def plot_macro_f1_train_cv_test(table, out_path):
    x = np.arange(len(table))
    width = 0.26
    plt.figure(figsize=(9, 4.8))
    plt.bar(x - width, table["cv_train_macro_f1_mean"], width, label="CV fold train", color="#54A24B")
    plt.bar(x, table["cv_macro_f1_mean"], width, label="GroupKFold CV", color="#4C78A8")
    plt.bar(x + width, table["macro_f1"], width, label="Internal test", color="#F58518")
    plt.ylim(0, 1)
    plt.ylabel("Macro-F1")
    plt.title("Macro-F1: fold train vs GroupKFold CV vs internal test")
    plt.xticks(x, table["run"], rotation=22, ha="right")
    plt.legend()
    save_figure(out_path)


def main():
    parser = argparse.ArgumentParser(description="Build internal CV-vs-test figures.")
    parser.add_argument("--metrics", default="results/tables/ml_train_sleep.csv")
    parser.add_argument(
        "--summary-out",
        default=None,
        help="Optional CSV summary path. By default, only figures are written.",
    )
    parser.add_argument("--out-dir", default="results/figures/train/summary")
    args = parser.parse_args()

    table = load_internal_runs(args.metrics)
    if args.summary_out:
        summary = write_summary(table, args.summary_out)
    else:
        summary = None
    out_dir = ensure_dir(args.out_dir)

    for test_col, cv_col, label in METRICS:
        plot_cv_vs_test(
            table, test_col, cv_col, label,
            out_dir / f"groupkfold_vs_internal_test_{test_col}.png",
        )
    plot_macro_f1_train_cv_test(
        table,
        out_dir / "macro_f1_fold_train_vs_groupkfold_vs_internal_test.png",
    )
    logger.info("Saved CV-vs-test figures for %d rows to %s", len(table), out_dir)
    if summary is not None:
        logger.info("Saved CV-vs-test summary to %s", args.summary_out)


if __name__ == "__main__":
    main()
