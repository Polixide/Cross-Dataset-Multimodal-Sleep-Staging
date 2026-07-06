"""Run the training and evaluation stages of the feature-based ML workflow.

Sleep-EDF and HMC feature files must already exist. The workflow trains and
tunes all ML models with balanced class weights, selects the best model using
train-only GroupKFold CV, compares class weighting with SMOTE only on that model,
evaluates frozen models on HMC, generates figures plus SHAP explanations, and
finishes with LOSO for the best model/imbalance strategy.

Usage:
    python scripts/ml/run_ml_workflow.py
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.common.utils import ensure_dir, get_logger, load_json

logger = get_logger("run_ml_workflow")
ROOT = Path(__file__).resolve().parents[2]


def run_command(*args):
    """Run one project command with the active Python and fail fast."""
    command = [sys.executable, *map(str, args)]
    logger.info("Running: %s", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def model_paths(model, suffix=""):
    tag = f"{model}{suffix}"
    return {
        "metrics": ROOT / f"results/logs/ml_metrics_{tag}.json",
        "model": ROOT / f"results/logs/ml_model_{tag}.pkl",
        "probs": ROOT / f"results/logs/ml_test_probs_{tag}.npz",
        "external": ROOT / f"results/tables/external_metrics_{tag}.json",
        "external_probs": ROOT / f"results/logs/external_test_probs_{tag}.npz",
        "figures": ROOT / f"results/figures/{tag}",
    }


def train_model(model, balance, data, folds, suffix=""):
    paths = model_paths(model, suffix)
    run_command(
        "scripts/ml/run_ml.py", "--data", data, "--model", model,
        "--balance", balance, "--folds", folds, "--tune",
        "--json-out", paths["metrics"],
        "--model-out", paths["model"],
        "--probs-out", paths["probs"],
    )
    return paths


def external_and_reports(model, paths, sleep_features, hmc_features, skip_shap):
    run_command(
        "scripts/ml/run_external.py",
        "--model-path", paths["model"],
        "--external", hmc_features,
        "--out", paths["external"],
        "--probs-out", paths["external_probs"],
    )
    run_command(
        "scripts/shared/make_figures.py",
        "--metrics", paths["metrics"],
        "--probs", paths["probs"],
        "--external", paths["external"],
        "--out-dir", paths["figures"],
    )
    if not skip_shap:
        run_command(
            "scripts/ml/run_shap.py",
            "--model-path", paths["model"],
            "--data", sleep_features,
            "--out-dir", paths["figures"],
        )


def summary_row(model, balance, paths):
    internal = load_json(paths["metrics"])
    external = load_json(paths["external"])
    overfitting = internal["overfitting"]
    return {
        "model": model,
        "balance": balance,
        "cv_macro_f1": internal["cv_summary"]["macro_f1"]["mean"],
        "cv_train_macro_f1": overfitting["cv_train_macro_f1_mean"],
        "overfitting_gap": overfitting["overfitting_gap"],
        "overfitted": overfitting["overfitted"],
        "overfitting_status": overfitting["overfitting_status"],
        "internal_macro_f1": internal["test_metrics"]["macro_f1"],
        "internal_balanced_accuracy": internal["test_metrics"]["balanced_accuracy"],
        "external_hmc_macro_f1": external["metrics"]["macro_f1"],
        "external_hmc_balanced_accuracy": external["metrics"]["balanced_accuracy"],
    }


def main():
    parser = argparse.ArgumentParser(description="Run the complete ML workflow.")
    parser.add_argument("--sleep-features", default="data/processed/sleep_edf_features.npz")
    parser.add_argument("--hmc-features", default="data/processed/hmc_features.npz")
    parser.add_argument("--models", nargs="+", default=["rf", "xgb", "logreg"],
                        choices=["rf", "xgb", "logreg"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--skip-shap", action="store_true",
                        help="Skip the potentially slow permutation-SHAP stage.")
    parser.add_argument("--skip-loso", action="store_true",
                        help="Skip the final, expensive LOSO robustness analysis.")
    parser.add_argument("--summary-out", default="results/tables/ml_workflow_summary.csv")
    args = parser.parse_args()

    sleep_features = ROOT / args.sleep_features
    hmc_features = ROOT / args.hmc_features
    if not sleep_features.exists():
        parser.error(f"Sleep-EDF feature dataset not found: {sleep_features}")
    if not hmc_features.exists():
        parser.error(f"HMC feature dataset not found: {hmc_features}")

    # Primary comparison: identical train-only CV/tuning and class weighting.
    primary = {}
    for model in args.models:
        primary[model] = train_model(
            model, "class_weight", args.sleep_features, args.folds
        )

    # Select without touching the held-out test: training-subject CV is the selector.
    best_model = max(
        args.models,
        key=lambda name: load_json(primary[name]["metrics"])["cv_summary"]["macro_f1"]["mean"],
    )
    logger.info("Best train-CV model: %s. Running its SMOTE ablation.", best_model)
    smote_paths = train_model(
        best_model, "smote", args.sleep_features, args.folds, suffix="_smote"
    )

    class_weight_cv = load_json(primary[best_model]["metrics"])["cv_summary"]["macro_f1"]["mean"]
    smote_cv = load_json(smote_paths["metrics"])["cv_summary"]["macro_f1"]["mean"]
    best_balance = "smote" if smote_cv > class_weight_cv else "class_weight"
    logger.info(
        "Best %s imbalance strategy by train CV: %s "
        "(class_weight=%.3f, SMOTE=%.3f).",
        best_model, best_balance, class_weight_cv, smote_cv,
    )

    configurations = [
        (model, "class_weight", primary[model]) for model in args.models
    ] + [(best_model, "smote", smote_paths)]

    for model, _, paths in configurations:
        external_and_reports(
            model, paths, args.sleep_features, args.hmc_features, args.skip_shap
        )

    loso_metrics = None
    if not args.skip_loso:
        loso_path = ROOT / f"results/logs/loso_metrics_{best_model}_{best_balance}.json"
        selected_paths = smote_paths if best_balance == "smote" else primary[best_model]
        run_command(
            "scripts/ml/run_ml.py",
            "--data", args.sleep_features,
            "--model", best_model,
            "--balance", best_balance,
            "--loso",
            "--params-from", selected_paths["metrics"],
            "--json-out", loso_path,
        )
        loso_metrics = load_json(loso_path)["loso_metrics"]
        run_command(
            "scripts/shared/make_figures.py",
            "--metrics", loso_path,
            "--out-dir", ROOT / f"results/figures/{best_model}_{best_balance}_loso",
        )

    summary = pd.DataFrame([
        summary_row(model, balance, paths)
        for model, balance, paths in configurations
    ])
    summary["selected_best_by_train_cv"] = (
        (summary["model"] == best_model) & (summary["balance"] == "class_weight")
    )
    summary["selected_best_imbalance_by_train_cv"] = (
        (summary["model"] == best_model) & (summary["balance"] == best_balance)
    )
    summary["loso_macro_f1"] = float("nan")
    summary["loso_balanced_accuracy"] = float("nan")
    summary["loso_cohen_kappa"] = float("nan")
    if loso_metrics is not None:
        selected = summary["selected_best_imbalance_by_train_cv"]
        summary.loc[selected, "loso_macro_f1"] = loso_metrics["macro_f1"]
        summary.loc[selected, "loso_balanced_accuracy"] = loso_metrics["balanced_accuracy"]
        summary.loc[selected, "loso_cohen_kappa"] = loso_metrics["cohen_kappa"]
    summary_path = ROOT / args.summary_out
    ensure_dir(summary_path.parent)
    summary.to_csv(summary_path, index=False)
    logger.info("Complete workflow finished. Summary: %s", summary_path)


if __name__ == "__main__":
    main()
