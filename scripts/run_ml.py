"""Train and evaluate the feature-based ML pipeline.

Extracts features from raw epochs, runs subject-wise cross-validation, fits the
final model on the training subjects, calibrates on the validation subjects, and
evaluates on the held-out internal test subjects. The fitted (calibrated) model
is saved for external validation, together with the test-set probabilities.

Examples:
    python scripts/run_ml.py --data data/processed/sleep_edf.npz --model rf
    python scripts/run_ml.py --data data/processed/sleep_edf.npz --model svm --balance smote --tune
    python scripts/run_ml.py --data data/processed/sleep_edf.npz --model rf --loso
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.data_loader import leave_one_subject_out, load_processed_dataset, subject_wise_split
from src.evaluate import compute_metrics, probabilistic_metrics, summarize_folds
from src.features import extract_features_dataset
from src.models_ml import build_estimator
from src.train import calibrate_classifier, cross_validate_ml, tune_ml
from src.utils import STAGE_NAMES, ensure_dir, get_logger, save_json, save_pickle, set_seed

logger = get_logger("run_ml")
LABELS = list(range(len(STAGE_NAMES)))


def get_feature_matrix(dataset, sfreq):
    """Return the feature matrix as a DataFrame, extracting it if data is raw epochs.

    The named columns flow into the fitted estimator (``feature_names_in_``) so
    SHAP and reporting can label features without a separate name list. A dataset
    that is already a 2D feature matrix is wrapped with generic column names.
    """
    if dataset.x.ndim == 3:
        logger.info("Extracting features from raw epochs (%d epochs, sfreq=%g Hz)...",
                    len(dataset.y), sfreq)
        return extract_features_dataset(dataset.x, sfreq)
    x = np.asarray(dataset.x)
    return pd.DataFrame(x, columns=[f"feature_{i}" for i in range(x.shape[1])])


def selected_feature_count(fitted_model):
    """Best-effort count of features surviving the pipeline's selection steps."""
    if not hasattr(fitted_model, "named_steps"):
        return None
    n = None
    for step_name in ("variance_filter", "correlation_pruner", "model_select"):
        step = fitted_model.named_steps.get(step_name)
        if step is not None and hasattr(step, "get_support"):
            n = int(np.sum(step.get_support()))
    return n


def run_loso(x, y, subjects, model, balance, feature_selection, out_path):
    """Leave-one-subject-out robustness analysis (secondary)."""
    y_true, y_pred = [], []
    folds = leave_one_subject_out(subjects)
    for train_idx, test_idx in tqdm(folds, desc="LOSO", unit="subject"):
        estimator = build_estimator(model, balance, feature_selection=feature_selection)
        estimator.fit(x.iloc[train_idx], y[train_idx])
        y_pred.append(estimator.predict(x.iloc[test_idx]))
        y_true.append(y[test_idx])
    metrics = compute_metrics(np.concatenate(y_true), np.concatenate(y_pred), labels=LABELS)
    save_json({"model": model, "balance": balance, "loso_metrics": metrics}, out_path)
    logger.info("LOSO macro-F1: %.3f | kappa: %.3f", metrics["macro_f1"], metrics["cohen_kappa"])


def main():
    parser = argparse.ArgumentParser(description="Run the feature-based ML benchmark.")
    parser.add_argument("--data", required=True, help="Processed .npz with (x, y, subjects).")
    parser.add_argument("--model", default="rf", choices=["svm", "rf", "gb", "logreg"])
    parser.add_argument("--balance", default="class_weight",
                        choices=["class_weight", "none", "smote", "oversample"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--tune", action="store_true", help="Grid-search hyperparameters.")
    parser.add_argument("--loso", action="store_true", help="Run leave-one-subject-out instead.")
    parser.add_argument("--feature-selection", action="store_true",
                        help="Prepend leakage-safe variance + correlation feature pruning.")
    parser.add_argument("--corr-threshold", type=float, default=0.95,
                        help="Prune features whose |correlation| exceeds this (with --feature-selection).")
    parser.add_argument("--select-k", type=int, default=None,
                        help="Optionally keep only the top-k features by RF importance.")
    parser.add_argument("--sfreq", type=float, default=None,
                        help="Override the sampling rate; defaults to the value stored in the .npz.")
    parser.add_argument("--out", default="results/tables/ml_metrics.json")
    parser.add_argument("--model-out", default="results/logs/ml_model.pkl")
    parser.add_argument("--probs-out", default="results/logs/ml_test_probs.npz")
    args = parser.parse_args()

    set_seed()
    dataset = load_processed_dataset(args.data)
    sfreq = args.sfreq or dataset.sfreq or 100.0
    x = get_feature_matrix(dataset, sfreq)
    feature_names = list(x.columns)
    y, subjects = dataset.y, dataset.subjects
    logger.info("Loaded %d epochs from %d subjects (%d features).",
                len(y), dataset.n_subjects, x.shape[1])

    # Leakage-safe feature selection: config lives in the estimator, so it is
    # re-fit on the training portion of every CV fold (never sees val/test).
    feature_selection = None
    if args.feature_selection:
        feature_selection = {"correlation_threshold": args.corr_threshold, "k_best": args.select_k}
        logger.info("Feature selection ON (|corr|>%.2f%s).", args.corr_threshold,
                    f", top-{args.select_k}" if args.select_k else "")

    if args.loso:
        run_loso(x, y, subjects, args.model, args.balance, feature_selection, args.out)
        return

    train_idx, val_idx, test_idx = subject_wise_split(subjects)
    dev_idx = np.concatenate([train_idx, val_idx])

    # Cross-validation on the development subjects (for tuning / robustness).
    n_splits = min(args.folds, len(np.unique(subjects[dev_idx])))
    cv_metrics = cross_validate_ml(
        lambda: build_estimator(args.model, args.balance, feature_selection=feature_selection),
        x.iloc[dev_idx], y[dev_idx], subjects[dev_idx], n_splits=n_splits, labels=LABELS,
    )
    cv_summary = summarize_folds(cv_metrics)
    logger.info("CV macro-F1: %.3f +/- %.3f",
                cv_summary["macro_f1"]["mean"], cv_summary["macro_f1"]["std"])

    # Fit the final model on the training subjects (optionally tuned).
    best_params = None
    if args.tune:
        tune_splits = min(args.folds, len(np.unique(subjects[train_idx])))
        final_model, best_params, best_score = tune_ml(
            build_estimator(args.model, args.balance, feature_selection=feature_selection),
            args.model, x.iloc[train_idx], y[train_idx], subjects[train_idx], n_splits=tune_splits,
        )
        logger.info("Best params: %s (CV macro-F1 %.3f)", best_params, best_score)
    else:
        final_model = build_estimator(args.model, args.balance, feature_selection=feature_selection)
        final_model.fit(x.iloc[train_idx], y[train_idx])

    # Calibrate on the validation subjects only, then evaluate on the held-out test.
    calibrated = calibrate_classifier(final_model, x.iloc[val_idx], y[val_idx])
    y_pred = calibrated.predict(x.iloc[test_idx])
    test_metrics = compute_metrics(y[test_idx], y_pred, labels=LABELS)
    prob_raw = final_model.predict_proba(x.iloc[test_idx])
    prob_cal = calibrated.predict_proba(x.iloc[test_idx])
    metrics_raw = probabilistic_metrics(y[test_idx], prob_raw)
    metrics_cal = probabilistic_metrics(y[test_idx], prob_cal)
    logger.info("Test macro-F1: %.3f | ECE raw %.3f -> calibrated %.3f",
                test_metrics["macro_f1"], metrics_raw["ece"], metrics_cal["ece"])

    save_pickle(calibrated, args.model_out)
    ensure_dir(Path(args.probs_out).parent)
    np.savez_compressed(args.probs_out, y_true=y[test_idx], y_prob=prob_cal, y_prob_raw=prob_raw)
    save_json({
        "model": args.model, "balance": args.balance, "tuned": args.tune, "best_params": best_params,
        "sfreq": sfreq, "n_features_total": int(x.shape[1]), "feature_names": feature_names,
        "feature_selection": feature_selection,
        "n_features_selected": selected_feature_count(final_model),
        "cv_summary": cv_summary, "test_metrics": test_metrics,
        "test_prob_metrics_raw": metrics_raw, "test_prob_metrics_calibrated": metrics_cal,
    }, args.out)
    logger.info("Saved metrics to %s and model to %s", args.out, args.model_out)


if __name__ == "__main__":
    main()
