"""Train and evaluate the feature-based ML pipeline.

Extracts features from raw epochs, runs subject-wise cross-validation, fits the
final model on the training subjects, calibrates on the validation subjects, and
evaluates on the held-out internal test subjects. The fitted (calibrated) model
is saved for external validation, together with the test-set probabilities.

Headline scalar metrics are appended to a shared CSV comparison table
(results/tables/ml_train_sleep.csv), one row per model run, so several models can be
compared side by side. The detailed nested metrics (confusion matrix, per-class
arrays) needed by make_figures are saved per model under results/logs/.

Examples:
    python scripts/ml/run_ml.py --data data/processed/sleep_edf.npz --model rf
    python scripts/ml/run_ml.py --data data/processed/sleep_edf.npz --model logreg --balance smote --tune
    python scripts/ml/run_ml.py --data data/processed/sleep_edf.npz --model rf --loso
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from scipy.special import softmax
from sklearn.base import clone
from tqdm import tqdm

from src.common.data import leave_one_subject_out, load_processed_dataset, subject_wise_split
from src.common.evaluation import compute_metrics, probabilistic_metrics, summarize_folds
from src.ml.features import extract_features_dataset
from src.ml.models import adapt_params_to_estimator, build_estimator
from src.ml.train import calibrate_classifier, cross_validate_ml, tune_ml
from src.common.utils import (
    STAGE_NAMES,
    append_metrics_row,
    ensure_dir,
    get_logger,
    load_json,
    save_json,
    save_pickle,
    set_seed,
)

logger = get_logger("run_ml")
LABELS = list(range(len(STAGE_NAMES)))

# Columns that identify a run in the comparison table. Re-running the same
# configuration replaces its row; different configs accumulate as new rows.
KEY_COLS = ["model", "balance", "calibration", "tuned", "loso"]


def metrics_row(model, balance, calibration, tuned, loso, test_metrics,
                cv_summary=None, prob_metrics=None, best_params=None,
                overfitting=None):
    """Flatten the headline scalar metrics into one row for the CSV table."""
    row = {
        "model": model,
        "balance": balance,
        "calibration": calibration,
        "tuned": tuned,
        "loso": loso,
        "best_params": "" if best_params is None else str(best_params),
        "accuracy": test_metrics["accuracy"],
        "balanced_accuracy": test_metrics["balanced_accuracy"],
        "macro_f1": test_metrics["macro_f1"],
        "weighted_f1": test_metrics["weighted_f1"],
        "cohen_kappa": test_metrics["cohen_kappa"],
    }
    for name, f1 in zip(STAGE_NAMES, test_metrics["per_class_f1"]):
        row[f"f1_{name}"] = f1
    if cv_summary is not None:
        row["cv_macro_f1_mean"] = cv_summary["macro_f1"]["mean"]
        row["cv_macro_f1_std"] = cv_summary["macro_f1"]["std"]
        row["cv_cohen_kappa_mean"] = cv_summary["cohen_kappa"]["mean"]
    if prob_metrics is not None:
        row["macro_auprc"] = prob_metrics["macro_auprc"]
        row["macro_roc_auc"] = prob_metrics["macro_roc_auc"]
        row["ece"] = prob_metrics["ece"]
        row["brier"] = prob_metrics["brier"]
    if overfitting is not None:
        row.update(overfitting)
    return row


def get_feature_matrix(dataset, sfreq):
    """Return a 2D feature matrix, extracting features if the data is raw epochs."""
    if dataset.x.ndim == 3:
        logger.info("Extracting features from raw epochs (%d epochs)...", len(dataset.y))
        features, _ = extract_features_dataset(dataset.x, sfreq)
        return features
    return dataset.x


def uncalibrated_probabilities(estimator, x):
    """Return native probabilities or a softmax decision-score proxy.

    Estimators without native probabilities use a softmax decision-score proxy
    only for the diagnostic raw-vs-calibrated comparison. Downstream use should
    prefer the calibrated probabilities.
    """
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(x)
    scores = estimator.decision_function(x)
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    return softmax(scores, axis=1)


def suffixed_path(path, suffix):
    """Insert a suffix before the file extension."""
    path = Path(path)
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


def output_paths(args, calibration, multi_output):
    """Return artifact paths for one calibration variant."""
    suffix = "raw" if calibration == "none" else calibration
    if not multi_output:
        return Path(args.json_out), Path(args.model_out), Path(args.probs_out)
    return (
        suffixed_path(args.json_out, suffix),
        suffixed_path(args.model_out, suffix),
        suffixed_path(args.probs_out, suffix),
    )


def save_run_outputs(args, calibration, fitted_model, y_true, y_pred, y_pred_raw,
                     y_prob, y_prob_raw, test_metrics, test_metrics_raw,
                     prob_metrics, prob_metrics_raw, cv_summary, overfitting,
                     tuning_selection, best_params, multi_output):
    """Persist model, probabilities, JSON metrics, and the CSV row for one run."""
    json_out, model_out, probs_out = output_paths(args, calibration, multi_output)

    save_pickle(fitted_model, model_out)
    ensure_dir(probs_out.parent)
    np.savez_compressed(
        probs_out,
        y_true=y_true,
        y_pred_raw=y_pred_raw,
        y_pred_calibrated=y_pred,
        y_prob_raw=y_prob_raw,
        y_prob=y_prob,
        y_prob_calibrated=y_prob,
    )
    save_json({
        "model": args.model, "balance": args.balance,
        "tuned": bool(args.tune or args.params_from), "best_params": best_params,
        "calibration": calibration,
        "protocol": {
            "cross_validation": "training_subjects_only",
            "validation_role": (
                "probability_calibration_only" if calibration != "none"
                else "not_used_for_uncalibrated_output"
            ),
            "test_role": "final_evaluation_only",
        },
        "cv_summary": cv_summary,
        "overfitting": overfitting,
        "tuning_selection": tuning_selection,
        "test_metrics": test_metrics,
        "test_metrics_raw": test_metrics_raw,
        "test_metrics_calibrated": test_metrics if calibration != "none" else None,
        "test_prob_metrics_raw": prob_metrics_raw,
        "test_prob_metrics_calibrated": prob_metrics if calibration != "none" else None,
    }, json_out)

    row = metrics_row(
        args.model, args.balance, calibration,
        bool(args.tune or args.params_from), loso=False,
        test_metrics=test_metrics, cv_summary=cv_summary,
        prob_metrics=prob_metrics, best_params=best_params,
        overfitting=overfitting,
    )
    append_metrics_row(row, args.out, KEY_COLS)
    logger.info(
        "Saved %s metrics to %s and model to %s",
        calibration, json_out, model_out,
    )


def run_loso(x, y, subjects, model, balance, out_path, json_path, best_params=None):
    """Leave-one-subject-out robustness analysis (secondary)."""
    y_true, y_pred = [], []
    folds = leave_one_subject_out(subjects)
    for train_idx, test_idx in tqdm(
        folds,
        total=len(np.unique(subjects)),
        desc="LOSO subjects",
        unit="subject",
    ):
        estimator = build_estimator(model, balance)
        if best_params:
            estimator.set_params(**adapt_params_to_estimator(estimator, best_params))
        estimator.fit(x[train_idx], y[train_idx])
        y_pred.append(estimator.predict(x[test_idx]))
        y_true.append(y[test_idx])
    metrics = compute_metrics(np.concatenate(y_true), np.concatenate(y_pred), labels=LABELS)
    save_json({
        "model": model,
        "balance": balance,
        "tuned": bool(best_params),
        "best_params": best_params,
        "loso_metrics": metrics,
    }, json_path)
    row = metrics_row(model, balance, calibration="", tuned=bool(best_params), loso=True,
                      test_metrics=metrics, best_params=best_params)
    append_metrics_row(row, out_path, KEY_COLS)
    logger.info("LOSO macro-F1: %.3f | kappa: %.3f", metrics["macro_f1"], metrics["cohen_kappa"])


def main():
    parser = argparse.ArgumentParser(description="Run the feature-based ML benchmark.")
    parser.add_argument("--data", required=True, help="Processed .npz with (x, y, subjects).")
    parser.add_argument("--model", default="rf", choices=["logreg", "rf", "xgb"])
    parser.add_argument("--balance", default="sample_weight",
                        choices=["sample_weight", "class_weight", "none", "smote",
                                 "smote_half", "oversample"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--tune", action="store_true", help="Grid-search hyperparameters.")
    parser.add_argument("--skip-cv", action="store_true",
                        help="Skip GroupKFold evaluation and fit one final model.")
    parser.add_argument("--loso", action="store_true", help="Run leave-one-subject-out instead.")
    parser.add_argument("--params-from", default=None,
                        help="Metrics JSON whose best_params are reused without tuning.")
    parser.add_argument(
        "--calibration",
        default=None,
        choices=["none", "sigmoid", "isotonic"],
        help=("Probability calibration fitted on validation subjects. If omitted, "
              "both uncalibrated ('none') and sigmoid-calibrated outputs are saved "
              "after a single GroupKFold tuning/CV pass. "
              "Use 'none' to keep the estimator's native probabilities. "
              "Sigmoid is the safer default; isotonic is more flexible but can alter "
              "minority-class decisions aggressively."),
    )
    parser.add_argument("--sfreq", type=float, default=100.0)
    parser.add_argument("--out", default="results/tables/ml_train_sleep.csv",
                        help="Comparison table (CSV). Accumulates one row per "
                             "model run instead of overwriting.")
    parser.add_argument("--json-out", default=None,
                        help="Detailed metrics JSON (used by make_figures). "
                             "Defaults to results/logs/ml_metrics_<model>.json.")
    parser.add_argument("--model-out", default=None,
                        help="Fitted model path. Defaults to "
                             "results/logs/ml_model_<model>.pkl.")
    parser.add_argument("--probs-out", default=None,
                        help="Test-set probabilities path. Defaults to "
                             "results/logs/ml_test_probs_<model>.npz.")
    args = parser.parse_args()
    if args.tune and args.params_from:
        parser.error("Use either --tune or --params-from, not both.")

    # Per-model detailed artifacts so different models don't overwrite each
    # other; the CSV comparison table (--out) accumulates a row per model.
    if args.json_out is None:
        args.json_out = f"results/logs/ml_metrics_{args.model}.json"
    if args.model_out is None:
        args.model_out = f"results/logs/ml_model_{args.model}.pkl"
    if args.probs_out is None:
        args.probs_out = f"results/logs/ml_test_probs_{args.model}.npz"

    set_seed()
    dataset = load_processed_dataset(args.data)
    x = get_feature_matrix(dataset, args.sfreq)
    y, subjects = dataset.y, dataset.subjects
    logger.info("Loaded %d epochs from %d subjects (%d features).",
                len(y), dataset.n_subjects, x.shape[1])

    if args.loso:
        loso_params = None
        if args.params_from:
            loso_params = load_json(args.params_from).get("best_params")
            if not loso_params:
                parser.error(f"No best_params found in {args.params_from}")
        run_loso(
            x, y, subjects, args.model, args.balance, args.out, args.json_out,
            best_params=loso_params,
        )
        return

    train_idx, val_idx, test_idx = subject_wise_split(subjects)
    logger.info(
        "Subject-wise split: train=%d, validation=%d, test=%d subjects.",
        len(np.unique(subjects[train_idx])),
        len(np.unique(subjects[val_idx])),
        len(np.unique(subjects[test_idx])),
    )

    # Tune only on training subjects. Configurations with a train-validation
    # Macro-F1 gap above 0.10 are rejected by the selector.
    n_splits = min(args.folds, len(np.unique(subjects[train_idx])))
    best_params = None
    tuning_selection = None
    if args.tune:
        final_model, best_params, tuning_selection = tune_ml(
            build_estimator(args.model, args.balance), args.model,
            x[train_idx], y[train_idx], subjects[train_idx], n_splits=n_splits,
        )
        logger.info(
            "Best constrained params: %s (CV macro-F1 %.3f | gap %.3f | constraint=%s)",
            best_params, tuning_selection["cv_macro_f1"],
            tuning_selection["overfitting_gap"],
            tuning_selection["constraint_satisfied"],
        )
        model_builder = lambda: clone(final_model)
    else:
        final_model = build_estimator(args.model, args.balance)
        if args.params_from:
            source = load_json(args.params_from)
            best_params = source.get("best_params")
            if not best_params:
                parser.error(f"No best_params found in {args.params_from}")
            adapted = adapt_params_to_estimator(final_model, best_params)
            final_model.set_params(**adapted)
            logger.info("Reusing parameters from %s: %s", args.params_from, adapted)
        model_builder = lambda: clone(final_model)

    # Measure the selected model with subject-wise folds. Validation subjects
    # remain untouched until calibration and test subjects until final scoring.
    cv_summary = None
    overfitting = None
    if not args.skip_cv:
        cv_metrics = cross_validate_ml(
            model_builder,
            x[train_idx], y[train_idx], subjects[train_idx],
            n_splits=n_splits, labels=LABELS,
        )
        cv_summary = summarize_folds(cv_metrics)
        cv_train_macro_f1 = float(np.mean([fold["train_macro_f1"] for fold in cv_metrics]))
        overfitting_gap = cv_train_macro_f1 - cv_summary["macro_f1"]["mean"]
        overfitting = {
            "cv_train_macro_f1_mean": cv_train_macro_f1,
            "overfitting_gap": overfitting_gap,
            "overfitted": bool(overfitting_gap > 0.10),
            "overfitting_status": "overfitted" if overfitting_gap > 0.10 else "not_overfitted",
        }
        logger.info("CV macro-F1: %.3f +/- %.3f",
                    cv_summary["macro_f1"]["mean"], cv_summary["macro_f1"]["std"])
        logger.info(
            "CV train macro-F1: %.3f | gap: %.3f | %s",
            cv_train_macro_f1, overfitting_gap, overfitting["overfitting_status"],
        )
    else:
        logger.info("Skipping GroupKFold; fitting the final model once.")

    # The tuned estimator is already refit on all training subjects.
    if not args.tune:
        final_model.fit(x[train_idx], y[train_idx])

    # Evaluate the uncalibrated classifier and its probabilities separately.
    # Calibration is intended to improve probability quality (ECE/Brier), and is
    # not guaranteed to improve the argmax decision or macro-F1.
    y_pred_raw = final_model.predict(x[test_idx])
    prob_raw = uncalibrated_probabilities(final_model, x[test_idx])
    test_metrics_raw = compute_metrics(y[test_idx], y_pred_raw, labels=LABELS)
    prob_metrics_raw = probabilistic_metrics(y[test_idx], prob_raw)

    calibration_modes = ["none", "sigmoid"] if args.calibration is None else [args.calibration]
    multi_output = len(calibration_modes) > 1
    for calibration in calibration_modes:
        if calibration == "none":
            logger.info("Calibration disabled; retaining native model probabilities.")
            save_run_outputs(
                args, calibration, final_model,
                y[test_idx], y_pred_raw, y_pred_raw, prob_raw, prob_raw,
                test_metrics_raw, test_metrics_raw, prob_metrics_raw, prob_metrics_raw,
                cv_summary, overfitting, tuning_selection, best_params, multi_output,
            )
            continue

        calibrated_model = calibrate_classifier(
            final_model, x[val_idx], y[val_idx], method=calibration
        )
        y_pred_cal = calibrated_model.predict(x[test_idx])
        prob_cal = calibrated_model.predict_proba(x[test_idx])
        test_metrics_cal = compute_metrics(y[test_idx], y_pred_cal, labels=LABELS)
        prob_metrics_cal = probabilistic_metrics(y[test_idx], prob_cal)
        logger.info(
            "Test macro-F1 raw %.3f -> %s %.3f | ECE raw %.3f -> %s %.3f",
            test_metrics_raw["macro_f1"], calibration, test_metrics_cal["macro_f1"],
            prob_metrics_raw["ece"], calibration, prob_metrics_cal["ece"],
        )
        save_run_outputs(
            args, calibration, calibrated_model,
            y[test_idx], y_pred_cal, y_pred_raw, prob_cal, prob_raw,
            test_metrics_cal, test_metrics_raw, prob_metrics_cal, prob_metrics_raw,
            cv_summary, overfitting, tuning_selection, best_params, multi_output,
        )

    logger.info("Appended metrics to %s", args.out)


if __name__ == "__main__":
    main()
