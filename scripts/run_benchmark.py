"""Feature-based ML benchmark: every model x every imbalance strategy (INTERNAL).

For each (model, balance) it runs subject-wise cross-validation on the
development subjects, fits on the training subjects, calibrates on validation,
and evaluates on the held-out INTERNAL test subjects. The single best config by
CV macro-F1 is then re-tuned. All results go to one comparison table
(CSV + JSON) plus the best fitted model.

External validation (HMC) is deliberately NOT done here: the external set must be
touched only once, with the final frozen model (scripts/run_external.py), so it
never influences model selection. Running it inside the sweep would leak it.

Features are extracted once and cached to disk, so the 16-config sweep does not
recompute the ~96 features per run.

Usage:
    python scripts/run_benchmark.py --data data/processed/sleep_edf.npz
    python scripts/run_benchmark.py --data data/processed/sleep_edf.npz --models rf gb logreg
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.data_loader import load_processed_dataset, subject_wise_split
from src.evaluate import compute_metrics, probabilistic_metrics, summarize_folds
from src.features import load_or_extract_features
from src.models_ml import build_estimator
from src.train import calibrate_classifier, cross_validate_ml, tune_ml
from src.utils import STAGE_NAMES, ensure_dir, get_logger, save_json, save_pickle, set_seed

logger = get_logger("run_benchmark")
LABELS = list(range(len(STAGE_NAMES)))
MODELS = ["rf", "gb", "logreg", "svm"]
BALANCES = ["none", "class_weight", "oversample", "smote"]


def evaluate_config(model, balance, feature_selection, x, y, subjects, split,
                    n_splits, tune=False, cv_summary=None):
    """Train/evaluate one (model, balance) config on the internal split.

    Returns (record, calibrated_model, (y_test, prob_calibrated)). ``cv_summary``
    can be passed in to reuse an already-computed CV (e.g. for the tuned re-run of
    the best config).
    """
    train_idx, val_idx, test_idx, dev_idx = split

    # Subject-wise CV on the development subjects: the leakage-safe selection
    # signal (it never touches the held-out internal test).
    if cv_summary is None:
        cv = cross_validate_ml(
            lambda: build_estimator(model, balance, feature_selection=feature_selection),
            x.iloc[dev_idx], y[dev_idx], subjects[dev_idx], n_splits=n_splits, labels=LABELS,
        )
        cv_summary = summarize_folds(cv)

    best_params = None
    if tune:
        fitted, best_params, _ = tune_ml(
            build_estimator(model, balance, feature_selection=feature_selection),
            model, x.iloc[train_idx], y[train_idx], subjects[train_idx],
            n_splits=min(n_splits, len(np.unique(subjects[train_idx]))),
        )
    else:
        fitted = build_estimator(model, balance, feature_selection=feature_selection)
        fitted.fit(x.iloc[train_idx], y[train_idx])

    # Calibrate on validation only, then evaluate once on the internal test.
    calibrated = calibrate_classifier(fitted, x.iloc[val_idx], y[val_idx])
    y_pred = calibrated.predict(x.iloc[test_idx])
    prob_cal = calibrated.predict_proba(x.iloc[test_idx])
    prob_raw = fitted.predict_proba(x.iloc[test_idx])

    record = {
        "model": model, "balance": balance, "tuned": tune, "best_params": best_params,
        "cv_summary": cv_summary,
        "internal_test": compute_metrics(y[test_idx], y_pred, labels=LABELS),
        "internal_test_prob_calibrated": probabilistic_metrics(y[test_idx], prob_cal),
        "internal_test_prob_raw": probabilistic_metrics(y[test_idx], prob_raw),
    }
    return record, calibrated, (y[test_idx], prob_cal)


def flat_row(record):
    """Flatten a config record into one row of the comparison table."""
    test = record["internal_test"]
    cal = record["internal_test_prob_calibrated"]
    raw = record["internal_test_prob_raw"]
    cv = record["cv_summary"]
    row = {
        "model": record["model"], "balance": record["balance"], "tuned": record["tuned"],
        "cv_macro_f1": cv["macro_f1"]["mean"], "cv_macro_f1_std": cv["macro_f1"]["std"],
        "cv_kappa": cv["cohen_kappa"]["mean"],
        "test_macro_f1": test["macro_f1"], "test_weighted_f1": test["weighted_f1"],
        "test_bal_acc": test["balanced_accuracy"], "test_kappa": test["cohen_kappa"],
        "test_ece_raw": raw["ece"], "test_ece_cal": cal["ece"],
        "test_brier_cal": cal["brier"], "test_macro_auprc": cal["macro_auprc"],
    }
    for i, name in enumerate(STAGE_NAMES):
        row[f"test_f1_{name}"] = test["per_class_f1"][i]
    return row


def main():
    parser = argparse.ArgumentParser(description="Internal feature-based ML benchmark (all models x all imbalance strategies).")
    parser.add_argument("--data", required=True, help="Development processed .npz (raw epochs).")
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS,
                        help="Models to benchmark (default: all four).")
    parser.add_argument("--balances", nargs="+", default=BALANCES, choices=BALANCES,
                        help="Imbalance strategies to compare (default: all four).")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--feature-selection", action="store_true",
                        help="Prepend leakage-safe variance + correlation pruning to every config.")
    parser.add_argument("--corr-threshold", type=float, default=0.95)
    parser.add_argument("--sfreq", type=float, default=None)
    parser.add_argument("--feature-cache", default=None,
                        help="Feature cache .npy path (default: data/processed/cache/<stem>_features.npy).")
    parser.add_argument("--out-csv", default="results/tables/benchmark_ml.csv")
    parser.add_argument("--out-json", default="results/tables/benchmark_ml.json")
    parser.add_argument("--model-out", default="results/logs/benchmark_best_model.pkl")
    parser.add_argument("--probs-out", default="results/logs/benchmark_best_test_probs.npz")
    args = parser.parse_args()

    set_seed()
    dataset = load_processed_dataset(args.data)
    sfreq = args.sfreq or dataset.sfreq or 100.0
    y, subjects = dataset.y, dataset.subjects

    cache = args.feature_cache or f"data/processed/cache/{Path(args.data).stem}_features.npy"
    logger.info("Loading/extracting features (cache: %s)...", cache)
    x = load_or_extract_features(dataset.x, sfreq, cache)
    logger.info("Features: %d epochs x %d features from %d subjects.",
                x.shape[0], x.shape[1], dataset.n_subjects)

    train_idx, val_idx, test_idx = subject_wise_split(subjects)
    dev_idx = np.concatenate([train_idx, val_idx])
    n_splits = min(args.folds, len(np.unique(subjects[dev_idx])))
    split = (train_idx, val_idx, test_idx, dev_idx)
    feature_selection = {"correlation_threshold": args.corr_threshold} if args.feature_selection else None

    records, rows = [], []
    configs = [(m, b) for m in args.models for b in args.balances]
    for model, balance in tqdm(configs, desc="Benchmark", unit="config"):
        record, _, _ = evaluate_config(model, balance, feature_selection, x, y, subjects, split, n_splits)
        records.append(record)
        rows.append(flat_row(record))
        logger.info("model=%s balance=%s | CV macroF1=%.3f | test macroF1=%.3f kappa=%.3f",
                    model, balance, record["cv_summary"]["macro_f1"]["mean"],
                    record["internal_test"]["macro_f1"], record["internal_test"]["cohen_kappa"])

    # Select the best config by CV macro-F1 (never by the internal test), then tune it.
    best_i = int(np.argmax([r["cv_summary"]["macro_f1"]["mean"] for r in records]))
    best = records[best_i]
    logger.info("Best config by CV: %s + %s (CV macroF1 %.3f). Tuning it...",
                best["model"], best["balance"], best["cv_summary"]["macro_f1"]["mean"])
    tuned_record, tuned_model, (y_test, prob_test) = evaluate_config(
        best["model"], best["balance"], feature_selection, x, y, subjects, split, n_splits,
        tune=True, cv_summary=best["cv_summary"],
    )
    records.append(tuned_record)
    rows.append(flat_row(tuned_record))
    logger.info("Tuned best: params=%s | test macroF1 %.3f (untuned %.3f)",
                tuned_record["best_params"], tuned_record["internal_test"]["macro_f1"],
                best["internal_test"]["macro_f1"])

    table = pd.DataFrame(rows)
    ensure_dir(Path(args.out_csv).parent)
    table.to_csv(args.out_csv, index=False)
    save_json({
        "data": args.data, "sfreq": sfreq, "n_features": int(x.shape[1]),
        "feature_names": list(x.columns), "feature_selection": feature_selection,
        "best": {"model": best["model"], "balance": best["balance"],
                 "best_params": tuned_record["best_params"]},
        "configs": records,
    }, args.out_json)
    save_pickle(tuned_model, args.model_out)
    ensure_dir(Path(args.probs_out).parent)
    np.savez_compressed(args.probs_out, y_true=y_test, y_prob=prob_test)

    logger.info("Saved benchmark table -> %s | details -> %s | best model -> %s",
                args.out_csv, args.out_json, args.model_out)
    summary_cols = ["model", "balance", "tuned", "cv_macro_f1", "test_macro_f1", "test_kappa", "test_ece_cal"]
    logger.info("Benchmark summary (internal):\n%s", table[summary_cols].to_string(index=False))


if __name__ == "__main__":
    main()
