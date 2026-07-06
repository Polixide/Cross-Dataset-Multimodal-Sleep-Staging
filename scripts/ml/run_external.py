"""External validation of a frozen ML model on a second dataset (e.g. SHHS).

Loads a model trained on the development dataset (Sleep-EDF) and evaluates it,
without any re-tuning, on an external processed dataset — quantifying
cross-dataset generalization (master plan: Internal vs external testing). For a
deep-learning model, use scripts/dl/run_dl.py --external instead.

Usage:
    python scripts/ml/run_external.py --model-path results/logs/ml_model.pkl \
        --external data/processed/shhs.npz
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from src.common.data import load_processed_dataset
from src.common.evaluation import compute_metrics, probabilistic_metrics
from src.ml.features import extract_features_dataset
from src.common.utils import STAGE_NAMES, ensure_dir, get_logger, load_pickle, save_json, set_seed

logger = get_logger("run_external")
LABELS = list(range(len(STAGE_NAMES)))


def main():
    parser = argparse.ArgumentParser(description="Evaluate a frozen ML model on an external dataset.")
    parser.add_argument("--model-path", required=True, help="Pickled fitted estimator.")
    parser.add_argument("--external", required=True, help="External processed .npz (x, y, subjects).")
    parser.add_argument("--sfreq", type=float, default=100.0)
    parser.add_argument("--out", default="results/tables/external_metrics.json")
    parser.add_argument("--probs-out", default="results/logs/external_test_probs.npz")
    args = parser.parse_args()

    set_seed()
    model = load_pickle(args.model_path)
    dataset = load_processed_dataset(args.external)
    logger.info("External set: %d epochs from %d subjects.", len(dataset.y), dataset.n_subjects)

    # Extract the same features used in training if the external data is raw epochs.
    x = dataset.x
    if x.ndim == 3:
        logger.info("Extracting features from raw external epochs...")
        x, _ = extract_features_dataset(x, args.sfreq)

    y_pred = model.predict(x)
    metrics = compute_metrics(dataset.y, y_pred, labels=LABELS)
    logger.info("External macro-F1: %.3f | kappa: %.3f", metrics["macro_f1"], metrics["cohen_kappa"])

    output = {"external_data": args.external, "metrics": metrics}
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(x)
        output["prob_metrics"] = probabilistic_metrics(dataset.y, prob)
        ensure_dir(Path(args.probs_out).parent)
        np.savez_compressed(args.probs_out, y_true=dataset.y, y_prob=prob)

    save_json(output, args.out)
    logger.info("Saved external metrics to %s", args.out)


if __name__ == "__main__":
    main()
