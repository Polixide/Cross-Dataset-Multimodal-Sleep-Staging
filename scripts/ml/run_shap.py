"""Generate global SHAP feature importance for a fitted ML model.

Example:
    python scripts/ml/run_shap.py --model-path results/logs/ml_model_rf.pkl \
        --data data/processed/sleep_edf_features.npz --out-dir results/figures/rf
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from src.common.data import load_processed_dataset, subject_wise_split
from src.ml.explainability import shap_feature_importance
from src.common.utils import get_logger, load_pickle

logger = get_logger("run_shap")


def sample_rows(x, indices, size, rng):
    """Sample rows without replacement from a subject-wise split."""
    size = min(size, len(indices))
    return x[rng.choice(indices, size=size, replace=False)]


def main():
    parser = argparse.ArgumentParser(description="Generate SHAP importance for an ML model.")
    parser.add_argument("--model-path", required=True, help="Fitted calibrated model pickle.")
    parser.add_argument("--data", required=True, help="Processed 2D feature dataset (.npz).")
    parser.add_argument("--out-dir", required=True, help="Directory for the SHAP figure.")
    parser.add_argument("--background-size", type=int, default=50)
    parser.add_argument("--explain-size", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = load_processed_dataset(args.data)
    if dataset.x.ndim != 2:
        parser.error("--data must contain cached 2D features; run cache_ml_features.py first.")

    with np.load(args.data, allow_pickle=True) as data:
        feature_names = data["feature_names"].tolist() if "feature_names" in data else None

    train_idx, _, test_idx = subject_wise_split(dataset.subjects, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    background = sample_rows(dataset.x, train_idx, args.background_size, rng)
    explained = sample_rows(dataset.x, test_idx, args.explain_size, rng)
    model = load_pickle(args.model_path)

    logger.info(
        "Computing SHAP with %d background and %d held-out test epochs.",
        len(background), len(explained),
    )
    path = shap_feature_importance(
        model,
        background,
        explained,
        feature_names=feature_names,
        out_dir=args.out_dir,
        top_k=args.top_k,
    )
    logger.info("Saved SHAP feature importance to %s.", path)


if __name__ == "__main__":
    main()
