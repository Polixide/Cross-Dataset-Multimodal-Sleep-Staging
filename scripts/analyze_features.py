"""Feature-analysis figures for the feature-based ML pipeline (EDA).

Produces the material that motivates (and documents) feature selection:
- a correlation-matrix heatmap of the engineered features,
- Random-Forest impurity importance and permutation importance bar charts,
- a report of which features the leakage-safe correlation pruning keeps/drops.

Everything is computed on the TRAINING subjects only, so the analysis never sees
validation/test data (master plan: Validation rules). Importance uses a Random
Forest fit on the training split; permutation importance is measured on the
validation split.

Usage:
    python scripts/analyze_features.py --data data/processed/sleep_edf.npz
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

from src.data_loader import load_processed_dataset, subject_wise_split
from src.feature_selection import correlation_pruned_indices
from src.features import extract_features_dataset
from src.utils import SLEEP_EDF_CHANNELS, ensure_dir, get_logger, save_json

logger = get_logger("analyze_features")


def plot_correlation_matrix(corr, feature_names, out_path):
    """Heatmap of the absolute feature-feature correlation matrix."""
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(np.abs(corr), cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_title(f"Absolute feature correlation ({len(feature_names)} features)")
    # Sparse ticks: too many features to label individually.
    step = max(1, len(feature_names) // 12)
    ticks = range(0, len(feature_names), step)
    ax.set_xticks(list(ticks))
    ax.set_xticklabels([feature_names[i] for i in ticks], rotation=90, fontsize=6)
    ax.set_yticks(list(ticks))
    ax.set_yticklabels([feature_names[i] for i in ticks], fontsize=6)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="|Pearson r|")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_importance(importances, feature_names, title, out_path, top_k=20, errors=None):
    """Horizontal bar chart of the top-k most important features."""
    order = np.argsort(importances)[-top_k:]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.32 * len(order))))
    ax.barh([feature_names[i] for i in order], importances[order],
            xerr=None if errors is None else errors[order])
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Feature correlation and importance analysis.")
    parser.add_argument("--data", required=True, help="Processed .npz with raw epochs (x, y, subjects).")
    parser.add_argument("--channels", nargs="+", default=SLEEP_EDF_CHANNELS,
                        help="Channel names for feature labelling.")
    parser.add_argument("--corr-threshold", type=float, default=0.95)
    parser.add_argument("--sfreq", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--out-dir", default="results/figures")
    parser.add_argument("--table-out", default="results/tables/feature_analysis.json")
    args = parser.parse_args()

    dataset = load_processed_dataset(args.data)
    if dataset.x.ndim != 3:
        raise ValueError("analyze_features expects raw epochs (n_epochs, n_channels, n_samples).")
    sfreq = args.sfreq or dataset.sfreq or 100.0

    logger.info("Extracting features from %d epochs (sfreq=%g Hz)...", len(dataset.y), sfreq)
    features, feature_names = extract_features_dataset(dataset.x, sfreq, args.channels)

    # Use training subjects only for every statistic below (leakage-safe).
    train_idx, val_idx, _ = subject_wise_split(dataset.subjects)
    x_train, y_train = features[train_idx], dataset.y[train_idx]
    x_val, y_val = features[val_idx], dataset.y[val_idx]
    logger.info("Analysis on %d training + %d validation epochs.", len(train_idx), len(val_idx))

    out_dir = ensure_dir(args.out_dir)

    # 1. Correlation matrix (redundancy structure).
    corr = np.nan_to_num(np.corrcoef(x_train, rowvar=False))
    plot_correlation_matrix(corr, feature_names, out_dir / "feature_correlation_matrix.png")

    # 2. Importance: RF impurity (train) + permutation (validation).
    forest = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=42
    ).fit(x_train, y_train)
    plot_importance(forest.feature_importances_, feature_names,
                    "Random-Forest impurity importance",
                    out_dir / "feature_importance_impurity.png", args.top_k)

    perm = permutation_importance(forest, x_val, y_val, n_repeats=10,
                                  random_state=42, scoring="f1_macro", n_jobs=-1)
    plot_importance(perm.importances_mean, feature_names,
                    "Permutation importance (validation, macro-F1 drop)",
                    out_dir / "feature_importance_permutation.png", args.top_k,
                    errors=perm.importances_std)

    # 3. Correlation-pruning preview (what feature selection would keep).
    kept, dropped = correlation_pruned_indices(x_train, threshold=args.corr_threshold)
    logger.info("Correlation pruning (|r|>%.2f): keep %d / drop %d of %d features.",
                args.corr_threshold, len(kept), len(dropped), len(feature_names))

    ranking = np.argsort(forest.feature_importances_)[::-1]
    save_json({
        "n_features": len(feature_names),
        "corr_threshold": args.corr_threshold,
        "kept_features": [feature_names[i] for i in kept],
        "dropped_features": [feature_names[i] for i in dropped],
        "impurity_importance_ranking": [feature_names[i] for i in ranking],
    }, args.table_out)
    logger.info("Saved figures to %s and feature report to %s", out_dir, args.table_out)


if __name__ == "__main__":
    main()
