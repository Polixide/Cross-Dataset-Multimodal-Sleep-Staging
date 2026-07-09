"""Build side-by-side SHAP feature-importance comparisons between datasets."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import shap

from src.common.data import load_processed_dataset, subject_wise_split
from src.common.utils import ensure_dir, get_logger, load_pickle

logger = get_logger("make_shap_dataset_comparison")


DEFAULT_MODELS = {
    "logreg_calibrated": "results/logs/train/logreg_class-weight_tune_sleep_train_sigmoid.pkl",
    "rf_calibrated": "results/logs/train/rf_class-weight_tune_sleep_train_sigmoid.pkl",
    "xgb_calibrated": "results/logs/train/xgb_class-weight_tune_sleep_train_sigmoid.pkl",
    "xgb_smote_half_calibrated": "results/logs/train/xgb_half-smote_tune_sigmoid_sleep_train.pkl",
}


def sample_rows(x, indices, size, rng):
    size = min(size, len(indices))
    return x[rng.choice(indices, size=size, replace=False)]


def load_feature_names(path):
    with np.load(path, allow_pickle=True) as data:
        return data["feature_names"].tolist() if "feature_names" in data else None


def dataset_samples(path, background_size, explain_size, seed):
    dataset = load_processed_dataset(path)
    if dataset.x.ndim != 2:
        raise ValueError(f"{path} must contain cached 2D features, got {dataset.x.shape}")
    train_idx, _, test_idx = subject_wise_split(dataset.subjects, seed=seed)
    rng = np.random.default_rng(seed)
    return (
        sample_rows(dataset.x, train_idx, background_size, rng),
        sample_rows(dataset.x, test_idx, explain_size, rng),
        load_feature_names(path),
    )


def shap_importance(model, background, explained, feature_names, max_evals):
    explainer = shap.Explainer(
        model.predict_proba,
        background,
        algorithm="permutation",
        feature_names=feature_names,
    )
    if max_evals is None:
        max_evals = 2 * background.shape[1] + 1
    values = np.abs(explainer(explained, max_evals=max_evals).values)
    if values.ndim == 3:
        values = values.mean(axis=2)
    return values.mean(axis=0)


def plot_comparison(left, right, feature_names, left_label, right_label, model_label, top_k, out_path):
    combined = (left + right) / 2
    top = np.argsort(combined)[-top_k:]
    top = top[np.argsort(combined[top])]

    names = [feature_names[i] for i in top]
    y = np.arange(len(top))
    height = 0.38

    plt.figure(figsize=(11, max(5, 0.38 * len(top))))
    plt.barh(y - height / 2, left[top], height, label=left_label, color="#4C78A8")
    plt.barh(y + height / 2, right[top], height, label=right_label, color="#E15759")
    plt.yticks(y, names)
    plt.xlabel("Mean |SHAP value|")
    plt.title(f"SHAP {left_label} vs {right_label}: {model_label}")
    plt.legend()
    plt.tight_layout()

    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    out_path.unlink(missing_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def model_items(model_args):
    if not model_args:
        return DEFAULT_MODELS.items()
    items = []
    for item in model_args:
        if "=" not in item:
            raise ValueError(f"Model spec must be label=path, got {item!r}")
        label, path = item.split("=", 1)
        items.append((label, path))
    return items


def main():
    parser = argparse.ArgumentParser(
        description="Build one grouped SHAP bar plot per model comparing two datasets."
    )
    parser.add_argument("--left-data", default="data/processed/sleep_edf_features.npz")
    parser.add_argument("--right-data", default="data/processed/hmc_features.npz")
    parser.add_argument("--left-label", default="Sleep-EDF")
    parser.add_argument("--right-label", default="HMC")
    parser.add_argument("--out-dir", default="results/figures/shap/comparison")
    parser.add_argument("--background-size", type=int, default=50)
    parser.add_argument("--explain-size", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-evals", type=int, default=None)
    parser.add_argument(
        "--model",
        action="append",
        help="Optional model spec label=path. Can be repeated. Defaults to calibrated ML models.",
    )
    args = parser.parse_args()

    left_background, left_explained, left_names = dataset_samples(
        args.left_data, args.background_size, args.explain_size, args.seed
    )
    right_background, right_explained, right_names = dataset_samples(
        args.right_data, args.background_size, args.explain_size, args.seed
    )
    if left_names != right_names:
        raise ValueError("Feature names differ between datasets; cannot compare SHAP importances safely.")

    out_dir = ensure_dir(args.out_dir)
    for label, model_path in model_items(args.model):
        model_path = Path(model_path)
        if not model_path.exists():
            logger.warning("Skipping %s; missing model: %s", label, model_path)
            continue
        logger.info("Computing comparative SHAP for %s.", label)
        model = load_pickle(model_path)
        left = shap_importance(model, left_background, left_explained, left_names, args.max_evals)
        right = shap_importance(model, right_background, right_explained, right_names, args.max_evals)
        out_path = out_dir / f"shap_{args.left_label.lower().replace('-', '_')}_vs_{args.right_label.lower()}_{label}.png"
        plot_comparison(left, right, left_names, args.left_label, args.right_label, label, args.top_k, out_path)
        logger.info("Saved %s", out_path)


if __name__ == "__main__":
    main()
