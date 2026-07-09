"""Generate SHAP figures for Sleep-EDF, HMC, and cross-model comparisons.

The script expects cached 2D feature datasets and fitted model pickles. Missing
inputs are skipped so the same command can be reused while artifacts are being
created incrementally.
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.common.data import load_processed_dataset, subject_wise_split
from src.common.utils import ensure_dir, get_logger, load_pickle
from src.ml.explainability import shap_feature_importance

logger = get_logger("make_shap_suite")


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


def run_one(model_label, model_path, dataset_label, dataset_path, out_root, args):
    if not model_path.exists():
        logger.warning("Skipping %s on %s; model missing: %s", model_label, dataset_label, model_path)
        return None
    if not dataset_path.exists():
        logger.warning("Skipping %s on %s; dataset missing: %s", model_label, dataset_label, dataset_path)
        return None

    background, explained, feature_names = dataset_samples(
        dataset_path, args.background_size, args.explain_size, args.seed
    )
    model = load_pickle(model_path)
    out_dir = ensure_dir(out_root / dataset_label / model_label)
    logger.info("Computing SHAP for %s on %s.", model_label, dataset_label)
    path = shap_feature_importance(
        model,
        background,
        explained,
        feature_names=feature_names,
        out_dir=out_dir,
        top_k=args.top_k,
    )
    renamed = out_dir / f"{model_label}_{dataset_label}_shap_feature_importance.png"
    path.replace(renamed)
    return renamed


def copy_existing_sleep_shap(out_root):
    """Seed the new suite layout with already generated Sleep-EDF SHAP PNGs."""
    existing = {
        "logreg_calibrated": Path(
            "results/figures/shap/single_models/"
            "logreg_class-weight_tune_sleep_train_shap_feature_importance.png"
        ),
        "logreg_raw": Path(
            "results/figures/shap/single_models/"
            "logreg_raw_class-weight_tune_sleep_train_shap_feature_importance.png"
        ),
        "rf_calibrated": Path(
            "results/figures/shap/single_models/"
            "rf_class-weight_tune_sleep_train_shap_feature_importance.png"
        ),
        "xgb_calibrated": Path(
            "results/figures/shap/single_models/"
            "xgb_class-weight_tune_sleep_train_shap_feature_importance.png"
        ),
    }
    copied = {}
    for label, source in existing.items():
        if not source.exists():
            continue
        out_dir = ensure_dir(out_root / "sleep" / label)
        target = out_dir / f"{label}_sleep_shap_feature_importance.png"
        shutil.copyfile(source, target)
        copied[(label, "sleep")] = target
    return copied


def make_montage(items, out_path, title):
    images = []
    labels = []
    for label, path in items:
        if not path or not Path(path).exists():
            continue
        image = Image.open(path).convert("RGB")
        image.thumbnail((850, 500))
        images.append(image.copy())
        labels.append(label)
    if not images:
        return None

    cols = 2
    rows = int(np.ceil(len(images) / cols))
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    pad = 28
    title_h = 48
    header_h = 58
    canvas = Image.new(
        "RGB",
        (cols * width + (cols + 1) * pad, rows * (height + title_h) + (rows + 1) * pad + header_h),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    try:
        title_font = ImageFont.truetype("arial.ttf", 28)
        label_font = ImageFont.truetype("arial.ttf", 22)
    except OSError:
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
    draw.text((pad, pad), title, fill=(20, 20, 20), font=title_font)
    for index, (label, image) in enumerate(zip(labels, images)):
        row = index // cols
        col = index % cols
        x = pad + col * (width + pad)
        y = header_h + pad + row * (height + title_h + pad)
        draw.text((x, y), label, fill=(20, 20, 20), font=label_font)
        canvas.paste(image, (x, y + title_h))
    ensure_dir(Path(out_path).parent)
    canvas.save(out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Build Sleep/HMC SHAP figure suite.")
    parser.add_argument("--sleep-data", default="data/processed/sleep_edf_features.npz")
    parser.add_argument("--hmc-data", default="data/processed/hmc_features.npz")
    parser.add_argument("--out-root", default="results/figures/shap")
    parser.add_argument("--background-size", type=int, default=50)
    parser.add_argument("--explain-size", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logreg-model", default="results/logs/train/logreg_class-weight_tune_sleep_train_sigmoid.pkl")
    parser.add_argument("--rf-model", default="results/logs/train/rf_class-weight_tune_sleep_train_sigmoid.pkl")
    parser.add_argument("--xgb-model", default="results/logs/train/xgb_class-weight_tune_sleep_train_sigmoid.pkl")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    generated = copy_existing_sleep_shap(out_root)
    models = {
        "logreg_calibrated": Path(args.logreg_model),
        "rf_calibrated": Path(args.rf_model),
        "xgb_calibrated": Path(args.xgb_model),
    }
    datasets = {
        "sleep": Path(args.sleep_data),
        "hmc": Path(args.hmc_data),
    }

    for model_label, model_path in models.items():
        for dataset_label, dataset_path in datasets.items():
            path = run_one(model_label, model_path, dataset_label, dataset_path, out_root, args)
            if path is not None:
                generated[(model_label, dataset_label)] = path

    comparison_dir = ensure_dir(out_root / "comparison")
    for dataset_label in datasets:
        make_montage(
            [(model_label, generated.get((model_label, dataset_label))) for model_label in models],
            comparison_dir / f"shap_model_comparison_{dataset_label}.png",
            f"SHAP model comparison: {dataset_label.upper()}",
        )
    for model_label in models:
        make_montage(
            [
                ("Sleep-EDF", generated.get((model_label, "sleep"))),
                ("HMC", generated.get((model_label, "hmc"))),
            ],
            comparison_dir / f"shap_sleep_vs_hmc_{model_label}.png",
            f"SHAP Sleep-EDF vs HMC: {model_label}",
        )


if __name__ == "__main__":
    main()
