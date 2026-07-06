"""Extract and cache handcrafted features for all feature-based ML models.

The input must contain raw epochs with shape (epochs, channels, samples).  The
output keeps the same labels and subject identifiers, but replaces raw signals
with a float32 feature matrix that can be reused by multinomial logistic
regression, random forest and XGBoost.

Example:
    python scripts/ml/cache_features.py \
        --data data/processed/sleep_edf.npz \
        --out data/processed/sleep_edf_features.npz \
        --sfreq 100
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from src.common.data import load_processed_dataset
from src.ml.features import extract_features_dataset
from src.common.utils import SLEEP_EDF_CHANNELS, ensure_dir, get_logger

logger = get_logger("cache_ml_features")


def main():
    parser = argparse.ArgumentParser(
        description="Extract handcrafted ML features once and save a reusable dataset."
    )
    parser.add_argument("--data", required=True, help="Input .npz containing raw epochs.")
    parser.add_argument("--out", required=True, help="Output .npz containing 2D features.")
    parser.add_argument("--sfreq", type=float, default=100.0, help="Sampling rate in Hz.")
    parser.add_argument(
        "--channel-names",
        nargs="+",
        default=None,
        help="Optional channel names in the same order as the input channels.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Deprecated compatibility flag; existing outputs are always replaced.",
    )
    args = parser.parse_args()

    input_path = Path(args.data).resolve()
    output_path = Path(args.out).resolve()
    if input_path == output_path:
        parser.error("--out must be different from --data; raw data must be preserved.")
    if output_path.exists():
        logger.info("Replacing existing feature cache: %s", output_path)

    dataset = load_processed_dataset(input_path)
    if dataset.x.ndim != 3:
        parser.error(
            f"Expected raw epochs with 3 dimensions, got shape {dataset.x.shape}. "
            "This dataset may already contain cached features."
        )

    n_channels = dataset.x.shape[1]
    channel_names = args.channel_names
    if channel_names is None and n_channels == len(SLEEP_EDF_CHANNELS):
        channel_names = SLEEP_EDF_CHANNELS
    if channel_names is not None and len(channel_names) != n_channels:
        parser.error(
            f"Received {len(channel_names)} channel names for {n_channels} input channels."
        )

    logger.info(
        "Extracting features from %d epochs, %d channels at %.1f Hz.",
        len(dataset.y), n_channels, args.sfreq,
    )
    features, feature_names = extract_features_dataset(
        dataset.x, args.sfreq, channel_names=channel_names
    )
    features = features.astype(np.float32, copy=False)

    ensure_dir(output_path.parent)
    np.savez_compressed(
        output_path,
        x=features,
        y=dataset.y,
        subjects=dataset.subjects,
        feature_names=np.asarray(feature_names, dtype=str),
        sfreq=np.asarray(args.sfreq),
    )
    logger.info(
        "Saved reusable feature dataset to %s with shape %s and dtype %s.",
        output_path, features.shape, features.dtype,
    )


if __name__ == "__main__":
    main()
