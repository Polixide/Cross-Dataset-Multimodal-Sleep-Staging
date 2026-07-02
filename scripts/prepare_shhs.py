"""Convert raw SHHS recordings into model-ready epochs (external test set).

SHHS provides EDF signals with NSRR XML hypnograms. This script harmonizes SHHS
to the SAME channel count, sampling rate, epoching and 5-class label space as the
Sleep-EDF development data, so external validation is a fair comparison.

Channel names and the XML layout vary slightly between SHHS releases; the
defaults below match the common NSRR export and can be overridden with --channels.

Usage:
    python scripts/prepare_shhs.py --raw-dir data/raw/shhs \
        --out data/processed/shhs.npz
"""
import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mne
import numpy as np

from src.utils import EPOCH_SECONDS, get_logger, ensure_dir

logger = get_logger("prepare_shhs")

# SHHS stage code (the number after "|" in EventConcept) -> 5-class index. N4 -> N3.
SHHS_STAGE_TO_INDEX = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 4}
# Two EEG, one EOG, one EMG channel, matching the Sleep-EDF channel structure.
SHHS_CHANNELS = ["EEG", "EEG(sec)", "EOG(L)", "EMG"]


def read_shhs_labels(xml_path):
    """Read one 5-class label per 30 s epoch from an NSRR SHHS annotation file.

    Unknown stage codes become None so the epoch can be dropped.
    """
    root = ET.parse(xml_path).getroot()
    labels = []
    for event in root.iter("ScoredEvent"):
        if "Stages" not in (event.findtext("EventType") or ""):
            continue
        code = int((event.findtext("EventConcept") or "|-1").split("|")[-1])
        duration = float(event.findtext("Duration") or 0.0)
        n_epochs = round(duration / EPOCH_SECONDS)
        labels.extend([SHHS_STAGE_TO_INDEX.get(code)] * n_epochs)
    return labels


def process_recording(edf_path, xml_path, channels, target_sfreq):
    """Return (epochs, labels) for one SHHS EDF + XML pair, resampled and epoched."""
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.pick(channels)
    if raw.info["sfreq"] != target_sfreq:
        raw.resample(target_sfreq, verbose=False)

    samples_per_epoch = int(target_sfreq * EPOCH_SECONDS)
    signals = raw.get_data()  # (n_channels, n_total_samples)
    n_epochs = signals.shape[1] // samples_per_epoch
    signals = signals[:, : n_epochs * samples_per_epoch]
    # -> (n_epochs, n_channels, n_samples)
    epochs = signals.reshape(len(channels), n_epochs, samples_per_epoch).transpose(1, 0, 2)

    labels = read_shhs_labels(xml_path)
    n = min(len(labels), n_epochs)
    epochs, labels = epochs[:n], labels[:n]

    valid = np.array([label is not None for label in labels])
    x = epochs[valid]
    y = np.array([label for label in labels if label is not None], dtype=int)
    return x, y


def main():
    parser = argparse.ArgumentParser(description="Prepare SHHS epochs for external validation.")
    parser.add_argument("--raw-dir", default="data/raw/shhs")
    parser.add_argument("--out", default="data/processed/shhs.npz")
    parser.add_argument("--channels", nargs="+", default=SHHS_CHANNELS)
    parser.add_argument("--target-sfreq", type=float, default=100.0,
                        help="Resample SHHS to match the development dataset.")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    edf_files = sorted(raw_dir.rglob("*.edf"))  # recursive: handles subfolders
    if not edf_files:
        raise FileNotFoundError(f"No .edf files found in {raw_dir}.")

    all_x, all_y, all_subjects = [], [], []
    for edf_path in edf_files:
        xml_files = sorted(raw_dir.rglob(f"{edf_path.stem}*.xml"))
        if not xml_files:
            logger.warning("No annotation XML for %s, skipping.", edf_path.name)
            continue

        x, y = process_recording(edf_path, xml_files[0], args.channels, args.target_sfreq)
        subject = edf_path.stem  # one recording per subject in SHHS
        all_x.append(x)
        all_y.append(y)
        all_subjects.append(np.full(len(y), subject))
        logger.info("Processed %s: %d epochs.", edf_path.name, len(y))

    x = np.concatenate(all_x)
    y = np.concatenate(all_y)
    subjects = np.concatenate(all_subjects)

    ensure_dir(Path(args.out).parent)
    np.savez_compressed(args.out, x=x, y=y, subjects=subjects)
    logger.info("Saved %d epochs from %d subjects to %s",
                len(y), len(np.unique(subjects)), args.out)


if __name__ == "__main__":
    main()
