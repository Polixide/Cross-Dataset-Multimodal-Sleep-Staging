"""Convert raw Sleep-EDF Expanded recordings into model-ready epochs.

For each PSG recording it picks the channels of interest, reads the paired
hypnogram, segments into 30 s epochs, maps annotations to the 5-class space, and
saves one .npz (x, y, subjects) under data/processed/.

Sleep-EDF file names look like `SC4001E0-PSG.edf` / `SC4001EC-Hypnogram.edf`.
The subject id is the two-digit code at positions 3-4 (two nights per subject),
so both nights of one subject share the same id for subject-wise splitting.

Usage:
    python scripts/prepare_sleep_edf.py --raw-dir data/raw/sleep_edf \
        --out data/processed/sleep_edf.npz
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mne
import numpy as np

from src.preprocessing import RAW_LABEL_TO_INDEX
from src.utils import EPOCH_SECONDS, SLEEP_EDF_CHANNELS, ensure_dir, get_logger

logger = get_logger("prepare_sleep_edf")

# Map each kept annotation to a temporary event code (start at 1; MNE avoids 0).
ANNOTATION_TO_CODE = {desc: code for code, desc in enumerate(RAW_LABEL_TO_INDEX, start=1)}
CODE_TO_CLASS = {code: RAW_LABEL_TO_INDEX[desc] for desc, code in ANNOTATION_TO_CODE.items()}


def process_recording(psg_path, hypnogram_path, channels):
    """Return (epochs, labels) for one PSG + hypnogram pair.

    epochs has shape (n_epochs, n_channels, n_samples); labels are in [0, 4].
    """
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
    raw.pick(channels)

    annotations = mne.read_annotations(hypnogram_path)
    raw.set_annotations(annotations, emit_warning=False)

    # chunk_duration splits each stage annotation into consecutive 30 s events.
    events, _ = mne.events_from_annotations(
        raw, event_id=ANNOTATION_TO_CODE, chunk_duration=float(EPOCH_SECONDS), verbose=False
    )
    tmax = EPOCH_SECONDS - 1.0 / raw.info["sfreq"]
    epochs = mne.Epochs(
        raw, events, event_id=ANNOTATION_TO_CODE, tmin=0.0, tmax=tmax,
        baseline=None, preload=True, verbose=False,
    )

    x = epochs.get_data()
    y = np.array([CODE_TO_CLASS[code] for code in epochs.events[:, 2]], dtype=int)
    return x, y


def main():
    parser = argparse.ArgumentParser(description="Prepare Sleep-EDF Expanded epochs.")
    parser.add_argument("--raw-dir", default="data/raw/sleep_edf")
    parser.add_argument("--out", default="data/processed/sleep_edf.npz")
    parser.add_argument("--channels", nargs="+", default=SLEEP_EDF_CHANNELS)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    psg_files = sorted(raw_dir.rglob("*-PSG.edf"))  # recursive: handles subfolders
    if not psg_files:
        raise FileNotFoundError(f"No *-PSG.edf files found in {raw_dir}.")

    all_x, all_y, all_subjects = [], [], []
    for psg_path in psg_files:
        recording_id = psg_path.name[:6]
        hypnograms = sorted(raw_dir.rglob(f"{recording_id}*-Hypnogram.edf"))
        if not hypnograms:
            logger.warning("No hypnogram for %s, skipping.", psg_path.name)
            continue

        x, y = process_recording(psg_path, hypnograms[0], args.channels)
        subject = psg_path.name[3:5]  # night-independent subject id
        all_x.append(x)
        all_y.append(y)
        all_subjects.append(np.full(len(y), subject))
        logger.info("Processed %s: %d epochs (subject %s).", psg_path.name, len(y), subject)

    x = np.concatenate(all_x)
    y = np.concatenate(all_y)
    subjects = np.concatenate(all_subjects)

    ensure_dir(Path(args.out).parent)
    np.savez_compressed(args.out, x=x, y=y, subjects=subjects)
    logger.info("Saved %d epochs from %d subjects to %s",
                len(y), len(np.unique(subjects)), args.out)


if __name__ == "__main__":
    main()
