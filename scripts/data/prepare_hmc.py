"""Convert raw HMC (Haaglanden Medisch Centrum) recordings into model-ready epochs.

Open external PSG dataset (PhysioNet). Each recording is a signal EDF
`SNxxx.edf` plus a scoring EDF `SNxxx_sleepscoring.edf` with AASM annotations.
This script selects 2 EEG + 1 EOG + 1 EMG channels, resamples and epochs to match
the Sleep-EDF development data, and saves one .npz (x, y, subjects).

Channel names vary; if `raw.pick` fails, print `mne.io.read_raw_edf(...).ch_names`
and pass the right ones with --channels.

Usage:
    python scripts/data/prepare_hmc.py --raw-dir data/raw/hmc --out data/processed/hmc.npz
"""
import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mne
import numpy as np
from tqdm import tqdm

from src.common.preprocessing import filter_psg_channels, stage_label_from_text
from src.common.utils import EPOCH_SECONDS, ensure_dir, get_logger

logger = get_logger("prepare_hmc")
mne.set_log_level("ERROR")

# Two EEG, one EOG, one EMG channel, matching the Sleep-EDF channel structure.
HMC_CHANNELS = ["EEG C4-M1", "EEG C3-M2", "EOG E1-M2", "EMG chin"]


def find_scoring_file(signal_path, raw_dir):
    """Return the sleep-scoring EDF paired with a signal EDF, or None."""
    matches = list(raw_dir.rglob(f"{signal_path.stem}_sleepscoring.edf"))
    return matches[0] if matches else None


def process_recording(signal_path, scoring_path, channels, target_sfreq):
    """Return (epochs, labels) for one HMC signal + scoring pair."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        raw = mne.io.read_raw_edf(signal_path, preload=True, verbose=False)
        raw.pick(channels)
        raw._data[:] = filter_psg_channels(
            raw.get_data(), raw.info["sfreq"], raw.ch_names
        )
        if raw.info["sfreq"] != target_sfreq:
            raw.resample(target_sfreq, verbose=False)

        annotations = mne.read_annotations(scoring_path)
        raw.set_annotations(annotations, emit_warning=False)

        # Keep only descriptions that map to a real sleep stage.
        desc_to_class = {}
        for description in set(annotations.description):
            label = stage_label_from_text(description)
            if label is not None:
                desc_to_class[description] = label
        if not desc_to_class:
            raise ValueError(
                f"No sleep-stage annotations recognized in {scoring_path.name}. "
                f"Found: {sorted(set(annotations.description))}"
            )

        desc_to_code = {description: code for code, description in enumerate(desc_to_class, start=1)}
        events, _ = mne.events_from_annotations(
            raw, event_id=desc_to_code, chunk_duration=float(EPOCH_SECONDS), verbose=False
        )
        tmax = EPOCH_SECONDS - 1.0 / raw.info["sfreq"]
        epochs = mne.Epochs(
            raw, events, event_id=desc_to_code, tmin=0.0, tmax=tmax,
            baseline=None, preload=True, verbose=False, on_missing="warn",
        )

    code_to_class = {code: desc_to_class[desc] for desc, code in desc_to_code.items()}
    x = epochs.get_data().astype(np.float32)
    y = np.array([code_to_class[code] for code in epochs.events[:, 2]], dtype=int)
    return x, y


def main():
    parser = argparse.ArgumentParser(description="Prepare HMC epochs for external validation.")
    parser.add_argument("--raw-dir", default="data/raw/hmc")
    parser.add_argument("--out", default="data/processed/hmc.npz")
    parser.add_argument("--channels", nargs="+", default=HMC_CHANNELS)
    parser.add_argument("--target-sfreq", type=float, default=100.0,
                        help="Resample HMC to match the development dataset.")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    signal_files = sorted(f for f in raw_dir.rglob("*.edf") if not f.name.endswith("_sleepscoring.edf"))
    if not signal_files:
        raise FileNotFoundError(f"No signal .edf files found in {raw_dir}.")

    all_x, all_y, all_subjects = [], [], []
    progress = tqdm(signal_files, desc="hmc", unit="file")
    for signal_path in progress:
        scoring_path = find_scoring_file(signal_path, raw_dir)
        if scoring_path is None:
            tqdm.write(f"No sleep-scoring file for {signal_path.name}, skipping.")
            continue

        x, y = process_recording(signal_path, scoring_path, args.channels, args.target_sfreq)
        subject = signal_path.stem  # one recording per subject in HMC
        all_x.append(x)
        all_y.append(y)
        all_subjects.append(np.full(len(y), subject))
        progress.set_postfix(epochs=len(y), subject=subject)

    x = np.concatenate(all_x)
    y = np.concatenate(all_y)
    subjects = np.concatenate(all_subjects)

    ensure_dir(Path(args.out).parent)
    np.savez_compressed(args.out, x=x, y=y, subjects=subjects)
    logger.info("Saved %d epochs from %d recordings to %s",
                len(y), len(np.unique(subjects)), args.out)


if __name__ == "__main__":
    main()
