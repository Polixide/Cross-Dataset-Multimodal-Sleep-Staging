"""Convert raw Sleep-EDF Expanded recordings into model-ready epochs.

For each PSG recording it picks the channels of interest, reads the paired
hypnogram, segments into 30 s epochs, maps annotations to the 5-class space, and
saves one .npz (x, y, subjects) under data/processed/.

Sleep-EDF file names look like `SC4001E0-PSG.edf` / `SC4001EC-Hypnogram.edf`.
The subject id is the cohort prefix (SC/ST) plus the two-digit code at positions
3-4 (two nights per subject), so both nights of one subject share the same id
for subject-wise splitting, and cassette/telemetry subjects never collide (both
cohorts number their subjects starting from 00).

Usage:
    python scripts/data/prepare_sleep_edf.py --raw-dir data/raw/sleep_edf \
        --out data/processed/sleep_edf.npz
"""
import argparse
import gc
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mne
import numpy as np
from tqdm import tqdm

from src.common.preprocessing import RAW_LABEL_TO_INDEX, filter_psg_channels
from src.common.utils import EPOCH_SECONDS, SLEEP_EDF_CHANNELS, ensure_dir, get_logger

logger = get_logger("prepare_sleep_edf")
mne.set_log_level("ERROR")

# Map each kept annotation to a temporary event code (start at 1; MNE avoids 0).
ANNOTATION_TO_CODE = {desc: code for code, desc in enumerate(RAW_LABEL_TO_INDEX, start=1)}
CODE_TO_CLASS = {code: RAW_LABEL_TO_INDEX[desc] for desc, code in ANNOTATION_TO_CODE.items()}


def count_epochs(psg_path, hypnogram_path):
    """Return (n_epochs, sfreq) for one PSG + hypnogram pair without loading signal data."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        raw = mne.io.read_raw_edf(psg_path, preload=False, verbose=False)
        annotations = mne.read_annotations(hypnogram_path)
        raw.set_annotations(annotations, emit_warning=False)
        events, _ = mne.events_from_annotations(
            raw, event_id=ANNOTATION_TO_CODE, chunk_duration=float(EPOCH_SECONDS), verbose=False
        )
    return len(events), raw.info["sfreq"]


def process_recording(psg_path, hypnogram_path, channels):
    """Return (epochs, labels) for one PSG + hypnogram pair.

    epochs has shape (n_epochs, n_channels, n_samples); labels are in [0, 4].
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
        raw.pick(channels)
        raw._data[:] = filter_psg_channels(
            raw.get_data(), raw.info["sfreq"], raw.ch_names
        )

        annotations = mne.read_annotations(hypnogram_path)
        raw.set_annotations(annotations, emit_warning=False)

        # chunk_duration splits each stage annotation into consecutive 30 s events.
        events, _ = mne.events_from_annotations(
            raw, event_id=ANNOTATION_TO_CODE, chunk_duration=float(EPOCH_SECONDS), verbose=False
        )
        tmax = EPOCH_SECONDS - 1.0 / raw.info["sfreq"]
        epochs = mne.Epochs(
            raw, events, event_id=ANNOTATION_TO_CODE, tmin=0.0, tmax=tmax,
            baseline=None, preload=True, verbose=False, on_missing="warn",
        )

    x = epochs.get_data().astype(np.float32)
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

    # Pass 1: find each recording's hypnogram and count epochs, without loading signals,
    # so the final array can be pre-allocated once instead of concatenated at the end
    # (concatenating a list of per-file arrays needs double the peak memory).
    manifest = []
    total_epochs = 0
    sfreq = None
    count_progress = tqdm(psg_files, desc="sleep_edf (counting)", unit="file")
    for psg_path in count_progress:
        recording_id = psg_path.name[:6]
        hypnograms = sorted(raw_dir.rglob(f"{recording_id}*-Hypnogram.edf"))
        if not hypnograms:
            tqdm.write(f"No hypnogram for {psg_path.name}, skipping.")
            continue
        n_epochs, file_sfreq = count_epochs(psg_path, hypnograms[0])
        if n_epochs == 0:
            continue
        sfreq = sfreq or file_sfreq
        # Cohort prefix (SC/ST) + the 2-digit code at positions 3-4: cassette and
        # telemetry each number subjects from 00, so without the prefix a cassette
        # subject and an unrelated telemetry subject would collide onto one id.
        subject = psg_path.name[:2] + psg_path.name[3:5]
        manifest.append((psg_path, hypnograms[0], subject, n_epochs))
        total_epochs += n_epochs

    # Pass 2: process each recording and write straight into a disk-backed memmap for x
    # (the big raw-signal array). Only x is memmapped: y/subjects are tiny (a few MB)
    # and stay in RAM. This keeps peak memory to ~one recording at a time, not the
    # whole dataset, however large it grows.
    n_channels = len(args.channels)
    n_samples = round(EPOCH_SECONDS * sfreq)
    ensure_dir(Path(args.out).parent)
    tmp_x_path = Path(args.out).with_suffix(".x.tmp.npy")
    x = np.lib.format.open_memmap(
        tmp_x_path, mode="w+", dtype=np.float32, shape=(total_epochs, n_channels, n_samples)
    )
    y = np.empty(total_epochs, dtype=int)
    subjects = np.empty(total_epochs, dtype="<U4")

    try:
        offset = 0
        fill_progress = tqdm(manifest, desc="sleep_edf (processing)", unit="file")
        for psg_path, hypnogram_path, subject, n_epochs in fill_progress:
            x_i, y_i = process_recording(psg_path, hypnogram_path, args.channels)
            x[offset:offset + n_epochs] = x_i
            y[offset:offset + n_epochs] = y_i
            subjects[offset:offset + n_epochs] = subject
            offset += n_epochs
            fill_progress.set_postfix(epochs=n_epochs, subject=subject)

        x.flush()
        logger.info("Compressing and writing %s (this can take a while, no progress bar)...",
                    args.out)
        np.savez_compressed(args.out, x=x, y=y, subjects=subjects)
        logger.info("Saved %d epochs from %d subjects to %s",
                    len(y), len(np.unique(subjects)), args.out)
    finally:
        del x
        gc.collect()  # release the memmap so Windows allows deleting the temp file
        tmp_x_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
