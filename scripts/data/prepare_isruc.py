"""Convert raw ISRUC-Sleep subgroup I recordings into model-ready epochs.

ISRUC-Sleep-I stores one PSG recording per subject as ``<id>.rec`` and sleep
stage scores as text files, usually ``<id>_1.txt`` and ``<id>_2.txt``. Each line
of the score file is one 30 s epoch in the ISRUC convention:

0 Wake, 1 N1, 2 N2, 3 N3, 4 N4, 5 REM. N4 is merged into N3 to match the shared
5-class project label space.

The output matches the other datasets: one compressed .npz with x, y, subjects.
"""
import argparse
import gc
import os
import sys
import uuid
import warnings
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mne
import numpy as np
from tqdm import tqdm

from src.common.preprocessing import filter_psg_channels
from src.common.utils import EPOCH_SECONDS, STAGE_TO_INDEX, ensure_dir, get_logger

logger = get_logger("prepare_isruc")
mne.set_log_level("ERROR")

ISRUC_CHANNELS = ["C3-A2", "C4-A1", "LOC-A2", "X1"]
ISRUC_CHANNEL_ALIASES = ["EEG C3-A2", "EEG C4-A1", "EOG LOC-A2", "EMG X1"]

# ISRUC-Sleep-I is not internally consistent in how it labels channels. Across
# subjects we see the same signals under several naming schemes:
#   * auricular EEG/EOG labels (``*-A2``/``*-A1``, ``LOC``/``ROC``);
#   * equivalent mastoid labels (``*-M2``/``*-M1``, ``E1``/``E2``);
#   * EMG/auxiliary channels exported as numbers (``24``..``31``) instead of
#     ``X1``..``X8`` — same signals, same order;
#   * a few recordings store raw monopolar electrodes (``C3``, ``A2``, ...)
#     rather than the bipolar derivations, which are rebuilt by subtraction.
# Each key maps to the interchangeable variants; we pick whichever a recording
# actually holds (see resolve_channel_source).
ISRUC_CHANNEL_EQUIVALENTS = {
    "LOC-A2": ("LOC-A2", "E1-M2"),
    "ROC-A1": ("ROC-A1", "E2-M1"),
    "F3-A2": ("F3-A2", "F3-M2"),
    "C3-A2": ("C3-A2", "C3-M2"),
    "O1-A2": ("O1-A2", "O1-M2"),
    "F4-A1": ("F4-A1", "F4-M1"),
    "C4-A1": ("C4-A1", "C4-M1"),
    "O2-A1": ("O2-A1", "O2-M1"),
    "X1": ("X1", "24"),
    "X2": ("X2", "25"),
    "X3": ("X3", "26"),
    "X4": ("X4", "27"),
    "X5": ("X5", "28"),
    "X6": ("X6", "29"),
    "X7": ("X7", "30"),
    "X8": ("X8", "31"),
}
ISRUC_LABEL_TO_INDEX = {
    0: STAGE_TO_INDEX["Wake"],
    1: STAGE_TO_INDEX["N1"],
    2: STAGE_TO_INDEX["N2"],
    3: STAGE_TO_INDEX["N3"],
    4: STAGE_TO_INDEX["N3"],
    5: STAGE_TO_INDEX["REM"],
}


@contextmanager
def edf_extension_link(recording_path, tmp_dir):
    """Expose a .rec recording through a temporary .edf path for MNE."""
    tmp_dir = ensure_dir(tmp_dir)
    tmp_path = tmp_dir / f"{recording_path.stem}_{uuid.uuid4().hex}.edf"
    try:
        os.link(recording_path, tmp_path)
    except OSError:
        # Fallback for filesystems that do not allow hard links. This is slower
        # but still only holds one recording on disk at a time.
        import shutil

        shutil.copy2(recording_path, tmp_path)
    try:
        yield tmp_path
    finally:
        tmp_path.unlink(missing_ok=True)


def read_score_file(score_path):
    """Read ISRUC one-label-per-line scores and map them to the project labels."""
    labels = []
    with open(score_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                raw_label = int(float(text))
            except ValueError as exc:
                raise ValueError(f"Invalid ISRUC score {text!r} in {score_path}:{line_number}") from exc
            labels.append(ISRUC_LABEL_TO_INDEX.get(raw_label))
    return np.asarray(labels, dtype=object)


def count_epochs(recording_path, score_path, target_sfreq, tmp_dir):
    """Return the usable epoch count for one recording without loading signals."""
    labels = read_score_file(score_path)
    with edf_extension_link(recording_path, tmp_dir) as edf_path:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
        n_samples = round(EPOCH_SECONDS * target_sfreq)
        n_signal_epochs = int((raw.n_times / raw.info["sfreq"] * target_sfreq) // n_samples)
    n_epochs = min(len(labels), n_signal_epochs)
    return int(sum(label is not None for label in labels[:n_epochs])), raw.info["sfreq"]


def resolve_channel_source(available, name):
    """Describe how to obtain channel ``name`` from a recording.

    Returns ``("pick", actual_name)`` when a naming variant is present directly,
    or ``("bipolar", anode, cathode)`` when ``name`` is a bipolar derivation that
    must be rebuilt from two monopolar electrodes (some ISRUC recordings store
    the raw electrodes instead of the ready-made derivation).
    """
    available_set = set(available)
    candidates = ISRUC_CHANNEL_EQUIVALENTS.get(name, (name,))
    for candidate in candidates:
        if candidate in available_set:
            return ("pick", candidate)
    if "-" in name:
        anode, cathode = name.split("-", 1)
        if anode in available_set and cathode in available_set:
            return ("bipolar", anode, cathode)
    raise ValueError(
        f"Cannot obtain channel {name!r}: neither a known naming variant "
        f"{candidates} nor its bipolar electrodes are present. "
        f"Available channels: {sorted(available_set)}"
    )


def build_channel_matrix(raw, channels):
    """Return a (n_channels, n_times) array for ``channels`` from ``raw``.

    Handles the direct/equivalent naming variants and rebuilds bipolar
    derivations from monopolar electrodes where needed.
    """
    raw_data = raw.get_data()
    index = {ch: i for i, ch in enumerate(raw.ch_names)}
    rows = []
    for name in channels:
        source = resolve_channel_source(raw.ch_names, name)
        if source[0] == "pick":
            rows.append(raw_data[index[source[1]]])
        else:
            _, anode, cathode = source
            rows.append(raw_data[index[anode]] - raw_data[index[cathode]])
    return np.asarray(rows)


def process_recording(recording_path, score_path, channels, channel_aliases, target_sfreq, tmp_dir):
    """Return (epochs, labels) for one ISRUC recording."""
    labels = read_score_file(score_path)
    with edf_extension_link(recording_path, tmp_dir) as edf_path:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            sfreq = raw.info["sfreq"]
            data = build_channel_matrix(raw, channels)
            data = filter_psg_channels(data, sfreq, channel_aliases)
            if sfreq != target_sfreq:
                info = mne.create_info(list(channel_aliases), sfreq, ch_types="misc")
                resampled = mne.io.RawArray(data, info, verbose=False)
                resampled.resample(target_sfreq, verbose=False)
                data = resampled.get_data()

    n_samples = round(EPOCH_SECONDS * target_sfreq)
    n_signal_epochs = data.shape[1] // n_samples
    n_epochs = min(len(labels), n_signal_epochs)
    labels = labels[:n_epochs]
    valid_mask = np.asarray([label is not None for label in labels], dtype=bool)
    if not valid_mask.any():
        return np.empty((0, len(channels), n_samples), dtype=np.float32), np.empty(0, dtype=int)

    data = data[:, : n_epochs * n_samples].astype(np.float32, copy=False)
    x = data.reshape(len(channels), n_epochs, n_samples).transpose(1, 0, 2)
    y = np.asarray([label for label in labels if label is not None], dtype=int)
    return x[valid_mask], y


def main():
    parser = argparse.ArgumentParser(description="Prepare ISRUC-Sleep-I epochs.")
    parser.add_argument("--raw-dir", default="data/raw/ISRUC-Sleep-I")
    parser.add_argument("--out", default="data/processed/isruc.npz")
    parser.add_argument("--scorer", choices=["1", "2"], default="1",
                        help="Use <subject>_<scorer>.txt sleep-stage labels.")
    parser.add_argument("--channels", nargs="+", default=ISRUC_CHANNELS,
                        help="Original ISRUC channels to keep.")
    parser.add_argument("--channel-aliases", nargs="+", default=ISRUC_CHANNEL_ALIASES,
                        help="Modality-aware aliases used by the shared filters/features.")
    parser.add_argument("--target-sfreq", type=float, default=100.0,
                        help="Resample ISRUC to match the development dataset.")
    args = parser.parse_args()

    if len(args.channels) != len(args.channel_aliases):
        parser.error("--channels and --channel-aliases must have the same length.")

    raw_dir = Path(args.raw_dir)
    subject_dirs = sorted([path for path in raw_dir.iterdir() if path.is_dir()], key=lambda p: int(p.name))
    if not subject_dirs:
        raise FileNotFoundError(f"No ISRUC subject folders found in {raw_dir}.")

    tmp_dir = Path(args.out).parent / "_tmp_isruc_edf_links"
    manifest = []
    total_epochs = 0
    count_progress = tqdm(subject_dirs, desc="isruc (counting)", unit="subject")
    for subject_dir in count_progress:
        subject = subject_dir.name
        recording_path = subject_dir / f"{subject}.rec"
        score_path = subject_dir / f"{subject}_{args.scorer}.txt"
        if not recording_path.exists() or not score_path.exists():
            tqdm.write(f"Missing recording or scorer {args.scorer} labels for subject {subject}, skipping.")
            continue
        n_epochs, _ = count_epochs(recording_path, score_path, args.target_sfreq, tmp_dir)
        if n_epochs == 0:
            continue
        manifest.append((recording_path, score_path, subject, n_epochs))
        total_epochs += n_epochs
        count_progress.set_postfix(epochs=n_epochs, subject=subject)

    if total_epochs == 0:
        raise RuntimeError("No usable ISRUC epochs found.")

    ensure_dir(Path(args.out).parent)
    n_channels = len(args.channels)
    n_samples = round(EPOCH_SECONDS * args.target_sfreq)
    tmp_x_path = Path(args.out).with_suffix(".x.tmp.npy")
    x = np.lib.format.open_memmap(
        tmp_x_path, mode="w+", dtype=np.float32, shape=(total_epochs, n_channels, n_samples)
    )
    y = np.empty(total_epochs, dtype=int)
    subject_width = max(len(subject) for _, _, subject, _ in manifest)
    subjects = np.empty(total_epochs, dtype=f"<U{subject_width}")
    x_out = y_out = subjects_out = None

    try:
        offset = 0
        fill_progress = tqdm(manifest, desc="isruc (processing)", unit="subject")
        for recording_path, score_path, subject, expected_epochs in fill_progress:
            x_i, y_i = process_recording(
                recording_path, score_path, args.channels, args.channel_aliases,
                args.target_sfreq, tmp_dir
            )
            n_epochs = len(y_i)
            if n_epochs != expected_epochs:
                tqdm.write(
                    f"Subject {subject}: counted {expected_epochs} epochs, processed {n_epochs}; using processed count."
                )
            x[offset:offset + n_epochs] = x_i
            y[offset:offset + n_epochs] = y_i
            subjects[offset:offset + n_epochs] = subject
            offset += n_epochs
            fill_progress.set_postfix(epochs=n_epochs, subject=subject)

        x.flush()
        if offset != total_epochs:
            x_out = np.asarray(x[:offset])
            y_out = y[:offset]
            subjects_out = subjects[:offset]
        else:
            x_out = x
            y_out = y
            subjects_out = subjects

        logger.info("Compressing and writing %s (this can take a while, no progress bar)...",
                    args.out)
        np.savez_compressed(args.out, x=x_out, y=y_out, subjects=subjects_out)
        logger.info("Saved %d epochs from %d subjects to %s",
                    len(y_out), len(np.unique(subjects_out)), args.out)
    finally:
        x_out = y_out = subjects_out = None
        x = None
        gc.collect()  # release memmap references so Windows allows deleting the temp file
        tmp_x_path.unlink(missing_ok=True)
        try:
            tmp_dir.rmdir()
        except OSError:
            pass


if __name__ == "__main__":
    main()
