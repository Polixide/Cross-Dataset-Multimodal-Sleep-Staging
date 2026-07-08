"""Convert raw ISRUC-Sleep (Cohort/Subgroup I) recordings into model-ready epochs.

Open external PSG dataset (University of Coimbra, sleeptight.isr.uc.pt). Cohort I
holds 100 single-session subjects, distributed one folder per subject, e.g.::

    data/raw/isruc/1/1.rec        # PSG recording (EDF content, .rec extension)
    data/raw/isruc/1/1_1.txt      # hypnogram scored by expert 1 (one code / line)
    data/raw/isruc/1/1_2.txt      # hypnogram scored by expert 2

The hypnogram is a plain-text file with one integer sleep-stage code per 30 s
epoch (0=Wake, 1=N1, 2=N2, 3=N3, 5=REM), unlike the free-text annotations of
Sleep-EDF/HMC. This script selects 2 EEG + 1 EOG + 1 EMG channels, band-pass
filters, resamples and epochs the recording to match the Sleep-EDF development
data, aligns the expert hypnogram, maps to the shared 5-class space, and saves
one .npz (x, y, subjects) — so it drops straight into run_external.py /
eval_dl_external.py as a second external test set, exactly like HMC.

Channel names vary across ISRUC recordings. Defaults below are resolved
case/whitespace-insensitively; if a channel is missing the script prints every
available channel name so you can pass the right ones with --channels.

Usage:
    python scripts/prepare_isruc.py --raw-dir data/raw/isruc --out data/processed/isruc.npz
    python scripts/prepare_isruc.py --raw-dir data/raw/isruc --scorer 2 \
        --channels C3-A2 C4-A1 ROC-A1 X1
"""
import argparse
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mne
import numpy as np
from tqdm.auto import tqdm

from src.data_loader import EpochStreamWriter
from src.preprocessing import (
    DEFAULT_H_FREQ,
    DEFAULT_L_FREQ,
    DEFAULT_TARGET_SFREQ,
    filter_and_resample_raw,
    stage_label_from_code,
)
from src.utils import EPOCH_SECONDS, get_logger

logger = get_logger("prepare_isruc")

# ISRUC recordings mix naming conventions across subjects: some reference EEG to
# A1/A2 (auricle) and label EOG LOC/ROC, others reference M1/M2 (mastoid) and label
# EOG E1/E2 — electrically the same montage. So each modality slot is resolved from
# a list of aliases (first match wins), giving the shared 2 EEG + 1 EOG + 1 EMG
# layout regardless of the per-recording labels. X1 is the chin EMG (consistent).
ISRUC_CHANNEL_ALIASES = [
    ["C3-A2", "C3-M2"],                         # EEG central-left
    ["C4-A1", "C4-M1"],                         # EEG central-right
    ["LOC-A2", "E1-M2", "ROC-A1", "E2-M1"],     # EOG (prefer left)
    ["X1"],                                     # chin EMG
]


def read_psg_raw(path):
    """Read an ISRUC PSG recording, coping with the non-standard .rec extension.

    ISRUC ships EDF-content files named ``*.rec``; some MNE versions reject that
    extension even though the bytes are valid EDF. On failure we retry through a
    temporary ``.edf`` copy (preload pulls the samples into memory before the copy
    is deleted).
    """
    try:
        return mne.io.read_raw_edf(path, preload=True, verbose=False)
    except Exception:
        with tempfile.TemporaryDirectory() as tmp:
            edf_copy = Path(tmp) / (path.stem + ".edf")
            shutil.copyfile(path, edf_copy)
            return mne.io.read_raw_edf(edf_copy, preload=True, verbose=False)


def resolve_channels(raw, requested):
    """Map requested channel names to raw's actual names (case/whitespace-tolerant).

    Returns the actual names in the requested order so the modality layout stays
    EEG, EEG, EOG, EMG. Raises with the full channel list if any is missing.
    """
    lookup = {}
    for name in raw.ch_names:
        lookup.setdefault(name.strip().upper(), name)
    resolved, missing = [], []
    for req in requested:
        actual = lookup.get(req.strip().upper())
        (resolved if actual is not None else missing).append(actual if actual is not None else req)
    if missing:
        raise ValueError(
            f"Channels {missing} not found. Available channels: {raw.ch_names}. "
            "Pass the right names with --channels."
        )
    return resolved


def resolve_channel_slots(raw, slot_aliases):
    """Resolve one channel per modality slot from a list of acceptable aliases.

    Handles ISRUC's per-recording naming variability (A1/A2 vs M1/M2, LOC/ROC vs
    E1/E2): for each slot the first alias present in the recording is picked, so the
    output is always the same 2 EEG + 1 EOG + 1 EMG layout, in order.
    """
    lookup = {}
    for name in raw.ch_names:
        lookup.setdefault(name.strip().upper(), name)
    resolved = []
    for aliases in slot_aliases:
        match = next((lookup[a.strip().upper()] for a in aliases if a.strip().upper() in lookup), None)
        if match is None:
            raise ValueError(
                f"No channel from {aliases} found. Available channels: {raw.ch_names}. "
                "Pass explicit names with --channels."
            )
        resolved.append(match)
    return resolved


# Prefixes of the 8 standard EEG/EOG derivations, used to find where that block
# ends so the chin EMG (the channel right after it) can be located even when a
# recording labels its auxiliary channels numerically (24, 25, ...) instead of X1.
_EEG_EOG_PREFIXES = ("F3", "F4", "C3", "C4", "O1", "O2", "E1", "E2", "LOC", "ROC")


def positional_emg(raw):
    """Chin EMG by position: the channel right after the standard EEG/EOG block.

    ISRUC records the 8 EEG/EOG derivations first, then X1 = chin (submental) EMG.
    A subset of recordings label the auxiliary channels numerically instead of
    X1/X2/..., but the chin EMG is still the first channel after that block.
    """
    montage_idx = [i for i, name in enumerate(raw.ch_names)
                   if name.strip().upper().split("-")[0] in _EEG_EOG_PREFIXES]
    if not montage_idx or max(montage_idx) + 1 >= len(raw.ch_names):
        raise ValueError(f"Cannot locate the chin EMG in {raw.ch_names}. Pass it with --channels.")
    return raw.ch_names[max(montage_idx) + 1]


def resolve_epoch_channels(raw, channels):
    """Explicit --channels list if given, else ISRUC alias + positional resolution.

    EEG (x2) and EOG are resolved from name aliases (A1/A2 vs M1/M2, LOC/ROC vs
    E1/E2). The chin EMG is resolved by name (X1) when present, otherwise by
    position for the recordings whose auxiliary channels are numerically labeled.
    """
    if channels:
        return resolve_channels(raw, channels)
    eeg_eog = resolve_channel_slots(raw, ISRUC_CHANNEL_ALIASES[:3])   # 2 EEG + 1 EOG
    try:
        emg = resolve_channel_slots(raw, [ISRUC_CHANNEL_ALIASES[3]])[0]
    except ValueError:
        emg = positional_emg(raw)
    return eeg_eog + [emg]


def select_recordings(raw_dir):
    """Return one recording per subject folder (ISRUC ships one per numbered folder).

    Prefers the file named after its folder, so a stray/renamed extra recording —
    e.g. a non-anonymized duplicate the distribution left behind — is skipped
    instead of duplicating the subject. Falls back to ``.edf`` for distributions
    that use that extension.
    """
    recordings = sorted(raw_dir.rglob("*.rec"))
    if not recordings:
        recordings = sorted(f for f in raw_dir.rglob("*.edf")
                            if "sleepscoring" not in f.name.lower())

    by_folder = {}
    for recording in recordings:
        by_folder.setdefault(recording.parent, []).append(recording)

    selected = []
    for folder, files in sorted(by_folder.items()):
        if len(files) == 1:
            selected.append(files[0])
            continue
        preferred = [f for f in files if f.stem == folder.name]
        chosen = preferred[0] if preferred else sorted(files)[0]
        skipped = [f.name for f in files if f != chosen]
        logger.warning("Folder %s has %d recordings; using %s, skipping %s (likely a duplicate).",
                       folder.name, len(files), chosen.name, skipped)
        selected.append(chosen)
    return sorted(selected)


def subject_id(rec_path, raw_dir):
    """Subject id for a recording: the enclosing subject folder name (ISRUC's layout),
    or the file stem when recordings sit directly under ``raw_dir`` (flat layout)."""
    return rec_path.parent.name if rec_path.parent != raw_dir else rec_path.stem


def find_hypnogram(rec_path, scorer):
    """Return the expert `scorer` hypnogram paired with a recording, or None.

    Handles the common ISRUC layouts: ``<stem>_<scorer>.txt`` next to the
    recording, a bare ``<scorer>.txt`` inside the subject folder, or any single
    ``*_<scorer>.txt`` in that folder.
    """
    directory = rec_path.parent
    candidates = [directory / f"{rec_path.stem}_{scorer}.txt", directory / f"{scorer}.txt"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(directory.glob(f"*_{scorer}.txt"))
    return matches[0] if matches else None


def read_hypnogram(path):
    """Read an ISRUC text hypnogram into a list of integer stage codes (one / epoch)."""
    codes = []
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            tokens = line.split()
            if not tokens:
                continue  # ISRUC files often end with a blank line
            try:
                codes.append(int(float(tokens[0])))
            except ValueError:
                continue
    return codes


def process_recording(rec_path, hypnogram_path, channels, target_sfreq,
                      l_freq=DEFAULT_L_FREQ, h_freq=DEFAULT_H_FREQ):
    """Return (epochs, labels) for one ISRUC recording + expert hypnogram.

    The continuous recording is band-pass filtered and resampled to the
    development sampling rate, then cut into consecutive non-overlapping 30 s
    epochs (fixed length, since the labels live in a separate text file). Signal
    epochs and hypnogram codes are truncated to a common length and unknown-stage
    epochs are dropped, keeping signals and labels aligned.
    """
    raw = read_psg_raw(rec_path)
    resolved = resolve_epoch_channels(raw, channels)
    raw.pick(resolved)
    raw.reorder_channels(resolved)  # guarantee EEG, EEG, EOG, EMG order
    filter_and_resample_raw(raw, l_freq, h_freq, target_sfreq)

    mapped = [stage_label_from_code(code) for code in read_hypnogram(hypnogram_path)]

    # Fixed-length events + the same tmax formula as prepare_sleep_edf/prepare_hmc,
    # so every epoch has exactly EPOCH_SECONDS * sfreq samples across datasets.
    events = mne.make_fixed_length_events(raw, id=1, duration=float(EPOCH_SECONDS))
    tmax = EPOCH_SECONDS - 1.0 / raw.info["sfreq"]
    epochs = mne.Epochs(raw, events, tmin=0.0, tmax=tmax, baseline=None,
                        preload=True, verbose=False)

    x = epochs.get_data()
    n = min(len(x), len(mapped))
    if len(x) != len(mapped):
        logger.warning("%s: %d signal epochs vs %d hypnogram labels; truncating to %d.",
                       rec_path.name, len(x), len(mapped), n)
    keep = [i for i in range(n) if mapped[i] is not None]
    x = x[keep]
    y = np.array([mapped[i] for i in keep], dtype=int)
    return x, y


def main():
    parser = argparse.ArgumentParser(description="Prepare ISRUC-Sleep (Cohort I) epochs for external validation.")
    parser.add_argument("--raw-dir", default="data/raw/isruc")
    parser.add_argument("--out", default="data/processed/isruc.npz")
    parser.add_argument("--channels", nargs="+", default=None,
                        help="Explicit 4 channels (2 EEG, 1 EOG, 1 EMG, in order). "
                             "Default: auto-resolve ISRUC aliases across recordings.")
    parser.add_argument("--scorer", default="1", choices=["1", "2"],
                        help="Which expert hypnogram to use (ISRUC provides two).")
    parser.add_argument("--l-freq", type=float, default=DEFAULT_L_FREQ)
    parser.add_argument("--h-freq", type=float, default=DEFAULT_H_FREQ)
    parser.add_argument("--target-sfreq", type=float, default=DEFAULT_TARGET_SFREQ,
                        help="Resample ISRUC to match the development dataset.")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    signal_files = select_recordings(raw_dir)
    if not signal_files:
        raise FileNotFoundError(f"No .rec/.edf recordings found in {raw_dir}.")

    # Stream each recording straight to disk (see prepare_sleep_edf): peak memory
    # stays around one recording instead of the whole dataset.
    writer = EpochStreamWriter(args.out)
    for rec_path in tqdm(signal_files, desc="ISRUC recordings", unit="rec"):
        hypnogram_path = find_hypnogram(rec_path, args.scorer)
        if hypnogram_path is None:
            logger.warning("No expert-%s hypnogram for %s, skipping.", args.scorer, rec_path.name)
            continue

        try:
            x, y = process_recording(rec_path, hypnogram_path, args.channels,
                                     args.target_sfreq, args.l_freq, args.h_freq)
        except Exception as exc:
            # One oddly-recorded file (e.g. a monopolar/unreferenced montage that
            # doesn't provide the standard derivations) must not kill the whole run.
            logger.warning("Skipping %s (%s): %s", rec_path.name, type(exc).__name__, exc)
            continue
        if len(y) == 0:
            logger.warning("No usable epochs in %s, skipping.", rec_path.name)
            continue
        subject = subject_id(rec_path, raw_dir)  # the subject folder name (ISRUC layout)
        writer.add(x, y, np.full(len(y), subject))
        logger.info("Processed %s: %d epochs (subject %s).", rec_path.name, len(y), subject)

    n_epochs, n_subjects = writer.finalize(args.target_sfreq)
    logger.info("Saved %d epochs from %d recordings to %s (signals in %s, sfreq=%g Hz)",
                n_epochs, n_subjects, args.out, writer.x_path.name, args.target_sfreq)


if __name__ == "__main__":
    main()
