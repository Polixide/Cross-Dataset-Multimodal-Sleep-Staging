"""Dataset loading, streaming save, and SUBJECT-WISE splitting.

This module is the single source of truth for how data is partitioned. All
splits are subject-wise to prevent subject leakage, the central methodological
rule of the project.

It also owns the on-disk format of the processed datasets. Because Sleep-EDF
expands to hundreds of thousands of epochs (tens of GB as one float array), the
signals are stored as a standalone memory-mappable ``.npy`` next to a small
``.npz`` holding the labels/subjects/metadata. `EpochStreamWriter` fills that
``.npy`` one recording at a time (peak RAM stays around a single recording) and
`load_processed_dataset` memory-maps it back, so no stage of the pipeline needs
the whole dataset resident in RAM.
"""
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from src.utils import ensure_dir


@dataclass
class EpochDataset:
    """Epoch-level data plus the subject each epoch belongs to.

    x        : signals (n_epochs, n_channels, n_samples) or features (n_epochs, n_features)
    y        : integer sleep-stage labels (n_epochs,)
    subjects : subject id per epoch (n_epochs,), used as the grouping key for splits
    sfreq    : sampling rate (Hz) the epochs were resampled to, or None if unknown.
               Stored by the prepare_*.py scripts so feature extraction uses the
               real rate instead of a hardcoded guess.
    """

    x: np.ndarray
    y: np.ndarray
    subjects: np.ndarray
    sfreq: float = None

    def __post_init__(self):
        if not (len(self.x) == len(self.y) == len(self.subjects)):
            raise ValueError("x, y and subjects must have the same length.")

    @property
    def n_subjects(self):
        return len(np.unique(self.subjects))


def load_processed_dataset(path, mmap=True):
    """Load a preprocessed dataset written by the prepare_*.py scripts.

    Two on-disk layouts are supported:

    - streaming layout (current): the .npz holds y/subjects/sfreq and an
      ``x_path`` pointing at a sibling ``*_x.npy``; the signals are memory-mapped
      (``mmap=True``) so they are never fully loaded into RAM.
    - legacy layout: a single .npz that also contains ``x`` inline.
    """
    path = Path(path)
    data = np.load(path, allow_pickle=True)
    sfreq = float(data["sfreq"]) if "sfreq" in data.files else None
    if "x" in data.files:
        x = data["x"]
    else:
        x_path = path.parent / str(data["x_path"])
        x = np.load(x_path, mmap_mode="r" if mmap else None)
    return EpochDataset(x=x, y=data["y"], subjects=data["subjects"], sfreq=sfreq)


class EpochStreamWriter:
    """Persist epochs recording-by-recording so peak RAM stays ~one recording.

    Each `add` writes that recording's signals to a temporary ``.npy`` shard and
    keeps only the (tiny) labels and subject ids in memory. `finalize` assembles
    the shards into a single memory-mapped ``*_x.npy`` next to the ``.npz``
    metadata file. This avoids ever holding the whole dataset in RAM, which for
    the ~450k Sleep-EDF epochs would be tens of GB.

    Signals are stored as float32 (half the size of MNE's float64) — ample
    precision for filtered, z-scored biosignals.
    """

    def __init__(self, out_path, dtype=np.float32):
        self.out_path = Path(out_path)
        self.dtype = dtype
        self.x_path = self.out_path.with_name(self.out_path.stem + "_x.npy")
        ensure_dir(self.out_path.parent)
        self._tmpdir = Path(tempfile.mkdtemp(prefix="epochs_", dir=self.out_path.parent))
        self._shards = []            # list of (shard_path, n_rows)
        self._y = []
        self._subjects = []
        self._sample_shape = None    # (n_channels, n_samples)

    def add(self, x, y, subjects):
        """Append one recording's epochs; only labels/ids stay in memory."""
        x = np.asarray(x, dtype=self.dtype)
        if self._sample_shape is None:
            self._sample_shape = x.shape[1:]
        elif x.shape[1:] != self._sample_shape:
            raise ValueError(f"Inconsistent epoch shape {x.shape[1:]} vs {self._sample_shape}.")
        shard = self._tmpdir / f"shard_{len(self._shards):04d}.npy"
        np.save(shard, x)
        self._shards.append((shard, len(x)))
        self._y.append(np.asarray(y))
        self._subjects.append(np.asarray(subjects))

    def finalize(self, sfreq):
        """Assemble shards into the memmapped .npy + metadata .npz.

        Returns (n_epochs, n_subjects).
        """
        if not self._shards:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
            raise RuntimeError("No epochs were added before finalize().")

        total = sum(n for _, n in self._shards)
        x_mm = np.lib.format.open_memmap(
            self.x_path, mode="w+", dtype=self.dtype, shape=(total, *self._sample_shape)
        )
        start = 0
        for shard, n in self._shards:
            x_mm[start:start + n] = np.load(shard)
            start += n
            shard.unlink()
        x_mm.flush()
        del x_mm
        shutil.rmtree(self._tmpdir, ignore_errors=True)

        y = np.concatenate(self._y)
        subjects = np.concatenate(self._subjects)
        np.savez_compressed(self.out_path, y=y, subjects=subjects, sfreq=sfreq,
                            x_path=self.x_path.name)
        return total, int(len(np.unique(subjects)))


def check_no_subject_overlap(*subject_groups):
    """Raise ValueError if any subject appears in more than one group.

    This is the explicit leakage guard reused across training and evaluation.
    """
    seen = set()
    for group in subject_groups:
        group = set(np.asarray(group).tolist())
        overlap = seen & group
        if overlap:
            raise ValueError(f"Subject leakage detected across splits: {sorted(overlap)}")
        seen |= group


def subject_wise_split(subjects, val_size=0.15, test_size=0.15, seed=42):
    """Split epoch indices into train/val/test by SUBJECT (never by epoch).

    Subjects are shuffled and partitioned, so no subject appears in two splits.
    Returns three index arrays into the epoch dimension.
    """
    subjects = np.asarray(subjects)
    unique_subjects = np.unique(subjects)

    rng = np.random.default_rng(seed)
    rng.shuffle(unique_subjects)

    n_subjects = len(unique_subjects)
    n_test = max(1, round(test_size * n_subjects))
    n_val = max(1, round(val_size * n_subjects))
    test_subjects = unique_subjects[:n_test]
    val_subjects = unique_subjects[n_test:n_test + n_val]
    train_subjects = unique_subjects[n_test + n_val:]

    check_no_subject_overlap(train_subjects, val_subjects, test_subjects)

    train_idx = np.where(np.isin(subjects, train_subjects))[0]
    val_idx = np.where(np.isin(subjects, val_subjects))[0]
    test_idx = np.where(np.isin(subjects, test_subjects))[0]
    return train_idx, val_idx, test_idx


def subject_wise_folds(subjects, n_splits=5):
    """Return a list of (train_idx, val_idx) for subject-wise k-fold CV.

    Uses GroupKFold so all epochs of a subject stay in the same fold.
    """
    subjects = np.asarray(subjects)
    group_kfold = GroupKFold(n_splits=n_splits)
    placeholder = np.zeros(len(subjects))
    return list(group_kfold.split(placeholder, groups=subjects))


def leave_one_subject_out(subjects):
    """Return a list of (train_idx, test_idx) for LOSO robustness analysis."""
    subjects = np.asarray(subjects)
    folds = []
    for subject in np.unique(subjects):
        test_idx = np.where(subjects == subject)[0]
        train_idx = np.where(subjects != subject)[0]
        folds.append((train_idx, test_idx))
    return folds


def make_epoch_sequences(x, y, subjects, seq_len, stride=None):
    """Build sequences of consecutive epochs WITHIN each subject (temporal context).

    Windows never cross subject boundaries, so building them within an already
    subject-wise split introduces no leakage. Windows are non-overlapping by
    default (stride = seq_len), so every epoch is used at most once; the trailing
    remainder shorter than seq_len is dropped.

    Returns
    -------
    x_seq : (n_windows, seq_len, ...) sequences of epochs.
    y_seq : (n_windows, seq_len) the label of every epoch in each window.
    window_subjects : (n_windows,) subject id per window.
    """
    if stride is None:
        stride = seq_len
    x = np.asarray(x)
    y = np.asarray(y)
    subjects = np.asarray(subjects)

    x_windows, y_windows, window_subjects = [], [], []
    for subject in np.unique(subjects):
        mask = subjects == subject
        x_subject, y_subject = x[mask], y[mask]
        for start in range(0, len(y_subject) - seq_len + 1, stride):
            x_windows.append(x_subject[start:start + seq_len])
            y_windows.append(y_subject[start:start + seq_len])
            window_subjects.append(subject)
    return np.array(x_windows), np.array(y_windows), np.array(window_subjects)


def build_sequence_windows(subjects, split_idx, seq_len, stride=None):
    """Global-index windows of consecutive epochs within each subject (temporal context).

    Same windowing rule as `make_epoch_sequences` (non-overlapping by default,
    never crossing subject boundaries, trailing remainder dropped) but it returns
    only INDICES into the full array instead of the signals, so the epochs stay on
    disk and can be read lazily by a memmap-backed Dataset. Because epochs of a
    subject are stored contiguously and in time order, sliding over that subject's
    sorted split indices reproduces the epoch ordering exactly.

    Parameters
    ----------
    subjects : subject id per epoch for the WHOLE dataset (n_epochs,).
    split_idx : integer indices selecting the epochs of one subject-wise split.
    seq_len : epochs per window.

    Returns
    -------
    (n_windows, seq_len) int array of global indices; empty if no subject in the
    split has at least ``seq_len`` epochs.
    """
    if stride is None:
        stride = seq_len
    subjects = np.asarray(subjects)
    split_idx = np.asarray(split_idx)
    split_subjects = subjects[split_idx]

    windows = []
    for subject in np.unique(split_subjects):
        subject_positions = split_idx[split_subjects == subject]  # sorted global indices
        for start in range(0, len(subject_positions) - seq_len + 1, stride):
            windows.append(subject_positions[start:start + seq_len])
    if not windows:
        return np.empty((0, seq_len), dtype=np.int64)
    return np.asarray(windows, dtype=np.int64)
