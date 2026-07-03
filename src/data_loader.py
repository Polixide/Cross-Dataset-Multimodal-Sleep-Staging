"""Dataset loading and SUBJECT-WISE splitting.

This module is the single source of truth for how data is partitioned. All
splits are subject-wise to prevent subject leakage, the central methodological
rule of the project.
"""
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import GroupKFold


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


def load_processed_dataset(path):
    """Load a preprocessed dataset saved as a single .npz with x, y, subjects[, sfreq]."""
    data = np.load(path, allow_pickle=True)
    sfreq = float(data["sfreq"]) if "sfreq" in data.files else None
    return EpochDataset(x=data["x"], y=data["y"], subjects=data["subjects"], sfreq=sfreq)


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
