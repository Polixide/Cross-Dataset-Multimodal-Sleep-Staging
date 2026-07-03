"""Signal preprocessing and label harmonization.

- map dataset-specific stage annotations to the shared 5-class space;
- basic signal conditioning (bandpass filtering, resampling) on the continuous
  recording, before epoching;
- per-channel z-score normalization fit on TRAINING data only.

The label mapping and normalization functions are implemented and tested,
because they are the leakage-sensitive parts of the pipeline.
"""
import numpy as np

from src.utils import STAGE_TO_INDEX

# Default band-pass edges (Hz) and harmonized sampling rate (master plan:
# Filtering and normalization). 0.3 Hz removes slow drift; 35 Hz keeps the
# sleep-relevant EEG/EOG/EMG band while dropping high-frequency noise.
DEFAULT_L_FREQ = 0.3
DEFAULT_H_FREQ = 35.0
DEFAULT_TARGET_SFREQ = 100.0

# Raw annotation string -> 5-class index. Legacy N4 is merged into N3.
RAW_LABEL_TO_INDEX = {
    "Sleep stage W": STAGE_TO_INDEX["Wake"],
    "Sleep stage 1": STAGE_TO_INDEX["N1"],
    "Sleep stage 2": STAGE_TO_INDEX["N2"],
    "Sleep stage 3": STAGE_TO_INDEX["N3"],
    "Sleep stage 4": STAGE_TO_INDEX["N3"],  # merge N4 into N3
    "Sleep stage R": STAGE_TO_INDEX["REM"],
}

# Epochs that are dropped rather than classified.
DROP_LABELS = {"Sleep stage ?", "Movement time"}


def map_stage_label(raw_label):
    """Map one raw annotation to a 5-class index, or None if it should be dropped."""
    return RAW_LABEL_TO_INDEX.get(raw_label)


def map_stage_labels(raw_labels):
    """Map a list of raw annotations to labels plus a validity mask.

    Returns
    -------
    labels : int array with the mapped epochs only.
    valid_mask : bool array over all epochs (True where mapped). Apply the same
        mask to the signals so signals and labels stay aligned.
    """
    mapped = [map_stage_label(label) for label in raw_labels]
    valid_mask = np.array([value is not None for value in mapped], dtype=bool)
    labels = np.array([value for value in mapped if value is not None], dtype=int)
    return labels, valid_mask


def stage_label_from_text(description):
    """Best-effort mapping of a free-text stage annotation to a 5-class index.

    Handles AASM ("Sleep stage N1", "REM") and R&K-style ("Stage 1", "S4")
    naming; returns None for non-stage markers. Legacy N4 is merged into N3.
    Useful for datasets (e.g. HMC) whose annotation strings differ from Sleep-EDF.
    """
    text = description.upper().replace("SLEEP", " ").replace("STAGE", " ")
    text = " ".join(text.split())
    if text in ("W", "WAKE"):
        return STAGE_TO_INDEX["Wake"]
    if text in ("N1", "S1", "1"):
        return STAGE_TO_INDEX["N1"]
    if text in ("N2", "S2", "2"):
        return STAGE_TO_INDEX["N2"]
    if text in ("N3", "S3", "N4", "S4", "3", "4"):
        return STAGE_TO_INDEX["N3"]
    if text in ("R", "REM"):
        return STAGE_TO_INDEX["REM"]
    return None


def filter_and_resample_raw(raw, l_freq=DEFAULT_L_FREQ, h_freq=DEFAULT_H_FREQ,
                            target_sfreq=DEFAULT_TARGET_SFREQ):
    """Band-pass filter (and optionally resample) a continuous MNE Raw in place.

    Filtering the whole recording BEFORE epoching avoids the edge artifacts that
    per-epoch filtering would introduce at every 30 s boundary, and resampling to
    a shared rate harmonizes datasets so cross-dataset models see epochs of the
    same length (master plan: Filtering and normalization).

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Preloaded raw recording; modified in place and also returned.
    l_freq, h_freq : float
        Band-pass edges in Hz. `h_freq` must stay below the target Nyquist.
    target_sfreq : float or None
        Resample to this rate; if None or already equal, no resampling is done.
    """
    raw.filter(l_freq, h_freq, verbose=False)
    if target_sfreq is not None and raw.info["sfreq"] != target_sfreq:
        raw.resample(target_sfreq, verbose=False)
    return raw


def fit_normalizer(x_train):
    """Compute per-channel mean and std from TRAINING epochs only.

    x_train has shape (n_epochs, n_channels, n_samples). Returns mean and std of
    shape (1, n_channels, 1), ready to reuse on validation/test.
    """
    x_train = np.asarray(x_train, dtype=float)
    mean = x_train.mean(axis=(0, 2), keepdims=True)
    std = x_train.std(axis=(0, 2), keepdims=True)
    std[std == 0] = 1.0
    return mean, std


def apply_normalizer(x, mean, std):
    """Apply a normalizer fit on training data. Never refit on validation/test."""
    x = np.asarray(x, dtype=float)
    return (x - mean) / std
