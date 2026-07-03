"""Tests for label harmonization, filtering/resampling and training-only normalization."""
import mne
import numpy as np

from src.preprocessing import (
    apply_normalizer,
    filter_and_resample_raw,
    fit_normalizer,
    map_stage_label,
    map_stage_labels,
    stage_label_from_text,
)


def _band_power(signal, sfreq, low, high):
    spectrum = np.abs(np.fft.rfft(signal)) ** 2
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / sfreq)
    return spectrum[(freqs >= low) & (freqs < high)].sum()


def test_label_mapping_merges_n3_n4_and_drops_unknown():
    assert map_stage_label("Sleep stage W") == 0
    assert map_stage_label("Sleep stage 1") == 1
    assert map_stage_label("Sleep stage 2") == 2
    assert map_stage_label("Sleep stage 3") == 3
    assert map_stage_label("Sleep stage 4") == 3  # legacy N4 merged into N3
    assert map_stage_label("Sleep stage R") == 4
    assert map_stage_label("Sleep stage ?") is None
    assert map_stage_label("Movement time") is None


def test_stage_label_from_text_handles_aasm_and_rk_naming():
    assert stage_label_from_text("Sleep stage W") == 0
    assert stage_label_from_text("Sleep stage N1") == 1
    assert stage_label_from_text("Stage 2") == 2
    assert stage_label_from_text("Sleep stage N3") == 3
    assert stage_label_from_text("S4") == 3          # legacy N4 merged into N3
    assert stage_label_from_text("REM") == 4
    assert stage_label_from_text("Sleep stage R") == 4
    assert stage_label_from_text("Movement time") is None
    assert stage_label_from_text("Lights off") is None


def test_map_stage_labels_returns_aligned_mask_and_labels():
    raw = ["Sleep stage W", "Movement time", "Sleep stage 2"]
    labels, mask = map_stage_labels(raw)
    assert mask.tolist() == [True, False, True]
    assert labels.tolist() == [0, 2]
    # mask length matches the raw sequence; labels length matches valid epochs
    assert len(mask) == len(raw)
    assert len(labels) == int(mask.sum())


def test_normalizer_fit_on_train_gives_zero_mean_unit_std():
    rng = np.random.default_rng(0)
    x_train = rng.normal(5.0, 2.0, size=(50, 2, 100))
    mean, std = fit_normalizer(x_train)
    x_norm = apply_normalizer(x_train, mean, std)
    assert np.allclose(x_norm.mean(axis=(0, 2)), 0.0, atol=1e-6)
    assert np.allclose(x_norm.std(axis=(0, 2)), 1.0, atol=1e-6)


def test_normalizer_uses_training_stats_on_new_data():
    # A constant test batch normalized with training stats must NOT become 0/1;
    # this guards against accidentally refitting statistics on val/test data.
    x_train = np.ones((10, 1, 20)) * 3.0
    x_train[:5] = 1.0  # give training data non-zero variance
    mean, std = fit_normalizer(x_train)
    x_test = np.ones((4, 1, 20)) * 10.0
    x_test_norm = apply_normalizer(x_test, mean, std)
    assert not np.allclose(x_test_norm.mean(), 0.0)


def test_filter_and_resample_changes_rate_and_attenuates_high_freq():
    sfreq = 200.0
    t = np.arange(0, 20, 1.0 / sfreq)
    # equal-amplitude 10 Hz (kept) and 45 Hz (above the 35 Hz cutoff -> removed)
    signal = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 45 * t)
    info = mne.create_info(["EEG"], sfreq, ch_types="eeg")
    raw = mne.io.RawArray(signal[np.newaxis, :], info, verbose=False)

    filter_and_resample_raw(raw, l_freq=0.3, h_freq=35.0, target_sfreq=100.0)

    assert raw.info["sfreq"] == 100.0
    out = raw.get_data()[0]
    # the 45 Hz component should be strongly attenuated relative to the 10 Hz one
    assert _band_power(out, 100.0, 40, 50) < 0.05 * _band_power(out, 100.0, 5, 15)
