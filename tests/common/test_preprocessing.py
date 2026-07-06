"""Tests for label harmonization and training-only normalization."""
import numpy as np

from src.common.preprocessing import (
    apply_normalizer,
    effective_bandpass,
    filter_psg_channels,
    fit_normalizer,
    map_stage_label,
    map_stage_labels,
    stage_label_from_text,
)


def test_modality_bands_and_nyquist_clipping():
    assert effective_bandpass("EEG Fpz-Cz", 100) == (0.3, 35.0)
    assert effective_bandpass("EOG horizontal", 100) == (0.1, 15.0)
    assert effective_bandpass("EMG submental", 100) == (10.0, 45.0)
    assert effective_bandpass("ECG", 100) == (0.5, 40.0)


def test_filter_psg_channels_preserves_shape_and_finite_values():
    rng = np.random.default_rng(0)
    signal = rng.normal(size=(4, 3000))
    filtered = filter_psg_channels(
        signal,
        sfreq=100,
        channel_names=["EEG C3", "EOG L", "EMG chin", "ECG"],
    )
    assert filtered.shape == signal.shape
    assert np.isfinite(filtered).all()


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
