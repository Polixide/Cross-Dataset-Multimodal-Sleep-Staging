"""Tests for feature extraction."""
import numpy as np
import pandas as pd

from src.features import extract_epoch_features, extract_features_dataset


def test_extract_features_dataset_returns_named_dataframe():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(6, 4, 300))  # 6 epochs, 4 channels, 300 samples
    features = extract_features_dataset(x, sfreq=100.0)
    assert isinstance(features, pd.DataFrame)
    assert features.shape[0] == 6
    assert features.shape[1] == len(features.columns)
    assert list(features.columns) == list(dict.fromkeys(features.columns))  # unique names
    assert np.isfinite(features.to_numpy()).all()


def test_feature_names_are_prefixed_per_channel():
    epoch = np.random.default_rng(1).normal(size=(2, 300))
    features = extract_epoch_features(epoch, sfreq=100.0, channel_names=["eeg", "eog"])
    assert any(name.startswith("eeg_") for name in features)
    assert any(name.startswith("eog_") for name in features)
    # each channel yields the same number of features
    eeg = [k for k in features if k.startswith("eeg_")]
    eog = [k for k in features if k.startswith("eog_")]
    assert len(eeg) == len(eog)
