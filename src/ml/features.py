"""Feature extraction for the feature-based ML pipeline.

Organized by the domains from the master plan: time, frequency, nonlinear and
time-frequency. `extract_features_dataset` turns a raw epoch array into the 2D
feature matrix the ML models consume.
"""
import numpy as np
from scipy.signal import stft, welch
from tqdm import tqdm

# Standard EEG frequency bands in Hz.
FREQ_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "sigma": (12.0, 16.0),
    "beta": (16.0, 30.0),
}


def time_domain_features(signal):
    """Amplitude statistics of a 1D signal (time domain)."""
    signal = np.asarray(signal, dtype=float)
    zero_crossing_rate = np.mean(np.abs(np.diff(np.sign(signal))) > 0)
    return {
        "mean": float(np.mean(signal)),
        "std": float(np.std(signal)),
        "rms": float(np.sqrt(np.mean(signal ** 2))),
        "ptp": float(np.ptp(signal)),  # peak-to-peak amplitude
        "zero_crossing_rate": float(zero_crossing_rate),
    }


def _welch_psd(signal, sfreq):
    """Welch power spectral density (shared by the frequency-domain features)."""
    return welch(signal, fs=sfreq, nperseg=min(len(signal), int(sfreq * 4)))


def _band_power(freqs, psd, band):
    low, high = band
    band_mask = (freqs >= low) & (freqs < high)
    return float(np.trapezoid(psd[band_mask], freqs[band_mask]))


def bandpower(signal, sfreq, band):
    """Absolute power within a frequency band using Welch's PSD."""
    freqs, psd = _welch_psd(signal, sfreq)
    return _band_power(freqs, psd, band)


def spectral_edge_frequency(freqs, psd, edge=0.95):
    """Frequency below which `edge` fraction of the total power lies."""
    cumulative = np.cumsum(psd)
    total = cumulative[-1] or 1e-12
    index = int(np.searchsorted(cumulative, edge * total))
    return float(freqs[min(index, len(freqs) - 1)])


def spectral_entropy(psd):
    """Shannon entropy of the normalized power spectrum."""
    psd_norm = psd / (psd.sum() or 1e-12)
    psd_norm = psd_norm[psd_norm > 0]
    return float(-np.sum(psd_norm * np.log2(psd_norm)))


def frequency_domain_features(signal, sfreq):
    """Band powers, relative band powers, spectral edge/entropy and band ratios."""
    freqs, psd = _welch_psd(signal, sfreq)
    powers = {name: _band_power(freqs, psd, band) for name, band in FREQ_BANDS.items()}
    total_power = sum(powers.values()) or 1e-12

    features = {f"bp_{name}": value for name, value in powers.items()}
    for name, value in powers.items():
        features[f"rbp_{name}"] = value / total_power
    features["spectral_edge_freq"] = spectral_edge_frequency(freqs, psd)
    features["spectral_entropy"] = spectral_entropy(psd)
    features["delta_theta_ratio"] = powers["delta"] / (powers["theta"] or 1e-12)
    features["slow_fast_ratio"] = (
        (powers["delta"] + powers["theta"]) /
        ((powers["alpha"] + powers["sigma"] + powers["beta"]) or 1e-12)
    )
    return features


def nonlinear_features(signal):
    """Hjorth parameters (activity, mobility, complexity)."""
    signal = np.asarray(signal, dtype=float)
    first_diff = np.diff(signal)
    second_diff = np.diff(first_diff)
    var_signal = np.var(signal) or 1e-12
    var_first = np.var(first_diff) or 1e-12
    var_second = np.var(second_diff)
    mobility = np.sqrt(var_first / var_signal)
    complexity = np.sqrt(var_second / var_first) / mobility if mobility else 0.0
    return {
        "hjorth_activity": float(var_signal),
        "hjorth_mobility": float(mobility),
        "hjorth_complexity": float(complexity),
    }


def time_frequency_features(signal, sfreq):
    """Summaries of how spectral power fluctuates over time within the epoch (STFT)."""
    _, _, coeffs = stft(signal, fs=sfreq, nperseg=min(len(signal), int(sfreq)))
    frame_power = (np.abs(coeffs) ** 2).sum(axis=0)  # total power per time frame
    return {
        "tf_power_mean": float(frame_power.mean()),
        "tf_power_std": float(frame_power.std()),
    }


def extract_epoch_features(epoch, sfreq, channel_names=None):
    """Extract the full feature dictionary for one multi-channel epoch.

    epoch has shape (n_channels, n_samples).
    """
    epoch = np.atleast_2d(epoch)
    if channel_names is None:
        channel_names = [f"ch{i}" for i in range(epoch.shape[0])]

    features = {}
    for name, channel in zip(channel_names, epoch):
        for group in (time_domain_features(channel),
                      frequency_domain_features(channel, sfreq),
                      nonlinear_features(channel),
                      time_frequency_features(channel, sfreq)):
            for key, value in group.items():
                features[f"{name}_{key}"] = value
    return features


def extract_features_dataset(x, sfreq, channel_names=None):
    """Turn a raw epoch array into a 2D feature matrix for the ML models.

    x has shape (n_epochs, n_channels, n_samples).
    Returns (features, feature_names): features is (n_epochs, n_features).
    """
    x = np.asarray(x)
    rows = []
    feature_names = None
    for epoch in tqdm(x, desc="extracting features", unit="epoch"):
        epoch_features = extract_epoch_features(epoch, sfreq, channel_names)
        if feature_names is None:
            feature_names = list(epoch_features)
        rows.append([epoch_features[name] for name in feature_names])
    return np.array(rows, dtype=float), feature_names
