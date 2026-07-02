"""Tests for the deep-learning models, focused on the Cross-Modal Transformer.

These check the shapes and the modality-masking mechanism that ablation and
missing-modality experiments rely on.
"""
import torch

from src.models_dl import build_dl_model, build_sequence_model


def test_all_models_expose_n_classes_and_output_shape():
    for name in ["cnn", "cnn_lstm", "transformer"]:
        model = build_dl_model(name, n_channels=4, n_classes=5)
        assert model.n_classes == 5
        logits = model(torch.randn(2, 4, 64))
        assert logits.shape == (2, 5)


def test_transformer_runs_with_a_single_modality():
    model = build_dl_model("transformer", n_channels=4, n_classes=5)
    x = torch.randn(2, 4, 64)
    # ablation / missing-modality: masking to EEG only must still produce logits
    eeg_only = model(x, active_modalities=["eeg"])
    assert eeg_only.shape == (2, 5)


def test_transformer_modality_masking_changes_output():
    torch.manual_seed(0)
    model = build_dl_model("transformer", n_channels=4, n_classes=5)
    model.eval()  # disable dropout so the comparison is deterministic
    x = torch.randn(2, 4, 64)
    full = model(x, active_modalities=["eeg", "eog", "emg"])
    eeg_only = model(x, active_modalities=["eeg"])
    assert not torch.allclose(full, eeg_only)


def test_sequence_model_predicts_one_label_per_epoch():
    for name in ["cnn", "cnn_lstm", "transformer"]:
        model = build_sequence_model(name, n_channels=4, n_classes=5, max_len=8)
        # (batch=2, seq_len=5, channels=4, samples=64) -> (2, 5, 5)
        out = model(torch.randn(2, 5, 4, 64))
        assert out.shape == (2, 5, 5)
        assert model.n_classes == 5
