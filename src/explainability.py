"""Explainability: SHAP for feature-based models, Grad-CAM for 1D-CNNs.

SHAP gives global feature importance for the ML pipeline; Grad-CAM highlights
the signal regions driving a CNN decision. Figures are saved under
results/figures for reproducibility.
"""
import matplotlib

matplotlib.use("Agg")  # non-interactive backend so figures save without a display

import matplotlib.pyplot as plt
import numpy as np
import shap
import torch

from src.utils import ensure_dir


def shap_feature_importance(model, x_background, x_explain, feature_names=None,
                            out_dir="results/figures", top_k=20):
    """Compute SHAP values for a fitted ML model and save a feature-importance bar plot.

    Importance is the mean absolute SHAP value per feature (averaged over classes
    for multi-class models). Returns the path to the saved figure.
    """
    out_dir = ensure_dir(out_dir)
    explainer = shap.Explainer(model, x_background)
    shap_values = np.abs(explainer(x_explain).values)
    if shap_values.ndim == 3:              # multi-class: average over the class axis
        shap_values = shap_values.mean(axis=2)
    importance = shap_values.mean(axis=0)  # one value per feature

    if feature_names is None:
        feature_names = [f"f{i}" for i in range(len(importance))]
    # top_k features, ascending so the most important ends up at the top of the bar chart
    top_features = np.argsort(importance)[-top_k:]

    plt.figure(figsize=(8, max(4, 0.3 * len(top_features))))
    plt.barh([feature_names[i] for i in top_features], importance[top_features])
    plt.xlabel("Mean |SHAP value|")
    plt.title("SHAP feature importance")
    plt.tight_layout()
    out_path = out_dir / "shap_feature_importance.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def _last_conv1d(model):
    """Return the last Conv1d layer of a model (Grad-CAM target)."""
    conv = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv1d):
            conv = module
    if conv is None:
        raise ValueError("No Conv1d layer found for Grad-CAM.")
    return conv


def grad_cam_1d(model, sample, target_class, target_layer=None):
    """Return a 1D Grad-CAM importance map (length = input length, values in [0, 1]).

    sample has shape (n_channels, n_samples) or (1, n_channels, n_samples).
    """
    model.eval()
    if target_layer is None:
        target_layer = _last_conv1d(model)

    activations, gradients = {}, {}

    def save_activation(module, inputs, output):
        activations["value"] = output.detach()

    def save_gradient(module, grad_input, grad_output):
        gradients["value"] = grad_output[0].detach()

    forward_handle = target_layer.register_forward_hook(save_activation)
    backward_handle = target_layer.register_full_backward_hook(save_gradient)

    x = torch.tensor(np.asarray(sample), dtype=torch.float32)
    if x.ndim == 2:
        x = x.unsqueeze(0)
    logits = model(x)
    model.zero_grad()
    logits[0, target_class].backward()

    forward_handle.remove()
    backward_handle.remove()

    weights = gradients["value"][0].mean(dim=1)                  # (channels,)
    cam = torch.relu((weights[:, None] * activations["value"][0]).sum(dim=0))
    cam = cam.numpy()

    input_length = x.shape[-1]
    cam = np.interp(np.linspace(0, len(cam) - 1, input_length), np.arange(len(cam)), cam)
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam
