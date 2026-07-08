"""Explainability: SHAP for feature-based models, Grad-CAM for deep models.

SHAP gives global feature importance for the ML pipeline. For the deep models we
provide two signal-level views, matched to each architecture:

- 1D-CNN (``cnn``): a single Grad-CAM over the last convolution — the canonical
  "which part of the epoch drove the decision" saliency.
- Cross-Modal Transformer (``transformer``): a *per-modality* Grad-CAM (one map
  for EEG, EOG and EMG, from each modality encoder) plus, best-effort, the
  CLS→modality cross-attention shares. Together they show not just *when* in the
  30 s epoch but *which modality* the fused decision relied on — a single
  last-conv Grad-CAM would only ever reflect the last encoder (EMG) and mislead.

The star visual is a stacked multi-channel overlay: every channel's signal drawn
over its own saliency heatmap on a shared time axis, so the highlighted regions
are read directly against the waveform. Figures are saved under results/figures
for reproducibility.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend so figures save without a display

import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
from matplotlib import gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from src.utils import MODALITY_CHANNELS, ensure_dir


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


def _is_cross_modal(model):
    """True for a Cross-Modal Transformer (has one Conv encoder per modality)."""
    return hasattr(model, "encoders") and hasattr(model, "modality_names")


def epoch_model(model):
    """Unwrap a SequenceSleepStager to its per-epoch encoder for epoch-level XAI."""
    return getattr(model, "epoch_encoder", model)


def _cam_from(activation, gradient, input_length):
    """Grad-CAM core: weight activation channels by mean gradient, ReLU, resize."""
    weights = gradient.mean(dim=1)                                   # (channels,)
    cam = torch.relu((weights[:, None] * activation).sum(dim=0)).numpy()
    return np.interp(np.linspace(0, len(cam) - 1, input_length), np.arange(len(cam)), cam)


def grad_cam_per_modality(model, sample, target_class):
    """Per-modality 1D Grad-CAM for the Cross-Modal Transformer.

    Hooks the last Conv1d of every modality encoder (EEG/EOG/EMG) and, from a
    single backward pass on the ``target_class`` logit, returns one saliency map
    per modality (dict name -> array of length = input length). The maps are on a
    common (unnormalized) scale so their relative heights are comparable — that is
    what makes "which modality mattered, and when" readable. Use `epoch_model`
    first if you hold a sequence model.
    """
    model = epoch_model(model)
    if not _is_cross_modal(model):
        raise ValueError("grad_cam_per_modality expects a Cross-Modal Transformer.")
    model.eval()

    activations, gradients, handles = {}, {}, []

    def make_forward(name):
        def hook(module, inputs, output):
            activations[name] = output.detach()
        return hook

    def make_backward(name):
        def hook(module, grad_input, grad_output):
            gradients[name] = grad_output[0].detach()
        return hook

    for name, encoder in model.encoders.items():
        conv = _last_conv1d(encoder)
        handles.append(conv.register_forward_hook(make_forward(name)))
        handles.append(conv.register_full_backward_hook(make_backward(name)))

    x = torch.tensor(np.asarray(sample), dtype=torch.float32)
    if x.ndim == 2:
        x = x.unsqueeze(0)
    logits = model(x)
    model.zero_grad()
    logits[0, target_class].backward()
    for handle in handles:
        handle.remove()

    input_length = x.shape[-1]
    return {name: _cam_from(activations[name][0], gradients[name][0], input_length)
            for name in model.modality_names}


def modality_importance_from_cams(cams):
    """Relative modality importance = each modality's share of total Grad-CAM mass."""
    mass = {name: float(np.sum(cam)) for name, cam in cams.items()}
    total = sum(mass.values()) or 1.0
    return {name: value / total for name, value in mass.items()}


def channel_saliency(model, sample, target_class):
    """Model-agnostic per-channel importance = mean |input x d(logit)/d(input)| over time.

    A cheap gradient-times-input attribution that works for any architecture
    (used to annotate the CNN figure with which channel drove the decision).
    """
    model = epoch_model(model)
    model.eval()
    x = torch.tensor(np.asarray(sample), dtype=torch.float32)
    if x.ndim == 2:
        x = x.unsqueeze(0)
    x.requires_grad_(True)
    logits = model(x)
    model.zero_grad()
    logits[0, target_class].backward()
    return (x.grad[0] * x[0]).abs().mean(dim=1).detach().numpy()   # (n_channels,)


def transformer_cls_attention(model, sample):
    """Best-effort CLS->token cross-attention for the Cross-Modal Transformer.

    Returns ``{"modality_share": {name: fraction}, "per_token": {name: array}}`` —
    how much the fused CLS token attends to each modality's tokens (averaged over
    heads and layers), and the within-epoch attention over that modality's token
    slots. Returns None if the running PyTorch build won't expose attention weights
    (the per-modality Grad-CAM already carries the explanation in that case).
    """
    model = epoch_model(model)
    if not _is_cross_modal(model):
        return None

    captured = []
    # Force the eager attention path so weights are actually returned (the fused
    # fast path discards them). Guarded: older torch lacks this switch.
    fastpath = None
    try:
        import torch.backends.mha as mha
        fastpath = mha.get_fastpath_enabled()
        mha.set_fastpath_enabled(False)
    except Exception:
        mha = None

    originals = []
    for layer in model.transformer.layers:
        attn = layer.self_attn
        original = attn.forward

        def wrapped(*args, _original=original, **kwargs):
            kwargs["need_weights"] = True
            kwargs["average_attn_weights"] = True
            output = _original(*args, **kwargs)
            if isinstance(output, tuple) and len(output) == 2 and output[1] is not None:
                captured.append(output[1].detach())
            return output

        attn.forward = wrapped
        originals.append((attn, original))

    try:
        x = torch.tensor(np.asarray(sample), dtype=torch.float32)
        if x.ndim == 2:
            x = x.unsqueeze(0)
        with torch.no_grad():
            model(x)
    finally:
        for attn, original in originals:
            attn.forward = original
        if mha is not None and fastpath is not None:
            mha.set_fastpath_enabled(fastpath)

    if not captured:
        return None

    # Average CLS row (query index 0) over layers -> attention to every token.
    cls_attention = torch.stack([w[0, 0] for w in captured]).mean(dim=0).numpy()
    n_tokens = model.n_tokens
    modality_share, per_token, start = {}, {}, 1  # index 0 is the CLS token itself
    for name in model.modality_names:
        block = cls_attention[start:start + n_tokens]
        per_token[name] = block
        modality_share[name] = float(block.sum())
        start += n_tokens
    total = sum(modality_share.values()) or 1.0
    modality_share = {name: value / total for name, value in modality_share.items()}
    return {"modality_share": modality_share, "per_token": per_token}


def cams_by_channel(cams, n_channels, modality_channels=None):
    """Expand a saliency spec into one map per channel for the overlay plot.

    ``cams`` is either a single array (used behind every channel, e.g. the CNN
    Grad-CAM) or a dict modality -> array (each modality map is placed behind its
    own channels, e.g. the transformer per-modality Grad-CAM).
    """
    if isinstance(cams, dict):
        modality_channels = modality_channels or MODALITY_CHANNELS
        length = len(next(iter(cams.values())))
        per_channel = [np.zeros(length) for _ in range(n_channels)]
        for name, cam in cams.items():
            for channel in modality_channels[name]:
                per_channel[channel] = cam
        return per_channel
    return [np.asarray(cams) for _ in range(n_channels)]


def plot_epoch_explanation(sample, cams, channel_names, out_path, title=None,
                           sfreq=100.0, importance=None, importance_label="Importance",
                           modality_channels=None):
    """Stacked multi-channel overlay: each channel's signal over its saliency heatmap.

    sample : (n_channels, n_samples) — the epoch the model saw.
    cams   : single array or {modality: array} (see `cams_by_channel`); every map is
             scaled by one global maximum so the color is comparable across channels.
    importance : optional {label: value} bar (which modality/channel drove the call).

    Saves a figure with one row per channel (waveform + red saliency background on a
    shared Time (s) axis) and an optional importance bar, then returns its path.
    """
    sample = np.asarray(sample)
    n_channels, n_samples = sample.shape
    time = np.arange(n_samples) / sfreq
    per_channel = cams_by_channel(cams, n_channels, modality_channels)
    global_max = max((c.max() for c in per_channel if c.size), default=0.0) or 1.0
    normalize = Normalize(vmin=0.0, vmax=1.0)

    rows = n_channels + (1 if importance else 0)
    heights = [1] * n_channels + ([0.8] if importance else [])
    fig = plt.figure(figsize=(10, 1.6 * rows + 1))
    grid = gridspec.GridSpec(rows, 1, height_ratios=heights, hspace=0.6)

    for i in range(n_channels):
        ax = fig.add_subplot(grid[i])
        signal = sample[i]
        cam = per_channel[i] / global_max
        ax.imshow(cam[None, :], aspect="auto", cmap="YlOrRd", norm=normalize,
                  extent=[time[0], time[-1], float(signal.min()), float(signal.max())],
                  origin="lower", alpha=0.6, zorder=0)
        ax.plot(time, signal, color="black", linewidth=0.6, zorder=1)
        ax.set_ylabel(channel_names[i], fontsize=8, rotation=0, ha="right", va="center")
        ax.set_xlim(time[0], time[-1])
        ax.set_yticks([])
        if i == n_channels - 1:
            ax.set_xlabel("Time (s)")
        else:
            ax.set_xticklabels([])

    if importance:
        ax = fig.add_subplot(grid[n_channels])
        labels = list(importance.keys())
        ax.bar(labels, [importance[k] for k in labels], color="#c1272d")
        ax.set_ylabel(importance_label, fontsize=8)
        ax.set_ylim(0, (max(importance.values()) * 1.2) or 1.0)
        ax.tick_params(labelsize=8)

    mappable = ScalarMappable(norm=normalize, cmap="YlOrRd")
    colorbar = fig.colorbar(mappable, ax=fig.axes, fraction=0.025, pad=0.02)
    colorbar.set_label("Grad-CAM saliency (normalized)", fontsize=8)

    if title:
        fig.suptitle(title, fontsize=11)
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
