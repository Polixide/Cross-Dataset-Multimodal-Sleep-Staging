"""Grad-CAM explainability for one-dimensional deep sleep models."""
import numpy as np
import torch


def _last_conv1d(model):
    conv = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv1d):
            conv = module
    if conv is None:
        raise ValueError("No Conv1d layer found for Grad-CAM.")
    return conv


def grad_cam_1d(model, sample, target_class, target_layer=None):
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
    weights = gradients["value"][0].mean(dim=1)
    cam = torch.relu((weights[:, None] * activations["value"][0]).sum(dim=0)).numpy()
    input_length = x.shape[-1]
    cam = np.interp(
        np.linspace(0, len(cam) - 1, input_length), np.arange(len(cam)), cam
    )
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam
