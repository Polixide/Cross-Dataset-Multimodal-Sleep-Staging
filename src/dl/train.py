"""Training, random search and temperature calibration for deep learning."""
import numpy as np
import torch
from scipy.optimize import minimize_scalar
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from src.common.evaluation import compute_metrics
from src.dl.models import FocalLoss


def compute_class_weights(y):
    """Inverse-frequency class weights (mean-normalized)."""
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    weights = counts.sum() / (len(classes) * counts)
    return {int(cls): float(weight) for cls, weight in zip(classes, weights)}


def make_dataloader(x, y, batch_size=64, shuffle=False, balanced=False):
    x_tensor = torch.tensor(np.asarray(x), dtype=torch.float32)
    y_tensor = torch.tensor(np.asarray(y), dtype=torch.long)
    dataset = TensorDataset(x_tensor, y_tensor)
    y_array = np.asarray(y)
    if balanced and y_array.ndim == 1:
        counts = np.bincount(y_array)
        sample_weights = 1.0 / counts[y_array]
        sampler = WeightedRandomSampler(
            torch.tensor(sample_weights, dtype=torch.double),
            len(sample_weights), replacement=True,
        )
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _build_weight_tensor(class_weights, n_classes, device):
    weights = torch.ones(n_classes)
    for cls, weight in class_weights.items():
        weights[cls] = weight
    return weights.to(device)


def evaluate_dl(model, loader, device="cpu"):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            logits = model(x_batch.to(device))
            y_pred.append(logits.argmax(dim=-1).reshape(-1).cpu().numpy())
            y_true.append(y_batch.reshape(-1).numpy())
    return compute_metrics(np.concatenate(y_true), np.concatenate(y_pred))


def predict_logits_dl(model, loader, device="cpu"):
    model.eval()
    y_true, logits = [], []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            batch_logits = model(x_batch.to(device))
            logits.append(batch_logits.reshape(-1, batch_logits.shape[-1]).cpu().numpy())
            y_true.append(y_batch.reshape(-1).numpy())
    return np.concatenate(y_true), np.concatenate(logits)


def train_dl(model, train_loader, val_loader, epochs=20, lr=1e-3,
             class_weights=None, use_focal=False, device="cpu"):
    model = model.to(device)
    weight_tensor = None
    if class_weights is not None:
        weight_tensor = _build_weight_tensor(class_weights, model.n_classes, device)
    criterion = (
        FocalLoss(weight=weight_tensor) if use_focal
        else torch.nn.CrossEntropyLoss(weight=weight_tensor)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history, best_macro_f1, best_state = [], -1.0, None
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), y_batch.reshape(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        val_metrics = evaluate_dl(model, val_loader, device)
        history.append({
            "epoch": epoch,
            "train_loss": epoch_loss / len(train_loader),
            "val_macro_f1": val_metrics["macro_f1"],
        })
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_state = {
                name: tensor.cpu().clone()
                for name, tensor in model.state_dict().items()
            }
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def random_search_dl(model_factory, train_loader, val_loader, n_trials=6,
                     epochs=10, class_weights=None, seed=42):
    rng = np.random.default_rng(seed)
    learning_rates = [1e-2, 3e-3, 1e-3, 3e-4]
    best, trials = None, []
    for _ in range(n_trials):
        lr = float(rng.choice(learning_rates))
        use_focal = bool(rng.integers(0, 2))
        model, history = train_dl(
            model_factory(), train_loader, val_loader, epochs=epochs, lr=lr,
            class_weights=class_weights, use_focal=use_focal,
        )
        score = max(step["val_macro_f1"] for step in history)
        trials.append({"lr": lr, "use_focal": use_focal, "val_macro_f1": score})
        if best is None or score > best["val_macro_f1"]:
            best = {
                "model": model, "lr": lr, "use_focal": use_focal,
                "val_macro_f1": score,
            }
    return best, trials


def fit_temperature(logits, y_true):
    logits = np.asarray(logits, dtype=float)
    y_true = np.asarray(y_true)

    def negative_log_likelihood(temperature):
        scaled = logits / temperature
        scaled = scaled - scaled.max(axis=1, keepdims=True)
        log_probs = scaled - np.log(np.exp(scaled).sum(axis=1, keepdims=True))
        return -log_probs[np.arange(len(y_true)), y_true].mean()

    result = minimize_scalar(
        negative_log_likelihood, bounds=(0.05, 10.0), method="bounded"
    )
    return float(result.x)


def softmax_with_temperature(logits, temperature=1.0):
    logits = np.asarray(logits, dtype=float) / temperature
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)
