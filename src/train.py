"""Training, hyperparameter tuning and calibration for the ML and DL pipelines.

- ML: subject-wise cross-validation, small grid-search tuning, post-hoc
  calibration (isotonic / Platt).
- DL: a training loop with class weighting or focal loss, a lightweight random
  search over training hyperparameters, and temperature-scaling calibration.

Random search is used for the DL model instead of Bayesian optimization to avoid
an extra dependency; it can be swapped for optuna without changing the callers.
"""
import numpy as np
import torch
from scipy.optimize import minimize_scalar
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import GridSearchCV, GroupKFold
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from src.data_loader import subject_wise_folds
from src.evaluate import compute_metrics
from src.models_dl import FocalLoss

# Small, defensible search grids per model (subject-wise CV).
ML_PARAM_GRIDS = {
    "svm": {"C": [1, 10], "gamma": ["scale", 0.01]},
    "rf": {"n_estimators": [200, 400], "max_depth": [None, 20]},
    "gb": {"max_depth": [None, 5], "learning_rate": [0.05, 0.1]},
    "logreg": {"C": [0.5, 1.0, 2.0]},
}


# --- Machine learning -------------------------------------------------------

def compute_class_weights(y):
    """Inverse-frequency class weights (mean-normalized) for weighted losses."""
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    weights = counts.sum() / (len(classes) * counts)
    return {int(cls): float(weight) for cls, weight in zip(classes, weights)}


def cross_validate_ml(model_builder, x, y, subjects, n_splits=5, labels=None):
    """Run subject-wise k-fold CV for a feature-based model.

    model_builder is a zero-argument function that returns a fresh estimator, so
    each fold trains an independent model. Returns a list of per-fold metric dicts.
    """
    fold_metrics = []
    for train_idx, val_idx in subject_wise_folds(subjects, n_splits=n_splits):
        model = model_builder()
        model.fit(x[train_idx], y[train_idx])
        y_pred = model.predict(x[val_idx])
        fold_metrics.append(compute_metrics(y[val_idx], y_pred, labels=labels))
    return fold_metrics


def tune_ml(estimator, name, x, y, subjects, n_splits=5, scoring="f1_macro"):
    """Grid-search the model's hyperparameters with subject-wise CV.

    Returns (best_estimator, best_params, best_score). The estimator's grid keys
    are prefixed with 'clf__' automatically when it is a pipeline.
    """
    grid = ML_PARAM_GRIDS[name.lower()]
    if hasattr(estimator, "named_steps") and "clf" in estimator.named_steps:
        grid = {f"clf__{key}": value for key, value in grid.items()}
    n_splits = min(n_splits, len(np.unique(subjects)))
    search = GridSearchCV(estimator, grid, scoring=scoring, cv=GroupKFold(n_splits=n_splits), n_jobs=-1)
    search.fit(x, y, groups=subjects)
    return search.best_estimator_, search.best_params_, float(search.best_score_)


def calibrate_classifier(fitted_estimator, x_val, y_val, method="isotonic"):
    """Fit post-hoc calibration on VALIDATION data only.

    method : 'isotonic' or 'sigmoid' (Platt scaling). FrozenEstimator keeps the
    base model fixed so calibration never re-trains it.
    """
    calibrated = CalibratedClassifierCV(FrozenEstimator(fitted_estimator), method=method)
    calibrated.fit(x_val, y_val)
    return calibrated


# --- Deep learning ----------------------------------------------------------

def make_dataloader(x, y, batch_size=64, shuffle=False, balanced=False):
    """Wrap epoch arrays (x, y) in a torch DataLoader.

    balanced=True draws class-balanced mini-batches with a WeightedRandomSampler.
    """
    x_tensor = torch.tensor(np.asarray(x), dtype=torch.float32)
    y_tensor = torch.tensor(np.asarray(y), dtype=torch.long)
    dataset = TensorDataset(x_tensor, y_tensor)
    y_array = np.asarray(y)
    # Balanced sampling is only defined for per-epoch (1D) labels, not sequences.
    if balanced and y_array.ndim == 1:
        counts = np.bincount(y_array)
        sample_weights = 1.0 / counts[y_array]
        sampler = WeightedRandomSampler(
            torch.tensor(sample_weights, dtype=torch.double), len(sample_weights), replacement=True
        )
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _build_weight_tensor(class_weights, n_classes, device):
    weights = torch.ones(n_classes)
    for cls, weight in class_weights.items():
        weights[cls] = weight
    return weights.to(device)


def evaluate_dl(model, loader, device="cpu"):
    """Run the model over a DataLoader and return the metric dict."""
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            logits = model(x_batch.to(device))
            # reshape(-1, ...) flattens sequence outputs; a no-op for per-epoch outputs.
            y_pred.append(logits.argmax(dim=-1).reshape(-1).cpu().numpy())
            y_true.append(y_batch.reshape(-1).numpy())
    return compute_metrics(np.concatenate(y_true), np.concatenate(y_pred))


def predict_logits_dl(model, loader, device="cpu"):
    """Return (y_true, logits) as numpy arrays for the whole loader.

    Sequence outputs (batch, seq_len, n_classes) are flattened to (N, n_classes).
    """
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
    """Train a deep model and keep the weights with the best validation macro-F1.

    Returns (model, history); history has per-epoch train loss and val macro-F1.
    """
    model = model.to(device)
    weight_tensor = None
    if class_weights is not None:
        weight_tensor = _build_weight_tensor(class_weights, model.n_classes, device)

    criterion = FocalLoss(weight=weight_tensor) if use_focal else torch.nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    best_macro_f1 = -1.0
    best_state = None
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            # reshape unifies per-epoch (batch, C) and sequence (batch, seq_len, C) outputs.
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
            best_state = {name: tensor.cpu().clone() for name, tensor in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def random_search_dl(model_factory, train_loader, val_loader, n_trials=6, epochs=10,
                     class_weights=None, seed=42):
    """Random search over learning rate and loss type (validation macro-F1).

    model_factory is a zero-argument callable returning a fresh model.
    Returns (best, trials): best has the trained model plus its hyperparameters.
    """
    rng = np.random.default_rng(seed)
    learning_rates = [1e-2, 3e-3, 1e-3, 3e-4]
    best, trials = None, []
    for _ in range(n_trials):
        lr = float(rng.choice(learning_rates))
        use_focal = bool(rng.integers(0, 2))
        model, history = train_dl(model_factory(), train_loader, val_loader,
                                  epochs=epochs, lr=lr, class_weights=class_weights, use_focal=use_focal)
        score = max(step["val_macro_f1"] for step in history)
        trials.append({"lr": lr, "use_focal": use_focal, "val_macro_f1": score})
        if best is None or score > best["val_macro_f1"]:
            best = {"model": model, "lr": lr, "use_focal": use_focal, "val_macro_f1": score}
    return best, trials


# --- Temperature scaling (DL calibration) -----------------------------------

def fit_temperature(logits, y_true):
    """Fit a single temperature by minimizing validation NLL (Guo et al., 2017)."""
    logits = np.asarray(logits, dtype=float)
    y_true = np.asarray(y_true)

    def negative_log_likelihood(temperature):
        scaled = logits / temperature
        scaled = scaled - scaled.max(axis=1, keepdims=True)
        log_probs = scaled - np.log(np.exp(scaled).sum(axis=1, keepdims=True))
        return -log_probs[np.arange(len(y_true)), y_true].mean()

    result = minimize_scalar(negative_log_likelihood, bounds=(0.05, 10.0), method="bounded")
    return float(result.x)


def softmax_with_temperature(logits, temperature=1.0):
    """Softmax of logits divided by a temperature (returns probabilities)."""
    logits = np.asarray(logits, dtype=float) / temperature
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)
