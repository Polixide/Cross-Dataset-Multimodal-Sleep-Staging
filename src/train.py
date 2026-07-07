"""Training, hyperparameter tuning and calibration for the ML and DL pipelines.

- ML: subject-wise cross-validation, small grid-search tuning, post-hoc
  calibration (isotonic / Platt).
- DL: a training loop with class weighting or focal loss, a lightweight random
  search over training hyperparameters, and temperature-scaling calibration.

Random search is used for the DL model instead of Bayesian optimization to avoid
an extra dependency; it can be swapped for optuna without changing the callers.
"""
import warnings
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from tqdm.auto import tqdm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import GridSearchCV, GroupKFold
from torch.utils.data import DataLoader, Dataset, TensorDataset, WeightedRandomSampler

from src.data_loader import subject_wise_folds
from src.evaluate import compute_metrics
from src.models_dl import FocalLoss
from src.utils import ensure_dir

# Small, defensible search grids per model (subject-wise CV).
ML_PARAM_GRIDS = {
    "svm": {"C": [1, 10], "gamma": ["scale", 0.01]},
    "rf": {"n_estimators": [200, 400], "max_depth": [None, 20]},
    "gb": {"max_depth": [None, 5], "learning_rate": [0.05, 0.1]},
    "logreg": {"C": [0.5, 1.0, 2.0]},
}


# --- Machine learning -------------------------------------------------------

def _take_rows(x, idx):
    """Row-subset ``x`` by integer positions for numpy arrays or DataFrames.

    Feature matrices are now pandas DataFrames (positional ``.iloc``) while the
    tests and legacy callers still pass numpy arrays (positional ``[]``); this
    keeps the CV loop agnostic to which one it received.
    """
    return x.iloc[idx] if hasattr(x, "iloc") else x[idx]


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
    folds = subject_wise_folds(subjects, n_splits=n_splits)
    for train_idx, val_idx in tqdm(folds, desc="CV folds", unit="fold"):
        model = model_builder()
        model.fit(_take_rows(x, train_idx), y[train_idx])
        y_pred = model.predict(_take_rows(x, val_idx))
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
    """Wrap in-memory epoch arrays (x, y) in a torch DataLoader.

    Loads the whole array into a tensor, so it is only for data that fits in RAM
    (tests, small subsets). For the full memmapped dataset use
    ``MemmapEpochDataset`` + ``make_lazy_dataloader`` instead.

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


class MemmapEpochDataset(Dataset):
    """Read epochs lazily from a memmapped array and z-score them per item.

    Only the item indices and the tiny normalizer live in RAM; each ``__getitem__``
    reads one epoch (or one sequence of epochs) straight from the memmap and
    normalizes it on the fly. This is the standard out-of-core pattern for torch
    datasets and is what keeps peak RAM at ~one mini-batch, so the full Sleep-EDF
    signal array (tens of GB) never needs to be resident.

    context == 1 -> item i is one epoch:      x (C, T),          y scalar
    context  > 1 -> item i is one sequence:   x (seq_len, C, T), y (seq_len,)

    Parameters
    ----------
    x : memmap/array of shape (n_epochs, n_channels, n_samples).
    y : integer labels for the whole dataset (n_epochs,).
    item_index : per-item indices into ``x``. Shape (n_items,) when context == 1,
        or (n_items, seq_len) of global indices when context > 1 (from
        ``build_sequence_windows``).
    mean, std : normalizer from ``fit_normalizer_streaming`` (shape (1, C, 1)).
    """

    def __init__(self, x, y, item_index, mean, std, context=1):
        self.x = x
        self.y = np.asarray(y)
        self.item_index = np.asarray(item_index)
        self.context = context
        self._mean = np.asarray(mean, dtype=np.float32).reshape(-1)  # (C,)
        self._std = np.asarray(std, dtype=np.float32).reshape(-1)

    def __len__(self):
        return len(self.item_index)

    @property
    def labels(self):
        """Flattened per-epoch labels of every item (for class weights / sampling)."""
        return self.y[self.item_index].reshape(-1)

    def __getitem__(self, i):
        gidx = self.item_index[i]
        signals = np.asarray(self.x[gidx], dtype=np.float32)
        if self.context > 1:
            signals = (signals - self._mean[None, :, None]) / self._std[None, :, None]
            return torch.from_numpy(signals), torch.from_numpy(self.y[gidx].astype(np.int64))
        signals = (signals - self._mean[:, None]) / self._std[:, None]
        return torch.from_numpy(signals), torch.tensor(int(self.y[gidx]), dtype=torch.long)


def make_lazy_dataloader(dataset, batch_size=64, shuffle=False, balanced=False,
                         num_workers=0, pin_memory=False):
    """Wrap a ``MemmapEpochDataset`` in a DataLoader (no full-array materialization).

    balanced=True draws class-balanced mini-batches (per-epoch datasets only).

    On a GPU run set ``num_workers>0`` so several worker processes read the memmap
    in parallel and keep the GPU fed (each ``__getitem__`` is a disk read), and
    ``pin_memory=True`` so host->device copies are faster. On Windows keep
    ``num_workers=0`` (process spawning is fragile there); on Colab/Linux a couple
    of workers is a clear win.
    """
    if balanced and dataset.context == 1:
        labels = dataset.labels
        counts = np.bincount(labels)
        sample_weights = 1.0 / counts[labels]
        sampler = WeightedRandomSampler(
            torch.tensor(sample_weights, dtype=torch.double), len(sample_weights), replacement=True
        )
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                          num_workers=num_workers, pin_memory=pin_memory)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=pin_memory)


def _build_weight_tensor(class_weights, n_classes, device):
    weights = torch.ones(n_classes)
    for cls, weight in class_weights.items():
        weights[cls] = weight
    return weights.to(device)


def _save_dl_checkpoint(path, model, meta, epoch, val_macro_f1):
    """Write a self-contained DL checkpoint that ``load_dl_checkpoint`` can reload.

    ``meta`` carries the architecture config + training normalizer (model name,
    context, channels, modalities, max_len, mean, std) so the exact model can be
    rebuilt later; ``epoch`` and ``val_macro_f1`` record where in training this
    snapshot was taken. ``temperature`` is a placeholder (1.0) because temperature
    scaling is only fitted after training — the final calibrated checkpoint saved
    by run_dl.py carries the real value.
    """
    path = Path(path)
    ensure_dir(path.parent)
    payload = dict(meta or {})
    payload.update({
        "state_dict": {name: tensor.cpu() for name, tensor in model.state_dict().items()},
        "epoch": int(epoch),
        "val_macro_f1": float(val_macro_f1),
    })
    payload.setdefault("temperature", 1.0)
    torch.save(payload, path)


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
        for x_batch, y_batch in tqdm(loader, desc="Predicting", unit="batch", leave=False):
            batch_logits = model(x_batch.to(device))
            logits.append(batch_logits.reshape(-1, batch_logits.shape[-1]).cpu().numpy())
            y_true.append(y_batch.reshape(-1).numpy())
    return np.concatenate(y_true), np.concatenate(logits)


def train_dl(model, train_loader, val_loader, epochs=20, lr=1e-3,
             class_weights=None, use_focal=False, device="cpu", patience=None,
             weight_decay=0.0, focal_gamma=2.0,
             checkpoint_dir=None, checkpoint_every=3, checkpoint_meta=None):
    """Train a deep model and keep the weights with the best validation macro-F1.

    Model selection is on validation macro-F1, so the returned model is the best
    epoch's, never the last. With ``patience`` set, training also stops early once
    validation macro-F1 has not improved for ``patience`` consecutive epochs (the
    best weights are still restored), which avoids wasting epochs after the
    validation curve has plateaued — worth it here because each epoch is a full
    pass over the memmapped dataset. ``patience=None`` disables early stopping and
    runs all ``epochs``.

    ``device`` selects CPU vs GPU ("cuda"); pass "cuda" on a GPU box (e.g. Colab)
    for the big speed-up. ``weight_decay`` (Adam L2) and ``focal_gamma`` are the
    training hyperparameters the search routines tune.

    Checkpointing: when ``checkpoint_dir`` is given, a reloadable snapshot is
    written every ``checkpoint_every`` epochs as ``epoch_NNN.pt`` (set
    ``checkpoint_every=0`` to skip the periodic ones), and ``last.pt`` / ``best.pt``
    are kept up to date every epoch so a disconnected run can resume and the best
    weights survive a crash. ``checkpoint_meta`` (architecture config + normalizer)
    is embedded so each file can be rebuilt by ``load_dl_checkpoint``.

    Returns (model, history); history has per-epoch train loss and val macro-F1.
    """
    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
    model = model.to(device)
    weight_tensor = None
    if class_weights is not None:
        weight_tensor = _build_weight_tensor(class_weights, model.n_classes, device)

    criterion = (FocalLoss(gamma=focal_gamma, weight=weight_tensor)
                 if use_focal else torch.nn.CrossEntropyLoss(weight=weight_tensor))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = []
    best_macro_f1 = -1.0
    best_state = None
    epochs_without_improvement = 0
    epoch_bar = tqdm(range(epochs), desc="Training", unit="epoch")
    for epoch in epoch_bar:
        model.train()
        epoch_loss = 0.0
        batch_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch", leave=False)
        for x_batch, y_batch in batch_bar:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(x_batch)
            # reshape unifies per-epoch (batch, C) and sequence (batch, seq_len, C) outputs.
            loss = criterion(logits.reshape(-1, logits.shape[-1]), y_batch.reshape(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batch_bar.set_postfix(loss=f"{loss.item():.3f}")

        val_metrics = evaluate_dl(model, val_loader, device)
        history.append({
            "epoch": epoch,
            "train_loss": epoch_loss / len(train_loader),
            "val_macro_f1": val_metrics["macro_f1"],
        })
        improved = val_metrics["macro_f1"] > best_macro_f1
        if improved:
            best_macro_f1 = val_metrics["macro_f1"]
            best_state = {name: tensor.cpu().clone() for name, tensor in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if checkpoint_dir is not None:
            val_f1 = val_metrics["macro_f1"]
            if checkpoint_every and (epoch + 1) % checkpoint_every == 0:
                _save_dl_checkpoint(checkpoint_dir / f"epoch_{epoch + 1:03d}.pt",
                                    model, checkpoint_meta, epoch + 1, val_f1)
            _save_dl_checkpoint(checkpoint_dir / "last.pt", model, checkpoint_meta, epoch + 1, val_f1)
            if improved:  # model currently holds the best weights
                _save_dl_checkpoint(checkpoint_dir / "best.pt", model, checkpoint_meta, epoch + 1, val_f1)

        epoch_bar.set_postfix(train_loss=f"{epoch_loss / len(train_loader):.3f}",
                              val_f1=f"{val_metrics['macro_f1']:.3f}",
                              best_f1=f"{best_macro_f1:.3f}",
                              stale=epochs_without_improvement)
        if patience is not None and epochs_without_improvement >= patience:
            epoch_bar.write(f"Early stopping at epoch {epoch + 1}/{epochs}: val macro-F1 did not "
                            f"improve for {patience} epoch(s) (best {best_macro_f1:.3f}).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def random_search_dl(model_factory, train_loader, val_loader, n_trials=6, epochs=10,
                     class_weights=None, seed=42, patience=None, device="cpu",
                     checkpoint_dir=None, checkpoint_every=3, checkpoint_meta=None):
    """Random search over learning rate and loss type (validation macro-F1).

    model_factory is a zero-argument callable returning a fresh model. ``patience``
    is forwarded to ``train_dl`` so each trial can stop early once its validation
    macro-F1 plateaus. Kept as a dependency-free fallback for ``bayesian_search_dl``.
    When ``checkpoint_dir`` is given each trial checkpoints into its own
    ``trial_NN/`` subfolder. Returns (best, trials): best has the trained model plus
    its hyperparameters.
    """
    rng = np.random.default_rng(seed)
    learning_rates = [1e-2, 3e-3, 1e-3, 3e-4]
    best, trials = None, []
    for trial_number in tqdm(range(n_trials), desc="Random search", unit="trial"):
        lr = float(rng.choice(learning_rates))
        use_focal = bool(rng.integers(0, 2))
        trial_ckpt = (Path(checkpoint_dir) / f"trial_{trial_number:02d}"
                      if checkpoint_dir is not None else None)
        model, history = train_dl(model_factory(), train_loader, val_loader, epochs=epochs, lr=lr,
                                  class_weights=class_weights, use_focal=use_focal,
                                  patience=patience, device=device,
                                  checkpoint_dir=trial_ckpt, checkpoint_every=checkpoint_every,
                                  checkpoint_meta=checkpoint_meta)
        score = max(step["val_macro_f1"] for step in history)
        trials.append({"lr": lr, "use_focal": use_focal, "val_macro_f1": score})
        if best is None or score > best["val_macro_f1"]:
            best = {"model": model, "history": history, "lr": lr,
                    "use_focal": use_focal, "val_macro_f1": score}
    return best, trials


def bayesian_search_dl(model_factory, train_loader, val_loader, n_trials=15, epochs=10,
                       class_weights=None, seed=42, patience=None, device="cpu",
                       checkpoint_dir=None, checkpoint_every=3, checkpoint_meta=None):
    """Bayesian optimization (Optuna TPE) over DL training hyperparameters.

    The master plan asks for Bayesian optimization rather than grid/random search
    for the deep models: the search space is larger and mixes continuous, categorical
    and conditional dimensions, where a Tree-structured Parzen Estimator is more
    sample-efficient than blind sampling. Each trial trains a fresh model from
    ``model_factory`` (a zero-argument callable, so the architecture is fixed) and
    varies only the training hyperparameters:

    - ``lr``            : learning rate, log-uniform in [1e-4, 1e-2]
    - ``weight_decay``  : Adam L2 penalty, log-uniform in [1e-6, 1e-3]
    - ``use_focal``     : weighted cross-entropy vs focal loss (categorical)
    - ``focal_gamma``   : focal focusing parameter in [1.0, 3.0], only when focal
                          (a conditional dimension — a natural fit for TPE)

    Selection is on validation macro-F1, mirroring ``train_dl``. If Optuna is not
    installed it falls back to ``random_search_dl`` (with a warning), so the
    pipeline still runs. Returns (best, trials) with the same shape as
    ``random_search_dl`` so callers are unchanged.
    """
    try:
        import optuna
    except ImportError:
        warnings.warn("Optuna not installed; falling back to random search. "
                      "Install it with `pip install optuna` for Bayesian optimization.",
                      RuntimeWarning)
        return random_search_dl(model_factory, train_loader, val_loader, n_trials=n_trials,
                                epochs=epochs, class_weights=class_weights, seed=seed,
                                patience=patience, device=device,
                                checkpoint_dir=checkpoint_dir, checkpoint_every=checkpoint_every,
                                checkpoint_meta=checkpoint_meta)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    best = {"val_macro_f1": -1.0}
    trials = []

    def objective(trial):
        nonlocal best
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        use_focal = trial.suggest_categorical("use_focal", [False, True])
        focal_gamma = trial.suggest_float("focal_gamma", 1.0, 3.0) if use_focal else 2.0
        trial_ckpt = (Path(checkpoint_dir) / f"trial_{trial.number:02d}"
                      if checkpoint_dir is not None else None)
        model, history = train_dl(model_factory(), train_loader, val_loader, epochs=epochs, lr=lr,
                                  weight_decay=weight_decay, class_weights=class_weights,
                                  use_focal=use_focal, focal_gamma=focal_gamma,
                                  patience=patience, device=device,
                                  checkpoint_dir=trial_ckpt, checkpoint_every=checkpoint_every,
                                  checkpoint_meta=checkpoint_meta)
        score = max(step["val_macro_f1"] for step in history)
        record = {"lr": lr, "weight_decay": weight_decay, "use_focal": use_focal,
                  "focal_gamma": focal_gamma, "val_macro_f1": score}
        trials.append(record)
        if score > best["val_macro_f1"]:
            best = {"model": model, "history": history, **record}
        return score

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
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
