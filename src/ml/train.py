"""Training, subject-wise tuning and post-hoc calibration for classical ML."""
import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import GridSearchCV, GroupKFold
from tqdm import tqdm

from src.common.data import subject_wise_folds
from src.common.evaluation import compute_metrics
from src.ml.models import AutoSampleWeightClassifier

ML_PARAM_GRIDS = {
    "logreg": {"C": [0.1, 0.5, 1.0, 2.0]},
    # Depth and leaf-size constraints prevent nearly pure, memorized leaves.
    "rf": {
        "n_estimators": [300, 500],
        "max_depth": [12, 18],
        "min_samples_leaf": [2, 5],
    },
    # Shallow trees plus child/L2/gamma penalties reduce boosting variance.
    "xgb": {
        "n_estimators": [300, 500],
        "max_depth": [3, 4],
        "learning_rate": [0.05],
        "min_child_weight": [3, 7],
        "gamma": [0.1],
        "reg_lambda": [5.0],
    },
}


def compute_class_weights(y):
    """Inverse-frequency class weights (mean-normalized)."""
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    weights = counts.sum() / (len(classes) * counts)
    return {int(cls): float(weight) for cls, weight in zip(classes, weights)}


def cross_validate_ml(model_builder, x, y, subjects, n_splits=5, labels=None):
    """Run subject-wise k-fold CV and retain train/validation macro-F1."""
    fold_metrics = []
    folds = subject_wise_folds(subjects, n_splits=n_splits)
    for train_idx, val_idx in tqdm(folds, desc="CV folds", unit="fold"):
        model = model_builder()
        model.fit(x[train_idx], y[train_idx])
        train_pred = model.predict(x[train_idx])
        val_pred = model.predict(x[val_idx])
        metrics = compute_metrics(y[val_idx], val_pred, labels=labels)
        metrics["train_macro_f1"] = compute_metrics(
            y[train_idx], train_pred, labels=labels
        )["macro_f1"]
        fold_metrics.append(metrics)
    return fold_metrics


def tune_ml(estimator, name, x, y, subjects, n_splits=5, scoring="f1_macro",
            max_overfitting_gap=0.10):
    """Tune with GroupKFold, rejecting configurations that overfit.

    Among configurations whose mean train-minus-validation score is at most
    ``max_overfitting_gap``, select the one with the highest validation score.
    If none satisfies the constraint, select the smallest-gap configuration and
    report that the constraint was not satisfied.
    """
    grid = ML_PARAM_GRIDS[name.lower()]
    prefix = ""
    base_estimator = estimator
    if isinstance(estimator, AutoSampleWeightClassifier):
        prefix = "estimator__"
        base_estimator = estimator.estimator
    if hasattr(base_estimator, "named_steps") and "clf" in base_estimator.named_steps:
        prefix += "clf__"
    if prefix:
        grid = {f"{prefix}{key}": value for key, value in grid.items()}
    n_splits = min(n_splits, len(np.unique(subjects)))
    search = GridSearchCV(
        estimator, grid, scoring=scoring,
        cv=GroupKFold(n_splits=n_splits), n_jobs=-1,
        return_train_score=True, refit=False,
    )
    search.fit(x, y, groups=subjects)
    results = search.cv_results_
    val_scores = np.asarray(results["mean_test_score"], dtype=float)
    train_scores = np.asarray(results["mean_train_score"], dtype=float)
    gaps = train_scores - val_scores
    eligible = np.flatnonzero(gaps <= max_overfitting_gap)
    constraint_satisfied = bool(len(eligible))
    if constraint_satisfied:
        selected = int(eligible[np.argmax(val_scores[eligible])])
    else:
        # Deterministic fallback: smallest gap, then highest validation score.
        min_gap = np.nanmin(gaps)
        candidates = np.flatnonzero(np.isclose(gaps, min_gap))
        selected = int(candidates[np.argmax(val_scores[candidates])])

    best_params = results["params"][selected]
    best_estimator = clone(estimator).set_params(**best_params)
    best_estimator.fit(x, y)
    selection = {
        "cv_macro_f1": float(val_scores[selected]),
        "cv_train_macro_f1": float(train_scores[selected]),
        "overfitting_gap": float(gaps[selected]),
        "max_overfitting_gap": float(max_overfitting_gap),
        "constraint_satisfied": constraint_satisfied,
    }
    return best_estimator, best_params, selection


def calibrate_classifier(fitted_estimator, x_val, y_val, method="isotonic"):
    """Calibrate a frozen ML estimator on validation subjects only."""
    calibrated = CalibratedClassifierCV(
        FrozenEstimator(fitted_estimator), method=method
    )
    calibrated.fit(x_val, y_val)
    return calibrated
