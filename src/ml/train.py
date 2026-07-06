"""Training, subject-wise tuning and post-hoc calibration for classical ML."""
import numpy as np
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


def tune_ml(estimator, name, x, y, subjects, n_splits=5, scoring="f1_macro"):
    """Grid-search model hyperparameters with subject-wise GroupKFold."""
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
    )
    search.fit(x, y, groups=subjects)
    return search.best_estimator_, search.best_params_, float(search.best_score_)


def calibrate_classifier(fitted_estimator, x_val, y_val, method="isotonic"):
    """Calibrate a frozen ML estimator on validation subjects only."""
    calibrated = CalibratedClassifierCV(
        FrozenEstimator(fitted_estimator), method=method
    )
    calibrated.fit(x_val, y_val)
    return calibrated
