"""Feature-based ML model factory (master plan: Model families -> ML).

Multinomial logistic regression, Random Forest and XGBoost. Class imbalance
can be handled with class weights, SMOTE, or random
oversampling — resampling is applied inside training folds only, never on
validation/test.

"""
import numpy as np
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from tqdm import tqdm

# Models that need standardized features; tree models are scale-invariant.
SCALE_SENSITIVE = {"logreg"}


class TqdmSMOTE(SMOTE):
    """SMOTE variant that makes the otherwise silent resampling stage visible."""

    def __init__(self, *, sampling_strategy="auto", random_state=None, k_neighbors=5):
        super().__init__(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=k_neighbors,
        )

    def fit_resample(self, x, y, **params):
        with tqdm(total=1, desc="SMOTE resampling", unit="stage") as progress:
            result = super().fit_resample(x, y, **params)
            progress.update(1)
        return result


def half_majority_strategy(y):
    """Oversample minority classes only up to 50% of the majority count."""
    classes, counts = np.unique(y, return_counts=True)
    target = int(counts.max() * 0.5)
    return {
        int(cls): target for cls, count in zip(classes, counts)
        if count < target
    }


class AutoSampleWeightClassifier(ClassifierMixin, BaseEstimator):
    """Fit any supported classifier with balanced per-sample weights.

    Weights are recomputed inside every training/CV fold from that fold's labels,
    preventing information from validation or test subjects entering training.
    """

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, x, y, **fit_params):
        weights = compute_sample_weight(class_weight="balanced", y=y)
        if hasattr(self.estimator, "named_steps") and "clf" in self.estimator.named_steps:
            fit_params["clf__sample_weight"] = weights
        else:
            fit_params["sample_weight"] = weights
        self.estimator.fit(x, y, **fit_params)
        self.classes_ = self.estimator.classes_
        return self

    def predict(self, x):
        return self.estimator.predict(x)

    def predict_proba(self, x):
        return self.estimator.predict_proba(x)


def _base_estimator(name, class_weight, seed):
    """Return the raw estimator (no scaling / resampling) by name."""
    name = name.lower()
    if name == "logreg":
        return LogisticRegression(
            solver="lbfgs", C=1.0, max_iter=2000,
            class_weight=class_weight, random_state=seed,
        )
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=18,
            min_samples_leaf=2,
            class_weight=class_weight,
            n_jobs=-1,
            random_state=seed,
        )
    if name == "xgb":
        # Import only when XGBoost is requested: RF/LogReg remain usable even
        # in lightweight environments where the optional package is absent.
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "XGBoost is not installed in the active Python environment. "
                "Run the workflow with the project venv Python."
            ) from exc
        return XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            min_child_weight=3,
            gamma=0.1,
            reg_lambda=5.0,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=seed,
        )
    raise ValueError(f"Unknown ML model: {name}")


def build_ml_model(name, class_weight="balanced", seed=42):
    """Return an unfitted estimator, adding a StandardScaler for scale-sensitive models.

    class_weight : 'balanced' or None (ignored by XGBoost; use resampling for it).
    """
    estimator = _base_estimator(name, class_weight, seed)
    if name.lower() in SCALE_SENSITIVE:
        return Pipeline([("scaler", StandardScaler()), ("clf", estimator)])
    return estimator


def build_balanced_pipeline(name, method="smote", seed=42):
    """Return a pipeline that oversamples the training fold before fitting.

    method : 'smote', 'smote_half', or 'oversample'. ``smote_half`` caps each
    minority at 50% of the majority. Resampling never touches validation/test.
    """
    if method == "smote":
        sampler = TqdmSMOTE(random_state=seed)
    elif method == "smote_half":
        sampler = TqdmSMOTE(
            sampling_strategy=half_majority_strategy, random_state=seed
        )
    else:
        sampler = RandomOverSampler(random_state=seed)
    steps = []
    if name.lower() in SCALE_SENSITIVE:
        steps.append(("scaler", StandardScaler()))
    steps.append(("resample", sampler))
    steps.append(("clf", _base_estimator(name, class_weight=None, seed=seed)))
    return ImbPipeline(steps)


def build_estimator(name, balance="sample_weight", seed=42):
    """Single entry point selecting the imbalance strategy.

    balance : 'sample_weight' | 'class_weight' | 'none' | 'smote' |
              'smote_half' | 'oversample'.
    """
    if balance == "sample_weight":
        return AutoSampleWeightClassifier(
            build_ml_model(name, class_weight=None, seed=seed)
        )
    if balance == "class_weight":
        # XGBoost has no native multiclass class_weight argument. Applying the
        # fold-local class weights to its per-row loss is the exact equivalent.
        if name.lower() == "xgb":
            return AutoSampleWeightClassifier(
                build_ml_model(name, class_weight=None, seed=seed)
            )
        return build_ml_model(name, class_weight="balanced", seed=seed)
    if balance == "none":
        return build_ml_model(name, class_weight=None, seed=seed)
    if balance in ("smote", "smote_half", "oversample"):
        return build_balanced_pipeline(name, method=balance, seed=seed)
    raise ValueError(f"Unknown balance strategy: {balance}")


def adapt_params_to_estimator(estimator, params):
    """Map saved hyperparameters onto a differently wrapped estimator.

    For example, XGBoost parameters saved as ``estimator__max_depth`` by the
    class-weight wrapper become ``clf__max_depth`` in the SMOTE pipeline.
    """
    available = estimator.get_params(deep=True)
    adapted = {}
    for key, value in params.items():
        if key in available:
            adapted[key] = value
            continue
        leaf = key.rsplit("__", 1)[-1]
        matches = [candidate for candidate in available
                   if candidate.rsplit("__", 1)[-1] == leaf]
        if len(matches) != 1:
            raise ValueError(
                f"Cannot map saved parameter {key!r} to {type(estimator).__name__}."
            )
        adapted[matches[0]] = value
    return adapted
