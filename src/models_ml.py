"""Feature-based ML model factory (master plan: Model families -> ML).

SVM (RBF), Random Forest, gradient boosting and logistic regression. Scale-
sensitive models (SVM, logistic regression) are wrapped in a StandardScaler
step. Class imbalance can be handled with class weights, SMOTE, or random
oversampling — resampling is applied inside training folds only, never on
validation/test. Optional leakage-safe feature selection can be prepended to the
pipeline; because it is part of the estimator it is re-fit on every CV fold.
"""
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.feature_selection import build_feature_selection_steps

# Models that need standardized features; tree models are scale-invariant.
SCALE_SENSITIVE = {"svm", "logreg"}
# Imbalance strategies handled by resampling the training fold.
RESAMPLE_STRATEGIES = {"smote", "oversample"}


def _base_estimator(name, class_weight, seed):
    """Return the raw estimator (no scaling / resampling) by name."""
    name = name.lower()
    if name == "svm":
        return SVC(kernel="rbf", probability=True, class_weight=class_weight, random_state=seed)
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=300, class_weight=class_weight, n_jobs=-1, random_state=seed
        )
    if name == "gb":
        return HistGradientBoostingClassifier(random_state=seed)
    if name == "logreg":
        return LogisticRegression(max_iter=1000, class_weight=class_weight, random_state=seed)
    raise ValueError(f"Unknown ML model: {name}")


def _resolve_feature_selection(feature_selection, seed):
    """Turn the feature_selection argument into a list of pipeline steps.

    Accepts False/None (off), True (sensible defaults) or a dict of overrides
    forwarded to build_feature_selection_steps.
    """
    if not feature_selection:
        return []
    config = {} if feature_selection is True else dict(feature_selection)
    return build_feature_selection_steps(seed=seed, **config)


def build_estimator(name, balance="class_weight", feature_selection=None, seed=42):
    """Build the ML estimator, assembling one flat pipeline.

    balance : 'class_weight' | 'none' | 'smote' | 'oversample'.
    feature_selection : False/None, True (defaults), or a dict of overrides for
        build_feature_selection_steps (variance_threshold, correlation_threshold,
        k_best). Selection steps run first, so they operate on raw features
        before scaling/resampling and are re-fit per CV fold (no leakage).

    Returns a bare estimator only for the simplest case (tree model, no scaling,
    no resampling, no selection); otherwise a Pipeline whose final step is 'clf'.
    """
    if balance not in RESAMPLE_STRATEGIES and balance not in ("class_weight", "none"):
        raise ValueError(f"Unknown balance strategy: {balance}")

    steps = _resolve_feature_selection(feature_selection, seed)
    if name.lower() in SCALE_SENSITIVE:
        steps.append(("scaler", StandardScaler()))

    if balance in RESAMPLE_STRATEGIES:
        sampler = SMOTE(random_state=seed) if balance == "smote" else RandomOverSampler(random_state=seed)
        steps.append(("resample", sampler))
        clf = _base_estimator(name, class_weight=None, seed=seed)
    else:
        class_weight = "balanced" if balance == "class_weight" else None
        clf = _base_estimator(name, class_weight, seed)

    if not steps:
        return clf  # bare tree estimator, nothing to wrap
    steps.append(("clf", clf))
    pipeline_cls = ImbPipeline if balance in RESAMPLE_STRATEGIES else Pipeline
    return pipeline_cls(steps)
