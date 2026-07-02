"""Feature-based ML model factory (master plan: Model families -> ML).

SVM (RBF), Random Forest, gradient boosting and logistic regression. Scale-
sensitive models (SVM, logistic regression) are wrapped in a StandardScaler
pipeline. Class imbalance can be handled with class weights, SMOTE, or random
oversampling — resampling is applied inside training folds only, never on
validation/test.
"""
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Models that need standardized features; tree models are scale-invariant.
SCALE_SENSITIVE = {"svm", "logreg"}


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


def build_ml_model(name, class_weight="balanced", seed=42):
    """Return an unfitted estimator, adding a StandardScaler for scale-sensitive models.

    class_weight : 'balanced' or None (ignored by gradient boosting).
    """
    estimator = _base_estimator(name, class_weight, seed)
    if name.lower() in SCALE_SENSITIVE:
        return Pipeline([("scaler", StandardScaler()), ("clf", estimator)])
    return estimator


def build_balanced_pipeline(name, method="smote", seed=42):
    """Return a pipeline that oversamples the training fold before fitting.

    method : 'smote' or 'oversample' (random oversampling). Uses an imblearn
    pipeline so resampling never touches validation/test data.
    """
    sampler = SMOTE(random_state=seed) if method == "smote" else RandomOverSampler(random_state=seed)
    steps = []
    if name.lower() in SCALE_SENSITIVE:
        steps.append(("scaler", StandardScaler()))
    steps.append(("resample", sampler))
    steps.append(("clf", _base_estimator(name, class_weight=None, seed=seed)))
    return ImbPipeline(steps)


def build_estimator(name, balance="class_weight", seed=42):
    """Single entry point selecting the imbalance strategy.

    balance : 'class_weight' | 'none' | 'smote' | 'oversample'.
    """
    if balance == "class_weight":
        return build_ml_model(name, class_weight="balanced", seed=seed)
    if balance == "none":
        return build_ml_model(name, class_weight=None, seed=seed)
    if balance in ("smote", "oversample"):
        return build_balanced_pipeline(name, method=balance, seed=seed)
    raise ValueError(f"Unknown balance strategy: {balance}")
