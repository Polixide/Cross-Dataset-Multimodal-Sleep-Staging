"""Leakage-safe feature selection for the feature-based ML pipeline.

The 96 engineered features (24 per channel x 4 channels) contain strong
redundancy: absolute vs relative band powers, RMS / std / Hjorth-activity, and
the various band ratios are largely collinear. Pruning them helps the scale-
sensitive models (SVM, logistic regression) and sharpens interpretability
(SHAP importance spreads across correlated features), even though the tree
models tolerate redundancy on their own.

Everything here is implemented as scikit-learn transformers so it lives INSIDE
the model pipeline and is therefore re-fit on the training portion of every
cross-validation fold. That is what keeps selection leakage-safe: the choice of
features never sees validation/test data (master plan: Validation rules).
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, VarianceThreshold


class CorrelationPruner(BaseEstimator, TransformerMixin):
    """Drop features that are near-duplicates of an already-kept feature.

    Fit computes the absolute Pearson correlation matrix on the training data and
    greedily keeps the first feature of every group, dropping any later feature
    whose correlation with a kept one exceeds ``threshold``. Constant features
    (undefined correlation) are left for the VarianceThreshold to remove.
    """

    def __init__(self, threshold=0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        n_features = X.shape[1]
        # nan_to_num handles constant columns (std == 0 -> undefined correlation);
        # errstate silences the expected divide-by-zero from those columns.
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.abs(np.nan_to_num(np.corrcoef(X, rowvar=False)))
        corr = np.atleast_2d(corr)

        keep = np.ones(n_features, dtype=bool)
        for i in range(n_features):
            if not keep[i]:
                continue
            for j in range(i + 1, n_features):
                if keep[j] and corr[i, j] > self.threshold:
                    keep[j] = False
        self.support_ = keep
        self.n_features_in_ = n_features
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.support_]

    def get_support(self, indices=False):
        """Return the boolean mask (or integer indices) of the kept features."""
        return np.where(self.support_)[0] if indices else self.support_


def build_feature_selection_steps(variance_threshold=1e-8, correlation_threshold=0.95,
                                  k_best=None, seed=42):
    """Return the ordered (name, transformer) steps for the selection block.

    variance_threshold : drop near-constant features (None to skip).
    correlation_threshold : prune collinear features above this |r| (None to skip).
    k_best : if set, keep only the top-k by Random-Forest importance (None to skip).
    """
    steps = []
    if variance_threshold is not None:
        steps.append(("variance_filter", VarianceThreshold(variance_threshold)))
    if correlation_threshold is not None:
        steps.append(("correlation_pruner", CorrelationPruner(correlation_threshold)))
    if k_best is not None:
        selector_model = RandomForestClassifier(
            n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=seed
        )
        steps.append(("model_select", SelectFromModel(selector_model, max_features=k_best, threshold=-np.inf)))
    return steps


def correlation_pruned_indices(x, threshold=0.95, variance_threshold=1e-8):
    """Return (kept_indices, dropped_indices) for a feature matrix.

    Convenience wrapper mirroring the pipeline's variance + correlation steps,
    used by the feature-analysis script to report which features a selection
    step would keep. Fit this on TRAINING features only to stay leakage-safe.
    """
    x = np.asarray(x, dtype=float)
    variances = x.var(axis=0)
    non_constant = np.where(variances > variance_threshold)[0]
    pruner = CorrelationPruner(threshold).fit(x[:, non_constant])
    kept = non_constant[pruner.get_support(indices=True)]
    dropped = np.array(sorted(set(range(x.shape[1])) - set(kept.tolist())), dtype=int)
    return kept, dropped
