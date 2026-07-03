"""Tests for leakage-safe feature selection."""
import numpy as np

from src.feature_selection import (
    CorrelationPruner,
    build_feature_selection_steps,
    correlation_pruned_indices,
)
from src.models_ml import build_estimator


def _redundant_matrix(seed=0):
    """Feature matrix where column 1 duplicates column 0 and column 3 is constant."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(200, 3))
    x = np.column_stack([base[:, 0], base[:, 0], base[:, 1], np.ones(200), base[:, 2]])
    return x  # columns: 0, copy-of-0, independent, constant, independent


def test_correlation_pruner_drops_duplicate_column():
    x = _redundant_matrix()
    pruner = CorrelationPruner(threshold=0.95).fit(x)
    support = pruner.get_support()
    # the duplicate of column 0 (index 1) must be dropped; column 0 stays
    assert support[0]
    assert not support[1]
    assert pruner.transform(x).shape[1] < x.shape[1]


def test_correlation_pruner_is_fit_only_on_given_data():
    # Fitting on train and transforming test must keep the SAME columns
    # (no refit), which is what makes selection leakage-safe in a pipeline.
    x_train = _redundant_matrix(seed=1)
    x_test = _redundant_matrix(seed=2)
    pruner = CorrelationPruner(threshold=0.95).fit(x_train)
    assert pruner.transform(x_test).shape[1] == int(pruner.get_support().sum())


def test_correlation_pruned_indices_drops_constant_and_duplicate():
    x = _redundant_matrix()
    kept, dropped = correlation_pruned_indices(x, threshold=0.95)
    assert 3 in dropped.tolist()        # constant column removed by variance filter
    assert 1 in dropped.tolist()        # duplicate of column 0 removed by correlation
    assert 0 in kept.tolist()
    assert len(kept) + len(dropped) == x.shape[1]


def test_build_steps_respects_toggles():
    steps = dict(build_feature_selection_steps(correlation_threshold=None, k_best=None))
    assert "variance_filter" in steps
    assert "correlation_pruner" not in steps


def test_build_estimator_with_selection_fits_and_predicts():
    x = _redundant_matrix()
    y = (x[:, 0] > 0).astype(int)
    model = build_estimator("rf", balance="none", feature_selection=True)
    model.fit(x, y)
    preds = model.predict(x)
    assert preds.shape == y.shape
    # the pipeline exposes the pruning step, and it removed at least one feature
    support = model.named_steps["correlation_pruner"].get_support()
    assert support.sum() < x.shape[1]
