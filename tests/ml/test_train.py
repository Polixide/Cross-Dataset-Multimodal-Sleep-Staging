"""Tests for ML imbalance handling."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.ml import train as ml_train
from src.ml.models import (
    AutoSampleWeightClassifier,
    adapt_params_to_estimator,
    build_estimator,
    half_majority_strategy,
)
from src.ml.train import calibrate_classifier, compute_class_weights, tune_ml


class RecordingClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, x, y, sample_weight=None):
        self.classes_ = np.unique(y)
        self.sample_weight_ = np.asarray(sample_weight)
        return self

    def predict(self, x):
        return np.repeat(self.classes_[0], len(x))


def test_class_weights_are_higher_for_rare_classes():
    y = np.array([0] * 90 + [1] * 10)  # class 1 is rare
    weights = compute_class_weights(y)
    assert weights[1] > weights[0]


def test_auto_sample_weights_are_computed_from_training_labels():
    y = np.array([0] * 9 + [1])
    model = AutoSampleWeightClassifier(RecordingClassifier()).fit(
        np.zeros((len(y), 2)), y
    )
    assert model.estimator.sample_weight_[-1] > model.estimator.sample_weight_[0]


def test_xgboost_weight_wrapper_can_be_calibrated_via_predict_proba():
    x, y = make_classification(
        n_samples=120, n_features=8, n_informative=5, n_classes=3,
        n_clusters_per_class=1, random_state=0,
    )
    fitted = AutoSampleWeightClassifier(
        XGBClassifier(n_estimators=5, max_depth=2, n_jobs=1, random_state=0)
    ).fit(x[:90], y[:90])
    calibrated = calibrate_classifier(fitted, x[90:], y[90:], method="sigmoid")
    probabilities = calibrated.predict_proba(x[90:])
    assert probabilities.shape == (30, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_tuning_enforces_train_validation_overfitting_constraint(monkeypatch):
    x, y = make_classification(
        n_samples=400, n_features=12, n_informative=4, n_redundant=2,
        flip_y=0.25, class_sep=0.7, random_state=1,
    )
    groups = np.repeat(np.arange(20), 20)
    monkeypatch.setitem(
        ml_train.ML_PARAM_GRIDS, "rf", {"max_depth": [1, None]}
    )
    _, params, selection = tune_ml(
        DecisionTreeClassifier(random_state=0), "rf", x, y, groups,
        n_splits=5, max_overfitting_gap=0.10,
    )
    assert params["max_depth"] == 1
    assert selection["constraint_satisfied"] is True
    assert selection["overfitting_gap"] <= 0.10


def test_saved_class_weight_params_are_adapted_to_smote_pipeline():
    smote_model = build_estimator("xgb", balance="smote")
    adapted = adapt_params_to_estimator(
        smote_model,
        {"estimator__max_depth": 4, "estimator__n_estimators": 500},
    )
    assert adapted == {"clf__max_depth": 4, "clf__n_estimators": 500}


def test_half_smote_caps_minorities_at_half_the_majority():
    y = np.array([0] * 100 + [1] * 10 + [2] * 60)
    assert half_majority_strategy(y) == {1: 50}
