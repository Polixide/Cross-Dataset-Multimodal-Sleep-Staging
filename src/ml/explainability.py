"""SHAP explainability for feature-based ML models."""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import shap

from src.common.utils import ensure_dir


def shap_feature_importance(model, x_background, x_explain, feature_names=None,
                            out_dir="results/figures", top_k=20, max_evals=None):
    out_dir = ensure_dir(out_dir)
    explainer = shap.Explainer(
        model.predict_proba, x_background, algorithm="permutation",
        feature_names=feature_names,
    )
    if max_evals is None:
        max_evals = 2 * x_background.shape[1] + 1
    shap_values = np.abs(explainer(x_explain, max_evals=max_evals).values)
    if shap_values.ndim == 3:
        shap_values = shap_values.mean(axis=2)
    importance = shap_values.mean(axis=0)
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(len(importance))]
    top_features = np.argsort(importance)[-top_k:]
    plt.figure(figsize=(8, max(4, 0.3 * len(top_features))))
    plt.barh([feature_names[i] for i in top_features], importance[top_features])
    plt.xlabel("Mean |SHAP value|")
    plt.title("SHAP feature importance")
    plt.tight_layout()
    out_path = out_dir / "shap_feature_importance.png"
    out_path.unlink(missing_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path
