"""SHAP value computation utilities."""

import numpy as np
import shap


def calculate_shap_values(model, X, model_type: str):
    """Compute SHAP values using an appropriate explainer for the model type.

    Parameters
    ----------
    model : fitted estimator
    X : ndarray
        Input features to explain.
    model_type : str
        One of ``"cnn"``, ``"rf"``, ``"rfr"``, ``"xgb"``, ``"xgbr"``, or
        a generic name (falls back to ``KernelExplainer``).

    Returns
    -------
    shap_values : ndarray
    """
    if model_type == "cnn":
        import torch

        X_t = torch.from_numpy(X).to("cpu")
        explainer = shap.DeepExplainer(model.module_.to("cpu"), X_t)
        return explainer.shap_values(X_t)

    if model_type in ("rf", "rfr", "xgb", "xgbr"):
        explainer = shap.TreeExplainer(model)
        return explainer.shap_values(X, check_additivity=False)

    explainer = shap.KernelExplainer(model.predict, X)
    return explainer.shap_values(X)
