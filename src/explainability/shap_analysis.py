"""SHAP analysis for model interpretability."""

import numpy as np
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

try:
    import shap

    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    logger.warning("SHAP not installed. Install with: pip install shap")


def compute_shap_values(
    model,
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
    background_size: int = 100,
) -> Dict[str, Any]:
    """Compute SHAP values for a model.

    Args:
        model: Trained model (LGBM, sklearn, or PyTorch)
        X: Input features array
        feature_names: Optional list of feature names
        background_size: Size of background dataset for non-tree models

    Returns:
        Dictionary with shap_values, expected_value, feature_names
    """
    if not HAS_SHAP:
        return {"error": "SHAP not installed"}

    try:
        # Try TreeExplainer first (for LGBM, XGBoost, etc.)
        if hasattr(model, "booster_") or hasattr(model, "model"):
            # LGBM wrapper
            lgbm_model = getattr(model, "model", model)
            explainer = shap.TreeExplainer(lgbm_model)
            shap_values = explainer.shap_values(X)
            expected_value = explainer.expected_value
        else:
            # Use KernelExplainer for other models
            background = X[: min(background_size, len(X))]

            # Create prediction function
            if hasattr(model, "predict"):
                predict_fn = model.predict
            else:
                # Assume it's a callable
                def predict_fn(x):
                    return model(x).reshape(-1, 1)

            explainer = shap.KernelExplainer(predict_fn, background)
            shap_values = explainer.shap_values(X[: min(100, len(X))])
            expected_value = explainer.expected_value

        # Ensure feature_names matches the number of features
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(X.shape[1])]
        elif len(feature_names) != X.shape[1]:
            logger.warning(
                f"Feature names count ({len(feature_names)}) doesn't match feature dimension ({X.shape[1]}). Using generic names."
            )
            feature_names = [f"f{i}" for i in range(X.shape[1])]

        # Handle multi-output shap_values (list of arrays)
        if isinstance(shap_values, list):
            # For multi-output, use first output or average
            shap_values = shap_values[0] if len(shap_values) > 0 else shap_values

        return {
            "shap_values": shap_values,
            "expected_value": expected_value,
            "feature_names": feature_names,
        }

    except Exception as e:
        logger.error(f"Failed to compute SHAP values: {e}")
        return {"error": str(e)}


def plot_shap_summary(
    shap_result: Dict[str, Any],
    X: np.ndarray,
    max_display: int = 10,
    save_path: Optional[str] = None,
):
    """Plot SHAP summary.

    Args:
        shap_result: Result from compute_shap_values()
        X: Feature data
        max_display: Maximum features to display
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    if not HAS_SHAP:
        logger.warning("SHAP not installed")
        return None

    if "error" in shap_result:
        logger.warning(f"SHAP error: {shap_result['error']}")
        return None

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    shap_values = shap_result["shap_values"]
    feature_names = shap_result["feature_names"]

    # Ensure shap_values is 2D (n_samples, n_features) for proper summary plot
    # If 3D with singleton last dimension (n_samples, n_features, 1), squeeze it
    shap_values = np.array(shap_values)
    if shap_values.ndim == 3 and shap_values.shape[-1] == 1:
        shap_values = shap_values.squeeze(axis=-1)

    shap.summary_plot(
        shap_values, X, feature_names=feature_names, max_display=max_display, show=False
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved SHAP summary to {save_path}")

    return fig


def plot_shap_waterfall(
    shap_result: Dict[str, Any], sample_idx: int = 0, save_path: Optional[str] = None
):
    """Plot SHAP waterfall for a single prediction.

    Args:
        shap_result: Result from compute_shap_values()
        sample_idx: Index of sample to explain
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    if not HAS_SHAP:
        return None

    if "error" in shap_result:
        return None

    import matplotlib.pyplot as plt

    shap_values = shap_result["shap_values"]
    expected_value = shap_result["expected_value"]
    feature_names = shap_result["feature_names"]

    # Create Explanation object
    explanation = shap.Explanation(
        values=shap_values[sample_idx],
        base_values=expected_value,
        feature_names=feature_names,
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(explanation, show=False)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def get_feature_importance(shap_result: Dict[str, Any]) -> Dict[str, float]:
    """Get mean absolute SHAP values as feature importance.

    Args:
        shap_result: Result from compute_shap_values()

    Returns:
        Dictionary mapping feature names to importance
    """
    if "error" in shap_result:
        return {}

    shap_values = shap_result["shap_values"]
    feature_names = shap_result["feature_names"]

    # Mean absolute SHAP value per feature
    importance = np.mean(np.abs(shap_values), axis=0)

    return {name: float(imp) for name, imp in zip(feature_names, importance)}
