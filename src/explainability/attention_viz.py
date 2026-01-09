"""Attention visualization for sequence models."""

import numpy as np
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


def plot_attention_weights(
    attention_weights: np.ndarray,
    x_labels: Optional[List[str]] = None,
    y_labels: Optional[List[str]] = None,
    title: str = "Attention Weights",
    save_path: Optional[str] = None,
):
    """Plot attention weights as heatmap.

    Args:
        attention_weights: Attention weights array of shape (seq_len, seq_len) or (heads, seq_len, seq_len)
        x_labels: Labels for x-axis (query positions)
        y_labels: Labels for y-axis (key positions)
        title: Plot title
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    import matplotlib.pyplot as plt

    # Handle multi-head attention
    if attention_weights.ndim == 3:
        # Average across heads
        attention_weights = attention_weights.mean(axis=0)

    seq_len = attention_weights.shape[0]

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(attention_weights, cmap="viridis", aspect="auto")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Attention Weight")

    # Labels
    if x_labels is None:
        x_labels = [f"t{i}" for i in range(seq_len)]
    if y_labels is None:
        y_labels = x_labels

    ax.set_xticks(range(len(x_labels)))
    ax.set_yticks(range(len(y_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_yticklabels(y_labels)

    ax.set_xlabel("Key Position")
    ax.set_ylabel("Query Position")
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved attention plot to {save_path}")

    return fig


def plot_attention_over_time(
    attention_weights: np.ndarray,
    time_values: Optional[np.ndarray] = None,
    focus_position: int = -1,
    title: str = "Attention Focus Over Time",
    save_path: Optional[str] = None,
):
    """Plot how attention is distributed from a specific position.

    Args:
        attention_weights: Attention weights
        time_values: Optional time values for x-axis
        focus_position: Query position to visualize (-1 for last)
        title: Plot title
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    import matplotlib.pyplot as plt

    if attention_weights.ndim == 3:
        attention_weights = attention_weights.mean(axis=0)

    weights = attention_weights[focus_position]
    seq_len = len(weights)

    if time_values is None:
        time_values = np.arange(seq_len)

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.bar(time_values, weights, alpha=0.7, color="steelblue")
    ax.set_xlabel("Time")
    ax.set_ylabel("Attention Weight")
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def explain_model_attention(model, samples, feature_names: List[str] = None):
    """Get attention explanations from a model.

    Args:
        model: Model with attention (LSTMAttentionModel)
        samples: Input samples
        feature_names: Feature names

    Returns:
        Dictionary with attention analysis
    """
    import torch

    if not hasattr(model, "explain"):
        return {"error": "Model does not have explain method"}

    # Get sample input
    if hasattr(samples[0], "x"):
        x = torch.stack([s.to_tensor().x for s in samples])
    else:
        x = samples

    # Get attention weights
    explanation = model.explain(x)

    if "attention_weights" not in explanation:
        return {"error": "No attention weights available"}

    attn = explanation["attention_weights"]

    # Analyze attention patterns
    analysis = {
        "attention_weights": attn,
        "mean_attention_per_position": attn.mean(axis=(0, 1)),
        "attention_entropy": -np.sum(attn * np.log(attn + 1e-10), axis=-1).mean(),
    }

    return analysis
