"""Base class for models."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import numpy as np


class BaseModel(ABC, nn.Module):
    """Abstract base for all neural network models.

    All models must:
    1. Accept Sample.x as input (tensor or dict)
    2. Return predictions compatible with Sample.y
    3. Optionally accept Sample.t for time-aware models
    4. Provide explain() hook for interpretability

    Note: LightGBM is a special case that doesn't inherit from nn.Module
    but follows the same interface pattern.
    """

    name: str = "base"

    def __init__(self, input_dim: int, output_dim: int = 1):
        """Initialize the model.

        Args:
            input_dim: Number of input features
            output_dim: Number of output predictions (default: 1 for SOH)
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    @abstractmethod
    def forward(
        self, x: torch.Tensor, t: Optional[torch.Tensor] = None, **kwargs
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Features tensor (batch, features) or (batch, seq_len, features)
            t: Optional time tensor for ODE models (batch, seq_len)
            **kwargs: Additional model-specific arguments

        Returns:
            predictions: Tensor of shape (batch, output_dim)
        """
        pass

    def predict(self, x: torch.Tensor, **kwargs) -> np.ndarray:
        """Inference with no gradient computation.

        Args:
            x: Input features tensor
            **kwargs: Additional arguments passed to forward

        Returns:
            Numpy array of predictions
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x, **kwargs).cpu().numpy()

    def explain(self, x: torch.Tensor, **kwargs) -> Dict[str, Any]:
        """Return interpretability information.

        Override in subclasses to provide model-specific explanations
        (attention weights, SHAP values, etc.).

        Args:
            x: Input features
            **kwargs: Additional arguments

        Returns:
            Dictionary with explanation data
        """
        return {}

    def count_parameters(self) -> int:
        """Count trainable parameters.

        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(input_dim={self.input_dim}, "
            f"output_dim={self.output_dim}, params={self.count_parameters():,})"
        )
