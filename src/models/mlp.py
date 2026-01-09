"""Simple MLP baseline model."""

import torch
import torch.nn as nn
from typing import List, Optional

from .base import BaseModel
from .registry import ModelRegistry


@ModelRegistry.register("mlp")
class MLPModel(BaseModel):
    """Simple MLP baseline for tabular features.

    Example usage:
        >>> model = MLPModel(input_dim=10, hidden_dims=[64, 32])
        >>> output = model(x)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dims: List[int] = None,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        """Initialize the model.

        Args:
            input_dim: Number of input features
            output_dim: Number of outputs
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
            activation: Activation function ("relu" or "tanh")
        """
        super().__init__(input_dim, output_dim)

        if hidden_dims is None:
            hidden_dims = [64, 32]

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU() if activation == "relu" else nn.Tanh(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor, t: Optional[torch.Tensor] = None, **kwargs
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, input_dim)
            t: Ignored (for interface compatibility)

        Returns:
            Output tensor of shape (batch, output_dim)
        """
        return self.net(x)
