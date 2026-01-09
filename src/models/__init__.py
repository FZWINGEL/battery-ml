"""Model implementations for battery degradation prediction."""

from .base import BaseModel
from .registry import ModelRegistry

# Import models to register them
from . import acla  # noqa: F401
from . import cnn_lstm  # noqa: F401
from . import lgbm  # noqa: F401
from . import mlp  # noqa: F401
from . import lstm_attn  # noqa: F401
from . import neural_ode  # noqa: F401

__all__ = [
    "BaseModel",
    "ModelRegistry",
]