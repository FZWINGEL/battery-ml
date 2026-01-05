"""Model implementations for battery degradation prediction."""

from .base import BaseModel
from .registry import ModelRegistry

__all__ = [
    "BaseModel",
    "ModelRegistry",
]

# Import models to register them
from . import acla  # noqa: F401