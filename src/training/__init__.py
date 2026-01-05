"""Training infrastructure."""

from .trainer import Trainer
from .metrics import compute_metrics, evaluate_by_group
from .losses import (
    LossRegistry,
    BaseLoss,
    MSELoss,
    PhysicsInformedLoss,
    HuberLoss,
    PercentageLoss,
    MAELoss,
)

__all__ = [
    "Trainer",
    "compute_metrics",
    "evaluate_by_group",
    "LossRegistry",
    "BaseLoss",
    "MSELoss",
    "PhysicsInformedLoss",
    "HuberLoss",
    "PercentageLoss",
    "MAELoss",
]
