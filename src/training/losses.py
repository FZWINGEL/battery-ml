"""Physics-informed losses for battery degradation models.

This module provides a registry-based loss function system that allows
configuration-driven selection of loss functions.

Example usage:
    >>> from src.training.losses import LossRegistry
    >>> loss = LossRegistry.get('physics_informed', monotonicity_weight=0.1)
    >>> computed_loss = loss(pred, target)
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Dict, Type


class BaseLoss(nn.Module, ABC):
    """Abstract base class for all loss functions.
    
    All loss functions must:
    1. Inherit from BaseLoss
    2. Implement forward(pred, target, t=None)
    3. Be registered with @LossRegistry.register()
    
    Example:
        >>> @LossRegistry.register("custom_loss")
        ... class CustomLoss(BaseLoss):
        ...     def forward(self, pred, target, t=None):
        ...         return torch.mean((pred - target) ** 2)
    """
    
    name: str = "base"
    
    @abstractmethod
    def forward(self, 
                pred: torch.Tensor, 
                target: torch.Tensor,
                t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute loss.
        
        Args:
            pred: Predictions
            target: Ground truth
            t: Optional time values for sequence data
        
        Returns:
            Loss value
        """
        pass


class LossRegistry:
    """Registry for loss function classes.
    
    Example usage:
        >>> @LossRegistry.register("mse")
        ... class MSELoss(BaseLoss):
        ...     pass
        
        >>> loss = LossRegistry.get("mse", reduction='mean')
        >>> print(LossRegistry.list_available())
        ['mse', 'physics_informed', 'huber', 'mape']
    """
    
    _losses: Dict[str, Type[BaseLoss]] = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register a loss function class.
        
        Args:
            name: Registry name for the loss function
        
        Returns:
            Decorator function
        """
        def decorator(loss_class: Type[BaseLoss]):
            cls._losses[name] = loss_class
            loss_class.name = name
            return loss_class
        return decorator
    
    @classmethod
    def get(cls, name: str, **kwargs) -> BaseLoss:
        """Get a loss function instance by name.
        
        Args:
            name: Registry name of the loss function
            **kwargs: Arguments to pass to loss constructor
        
        Returns:
            Loss function instance
        
        Raises:
            ValueError: If loss name is not found
        """
        if name not in cls._losses:
            available = list(cls._losses.keys())
            raise ValueError(f"Unknown loss: {name}. Available: {available}")
        return cls._losses[name](**kwargs)
    
    @classmethod
    def list_available(cls) -> list:
        """List all registered loss function names.
        
        Returns:
            List of loss function names
        """
        return list(cls._losses.keys())
    
    @classmethod
    def get_class(cls, name: str) -> Optional[Type[BaseLoss]]:
        """Get loss class by name (without instantiating).
        
        Args:
            name: Registry name of the loss function
        
        Returns:
            Loss class or None if not found
        """
        return cls._losses.get(name)


@LossRegistry.register("mse")
class MSELoss(BaseLoss):
    """Standard Mean Squared Error loss.
    
    Example usage:
        >>> loss_fn = MSELoss()
        >>> loss = loss_fn(pred, target)
    """
    
    def __init__(self, reduction: str = 'mean'):
        """Initialize MSE loss.
        
        Args:
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.reduction = reduction
        self.mse = nn.MSELoss(reduction=reduction)
    
    def forward(self, 
                pred: torch.Tensor, 
                target: torch.Tensor,
                t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute MSE loss."""
        return self.mse(pred, target)


@LossRegistry.register("physics_informed")
class PhysicsInformedLoss(BaseLoss):
    """MSE loss with optional physics-based regularization.
    
    Regularization terms:
    - Monotonicity: SOH should generally decrease over time
    - Smoothness: Predictions should be smooth
    - Arrhenius consistency: Temperature effects should follow Arrhenius
    
    Example usage:
        >>> loss_fn = PhysicsInformedLoss(monotonicity_weight=0.1)
        >>> loss = loss_fn(pred, target, t=times)
    """
    
    def __init__(self,
                 monotonicity_weight: float = 0.0,
                 smoothness_weight: float = 0.0,
                 reduction: str = 'mean'):
        """Initialize the loss.
        
        Args:
            monotonicity_weight: Weight for monotonicity regularization
            smoothness_weight: Weight for smoothness regularization
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.monotonicity_weight = monotonicity_weight
        self.smoothness_weight = smoothness_weight
        self.reduction = reduction
        self.mse = nn.MSELoss(reduction=reduction)
    
    def forward(self, 
                pred: torch.Tensor, 
                target: torch.Tensor,
                t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute loss.
        
        Args:
            pred: Predictions
            target: Ground truth
            t: Optional time values for sequence data
        
        Returns:
            Loss value
        """
        # Base MSE loss
        loss = self.mse(pred, target)
        
        # Monotonicity regularization (for sequences)
        if self.monotonicity_weight > 0 and pred.dim() >= 2:
            # Penalize increases in SOH over time
            diff = pred[:, 1:] - pred[:, :-1]
            monotonicity_penalty = torch.relu(diff).mean()
            loss = loss + self.monotonicity_weight * monotonicity_penalty
        
        # Smoothness regularization
        if self.smoothness_weight > 0 and pred.dim() >= 2:
            # Second derivative should be small
            diff2 = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]
            smoothness_penalty = (diff2 ** 2).mean()
            loss = loss + self.smoothness_weight * smoothness_penalty
        
        return loss


@LossRegistry.register("huber")
class HuberLoss(BaseLoss):
    """Huber loss (less sensitive to outliers than MSE).
    
    Uses L2 loss for small errors and L1 loss for large errors,
    providing robustness to outliers while maintaining smooth gradients.
    
    Example usage:
        >>> loss_fn = HuberLoss(delta=1.0)
        >>> loss = loss_fn(pred, target)
    """
    
    def __init__(self, delta: float = 1.0, reduction: str = 'mean'):
        """Initialize Huber loss.
        
        Args:
            delta: Threshold for switching between L1 and L2
            reduction: Reduction method
        """
        super().__init__()
        self.delta = delta
        self.reduction = reduction
    
    def forward(self, 
                pred: torch.Tensor, 
                target: torch.Tensor,
                t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute Huber loss."""
        diff = torch.abs(pred - target)
        
        # L2 for small errors, L1 for large errors
        loss = torch.where(
            diff < self.delta,
            0.5 * diff ** 2,
            self.delta * (diff - 0.5 * self.delta)
        )
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


@LossRegistry.register("mape")
class PercentageLoss(BaseLoss):
    """Mean Absolute Percentage Error loss.
    
    Useful when relative errors are more important than absolute errors.
    Returns loss as a percentage (0-100 scale).
    
    Example usage:
        >>> loss_fn = PercentageLoss()
        >>> loss = loss_fn(pred, target)  # Returns MAPE as percentage
    """
    
    def __init__(self, epsilon: float = 1e-8, reduction: str = 'mean'):
        """Initialize MAPE loss.
        
        Args:
            epsilon: Small value to prevent division by zero
            reduction: Reduction method
        """
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
    
    def forward(self, 
                pred: torch.Tensor, 
                target: torch.Tensor,
                t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute MAPE loss."""
        loss = torch.abs((target - pred) / (target.abs() + self.epsilon))
        
        if self.reduction == 'mean':
            return loss.mean() * 100
        elif self.reduction == 'sum':
            return loss.sum() * 100
        return loss * 100


@LossRegistry.register("mae")
class MAELoss(BaseLoss):
    """Mean Absolute Error loss (L1 loss).
    
    More robust to outliers than MSE, but has non-smooth gradients at zero.
    
    Example usage:
        >>> loss_fn = MAELoss()
        >>> loss = loss_fn(pred, target)
    """
    
    def __init__(self, reduction: str = 'mean'):
        """Initialize MAE loss.
        
        Args:
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.reduction = reduction
        self.l1 = nn.L1Loss(reduction=reduction)
    
    def forward(self, 
                pred: torch.Tensor, 
                target: torch.Tensor,
                t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute MAE loss."""
        return self.l1(pred, target)
