# Adding Loss Functions

This guide covers how to add custom loss functions to BatteryML.

## Overview

Loss functions in BatteryML follow the same registry pattern as pipelines and models, enabling easy extension and configuration via YAML.

## Quick Start

### 1. Create Your Loss Class

```python
# In src/training/losses.py or a new file

from src.training.losses import LossRegistry, BaseLoss
import torch
import torch.nn as nn
from typing import Optional

@LossRegistry.register("my_loss")
class MyLoss(BaseLoss):
    """My custom loss function."""
    
    def __init__(self, alpha: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, 
                pred: torch.Tensor, 
                target: torch.Tensor,
                t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute the loss."""
        loss = torch.mean((pred - target) ** 2) * self.alpha
        return loss
```

### 2. Create Configuration

```yaml
# configs/loss/my_loss.yaml
loss:
  name: "my_loss"
  alpha: 1.0
  reduction: "mean"
```

### 3. Use Your Loss

```python
from src.training import Trainer, LossRegistry

# Via registry
loss = LossRegistry.get("my_loss", alpha=2.0)

# Via trainer
trainer = Trainer(model, config, loss_config={'name': 'my_loss', 'alpha': 2.0})
```

## Interface Requirements

All loss functions must:

1. **Inherit from `BaseLoss`**
2. **Implement `forward(pred, target, t=None)`**
3. **Use the `@LossRegistry.register()` decorator**
4. **Return a scalar tensor**

### Required Method Signature

```python
def forward(self, 
            pred: torch.Tensor, 
            target: torch.Tensor,
            t: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Args:
        pred: Predicted values (batch, output_dim)
        target: Ground truth values (batch, output_dim)
        t: Optional time tensor (for ODE compatibility)
    
    Returns:
        Scalar loss tensor
    """
```

## Loss Categories

### Regression Losses

Standard losses for point predictions:

```python
@LossRegistry.register("weighted_mse")
class WeightedMSE(BaseLoss):
    def __init__(self, weights: Optional[List[float]] = None, reduction: str = 'mean'):
        super().__init__()
        self.weights = torch.tensor(weights) if weights else None
        self.reduction = reduction
    
    def forward(self, pred, target, t=None):
        error = (pred - target) ** 2
        if self.weights is not None:
            error = error * self.weights.to(error.device)
        
        if self.reduction == 'mean':
            return error.mean()
        elif self.reduction == 'sum':
            return error.sum()
        return error
```

### Physics-Informed Losses

Domain-specific regularization:

```python
@LossRegistry.register("degradation_aware")
class DegradationAwareLoss(BaseLoss):
    """Penalize predictions that violate battery degradation physics."""
    
    def __init__(self, 
                 base_weight: float = 1.0,
                 monotonicity_weight: float = 0.1,
                 reduction: str = 'mean'):
        super().__init__()
        self.base_weight = base_weight
        self.monotonicity_weight = monotonicity_weight
        self.mse = nn.MSELoss(reduction=reduction)
    
    def forward(self, pred, target, t=None):
        # Base MSE loss
        base_loss = self.mse(pred, target) * self.base_weight
        
        # Monotonicity: SOH should decrease over time
        if pred.dim() >= 2 and pred.shape[0] > 1:
            diffs = pred[1:] - pred[:-1]
            violations = torch.relu(diffs)  # Positive changes are violations
            mono_loss = violations.mean() * self.monotonicity_weight
        else:
            mono_loss = 0.0
        
        return base_loss + mono_loss
```

### Robust Losses

Handle outliers and noise:

```python
@LossRegistry.register("log_cosh")
class LogCoshLoss(BaseLoss):
    """Smooth approximation of Huber loss."""
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, pred, target, t=None):
        error = pred - target
        loss = torch.log(torch.cosh(error + 1e-12))
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
```

## Configuration Schema

Update `src/config_schema.py` to support your loss parameters:

```python
# In LossConfig class
class LossConfig(BaseModel):
    name: Literal["mse", "physics_informed", "huber", "mape", "mae", "my_loss"] = "mse"
    
    # Add your parameters
    alpha: Optional[float] = Field(default=None, description="My loss alpha parameter")
    
    # ... existing parameters
```

## Testing

Always add tests for new losses:

```python
# tests/test_losses.py

import pytest
import torch
from src.training import LossRegistry

def test_my_loss_registered():
    """Test that my_loss is registered."""
    assert "my_loss" in LossRegistry.list_available()

def test_my_loss_computation():
    """Test my_loss computes correctly."""
    loss_fn = LossRegistry.get("my_loss", alpha=2.0)
    
    pred = torch.tensor([0.9, 0.85, 0.8])
    target = torch.tensor([0.92, 0.87, 0.82])
    
    loss = loss_fn(pred, target)
    
    assert loss.dim() == 0  # Scalar
    assert not torch.isnan(loss)
    assert loss > 0

def test_my_loss_with_time():
    """Test my_loss handles time parameter."""
    loss_fn = LossRegistry.get("my_loss")
    
    pred = torch.tensor([0.9, 0.85, 0.8])
    target = torch.tensor([0.92, 0.87, 0.82])
    t = torch.tensor([0.0, 0.5, 1.0])
    
    # Should not raise
    loss = loss_fn(pred, target, t=t)
    assert not torch.isnan(loss)

def test_my_loss_gradient():
    """Test my_loss supports backpropagation."""
    loss_fn = LossRegistry.get("my_loss")
    
    pred = torch.tensor([0.9, 0.85, 0.8], requires_grad=True)
    target = torch.tensor([0.92, 0.87, 0.82])
    
    loss = loss_fn(pred, target)
    loss.backward()
    
    assert pred.grad is not None
```

## Checklist

When adding a new loss function:

- [ ] Create class inheriting from `BaseLoss`
- [ ] Add `@LossRegistry.register()` decorator
- [ ] Implement `forward(pred, target, t=None)` method
- [ ] Create YAML config in `configs/loss/`
- [ ] Update `LossConfig` schema if new parameters
- [ ] Add tests in `tests/test_losses.py`
- [ ] Update `__all__` in `src/training/__init__.py`
- [ ] Add docstring with usage example

## Next Steps

- [Custom Loss Tutorial](../examples/custom-loss.md) - Detailed examples
- [Training Guide](../user-guide/training.md) - Using losses in training
- [Design Patterns](../architecture/design-patterns.md) - Registry pattern details
