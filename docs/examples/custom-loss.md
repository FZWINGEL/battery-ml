# Creating a Custom Loss Function

This guide walks through creating a custom loss function step-by-step.

## Overview

Custom loss functions allow you to implement domain-specific objectives or regularization terms for battery degradation modeling.

## Step 1: Create Loss Class

Create your custom loss in `src/training/losses.py` or a new file:

```python
"""Custom loss function example."""

import torch
import torch.nn as nn
from typing import Optional

from src.training.losses import LossRegistry, BaseLoss


@LossRegistry.register("my_custom_loss")
class MyCustomLoss(BaseLoss):
    """Custom loss with domain-specific regularization.
    
    This loss demonstrates:
    - Custom loss computation
    - Additional regularization terms
    - Proper interface implementation
    """
    
    def __init__(self, 
                 alpha: float = 1.0, 
                 beta: float = 0.1,
                 reduction: str = 'mean'):
        """Initialize the loss.
        
        Args:
            alpha: Weight for base MSE loss
            beta: Weight for custom regularization
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction
        self.mse = nn.MSELoss(reduction=reduction)
    
    def forward(self, 
                pred: torch.Tensor, 
                target: torch.Tensor,
                t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute custom loss.
        
        Args:
            pred: Predictions
            target: Ground truth
            t: Optional time tensor
        
        Returns:
            Loss value
        """
        # Base MSE loss
        base_loss = self.alpha * self.mse(pred, target)
        
        # Custom regularization: penalize large predictions
        regularization = self.beta * torch.mean(pred ** 2)
        
        return base_loss + regularization
```

## Step 2: Register Loss

The `@LossRegistry.register("my_custom_loss")` decorator automatically registers your loss.

## Step 3: Use Your Loss

### Via Trainer Configuration

```python
from src.training import Trainer, LossRegistry

# Verify registration
print(LossRegistry.list_available())
# ['mse', 'physics_informed', 'huber', 'mape', 'mae', 'my_custom_loss']

# Use with trainer
trainer = Trainer(
    model, 
    config,
    loss_config={
        'name': 'my_custom_loss',
        'alpha': 1.0,
        'beta': 0.1
    }
)

history = trainer.fit(train_samples, val_samples)
```

### Direct Usage

```python
from src.training import LossRegistry

# Get loss instance
loss_fn = LossRegistry.get('my_custom_loss', alpha=1.0, beta=0.1)

# Use in training loop
pred = model(x)
loss = loss_fn(pred, target)
loss.backward()
```

## Step 4: Add Configuration

Create `configs/loss/my_custom_loss.yaml`:

```yaml
# Custom loss configuration
loss:
  name: "my_custom_loss"
  alpha: 1.0           # MSE weight
  beta: 0.1            # Regularization weight
  reduction: "mean"
```

Use with Hydra:

```bash
python run.py loss=my_custom_loss
```

## Available Base Losses

BatteryML provides these built-in losses to extend or use directly:

| Loss               | Use Case               | Key Parameters                              |
| ------------------ | ---------------------- | ------------------------------------------- |
| `mse`              | Standard regression    | `reduction`                                 |
| `physics_informed` | Battery degradation    | `monotonicity_weight`, `smoothness_weight`  |
| `huber`            | Robust to outliers     | `delta`                                     |
| `mape`             | Relative error focus   | `epsilon`                                   |
| `mae`              | L1 loss                | `reduction`                                 |

## Common Patterns

### Physics-Informed Regularization

```python
@LossRegistry.register("capacity_aware")
class CapacityAwareLoss(BaseLoss):
    """Loss that penalizes predictions outside physical bounds."""
    
    def __init__(self, min_soh: float = 0.5, max_soh: float = 1.0, 
                 boundary_weight: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.min_soh = min_soh
        self.max_soh = max_soh
        self.boundary_weight = boundary_weight
        self.mse = nn.MSELoss(reduction=reduction)
    
    def forward(self, pred, target, t=None):
        # Base loss
        base_loss = self.mse(pred, target)
        
        # Physical boundary penalty
        below_min = torch.relu(self.min_soh - pred)
        above_max = torch.relu(pred - self.max_soh)
        boundary_penalty = (below_min ** 2 + above_max ** 2).mean()
        
        return base_loss + self.boundary_weight * boundary_penalty
```

### Time-Weighted Loss

```python
@LossRegistry.register("time_weighted")
class TimeWeightedLoss(BaseLoss):
    """Weight recent predictions more heavily."""
    
    def __init__(self, time_weight: float = 0.5, reduction: str = 'mean'):
        super().__init__()
        self.time_weight = time_weight
    
    def forward(self, pred, target, t=None):
        # Base error
        error = (pred - target) ** 2
        
        if t is not None:
            # Weight by time (later = more important)
            t_normalized = t / t.max()
            weights = 1.0 + self.time_weight * t_normalized
            error = error * weights.unsqueeze(-1)
        
        return error.mean()
```

### Combined Loss

```python
@LossRegistry.register("combined")
class CombinedLoss(BaseLoss):
    """Combine multiple loss functions."""
    
    def __init__(self, mse_weight: float = 0.7, mae_weight: float = 0.3):
        super().__init__()
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
    
    def forward(self, pred, target, t=None):
        mse_loss = self.mse(pred, target)
        mae_loss = self.mae(pred, target)
        return self.mse_weight * mse_loss + self.mae_weight * mae_loss
```

## Interface Requirements

All custom losses must:

1. **Inherit from `BaseLoss`** (extends `nn.Module`)
2. **Implement `forward(pred, target, t=None)`**
3. **Accept `t` parameter** (even if unused, for ODE compatibility)
4. **Return a scalar tensor**

```python
class BaseLoss(nn.Module, ABC):
    @abstractmethod
    def forward(self, 
                pred: torch.Tensor, 
                target: torch.Tensor,
                t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """All losses must implement this signature."""
        pass
```

## Testing Your Loss

Create a test file:

```python
import pytest
import torch
from src.training import LossRegistry

def test_my_custom_loss():
    # Get loss from registry
    loss_fn = LossRegistry.get('my_custom_loss', alpha=1.0, beta=0.1)
    
    # Create test tensors
    pred = torch.tensor([0.9, 0.85, 0.8])
    target = torch.tensor([0.92, 0.87, 0.82])
    
    # Compute loss
    loss = loss_fn(pred, target)
    
    # Verify output
    assert loss.dim() == 0  # Scalar
    assert not torch.isnan(loss)
    assert loss > 0

def test_my_custom_loss_with_time():
    loss_fn = LossRegistry.get('my_custom_loss')
    
    pred = torch.tensor([0.9, 0.85, 0.8])
    target = torch.tensor([0.92, 0.87, 0.82])
    t = torch.tensor([0.0, 0.5, 1.0])
    
    # Should work with time parameter
    loss = loss_fn(pred, target, t=t)
    assert not torch.isnan(loss)
```

## Next Steps

- [Training Guide](../user-guide/training.md) - Complete training documentation
- [Training API](../api/training.md) - API reference for losses
- [Models Guide](../user-guide/models.md) - Model selection
