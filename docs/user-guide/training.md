# Training

This guide covers training workflows, metrics, callbacks, and best practices.

## Basic Training Workflow

### PyTorch Models (MLP, LSTM, Neural ODE)

```python
from src.training.trainer import Trainer
from src.tracking.dual_tracker import DualTracker

# Setup tracker
tracker = DualTracker(
    local_base_dir="artifacts/runs",
    use_tensorboard=True,
    mlflow_tracking_uri="file:./artifacts/mlruns"
)

# Create trainer
trainer = Trainer(
    model=model,
    config={
        'epochs': 200,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'weight_decay': 0.01,
        'early_stopping_patience': 20,
        'gradient_clip': 1.0,
        'use_amp': True
    },
    tracker=tracker
)

# Train
run_id = tracker.start_run("experiment_name", config)
history = trainer.fit(train_samples, val_samples)
tracker.end_run()
```

### LightGBM Models

```python
from src.models.lgbm import LGBMModel
import numpy as np

# Prepare arrays
X_train = np.vstack([s.x for s in train_samples])
y_train = np.vstack([s.y for s in train_samples])
X_val = np.vstack([s.x for s in val_samples])
y_val = np.vstack([s.y for s in val_samples])

# Train
model = LGBMModel(n_estimators=500, learning_rate=0.05)
model.fit(X_train, y_train, X_val, y_val)
```

## Training Configuration

### Key Parameters

| Parameter                 | Description                  | Default | Notes                  |
| ------------------------- | ---------------------------- | ------- | ---------------------- |
| `epochs`                  | Maximum training epochs      | 200     | Use early stopping     |
| `batch_size`              | Batch size                   | 32      | Adjust for GPU memory  |
| `learning_rate`           | Initial learning rate        | 1e-3    | Use scheduler          |
| `weight_decay`            | L2 regularization            | 0.01    | Prevent overfitting    |
| `early_stopping_patience` | Early stopping patience      | 20      | Stop if no improvement |
| `gradient_clip`           | Gradient clipping value      | 1.0     | For stability          |
| `use_amp`                 | Automatic mixed precision    | True    | Faster GPU training    |
| `scheduler_T0`            | Cosine annealing T0          | 50      | Learning rate schedule |

### Automatic Mixed Precision (AMP)

AMP speeds up training on GPU with minimal accuracy loss:

```python
config = {
    'use_amp': True,  # Enable AMP
    # ... other config
}
```

AMP is automatically disabled if CUDA is not available.

## Loss Functions

BatteryML provides a registry-based loss function system for flexible loss selection.

### Available Loss Functions

| Loss               | Class                  | Use Case                                         |
| ------------------ | ---------------------- | ------------------------------------------------ |
| `mse`              | `MSELoss`              | Standard regression (default)                    |
| `physics_informed` | `PhysicsInformedLoss`  | Battery degradation with physics regularization  |
| `huber`            | `HuberLoss`            | Robust to outliers                               |
| `mape`             | `PercentageLoss`       | When relative errors matter                      |
| `mae`              | `MAELoss`              | L1 loss, robust alternative                      |

### Using Loss Functions

#### Via Trainer Configuration

```python
from src.training import Trainer

# Default (MSE)
trainer = Trainer(model, config)

# With specific loss
trainer = Trainer(
    model, 
    config,
    loss_config={'name': 'huber', 'delta': 0.5}
)

# Physics-informed loss with regularization
trainer = Trainer(
    model,
    config,
    loss_config={
        'name': 'physics_informed',
        'monotonicity_weight': 0.1,  # Penalize SOH increases
        'smoothness_weight': 0.05    # Encourage smooth predictions
    }
)
```

#### Via YAML Configuration

```yaml
# configs/config.yaml
defaults:
  - loss: physics_informed  # Switch loss here

# Or override at runtime:
# python run.py loss=huber
```

#### Programmatic Usage

```python
from src.training import LossRegistry

# List available losses
print(LossRegistry.list_available())
# ['mse', 'physics_informed', 'huber', 'mape', 'mae']

# Get loss instance
loss_fn = LossRegistry.get('huber', delta=0.5)

# Use directly
loss = loss_fn(predictions, targets)
```

### Loss Function Parameters

| Loss               | Parameter             | Default  | Description                              |
| ------------------ | --------------------- | -------- | ---------------------------------------- |
| All                | `reduction`           | `"mean"` | Reduction method (`mean`, `sum`, `none`) |
| `huber`            | `delta`               | `1.0`    | Threshold for L1/L2 switch               |
| `physics_informed` | `monotonicity_weight` | `0.0`    | Weight for monotonicity penalty          |
| `physics_informed` | `smoothness_weight`   | `0.0`    | Weight for smoothness penalty            |
| `mape`             | `epsilon`             | `1e-8`   | Prevents division by zero                |

### Custom Loss Functions

Create custom losses by extending `BaseLoss`:

```python
from src.training import LossRegistry, BaseLoss
import torch

@LossRegistry.register("my_custom_loss")
class MyCustomLoss(BaseLoss):
    def __init__(self, alpha: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, pred, target, t=None):
        loss = torch.mean((pred - target) ** 2) * self.alpha
        return loss
```

## Metrics

BatteryML computes standard regression metrics:

### Available Metrics

- **RMSE** (Root Mean Squared Error): Lower is better
- **MAE** (Mean Absolute Error): Average absolute error
- **MAPE** (Mean Absolute Percentage Error): Percentage error
- **R²** (Coefficient of Determination): Explained variance (1.0 is perfect)

### Usage

```python
from src.training.metrics import compute_metrics, print_metrics

y_true = np.array([0.95, 0.90, 0.85])
y_pred = np.array([0.94, 0.91, 0.86])

metrics = compute_metrics(y_true, y_pred)
print_metrics(metrics)

# Output:
# RMSE: 0.0129
# MAE:  0.0100
# MAPE: 1.05%
# R²:   0.9876
```

### Logging Metrics

```python
# During training (automatic)
trainer.fit(train_samples, val_samples)  # Metrics logged automatically

# Manual logging
tracker.log_metrics({
    'train_rmse': 0.05,
    'val_rmse': 0.04
}, step=epoch)
```

## Early Stopping

Early stopping prevents overfitting by monitoring validation loss:

```python
config = {
    'early_stopping_patience': 20,  # Stop after 20 epochs without improvement
    # ... other config
}
```

The trainer automatically:

1. Monitors validation loss
2. Saves best model checkpoint
3. Restores best model after training

## Learning Rate Scheduling

Cosine annealing with warm restarts:

```python
config = {
    'scheduler_T0': 50,  # Initial period length
    # ... other config
}
```

The scheduler:

- Starts at `learning_rate`
- Decreases to 0 over `T0` epochs
- Restarts with period `T0 * 2`, `T0 * 4`, etc.

## Gradient Clipping

Prevents exploding gradients (important for RNNs and ODEs):

```python
config = {
    'gradient_clip': 1.0,  # Clip gradients to [-1.0, 1.0]
    # ... other config
}
```

## Callbacks

### Checkpointing

Models are automatically checkpointed:

```python
# Best model saved to:
# artifacts/runs/{run_id}/checkpoints/best.pt

# Load checkpoint
checkpoint = torch.load("artifacts/runs/{run_id}/checkpoints/best.pt")
model.load_state_dict(checkpoint['model_state_dict'])
```

### Custom Callbacks

Create custom callbacks by extending `BaseCallback`:

```python
from src.training.callbacks import BaseCallback

class MyCallback(BaseCallback):
    def on_epoch_end(self, epoch, metrics):
        # Custom logic
        pass
```

## Training Tips

### 1. Data Preparation

- **Normalize features**: Use pipeline normalization
- **Check data quality**: Verify no NaN/inf values
- **Balance datasets**: Ensure representative splits

### 2. Model Initialization

- **Start with pretrained**: If available
- **Xavier/He initialization**: Default for most models
- **Small learning rate**: Start conservative

### 3. Monitoring

- **Use TensorBoard**: Visualize training curves
- **Track validation metrics**: Watch for overfitting
- **Log hyperparameters**: Reproducibility

### 4. Debugging

- **Check gradients**: Use `torch.nn.utils.clip_grad_norm_`
- **Monitor loss**: Should decrease smoothly
- **Validate data**: Ensure samples are correct

### 5. Optimization

- **Use AMP**: Faster GPU training
- **Batch size**: Larger batches = faster (if memory allows)
- **DataLoader workers**: Parallel data loading

## Common Issues

### Loss Not Decreasing

- **Learning rate too high**: Reduce learning rate
- **Data issues**: Check data quality
- **Model capacity**: Increase model size

### Overfitting

- **Add dropout**: Increase dropout rate
- **Regularization**: Increase weight decay
- **Early stopping**: Reduce patience

### Out of Memory

- **Reduce batch size**: Smaller batches
- **Gradient accumulation**: Simulate larger batches
- **Use CPU**: Fallback if GPU memory insufficient

## Example: Complete Training Script

```python
from src.models.mlp import MLPModel
from src.training.trainer import Trainer
from src.tracking.dual_tracker import DualTracker
from src.data.splits import temperature_split

# Load and prepare data
# ... (data loading code) ...

# Split data
train_samples, val_samples = temperature_split(
    all_samples,
    train_temps=[10, 40],
    val_temps=[25]
)

# Create model
model = MLPModel(
    input_dim=train_samples[0].feature_dim,
    hidden_dims=[64, 32],
    dropout=0.1
)

# Setup tracking
tracker = DualTracker(
    local_base_dir="artifacts/runs",
    use_tensorboard=True,
    mlflow_tracking_uri="file:./artifacts/mlruns"
)

# Training config
config = {
    'epochs': 200,
    'batch_size': 32,
    'learning_rate': 1e-3,
    'weight_decay': 0.01,
    'early_stopping_patience': 20,
    'gradient_clip': 1.0,
    'use_amp': True
}

# Train
run_id = tracker.start_run("mlp_baseline", config)
trainer = Trainer(model, config, tracker)
history = trainer.fit(train_samples, val_samples)
tracker.end_run()

print(f"Training complete! Run ID: {run_id}")
```

## Next Steps

- [Tracking](tracking.md) - Experiment tracking and visualization
- [Models](models.md) - Model selection and hyperparameter tuning
- [Troubleshooting](../troubleshooting/training-issues.md) - Common training issues
