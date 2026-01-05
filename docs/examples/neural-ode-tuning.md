# Neural ODE Hyperparameter Tuning

This guide covers hyperparameter tuning for Neural ODE models.

## Overview

Neural ODEs have several hyperparameters that significantly affect performance and training time.

## Key Hyperparameters

### Architecture Parameters

- **`latent_dim`**: Latent state dimension (16-64)
- **`hidden_dim`**: ODE function hidden dimension (32-128)
- **`input_dim`**: Input feature dimension (determined by pipeline)

### Solver Parameters

- **`solver`**: ODE solver algorithm
  - `'dopri5'`: Adaptive Runge-Kutta (most accurate, slower)
  - `'euler'`: Euler method (fastest, less accurate)
  - `'rk4'`: 4th-order Runge-Kutta (balanced)
- **`rtol`**: Relative tolerance (default: 1e-4)
- **`atol`**: Absolute tolerance (default: 1e-5)
- **`use_adjoint`**: Use adjoint method for gradients (default: True)

### Training Parameters

- **`learning_rate`**: Initial learning rate (1e-4 to 1e-3)
- **`batch_size`**: Batch size (16-32 for ODEs)
- **`gradient_clip`**: Gradient clipping (0.5-2.0)

## Tuning Strategy

### Step 1: Start with Defaults

```python
from src.models.neural_ode import NeuralODEModel

model = NeuralODEModel(
    input_dim=5,
    latent_dim=32,
    hidden_dim=64,
    solver='dopri5',
    use_adjoint=True
)
```

### Step 2: Tune Latent Dimension

```python
# Try different latent dimensions
latent_dims = [16, 32, 64]

for latent_dim in latent_dims:
    model = NeuralODEModel(
        input_dim=5,
        latent_dim=latent_dim,
        hidden_dim=64,
        solver='dopri5'
    )
    
    # Train and evaluate
    trainer = Trainer(model, config, tracker)
    history = trainer.fit(train_samples, val_samples)
    
    # Log results
    print(f"Latent dim {latent_dim}: Val RMSE = {best_val_rmse}")
```

### Step 3: Tune Solver

```python
# Compare solvers
solvers = ['euler', 'rk4', 'dopri5']

for solver in solvers:
    model = NeuralODEModel(
        input_dim=5,
        latent_dim=32,
        hidden_dim=64,
        solver=solver
    )
    
    # Measure training time and accuracy
    start_time = time.time()
    trainer.fit(train_samples, val_samples)
    training_time = time.time() - start_time
    
    print(f"Solver {solver}: Time={training_time:.1f}s, RMSE={val_rmse:.4f}")
```

### Step 4: Tune Tolerances

```python
# Tune tolerances for accuracy vs speed tradeoff
tolerance_configs = [
    {'rtol': 1e-3, 'atol': 1e-4},  # Faster, less accurate
    {'rtol': 1e-4, 'atol': 1e-5},  # Default
    {'rtol': 1e-5, 'atol': 1e-6},  # Slower, more accurate
]

for tol_config in tolerance_configs:
    model = NeuralODEModel(
        input_dim=5,
        latent_dim=32,
        hidden_dim=64,
        solver='dopri5',
        **tol_config
    )
    
    # Train and compare
    # ...
```

## Hyperparameter Search

### Grid Search

```python
from itertools import product

# Define search space
latent_dims = [16, 32, 64]
hidden_dims = [32, 64, 128]
solvers = ['euler', 'rk4', 'dopri5']

best_score = float('inf')
best_config = None

for latent_dim, hidden_dim, solver in product(latent_dims, hidden_dims, solvers):
    model = NeuralODEModel(
        input_dim=5,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        solver=solver
    )
    
    trainer = Trainer(model, config, tracker)
    history = trainer.fit(train_samples, val_samples)
    
    val_rmse = min(history['val_rmse'])
    
    if val_rmse < best_score:
        best_score = val_rmse
        best_config = {
            'latent_dim': latent_dim,
            'hidden_dim': hidden_dim,
            'solver': solver
        }

print(f"Best config: {best_config}")
print(f"Best RMSE: {best_score:.4f}")
```

### Random Search

```python
import random

# Define ranges
configs = []
for _ in range(20):  # 20 random configurations
    configs.append({
        'latent_dim': random.choice([16, 32, 64, 128]),
        'hidden_dim': random.choice([32, 64, 128, 256]),
        'solver': random.choice(['euler', 'rk4', 'dopri5']),
    })

# Evaluate each
results = []
for config in configs:
    model = NeuralODEModel(input_dim=5, **config)
    # Train and evaluate
    # ...
```

## Performance Optimization

### Use Adjoint Method

```python
# Adjoint method saves memory
model = NeuralODEModel(
    input_dim=5,
    latent_dim=32,
    use_adjoint=True  # Recommended for memory efficiency
)
```

### Adjust Batch Size

```python
# Smaller batches for ODEs (memory intensive)
config = {
    'batch_size': 16,  # Smaller than typical 32
    # ...
}
```

### Gradient Clipping

```python
# ODEs can have unstable gradients
config = {
    'gradient_clip': 1.0,  # Clip gradients
    # ...
}
```

## Common Issues and Solutions

### Issue: Training Too Slow

**Solutions**:
- Use `'euler'` solver instead of `'dopri5'`
- Increase `rtol` and `atol` (less accurate but faster)
- Reduce `latent_dim` or `hidden_dim`
- Use smaller batch size

### Issue: Out of Memory

**Solutions**:
- Enable `use_adjoint=True`
- Reduce batch size
- Reduce `latent_dim` or `hidden_dim`

### Issue: NaN Losses

**Solutions**:
- Reduce learning rate
- Increase gradient clipping
- Check data for NaN/inf values
- Use more stable solver (`rk4` instead of `euler`)

### Issue: Poor Accuracy

**Solutions**:
- Increase `latent_dim` or `hidden_dim`
- Use `'dopri5'` solver
- Decrease `rtol` and `atol`
- Increase model capacity

## Recommended Configurations

### Fast Development

```python
model = NeuralODEModel(
    input_dim=5,
    latent_dim=16,
    hidden_dim=32,
    solver='euler',
    rtol=1e-3,
    atol=1e-4,
    use_adjoint=True
)
```

### Balanced

```python
model = NeuralODEModel(
    input_dim=5,
    latent_dim=32,
    hidden_dim=64,
    solver='rk4',
    rtol=1e-4,
    atol=1e-5,
    use_adjoint=True
)
```

### High Accuracy

```python
model = NeuralODEModel(
    input_dim=5,
    latent_dim=64,
    hidden_dim=128,
    solver='dopri5',
    rtol=1e-5,
    atol=1e-6,
    use_adjoint=True
)
```

## Monitoring Training

```python
# Track solver steps (indicator of complexity)
# Add to trainer callback
def on_epoch_end(self, epoch, metrics):
    if hasattr(model, 'ode_solver_steps'):
        tracker.log_metric('ode_steps', model.ode_solver_steps, step=epoch)
```

## Next Steps

- [Neural ODE Theory](../theory/neural-odes.md) - Neural ODE background
- [Models Guide](../user-guide/models.md) - Model selection
- [Training Guide](../user-guide/training.md) - Training workflows
