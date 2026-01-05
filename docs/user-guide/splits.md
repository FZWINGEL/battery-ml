# Data Splitting Strategies

BatteryML provides multiple data splitting strategies for different evaluation scenarios.

## Overview

Data splitting is crucial for:
- **Generalization assessment**: How well models generalize
- **Temperature interpolation**: Testing temperature extrapolation
- **Cell-to-cell transfer**: Testing across different cells
- **Temporal validation**: Testing on future time points

## Available Splits

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `temperature_split` | Split by temperature | Temperature generalization |
| `leave_one_cell_out` | Hold out one cell | Cell-to-cell transfer |
| `loco_cv_splits` | All LOCO splits | Cross-validation |
| `temporal_split` | Split by time | Temporal validation |

## Temperature Split

Train on extreme temperatures, validate on intermediate:

```python
from src.data.splits import temperature_split

train_samples, val_samples = temperature_split(
    all_samples,
    train_temps=[10, 40],  # Extreme temperatures
    val_temps=[25]          # Intermediate (interpolation)
)
```

### Use Case

Tests **temperature interpolation** capability:
- Can model predict at intermediate temperatures?
- Useful for real-world deployment scenarios

### Example

```python
# Expt 5: 8 cells at 3 temperatures
# Train: Cells A,B,C (10°C) + F,G,H (40°C)
# Val:   Cells D,E (25°C)

train_samples, val_samples = temperature_split(
    all_samples,
    train_temps=[10, 40],
    val_temps=[25]
)

print(f"Train: {len(train_samples)} samples")
print(f"Val:   {len(val_samples)} samples")
```

## Leave-One-Cell-Out (LOCO)

Hold out one cell for testing:

```python
from src.data.splits import leave_one_cell_out

train_samples, test_samples = leave_one_cell_out(
    all_samples,
    test_cell='A'  # Hold out cell A
)
```

### Use Case

Tests **cell-to-cell transfer**:
- Can model generalize to unseen cells?
- Important for production deployment

### Example

```python
# Hold out cell A
train_samples, test_samples = leave_one_cell_out(
    all_samples,
    test_cell='A'
)

# Train on cells B-H
# Test on cell A
```

## LOCO Cross-Validation

Generate all LOCO splits for cross-validation:

```python
from src.data.splits import loco_cv_splits

splits = loco_cv_splits(all_samples)

for cell_id, train_samples, test_samples in splits:
    # Train model
    model.fit(train_samples)
    
    # Evaluate
    metrics = model.evaluate(test_samples)
    print(f"Cell {cell_id}: RMSE = {metrics['rmse']:.4f}")
```

### Use Case

**Robust evaluation** across all cells:
- Average performance across cells
- Identify problematic cells
- More reliable than single split

### Example

```python
results = []
for cell_id, train, test in loco_cv_splits(all_samples):
    model = train_model(train)
    metrics = evaluate_model(model, test)
    results.append({
        'cell_id': cell_id,
        'rmse': metrics['rmse'],
        'mae': metrics['mae']
    })

# Average performance
avg_rmse = np.mean([r['rmse'] for r in results])
print(f"Average RMSE: {avg_rmse:.4f}")
```

## Temporal Split

Split by time (early vs. late):

```python
from src.data.splits import temporal_split

train_samples, val_samples, test_samples = temporal_split(
    all_samples,
    train_fraction=0.7,  # First 70% of time
    val_fraction=0.15     # Next 15%
    # Remaining 15% is test
)
```

### Use Case

Tests **temporal generalization**:
- Can model predict future degradation?
- Important for long-term forecasting

### Requirements

Samples must have `timestamp` or `cycle_idx` in `meta`:

```python
sample.meta['timestamp'] = 100.5  # Days since start
# or
sample.meta['cycle_idx'] = 50    # Cycle number
```

## Custom Splits

Create custom split functions:

```python
def custom_split(samples: List[Sample]) -> Tuple[List[Sample], List[Sample]]:
    """Custom split logic."""
    train = []
    val = []
    
    for sample in samples:
        # Your logic here
        if some_condition(sample):
            train.append(sample)
        else:
            val.append(sample)
    
    return train, val
```

## Best Practices

### 1. Match Research Question

Choose split strategy based on research question:
- **Temperature generalization**: Use `temperature_split`
- **Cell transfer**: Use `leave_one_cell_out`
- **Robust evaluation**: Use `loco_cv_splits`

### 2. Preserve Metadata

Ensure samples have required metadata:
- `temperature_C` for temperature splits
- `cell_id` for LOCO splits
- `timestamp` or `cycle_idx` for temporal splits

### 3. Check Split Sizes

Verify splits are reasonable:

```python
train, val = temperature_split(samples, train_temps=[10, 40], val_temps=[25])
print(f"Train: {len(train)} ({len(train)/len(samples)*100:.1f}%)")
print(f"Val:   {len(val)} ({len(val)/len(samples)*100:.1f}%)")
```

### 4. Stratify When Possible

For random splits, consider stratification:

```python
# Stratify by cell_id to ensure balanced splits
from sklearn.model_selection import train_test_split

# Group by cell_id
# ... (implement stratified split)
```

### 5. Document Split Strategy

Document why you chose a particular split:

```python
# Temperature holdout: Tests interpolation capability
# Train on extremes (10°C, 40°C), validate on intermediate (25°C)
train_samples, val_samples = temperature_split(
    all_samples,
    train_temps=[10, 40],
    val_temps=[25]
)
```

## Common Issues

### Missing Metadata

**Error**: `KeyError: 'temperature_C'`

**Solution**: Ensure samples have required metadata keys

### Empty Splits

**Error**: Split returns empty list

**Solution**: Check metadata values match split criteria

### Imbalanced Splits

**Issue**: One split much larger than other

**Solution**: Consider alternative split strategies or stratification

## Example: Complete Evaluation

```python
from src.data.splits import loco_cv_splits
from src.training.metrics import compute_metrics
import numpy as np

# LOCO cross-validation
results = []
for cell_id, train_samples, test_samples in loco_cv_splits(all_samples):
    # Train model
    model = train_model(train_samples)
    
    # Evaluate
    X_test = np.vstack([s.x for s in test_samples])
    y_test = np.vstack([s.y for s in test_samples])
    y_pred = model.predict(X_test)
    
    metrics = compute_metrics(y_test.flatten(), y_pred.flatten())
    results.append({
        'cell_id': cell_id,
        **metrics
    })

# Summary statistics
rmse_values = [r['rmse'] for r in results]
print(f"Mean RMSE: {np.mean(rmse_values):.4f}")
print(f"Std RMSE:  {np.std(rmse_values):.4f}")
print(f"Min RMSE:  {np.min(rmse_values):.4f}")
print(f"Max RMSE:  {np.max(rmse_values):.4f}")
```

## Next Steps

- [Training](training.md) - Training workflows
- [Examples](../examples/cross-validation.md) - Cross-validation examples
- [API Reference](../api/data.md) - Complete API documentation
