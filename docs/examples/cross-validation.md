# Cross-Validation Setup

This guide demonstrates how to set up cross-validation workflows.

## Overview

Cross-validation provides robust model evaluation by testing on multiple data splits.

## Leave-One-Cell-Out (LOCO) Cross-Validation

### Basic LOCO CV

```python
from src.data.splits import loco_cv_splits
from src.training.metrics import compute_metrics
import numpy as np

# Generate all LOCO splits
splits = loco_cv_splits(all_samples)

# Store results
results = []

for cell_id, train_samples, test_samples in splits:
    print(f"\nTraining on all cells except {cell_id}")
    
    # Prepare data
    X_train = np.vstack([s.x for s in train_samples])
    y_train = np.vstack([s.y for s in train_samples])
    X_test = np.vstack([s.x for s in test_samples])
    y_test = np.vstack([s.y for s in test_samples])
    
    # Train model
    model = LGBMModel(n_estimators=500)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test.flatten(), y_pred.flatten())
    
    results.append({
        'cell_id': cell_id,
        **metrics
    })
    
    print(f"  RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}")

# Summary statistics
rmse_values = [r['rmse'] for r in results]
print(f"\nCross-Validation Summary:")
print(f"  Mean RMSE: {np.mean(rmse_values):.4f} ± {np.std(rmse_values):.4f}")
print(f"  Min RMSE:  {np.min(rmse_values):.4f}")
print(f"  Max RMSE:  {np.max(rmse_values):.4f}")
```

## K-Fold Cross-Validation

### Custom K-Fold Implementation

```python
from sklearn.model_selection import KFold
import numpy as np

def kfold_cv(samples, n_splits=5, random_state=42):
    """K-fold cross-validation for samples.
    
    Args:
        samples: List of Sample objects
        n_splits: Number of folds
        random_state: Random seed
    
    Yields:
        (train_samples, val_samples) tuples
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Convert to arrays for indexing
    indices = np.arange(len(samples))
    
    for train_idx, val_idx in kf.split(indices):
        train_samples = [samples[i] for i in train_idx]
        val_samples = [samples[i] for i in val_idx]
        yield train_samples, val_samples

# Usage
results = []
for fold, (train_samples, val_samples) in enumerate(kfold_cv(all_samples, n_splits=5)):
    print(f"\nFold {fold + 1}/5")
    
    # Train and evaluate
    # ...
```

## Stratified Cross-Validation

### Stratify by Temperature

```python
from sklearn.model_selection import StratifiedKFold

def stratified_cv_by_temp(samples, n_splits=5):
    """Stratified CV ensuring balanced temperature distribution.
    
    Args:
        samples: List of Sample objects
        n_splits: Number of folds
    
    Yields:
        (train_samples, val_samples) tuples
    """
    # Create stratification labels (temperature)
    labels = [s.meta['temperature_C'] for s in samples]
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    indices = np.arange(len(samples))
    
    for train_idx, val_idx in skf.split(indices, labels):
        train_samples = [samples[i] for i in train_idx]
        val_samples = [samples[i] for i in val_idx]
        yield train_samples, val_samples

# Usage
for train_samples, val_samples in stratified_cv_by_temp(all_samples):
    # Train and evaluate
    # ...
```

## Nested Cross-Validation

### Outer CV for Evaluation, Inner CV for Hyperparameter Tuning

```python
def nested_cv(samples, n_outer=5, n_inner=3):
    """Nested cross-validation.
    
    Outer CV: Model evaluation
    Inner CV: Hyperparameter tuning
    """
    outer_results = []
    
    # Outer CV loop
    for outer_fold, (train_val_samples, test_samples) in enumerate(
        kfold_cv(samples, n_splits=n_outer)
    ):
        print(f"\nOuter Fold {outer_fold + 1}/{n_outer}")
        
        # Inner CV for hyperparameter tuning
        best_params = None
        best_score = float('inf')
        
        # Try different hyperparameters
        for lr in [0.01, 0.05, 0.1]:
            scores = []
            
            # Inner CV
            for train_samples, val_samples in kfold_cv(
                train_val_samples, n_splits=n_inner
            ):
                model = LGBMModel(learning_rate=lr)
                # Train and evaluate
                # ...
                scores.append(val_rmse)
            
            avg_score = np.mean(scores)
            if avg_score < best_score:
                best_score = avg_score
                best_params = {'learning_rate': lr}
        
        # Train with best params on full train_val set
        model = LGBMModel(**best_params)
        # Train on train_val_samples, evaluate on test_samples
        # ...
        
        outer_results.append(test_rmse)
    
    return outer_results
```

## Time Series Cross-Validation

### Temporal Splits

```python
from src.data.splits import temporal_split

def temporal_cv(samples, n_splits=5):
    """Temporal cross-validation (forward chaining).
    
    Each fold uses more historical data than the previous.
    """
    # Sort by timestamp
    sorted_samples = sorted(
        samples, 
        key=lambda s: s.meta.get('timestamp', s.meta.get('cycle_idx', 0))
    )
    
    n = len(sorted_samples)
    fold_size = n // (n_splits + 1)
    
    for i in range(1, n_splits + 1):
        split_idx = i * fold_size
        
        train_samples = sorted_samples[:split_idx]
        val_samples = sorted_samples[split_idx:split_idx + fold_size]
        
        yield train_samples, val_samples

# Usage
for train_samples, val_samples in temporal_cv(all_samples, n_splits=5):
    # Train and evaluate
    # ...
```

## Complete CV Workflow

```python
def run_cross_validation(samples, cv_method='loco'):
    """Run cross-validation with specified method.
    
    Args:
        samples: List of Sample objects
        cv_method: 'loco', 'kfold', 'stratified', 'temporal'
    
    Returns:
        Dictionary with CV results
    """
    if cv_method == 'loco':
        splits = loco_cv_splits(samples)
    elif cv_method == 'kfold':
        splits = kfold_cv(samples)
    elif cv_method == 'stratified':
        splits = stratified_cv_by_temp(samples)
    elif cv_method == 'temporal':
        splits = temporal_cv(samples)
    else:
        raise ValueError(f"Unknown CV method: {cv_method}")
    
    results = []
    for fold, (train_samples, val_samples) in enumerate(splits):
        # Train model
        model = train_model(train_samples)
        
        # Evaluate
        metrics = evaluate_model(model, val_samples)
        
        results.append({
            'fold': fold,
            **metrics
        })
    
    # Compute summary statistics
    summary = {
        'mean_rmse': np.mean([r['rmse'] for r in results]),
        'std_rmse': np.std([r['rmse'] for r in results]),
        'mean_mae': np.mean([r['mae'] for r in results]),
        'std_mae': np.std([r['mae'] for r in results]),
        'individual_results': results
    }
    
    return summary

# Usage
cv_results = run_cross_validation(all_samples, cv_method='loco')
print(f"CV RMSE: {cv_results['mean_rmse']:.4f} ± {cv_results['std_rmse']:.4f}")
```

## Best Practices

1. **Choose Appropriate CV**: Match CV method to research question
2. **Preserve Splits**: Save split indices for reproducibility
3. **Report Statistics**: Include mean, std, min, max
4. **Visualize Results**: Plot CV scores across folds
5. **Compare Methods**: Try different CV methods

## Next Steps

- [Splits Guide](../user-guide/splits.md) - Split strategies
- [Training Guide](../user-guide/training.md) - Training workflows
- [API Reference](../api/data.md) - Complete API docs
