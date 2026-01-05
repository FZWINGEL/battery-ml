"""Evaluation metrics for battery degradation prediction."""

import numpy as np
from typing import Dict, List
from collections import defaultdict

from ..pipelines.sample import Sample


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
    
    Returns:
        Dictionary with RMSE, MAE, MAPE, R² metrics
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # MAE
    mae = np.mean(np.abs(y_true - y_pred))
    
    # MAPE (avoid division by zero)
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = 0.0
    
    # R² (coefficient of determination)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Max absolute error
    max_ae = np.max(np.abs(y_true - y_pred))
    
    return {
        'rmse': float(rmse),
        'mae': float(mae),
        'mape': float(mape),
        'r2': float(r2),
        'max_ae': float(max_ae),
    }


def evaluate_by_group(samples: List[Sample], predictions: np.ndarray,
                      group_key: str = 'temperature_C') -> Dict[str, Dict[str, float]]:
    """Compute metrics per group (e.g., per temperature).
    
    Args:
        samples: List of Sample objects
        predictions: Predictions array
        group_key: Meta key to group by
    
    Returns:
        Dictionary mapping group values to metrics
    """
    groups = defaultdict(lambda: {'y_true': [], 'y_pred': []})
    
    predictions = np.asarray(predictions).flatten()
    
    for i, sample in enumerate(samples):
        group = sample.meta.get(group_key, 'unknown')
        y_true = sample.y.numpy().item() if hasattr(sample.y, 'numpy') else float(sample.y)
        y_pred = predictions[i] if i < len(predictions) else 0
        
        groups[group]['y_true'].append(y_true)
        groups[group]['y_pred'].append(y_pred)
    
    results = {}
    for group, data in groups.items():
        y_true = np.array(data['y_true'])
        y_pred = np.array(data['y_pred'])
        results[str(group)] = compute_metrics(y_true, y_pred)
    
    return results


def print_metrics(metrics: Dict[str, float], prefix: str = "") -> None:
    """Print metrics in a formatted way.
    
    Args:
        metrics: Metrics dictionary
        prefix: Optional prefix for lines
    """
    if prefix:
        print(f"\n=== {prefix} ===")
    
    print(f"  RMSE: {metrics['rmse']:.5f}")
    print(f"  MAE:  {metrics['mae']:.5f}")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    print(f"  R²:   {metrics['r2']:.4f}")


def print_grouped_metrics(grouped_metrics: Dict[str, Dict[str, float]], 
                           group_name: str = "Group") -> None:
    """Print metrics for each group.
    
    Args:
        grouped_metrics: Dictionary of group -> metrics
        group_name: Name for the grouping variable
    """
    for group, metrics in sorted(grouped_metrics.items()):
        print(f"\n{group_name} = {group}:")
        print(f"  RMSE: {metrics['rmse']:.5f}, MAE: {metrics['mae']:.5f}, R²: {metrics['r2']:.4f}")
