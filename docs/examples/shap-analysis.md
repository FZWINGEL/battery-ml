# SHAP Analysis Workflow

This guide demonstrates a complete SHAP (SHapley Additive exPlanations) analysis workflow for model interpretability.

## Overview

SHAP provides feature importance values that explain model predictions. This is especially useful for understanding which features drive battery degradation predictions.

## Prerequisites

```python
import shap
import numpy as np
import matplotlib.pyplot as plt
from src.models.lgbm import LGBMModel
from src.pipelines.summary_set import SummarySetPipeline
```

## Step 1: Train Model

```python
# Load data and create pipeline
pipeline = SummarySetPipeline(include_arrhenius=True, normalize=True)
train_samples = pipeline.fit_transform({'df': train_df})
val_samples = pipeline.transform({'df': val_df})

# Prepare arrays for LightGBM
X_train = np.vstack([s.x for s in train_samples])
y_train = np.vstack([s.y for s in train_samples])
X_val = np.vstack([s.x for s in val_samples])
y_val = np.vstack([s.y for s in val_samples])

# Train model
model = LGBMModel(n_estimators=500, learning_rate=0.05)
model.fit(X_train, y_train, X_val, y_val, 
          feature_names=pipeline.get_feature_names())
```

## Step 2: Create SHAP Explainer

```python
# Create SHAP explainer
explainer = shap.TreeExplainer(model.model)  # model.model is LightGBM object

# Compute SHAP values
shap_values = explainer.shap_values(X_val)
```

## Step 3: Visualize SHAP Values

### Summary Plot

```python
feature_names = pipeline.get_feature_names()

shap.summary_plot(
    shap_values,
    X_val,
    feature_names=feature_names,
    show=False
)
plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
plt.close()
```

### Bar Plot

```python
shap.summary_plot(
    shap_values,
    X_val,
    feature_names=feature_names,
    plot_type='bar',
    show=False
)
plt.savefig('shap_bar.png', dpi=300, bbox_inches='tight')
plt.close()
```

### Waterfall Plot (Single Prediction)

```python
# For a single sample
sample_idx = 0
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[sample_idx],
        base_values=explainer.expected_value,
        data=X_val[sample_idx],
        feature_names=feature_names
    ),
    show=False
)
plt.savefig('shap_waterfall.png', dpi=300, bbox_inches='tight')
plt.close()
```

## Step 4: Feature Importance Analysis

### Top Features

```python
# Compute mean absolute SHAP values
mean_shap = np.abs(shap_values).mean(axis=0)

# Sort by importance
feature_importance = list(zip(feature_names, mean_shap))
feature_importance.sort(key=lambda x: -x[1])

# Print top features
print("Top 10 Features by SHAP Importance:")
for name, importance in feature_importance[:10]:
    print(f"  {name}: {importance:.4f}")
```

### Feature Importance Comparison

```python
# Compare with LightGBM feature importance
lgbm_importance = model.feature_importances_

# Create comparison plot
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(feature_names))
width = 0.35

ax.bar(x - width/2, mean_shap, width, label='SHAP', alpha=0.8)
ax.bar(x + width/2, lgbm_importance, width, label='LightGBM', alpha=0.8)

ax.set_xlabel('Features')
ax.set_ylabel('Importance')
ax.set_title('Feature Importance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(feature_names, rotation=45, ha='right')
ax.legend()

plt.tight_layout()
plt.savefig('importance_comparison.png', dpi=300)
plt.close()
```

## Step 5: Partial Dependence Plots

```python
# Partial dependence for top feature
top_feature_idx = np.argmax(mean_shap)
top_feature_name = feature_names[top_feature_idx]

shap.partial_dependence_plot(
    top_feature_idx,
    model.model.predict,
    X_val,
    ice=False,
    model_expected_value=True,
    feature_expected_value=True,
    show=False
)
plt.savefig(f'pdp_{top_feature_name}.png', dpi=300, bbox_inches='tight')
plt.close()
```

## Step 6: Save Results

```python
import json

# Save SHAP values
np.save('shap_values.npy', shap_values)

# Save feature importance
importance_dict = {
    name: float(imp) 
    for name, imp in zip(feature_names, mean_shap)
}
with open('shap_importance.json', 'w') as f:
    json.dump(importance_dict, f, indent=2)
```

## Complete Workflow Function

```python
def run_shap_analysis(model, X_val, feature_names, save_dir='shap_results'):
    """Run complete SHAP analysis.
    
    Args:
        model: Trained LightGBM model
        X_val: Validation features
        feature_names: Feature names
        save_dir: Directory to save results
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Create explainer
    explainer = shap.TreeExplainer(model.model)
    shap_values = explainer.shap_values(X_val)
    
    # Summary plot
    shap.summary_plot(shap_values, X_val, feature_names=feature_names, 
                     show=False)
    plt.savefig(f'{save_dir}/summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Bar plot
    shap.summary_plot(shap_values, X_val, feature_names=feature_names,
                     plot_type='bar', show=False)
    plt.savefig(f'{save_dir}/bar.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Feature importance
    mean_shap = np.abs(shap_values).mean(axis=0)
    importance = dict(zip(feature_names, mean_shap))
    
    # Save
    np.save(f'{save_dir}/shap_values.npy', shap_values)
    with open(f'{save_dir}/importance.json', 'w') as f:
        json.dump(importance, f, indent=2)
    
    return shap_values, importance

# Usage
shap_values, importance = run_shap_analysis(
    model, X_val, pipeline.get_feature_names()
)
```

## Interpreting Results

### High SHAP Value
- Feature increases prediction when value is high
- Important for model decision

### Low SHAP Value
- Feature has little impact on prediction
- May be redundant

### Negative SHAP Value
- Feature decreases prediction
- Inverse relationship with target

## Best Practices

1. **Use Representative Data**: Compute SHAP on validation set
2. **Sample if Large**: For large datasets, sample for faster computation
3. **Compare Methods**: Compare SHAP with other importance methods
4. **Domain Knowledge**: Interpret results with battery degradation knowledge
5. **Save Results**: Save SHAP values for later analysis

## Next Steps

- [Explainability API](../api/explainability.md) - Complete API reference
- [Models Guide](../user-guide/models.md) - Model selection
- [Theory](../theory/battery-degradation.md) - Battery degradation theory
