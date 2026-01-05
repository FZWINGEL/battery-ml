# Models

BatteryML provides a model zoo with different architectures suited for various tasks. This guide covers model selection, usage, and hyperparameter tuning.

## Available Models

| Model | Type | Best For | Key Features |
|-------|------|----------|--------------|
| `LGBMModel` | Gradient Boosting | Fast baselines, SHAP analysis | Fast training, interpretable |
| `MLPModel` | Neural Network | Simple tabular data | Flexible architecture |
| `LSTMAttentionModel` | Sequence Model | Long sequences | Attention mechanism |
| `NeuralODEModel` | Continuous-Time | Physics-aware modeling | ODE integration |
| `ACLAModel` | Hybrid Sequence | Complex degradation patterns | Attention + CNN + LSTM + ANODE |

## Model Selection Guide

### When to Use LightGBM

- **Fast iteration**: Quick baseline experiments
- **Interpretability**: SHAP analysis and feature importance
- **Tabular data**: Static features (summary statistics, ICA peaks)
- **Small datasets**: Works well with limited data

### When to Use MLP

- **Simple neural baseline**: Compare against LightGBM
- **Tabular data**: Static features
- **Custom architectures**: Easy to modify hidden layers

### When to Use LSTM + Attention

- **Sequential data**: Time-series degradation trajectories
- **Variable length**: Handles sequences of different lengths
- **Attention visualization**: Understand which time steps matter

### When to Use Neural ODE

- **Continuous-time modeling**: Physics-informed degradation
- **Interpolation**: Predict at arbitrary time points
- **Trajectory analysis**: Understand degradation dynamics

### When to Use ACLA

- **Complex sequences**: Multi-component architecture for rich feature extraction
- **Attention analysis**: Understand which timesteps/features are important
- **Continuous-time dynamics**: ANODE component for physics-informed modeling
- **Hybrid approach**: Combines benefits of CNN (local patterns), LSTM (temporal), and ODE (continuous dynamics)

## LightGBM Model

### Usage

```python
from src.models.lgbm import LGBMModel
import numpy as np

model = LGBMModel(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    early_stopping_rounds=50
)

# Prepare data (numpy arrays)
X_train = np.vstack([s.x for s in train_samples])
y_train = np.vstack([s.y for s in train_samples])
X_val = np.vstack([s.x for s in val_samples])
y_val = np.vstack([s.y for s in val_samples])

# Train
model.fit(
    X_train, y_train,
    X_val, y_val,
    feature_names=pipeline.get_feature_names()
)

# Predict
y_pred = model.predict(X_val)
```

### Key Parameters

- **`n_estimators`**: Number of trees (default: 1000)
- **`learning_rate`**: Shrinkage rate (default: 0.05)
- **`max_depth`**: Maximum tree depth (default: 6)
- **`num_leaves`**: Number of leaves (default: 31)
- **`early_stopping_rounds`**: Early stopping patience (default: 50)

### Feature Importance

```python
importances = model.feature_importances_
feature_names = pipeline.get_feature_names()

for name, imp in zip(feature_names, importances):
    print(f"{name}: {imp:.1f}")
```

## MLP Model

### Usage

```python
from src.models.mlp import MLPModel
from src.training.trainer import Trainer

model = MLPModel(
    input_dim=15,  # Feature dimension
    hidden_dims=[64, 32],
    dropout=0.1,
    output_dim=1
)

trainer = Trainer(model, config, tracker)
history = trainer.fit(train_samples, val_samples)
```

### Key Parameters

- **`input_dim`**: Input feature dimension
- **`hidden_dims`**: List of hidden layer sizes (default: [64, 32])
- **`dropout`**: Dropout rate (default: 0.1)
- **`activation`**: Activation function (default: 'relu')

## LSTM + Attention Model

### Usage

```python
from src.models.lstm_attn import LSTMAttentionModel

model = LSTMAttentionModel(
    input_dim=5,  # Features per time step
    hidden_dim=64,
    num_layers=2,
    num_heads=4,
    dropout=0.1
)

trainer = Trainer(model, config, tracker)
history = trainer.fit(train_samples, val_samples)
```

### Key Parameters

- **`input_dim`**: Features per time step
- **`hidden_dim`**: LSTM hidden dimension (default: 64)
- **`num_layers`**: Number of LSTM layers (default: 2)
- **`num_heads`**: Attention heads (default: 4)
- **`dropout`**: Dropout rate (default: 0.1)

### Attention Visualization

```python
from src.explainability.attention_viz import visualize_attention

attention_weights = model.explain(x_batch)
visualize_attention(attention_weights, save_path="attention.png")
```

## Neural ODE Model

### Usage

```python
from src.models.neural_ode import NeuralODEModel

model = NeuralODEModel(
    input_dim=5,
    latent_dim=32,
    hidden_dim=64,
    solver='dopri5',      # Adaptive RK45
    use_adjoint=True      # Memory-efficient gradients
)

trainer = Trainer(model, config, tracker)
history = trainer.fit(train_samples, val_samples)
```

### Key Parameters

- **`input_dim`**: Features per time step
- **`latent_dim`**: Latent state dimension (default: 32)
- **`hidden_dim`**: ODE network hidden dimension (default: 64)
- **`solver`**: ODE solver - 'dopri5', 'euler', 'rk4' (default: 'dopri5')
- **`rtol`**: Relative tolerance (default: 1e-4)
- **`atol`**: Absolute tolerance (default: 1e-5)
- **`use_adjoint`**: Use adjoint method for gradients (default: True)

### Solver Selection

- **`dopri5`**: Adaptive Runge-Kutta (most accurate, slower)
- **`euler`**: Euler method (fastest, less accurate)
- **`rk4`**: 4th-order Runge-Kutta (balanced)

See [Neural ODE Tuning](../examples/neural-ode-tuning.md) for detailed tuning guide.

## ACLA Model

### Usage

```python
from src.models.acla import ACLAModel

model = ACLAModel(
    input_dim=20,
    output_dim=3,  # Multi-target: LAM_NE, LAM_PE, LLI
    hidden_dim=64,
    augment_dim=20,
    cnn_filters=[64, 32],
    solver='dopri5',
    use_adjoint=True
)

trainer = Trainer(model, config, tracker)
history = trainer.fit(train_samples, val_samples)
```

### Key Parameters

- **`input_dim`**: Features per time step
- **`output_dim`**: Number of output predictions (default: 1)
- **`hidden_dim`**: LSTM and ODE hidden dimension (default: 64)
- **`augment_dim`**: Augmented dimensions for ANODE (default: 20)
- **`cnn_filters`**: CNN filter sizes [first_layer, second_layer] (default: [64, 32])
- **`solver`**: ODE solver - 'dopri5', 'euler', 'rk4' (default: 'dopri5')
- **`use_adjoint`**: Use adjoint method for gradients (default: True)

### Architecture

ACLA combines multiple components:

1. **Attention**: Temporal attention across sequence timesteps
2. **CNN**: 1D convolutions for local pattern extraction
3. **LSTM**: Long-term temporal dependencies
4. **ANODE**: Augmented Neural ODE for continuous-time dynamics

### Attention Visualization

```python
attention_info = model.explain(x_batch)
attention_weights = attention_info['attention_weights']
# Visualize which timesteps the model focuses on
```

## Model Registry

List available models:

```python
from src.models.registry import ModelRegistry

available = ModelRegistry.list_available()
print(available)  # ['lgbm', 'mlp', 'lstm_attn', 'neural_ode', 'acla']
```

Get model by name:

```python
model = ModelRegistry.get("mlp", input_dim=10, hidden_dims=[64, 32])
```

## Hyperparameter Tuning

### LightGBM Tuning

Key hyperparameters to tune:

1. **`n_estimators`**: Start with 500, increase if underfitting
2. **`learning_rate`**: Lower (0.01-0.05) for better generalization
3. **`max_depth`**: Deeper trees (6-10) for complex patterns
4. **`num_leaves`**: Related to max_depth, typically `2^max_depth`

### Neural Network Tuning

1. **Learning rate**: Start with 1e-3, use learning rate finder
2. **Hidden dimensions**: Start small (32-64), increase if needed
3. **Dropout**: 0.1-0.3 for regularization
4. **Batch size**: 32-128 depending on GPU memory

### Neural ODE Tuning

1. **`latent_dim`**: 16-64, larger for complex dynamics
2. **`solver`**: Use `dopri5` for accuracy, `euler` for speed
3. **Tolerances**: Lower `rtol`/`atol` for accuracy (slower)

## Best Practices

1. **Start simple**: Begin with LightGBM baseline
2. **Validate splits**: Use temperature holdout or LOCO
3. **Monitor training**: Use TensorBoard/MLflow
4. **Early stopping**: Prevent overfitting
5. **Feature engineering**: Good features > complex models

## Next Steps

- [Training](training.md) - Training workflows and best practices
- [Neural ODE Tuning](../examples/neural-ode-tuning.md) - Detailed ODE tuning guide
- [API Reference](../api/models.md) - Complete API documentation
