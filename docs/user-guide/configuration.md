# Configuration

BatteryML uses Hydra for flexible, composable configuration management.

## Overview

Hydra allows you to:

- Compose configs from multiple files
- Override parameters from command line
- Switch components (models, pipelines) easily
- Maintain reproducible experiments

## Configuration Structure

```text
configs/
├── config.yaml              # Main config
├── data/
│   └── expt5.yaml          # Data configs
├── pipeline/
│   ├── summary_set.yaml
│   ├── ica_peaks.yaml
│   └── latent_ode_seq.yaml
├── model/
│   ├── lgbm.yaml
│   ├── mlp.yaml
│   ├── lstm_attn.yaml
│   └── neural_ode.yaml
├── split/
│   ├── temp_holdout.yaml
│   └── loco.yaml
├── tracking/
│   ├── local.yaml
│   ├── mlflow.yaml
│   └── dual.yaml
└── loss/
    ├── mse.yaml            # Mean Squared Error (default)
    ├── huber.yaml          # Robust to outliers
    ├── physics_informed.yaml
    ├── mape.yaml
    └── mae.yaml
```

## Main Config

```yaml
# configs/config.yaml
defaults:
  - data: expt5
  - pipeline: summary_set
  - model: lgbm
  - split: temp_holdout
  - tracking: dual
  - loss: mse
  - _self_

experiment:
  name: "battery_degradation"
  seed: 42

training:
  epochs: 200
  batch_size: 32
  learning_rate: 1e-3
  weight_decay: 0.01
  early_stopping_patience: 20
  gradient_clip: 1.0
  use_amp: true
  scheduler_T0: 50
```

## Component Configs

### Data Config

```yaml
# configs/data/expt5.yaml
experiment_id: 5
base_path: "Raw Data"
cells: ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
temp_map:
  10: ['A', 'B', 'C']
  25: ['D', 'E']
  40: ['F', 'G', 'H']
```

### Pipeline Config

```yaml
# configs/pipeline/summary_set.yaml
include_arrhenius: true
arrhenius_Ea: 50000.0
normalize: true
```

### Model Config

```yaml
# configs/model/mlp.yaml
input_dim: 15
hidden_dims: [64, 32]
dropout: 0.1
output_dim: 1
```

## Command Line Overrides

### Switch Components

```bash
# Different model
python run.py model=mlp

# Different pipeline
python run.py pipeline=ica_peaks

# Different split
python run.py split=loco

# Different loss function
python run.py loss=huber
python run.py loss=physics_informed
```

### Override Parameters

```bash
# Override nested parameters
python run.py model.learning_rate=0.01 training.epochs=500

# Override multiple parameters
python run.py model.hidden_dims=[128,64] training.batch_size=64
```

### Add Parameters

```bash
# Add new parameters
python run.py +custom_param=value
```

## Using Configs in Code

### Basic Usage

```python
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Access config values
    experiment_name = cfg.experiment.name
    learning_rate = cfg.training.learning_rate
    
    # Access component configs
    model_config = cfg.model
    pipeline_config = cfg.pipeline
    
    # Run experiment
    # ...
```

### Type Safety with Pydantic

```python
from src.config_schema import ExperimentConfig

@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Validate config
    validated_config = ExperimentConfig(**cfg)
    
    # Use validated config
    # ...
```

## Config Composition

### Multiple Configs

```yaml
# configs/config.yaml
defaults:
  - data: expt5
  - pipeline: summary_set
  - model: mlp
  - split: temp_holdout
  - tracking: dual
```

Hydra merges all configs into a single object.

### Config Groups

Organize related configs:

```yaml
# configs/model/mlp.yaml
# configs/model/lstm_attn.yaml
# configs/model/neural_ode.yaml
```

Select with: `model=mlp`

### Loss Config

```yaml
# configs/loss/physics_informed.yaml
loss:
  name: "physics_informed"
  monotonicity_weight: 0.1
  smoothness_weight: 0.05
  reduction: "mean"
```

Select with: `loss=physics_informed`

## Advanced Features

### Variable Interpolation

```yaml
base_dir: "Raw Data"
experiment_id: 5
experiment_path: "${base_dir}/Expt ${experiment_id}"
```

### Environment Variables

```yaml
mlflow_uri: ${oc.env:MLFLOW_TRACKING_URI,file:./artifacts/mlruns}
```

### Config Inheritance

```yaml
# configs/model/mlp_large.yaml
defaults:
  - mlp

hidden_dims: [128, 64, 32]  # Override
```

## Best Practices

### 1. Organize by Component

Keep related configs together:

- Data configs in `data/`
- Model configs in `model/`
- Pipeline configs in `pipeline/`

### 2. Use Defaults

Set sensible defaults in component configs:

```yaml
# configs/model/mlp.yaml
hidden_dims: [64, 32]  # Default architecture
dropout: 0.1           # Default dropout
```

### 3. Document Parameters

Add comments to explain parameters:

```yaml
# configs/pipeline/ica_peaks.yaml
sg_window: 51  # Must be odd for Savitzky-Golay
num_peaks: 3  # Number of ICA peaks to extract
```

### 4. Version Control

Commit configs to git for reproducibility:

```bash
git add configs/
git commit -m "Add MLP config"
```

### 5. Validate Configs

Use Pydantic schemas to validate:

```python
from src.config_schema import ModelConfig

config = ModelConfig(**cfg.model)  # Validates
```

## Example: Complete Workflow

```bash
# Run with default config
python run.py

# Switch to Neural ODE
python run.py model=neural_ode

# Override hyperparameters
python run.py model=neural_ode model.latent_dim=64 training.learning_rate=5e-4

# Different experiment
python run.py data=expt3 model=mlp
```

## Troubleshooting

### Config Not Found

**Error**: `Could not override 'model'`

**Solution**: Ensure config file exists in `configs/model/`

### Type Errors

**Error**: `TypeError: expected int, got str`

**Solution**: Use Pydantic validation or check YAML types

### Merge Conflicts

**Error**: `ConfigKeyError: Key not found`

**Solution**: Check config structure matches expected schema

## Next Steps

- [User Guide Overview](data-loading.md) - Other user guide sections
- [Examples](../examples/multi-experiment.md) - Config usage examples
- [API Reference](../api/config.md) - Config schema documentation
