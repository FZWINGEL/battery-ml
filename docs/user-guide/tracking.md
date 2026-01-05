# Experiment Tracking

BatteryML supports multiple experiment tracking backends for logging metrics, hyperparameters, and artifacts.

## Available Trackers

| Tracker | Description | Best For |
|---------|-------------|----------|
| `LocalTracker` | JSON + TensorBoard | Local development |
| `MLflowTracker` | MLflow server | Team collaboration |
| `DualTracker` | Both local + MLflow | Recommended default |

## Dual Tracker (Recommended)

The `DualTracker` logs to both local files and MLflow simultaneously:

```python
from src.tracking.dual_tracker import DualTracker

tracker = DualTracker(
    local_base_dir="artifacts/runs",
    use_tensorboard=True,
    mlflow_tracking_uri="file:./artifacts/mlruns",
    mlflow_experiment_name="battery_degradation"
)
```

### Benefits

- **Local files**: Fast access, no server needed
- **TensorBoard**: Real-time training visualization
- **MLflow**: Experiment comparison and management
- **Redundancy**: Data saved in multiple formats

## Basic Usage

### Starting a Run

```python
run_id = tracker.start_run(
    run_name="experiment_001",
    config={
        'model': 'mlp',
        'learning_rate': 1e-3,
        'batch_size': 32,
        # ... other hyperparameters
    }
)
```

### Logging Metrics

```python
# Single metric
tracker.log_metric("train_loss", 0.05, step=epoch)

# Multiple metrics
tracker.log_metrics({
    'train_rmse': 0.05,
    'val_rmse': 0.04,
    'train_mae': 0.03,
    'val_mae': 0.025
}, step=epoch)
```

### Logging Artifacts

```python
# Save model checkpoint
tracker.log_artifact("checkpoints/best.pt", "model.pt")

# Save plots
tracker.log_artifact("plots/attention.png", "attention_heatmap.png")
```

### Ending a Run

```python
tracker.end_run()
```

## Local Tracker

For local development without MLflow:

```python
from src.tracking.local import LocalTracker

tracker = LocalTracker(
    base_dir="artifacts/runs",
    use_tensorboard=True
)
```

### Output Structure

```
artifacts/runs/
└── {run_id}/
    ├── config.json          # Hyperparameters
    ├── metrics.json         # Metrics history
    ├── artifacts/           # Saved files
    └── tensorboard/         # TensorBoard logs
```

## MLflow Tracker

For team collaboration and experiment management:

```python
from src.tracking.mlflow_tracker import MLflowTracker

tracker = MLflowTracker(
    tracking_uri="file:./artifacts/mlruns",  # or remote URI
    experiment_name="battery_degradation"
)
```

### Remote MLflow Server

```python
tracker = MLflowTracker(
    tracking_uri="http://mlflow-server:5000",
    experiment_name="battery_degradation"
)
```

## Viewing Results

### TensorBoard

View training curves in real-time:

```bash
tensorboard --logdir artifacts/runs
```

Open browser to `http://localhost:6006`

**Features**:
- Loss curves
- Learning rate schedule
- Metric plots
- Histograms (if logged)

### MLflow UI

Compare experiments:

```bash
mlflow ui --backend-store-uri file:./artifacts/mlruns
```

Open browser to `http://localhost:5000`

**Features**:
- Experiment comparison
- Parameter search
- Metric visualization
- Artifact browsing
- Model registry

## Integration with Trainer

The `Trainer` automatically logs metrics:

```python
trainer = Trainer(model, config, tracker)
history = trainer.fit(train_samples, val_samples)
# Metrics logged automatically each epoch
```

## Best Practices

### 1. Consistent Naming

Use descriptive run names:

```python
run_name = f"{model_name}_{pipeline_name}_{split_strategy}_{timestamp}"
```

### 2. Log Everything

Log all hyperparameters and metrics:

```python
config = {
    'model': 'mlp',
    'model.hidden_dims': [64, 32],
    'pipeline': 'summary_set',
    'pipeline.include_arrhenius': True,
    'training.learning_rate': 1e-3,
    'training.batch_size': 32,
    # ... everything
}
```

### 3. Version Control

Tag important runs:

```python
tracker.set_tag("status", "baseline")
tracker.set_tag("dataset_version", "v1.0")
```

### 4. Artifact Management

Save important artifacts:

```python
# Model checkpoints
torch.save(model.state_dict(), "checkpoint.pt")
tracker.log_artifact("checkpoint.pt", "best_model.pt")

# Plots
fig.savefig("plot.png")
tracker.log_artifact("plot.png", "feature_importance.png")
```

### 5. Experiment Organization

Use MLflow experiments to organize runs:

```python
# Create experiment
tracker.create_experiment("temperature_generalization")

# Set experiment
tracker.set_experiment("temperature_generalization")
```

## Querying Results

### Local Tracker

```python
import json

# Load metrics
with open("artifacts/runs/{run_id}/metrics.json") as f:
    metrics = json.load(f)
```

### MLflow

```python
import mlflow

# Search runs
runs = mlflow.search_runs(
    experiment_ids=[experiment_id],
    filter_string="metrics.val_rmse < 0.05"
)

# Get best run
best_run = runs.loc[runs['metrics.val_rmse'].idxmin()]
```

## Troubleshooting

### TensorBoard Not Showing Data

- **Check path**: Ensure `--logdir` points to correct directory
- **Refresh**: TensorBoard may need refresh
- **Check logs**: Verify files exist in directory

### MLflow Connection Issues

- **Check URI**: Verify tracking URI is correct
- **Network**: Check network connectivity for remote servers
- **Permissions**: Ensure write permissions for file-based storage

### Missing Metrics

- **Check logging**: Ensure `log_metrics` is called
- **Step numbers**: Verify step numbers are sequential
- **Run active**: Ensure run is started before logging

## Next Steps

- [Configuration](configuration.md) - Hydra configuration system
- [Examples](../examples/shap-analysis.md) - Complete workflows
- [API Reference](../api/tracking.md) - Complete API documentation
