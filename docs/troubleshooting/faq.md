# Frequently Asked Questions

## General Questions

### What is BatteryML?

BatteryML is a modular machine learning platform for battery degradation modeling, designed for research on the LG M50T dataset from Oxford University's Battery Intelligence Lab.

### What Python version is required?

Python 3.8 or higher is required.

### Do I need a GPU?

No, BatteryML works on CPU, but GPU acceleration is recommended for neural network models (MLP, LSTM, Neural ODE).

### How do I cite BatteryML?

See [Citation](../reference/citation.md) for citation information.

## Data Questions

### Where do I get the dataset?

The LG M50T dataset is from Oxford University's Battery Intelligence Lab. Contact them for dataset access.

### What experiments are supported?

Experiments 1-5 are supported:
- Experiment 1: Si-based Degradation
- Experiment 2: C-based Degradation
- Experiment 3: Cathode Degradation and Li-Plating
- Experiment 4: Drive Cycle Aging (Control)
- Experiment 5: Standard Cycle Aging (Control)

### How do I load data from a different experiment?

```python
from src.data.tables import SummaryDataLoader
loader = SummaryDataLoader(experiment_id=1, base_path=Path("Raw Data"))
df = loader.load_all_cells(...)
```

## Model Questions

### Which model should I use?

- **LightGBM**: Fast baseline, good for tabular data
- **MLP**: Neural baseline, flexible architecture
- **LSTM**: For sequential data
- **Neural ODE**: For continuous-time modeling

See [Model Selection Guide](../user-guide/models.md) for details.

### How do I add a new model?

See [Adding Models](../contributing/adding-models.md) for step-by-step instructions.

### Can I use my own model?

Yes! See [Custom Model](../examples/custom-model.md) for examples.

## Pipeline Questions

### How do I add a new pipeline?

See [Adding Pipelines](../contributing/adding-pipelines.md) for step-by-step instructions.

### What is the Sample dataclass?

The `Sample` dataclass is the universal format that all pipelines produce and all models consume. See [Core Concepts](../getting-started/concepts.md) for details.

### How does caching work?

Expensive computations (especially ICA) are cached to disk with automatic invalidation based on input parameters. See [Core Concepts](../getting-started/concepts.md) for details.

## Training Questions

### How do I monitor training?

Use TensorBoard:
```bash
tensorboard --logdir artifacts/runs
```

Or MLflow:
```bash
mlflow ui --backend-store-uri file:./artifacts/mlruns
```

### How do I resume training?

Load checkpoint:
```python
checkpoint = torch.load("artifacts/runs/{run_id}/checkpoints/best.pt")
model.load_state_dict(checkpoint['model_state_dict'])
```

### How do I tune hyperparameters?

See [Neural ODE Tuning](../examples/neural-ode-tuning.md) for hyperparameter tuning guide.

## Configuration Questions

### How do I use Hydra configs?

```bash
python run.py model=mlp model.hidden_dim=128
```

See [Configuration Guide](../user-guide/configuration.md) for details.

### How do I override config parameters?

```bash
python run.py training.learning_rate=0.01 training.epochs=500
```

## Troubleshooting Questions

### My model isn't learning. What should I do?

1. Check learning rate
2. Verify data normalization
3. Check model capacity
4. Verify data quality

See [Training Issues](training-issues.md) for details.

### I'm getting out of memory errors. How do I fix it?

1. Reduce batch size
2. Enable mixed precision
3. Use gradient accumulation
4. Use CPU if GPU memory insufficient

See [Training Issues](training-issues.md) for details.

### How do I debug training issues?

1. Monitor training with TensorBoard
2. Check gradients
3. Validate data
4. Profile training

See [Training Issues](training-issues.md) for details.

## Contributing Questions

### How do I contribute?

See [Contributing Guide](../contributing/overview.md) for details.

### How do I add a new feature?

1. Create feature branch
2. Implement feature
3. Add tests
4. Update documentation
5. Create pull request

See [Contributing Guide](../contributing/overview.md) for details.

## Next Steps

- [Getting Started](../getting-started/installation.md) - Installation guide
- [User Guide](../user-guide/data-loading.md) - Usage documentation
- [Troubleshooting](common-issues.md) - Common issues
