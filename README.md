# BatteryML Research Platform

A modular machine learning platform for battery degradation modeling, designed for research on the LG M50T dataset from Oxford University's Battery Intelligence Lab.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue)](https://FZWINGEL.github.io/battery-ml/)

---

> 📚 **Full Documentation**: For comprehensive documentation, tutorials, API reference, and guides, visit the [BatteryML Documentation](https://FZWINGEL.github.io/battery-ml/).

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Project Structure](#project-structure)
6. [Core Concepts](#core-concepts)
7. [Pipelines](#pipelines)
8. [Models](#models)
9. [Experiment Tracking](#experiment-tracking)
10. [Milestones](#milestones)
11. [Configuration](#configuration)
12. [Testing](#testing)
13. [Citation](#citation)

---

## Overview

BatteryML addresses the key challenge in battery degradation research: **building reproducible, extensible ML pipelines** that can leverage multiple data modalities (summary statistics, ICA curves, time-series sequences) while supporting various model architectures.

### Research Goals

- **SOH Prediction**: Predict State of Health (remaining capacity) from operational data
- **Temperature Generalization**: Train on extreme temperatures (10°C, 40°C), validate on intermediate (25°C)
- **Degradation Mechanism Analysis**: Use SHAP and ICA features to understand degradation patterns
- **Continuous-Time Modeling**: Neural ODEs for physics-informed degradation trajectories

---

## Key Features

### Architecture

| Feature                       | Description                                                           |
| ----------------------------- | --------------------------------------------------------------------- |
| **Canonical Sample Schema**   | Universal `Sample` dataclass decoupling pipelines from models         |
| **Registry Pattern**          | Decorator-based registration for extensible pipelines and models      |
| **DataLoaderRegistry**        | Pipeline-to-data-format mapping for unified experiment orchestration   |
| **Hash-Based Caching**        | Expensive ICA computations cached to disk with automatic invalidation |
| **Hydra Configuration**       | Composable YAML configs for reproducible experiments                  |

### Data & Pipelines

| Feature                        | Description                                                            |
| ------------------------------ | ---------------------------------------------------------------------- |
| **Multi-Experiment Support**   | Path resolution for Experiments 1-5 with naming convention handling    |
| **Pipeline-Specific Loading**  | Automatic data format selection (summary vs. voltage curves)           |
| **Unit Normalization**         | Automatic mAh→Ah, °C→K conversions                                     |
| **Temperature Holdout Split**  | Train on 10°C+40°C, validate on 25°C for interpolation testing         |
| **LOCO Cross-Validation**      | Leave-One-Cell-Out for generalization assessment                       |
| **Optimized Data I/O**         | Per-temperature caching to reduce redundant file operations             |

### Models & Training

| Feature                       | Description                                                            |
| ----------------------------- | ---------------------------------------------------------------------- |
| **Model Zoo**                 | LightGBM, MLP, LSTM+Attention, Neural ODE, ACLA                        |
| **Modular Loss Functions**    | Registry-based loss selection (MSE, Huber, Physics-Informed, MAPE)     |
| **AMP Training**              | Automatic mixed precision for faster GPU training                      |
| **Early Stopping**            | Patience-based with best model restoration                             |
| **Gradient Clipping**         | Stability for ODE and RNN training                                     |

### Interpretability & Tracking

| Feature                       | Description                                             |
| ----------------------------- | ------------------------------------------------------- |
| **SHAP Analysis**             | Feature importance with summary and waterfall plots     |
| **Attention Visualization**   | Heatmaps for LSTM attention weights                     |
| **Dual Tracking**             | Simultaneous local JSON/TensorBoard + MLflow logging    |

---

## Installation

### Prerequisites

- Python 3.8 or higher
- (Optional) CUDA for GPU acceleration

### Setup

```bash
# Clone or navigate to the project
cd h:\Research_Module\battery-ml

# Install dependencies
pip install -r requirements.txt

# (Optional) Install development dependencies
pip install pytest pytest-cov
```

### Dependencies Overview

| Category            | Packages                                       |
| ------------------- | ---------------------------------------------- |
| Core                | `numpy`, `pandas`, `scipy`, `scikit-learn`     |
| Deep Learning       | `torch`, `torchdiffeq` (Neural ODEs)           |
| Gradient Boosting   | `lightgbm`                                     |
| Config              | `hydra-core`, `pydantic`, `omegaconf`          |
| Tracking            | `mlflow`, `tensorboard`                        |
| Explainability      | `shap`                                         |
| Visualization       | `matplotlib`, `seaborn`, `plotly`              |

---

## Quick Start

### 1. Verify Data Location

Ensure the LG M50T dataset is at:

```text
h:\Research_Module\battery-ml\Raw Data\
└── Expt 5 - Standard Cycle Aging (Control)\
    ├── Summary Data\
    │   ├── Performance Summary\
    │   └── Ageing Sets Summary\
    └── Processed Timeseries Data\
        └── 0.1C Voltage Curves\
```

### 2. Run Your First Experiment

```bash
# Milestone A: LightGBM baseline on summary features
python examples/milestone_a.py

# Or use the unified experiment runner
python examples/run.py pipeline=summary_set model=lgbm

# Try voltage curve features with LSTM
python examples/run.py pipeline=ica_peaks model=lstm_attn training.epochs=100
```

Expected output:

```text
============================================================
Experiment: battery_degradation
Model: lgbm
Pipeline: summary_set
Split: temperature_holdout
============================================================

[1/7] Setting up data loader...
  ✓ Data format: summary
  ✓ Experiment: 5
  ✓ Base path: Raw Data

[2/7] Loading data...
  ✓ Loaded 168 samples from 8 cells

[3/7] Setting up pipeline...
  ✓ Pipeline: SummarySetPipeline(include_arrhenius=True, ...)

[4/7] Creating samples...
  ✓ Created 168 samples
  ✓ Feature dimension: 5

[5/7] Splitting data...
  ✓ Temperature holdout: [10, 40] -> [25]
  ✓ Train: 126 samples
  ✓ Val:   42 samples

[6/7] Setting up model...
  ✓ Model: lgbm
  ✓ Parameters: {'n_estimators': 100, 'max_depth': 6, ...}

[7/7] Setting up tracking...
  ✓ Tracker: dual

[7/7] Training and evaluating...

========================================
Results
========================================
  RMSE: 0.03425
  MAE:  0.02891
  MAPE: 0.61%
  R²:   0.9847
```

### 3. View Results

```bash
# TensorBoard
tensorboard --logdir artifacts/runs

# MLflow UI
mlflow ui --backend-store-uri file:./artifacts/mlruns
```

---

## Project Structure

```text
battery-ml/
├── configs/                    # Hydra YAML configurations
│   ├── config.yaml             # Main config (defaults)
│   ├── data/
│   │   └── expt5.yaml          # Experiment 5 data config
│   ├── pipeline/
│   │   ├── summary_set.yaml    # Summary features pipeline
│   │   ├── ica_peaks.yaml      # ICA feature extraction
│   │   └── latent_ode_seq.yaml # Sequence pipeline for ODEs
│   ├── model/
│   │   ├── lgbm.yaml           # LightGBM hyperparams
│   │   ├── mlp.yaml            # MLP architecture
│   │   ├── lstm_attn.yaml      # LSTM + Attention
│   │   └── neural_ode.yaml     # Neural ODE config
│   ├── split/
│   │   ├── temp_holdout.yaml   # Temperature-based split
│   │   └── loco.yaml           # Leave-One-Cell-Out
│   └── tracking/
│       ├── local.yaml          # Local file tracking
│       ├── mlflow.yaml         # MLflow tracking
│       └── dual.yaml           # Combined tracking
│   └── loss/
│       ├── mse.yaml            # Mean Squared Error (default)
│       ├── huber.yaml          # Robust to outliers
│       ├── physics_informed.yaml # Physics regularization
│       ├── mape.yaml           # Percentage error
│       └── mae.yaml            # Mean Absolute Error
│
├── src/                        # Source code
│   ├── __init__.py
│   ├── config_schema.py        # Pydantic validation schemas
│   │
│   ├── data/                   # Data loading & processing
│   │   ├── expt_paths.py       # Experiment path resolution
│   │   ├── units.py            # Unit conversions
│   │   ├── tables.py           # CSV loaders
│   │   ├── registry.py         # DataLoaderRegistry for pipeline-specific loading
│   │   ├── splits.py           # Train/val/test splits
│   │   └── discovery.py        # File discovery utilities
│   │
│   ├── pipelines/              # Feature extraction
│   │   ├── sample.py           # Canonical Sample dataclass
│   │   ├── base.py             # BasePipeline ABC
│   │   ├── registry.py         # PipelineRegistry
│   │   ├── cache.py            # Hash-based caching
│   │   ├── summary_set.py      # Summary feature pipeline
│   │   ├── ica_peaks.py        # ICA dQ/dV features
│   │   └── latent_ode_seq.py   # ODE sequence pipeline
│   │
│   ├── models/                 # ML models
│   │   ├── base.py             # BaseModel ABC
│   │   ├── registry.py         # ModelRegistry
│   │   ├── lgbm.py             # LightGBM wrapper
│   │   ├── mlp.py              # Simple MLP
│   │   ├── lstm_attn.py        # BiLSTM + Self-Attention
│   │   └── neural_ode.py       # Latent Neural ODE
│   │
│   ├── training/               # Training infrastructure
│   │   ├── trainer.py          # Training loop (AMP, early stop)
│   │   ├── metrics.py          # RMSE, MAE, MAPE, R²
│   │   ├── losses.py           # LossRegistry + loss functions
│   │   └── callbacks.py        # Checkpointing, scheduling
│   │
│   ├── tracking/               # Experiment tracking
│   │   ├── base.py             # BaseTracker ABC
│   │   ├── local.py            # JSON + TensorBoard
│   │   ├── mlflow_tracker.py   # MLflow integration
│   │   └── dual_tracker.py     # Combined local + MLflow
│   │
│   └── explainability/         # Interpretability
│       ├── shap_analysis.py    # SHAP values & plots
│       └── attention_viz.py    # Attention heatmaps
│
│   └── experiments/             # Unified experiment orchestration
│       ├── run.py              # ExperimentRunner class
│       └── __init__.py         # Module exports

├── examples/                   # Runnable milestone scripts
│   ├── milestone_a.py          # LGBM baseline
│   ├── milestone_b.py          # ICA + SHAP
│   ├── milestone_c.py          # Neural ODE vs LSTM
│   └── run.py                  # Unified experiment runner entry point
│
├── tests/                      # pytest test suite
│   ├── conftest.py             # Fixtures
│   ├── test_pipelines.py
│   ├── test_models.py
│   └── test_cache.py
│
├── artifacts/                  # Generated outputs
│   ├── cache/                  # Pipeline cache (pickle)
│   └── runs/                   # Experiment runs
│
├── requirements.txt
└── README.md
```

---

## Core Concepts

### The `Sample` Dataclass

All pipelines produce `Sample` objects, and all models consume them. This **decouples data processing from modeling**.

```python
@dataclass
class Sample:
    meta: Dict[str, Any]    # Metadata (cell_id, temp, etc.) - not fed to model
    x: Tensor               # Features: (feature_dim,) or (seq_len, feature_dim)
    y: Tensor               # Target: SOH or capacity
    mask: Optional[Tensor]  # For variable-length sequences
    t: Optional[Tensor]     # Time vector for ODE models
```

### DataLoaderRegistry

Automatically selects the appropriate data loader based on pipeline requirements:

```python
from src.data import DataLoaderRegistry

# Pipeline-specific data loading
registry = DataLoaderRegistry(
    experiment_id=5,
    base_path=Path("Raw Data")
)

# Summary pipelines get summary data
data = registry.load_data("summary_set", cells=['A', 'B'], temp_map={25: ['A', 'B']})
# Returns: {'df': DataFrame}

# Curve pipelines get voltage curves + SOH targets
data = registry.load_data("ica_peaks", cells=['A', 'B'], temp_map={25: ['A', 'B']})
# Returns: {'curves': [(voltage, capacity, meta), ...], 'targets': {(cell, rpt): soh, ...}}
```

**Pipeline-to-Data Mapping:**
- `summary_set`, `summary_cycle` → Summary statistics CSV
- `ica_peaks`, `latent_ode_seq` → Voltage curves + SOH targets

### Registry Pattern

Pipelines and models self-register using decorators:

```python
@PipelineRegistry.register("summary_set")
class SummarySetPipeline(BasePipeline):
    ...

# Usage
pipeline = PipelineRegistry.get("summary_set", include_arrhenius=True)
```

### Loss Function Registry

Loss functions also support registry-based selection:

```python
from src.training import LossRegistry

# List available losses
print(LossRegistry.list_available())  # ['mse', 'physics_informed', 'huber', 'mape', 'mae']

# Get loss by name
loss = LossRegistry.get("huber", delta=0.5)

# Use in Trainer
trainer = Trainer(model, config, loss_config={'name': 'physics_informed', 'monotonicity_weight': 0.1})
```

### Unified Experiment Runner

The `ExperimentRunner` class orchestrates all components with automatic data loader selection:

```python
from src.experiments import ExperimentRunner

# Initialize with Hydra config
runner = ExperimentRunner(cfg)

# Run complete experiment
results = runner.run()
```

**Automatic Pipeline-to-Data Matching:**
- Detects pipeline type from config
- Selects appropriate data loader via `DataLoaderRegistry`
- Handles both summary and voltage curve data seamlessly

**Experiment Orchestration Steps:**
1. Setup data loader (pipeline-specific)
2. Load data with automatic format selection
3. Setup pipeline with extracted parameters
4. Create samples (fit + transform)
5. Split data (temperature holdout/LOCO)
6. Setup model with known input dimension
7. Setup tracking (local + MLflow)
8. Train and evaluate
9. Log results

### Caching

Expensive computations (ICA extraction) are cached:

```python
result = cache.get_or_compute(
    experiment_id=5, cell_id='A', rpt_id=3,
    pipeline_name='ica_peaks',
    pipeline_params={'sg_window': 51},
    compute_fn=lambda: extract_ica_features(...)
)
```

---

## Pipelines

### SummarySetPipeline

Extracts features from Performance Summary CSVs:

| Feature                 | Description                              |
| ----------------------- | ---------------------------------------- |
| Cumulative Throughput   | Charge/discharge throughput in Ah        |
| Resistance              | 0.1s and 10s resistance measurements     |
| Temperature             | Kelvin + Arrhenius factor `exp(-Ea/RT)`  |

```python
pipeline = SummarySetPipeline(include_arrhenius=True, normalize=True)
samples = pipeline.fit_transform({'df': df})
```

### ICAPeaksPipeline

Extracts dQ/dV (Incremental Capacity Analysis) peak features:

| Feature | Description |
| --------- | ------------- |
| Peak Voltage | Position of each peak (V) |
| Peak Height | Magnitude of dQ/dV at peak |
| Peak Width | FWHM of each peak |
| Total Area | Integrated dQ/dV curve |

```python
pipeline = ICAPeaksPipeline(sg_window=51, num_peaks=3, use_cache=True)
samples = pipeline.fit_transform({'curves': curves, 'targets': targets})
```

### LatentODESequencePipeline

Creates time series with explicit time vectors for Neural ODEs:

```python
pipeline = LatentODESequencePipeline(time_unit="days", max_seq_len=50)
samples = pipeline.fit_transform({'df': df})  # One sample per cell
```

---

## Models

| Model | Type | Best For | Key Parameters |
| ------- | ------ | ---------- | ---------------- |
| `LGBMModel` | Gradient Boosting | Fast baselines, SHAP | `n_estimators`, `max_depth` |
| `MLPModel` | Neural Network | Simple tabular data | `hidden_dims`, `dropout` |
| `LSTMAttentionModel` | Sequence Model | Long sequences | `hidden_dim`, `num_heads` |
| `NeuralODEModel` | Continuous-Time | Physics-aware modeling | `latent_dim`, `solver` |

### Example: Neural ODE

```python
model = NeuralODEModel(
    input_dim=5,
    latent_dim=32,
    hidden_dim=64,
    solver='dopri5',      # Adaptive RK45
    use_adjoint=True      # Memory-efficient gradients
)

trainer = Trainer(model, config, tracker)
trainer.fit(train_samples, val_samples)
```

---

## Experiment Tracking

### Dual Tracking (Recommended)

Logs to both local files and MLflow simultaneously:

```python
tracker = DualTracker(
    local_base_dir="artifacts/runs",
    use_tensorboard=True,
    mlflow_tracking_uri="file:./artifacts/mlruns",
    mlflow_experiment_name="battery_degradation"
)

run_id = tracker.start_run("experiment_name", config)
tracker.log_metrics({"rmse": 0.034}, step=epoch)
tracker.end_run()
```

### Viewing Results

```bash
# TensorBoard (training curves)
tensorboard --logdir artifacts/runs

# MLflow (experiment comparison)
mlflow ui --backend-store-uri file:./artifacts/mlruns
```

---

## Milestones

The project is structured around three progressive milestones:

### Milestone A: LGBM Baseline

**Goal**: End-to-end validation with simple features

```bash
python examples/milestone_a.py
```

- Summary features (throughput, resistance, Arrhenius)
- LightGBM model
- Temperature holdout split

### Milestone B: ICA + SHAP

**Goal**: Degradation-diagnostic features with interpretability

```bash
python examples/milestone_b.py
```

- ICA peak extraction from voltage curves
- SHAP feature importance analysis
- Degradation mechanism insights

### Milestone C: Neural ODE

**Goal**: Continuous-time degradation modeling

```bash
python examples/milestone_c.py
```

- Neural ODE vs LSTM comparison
- Time-aware sequences
- Latent trajectory visualization

---

## Configuration

### Hydra Compose

Swap components via command line with the unified experiment runner:

```bash
# Different model
python examples/run.py model=mlp

# Different pipeline (automatic data loader selection)
python examples/run.py pipeline=ica_peaks

# Different split
python examples/run.py split=loco

# Different loss function
python examples/run.py loss=huber
python examples/run.py loss=physics_informed

# Override parameters
python examples/run.py model.learning_rate=0.01 training.epochs=500

# Voltage curve features with LSTM
python examples/run.py pipeline=ica_peaks model=lstm_attn training.epochs=100

# Neural ODE with sequences
python examples/run.py pipeline=latent_ode_seq model=neural_ode
```

### Example Config

```yaml
# configs/config.yaml
defaults:
  - data: expt5
  - pipeline: summary_set
  - model: lgbm
  - split: temp_holdout
  - tracking: dual
  - loss: mse

experiment:
  name: "battery_degradation"
  seed: 42

training:
  epochs: 200
  batch_size: 32
  learning_rate: 0.001
```

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_pipelines.py -v
```

---

## Citation

If you use this codebase, please cite:

```bibtex
@misc{batteryml2024,
  title={BatteryML: A Modular Platform for Battery Degradation Modeling},
  author={Research Module},
  year={2024},
  publisher={GitHub}
}
```

### Dataset Citation

```bibtex
@article{enmg_data,
  title={Lithium-ion battery degradation: Measuring rapid loss of active silicon in silicon-graphite composite electrodes},
  author={ENMG Oxford},
  journal={Oxford Battery Intelligence Lab},
  year={2023}
}
```

---

## License

MIT License - see LICENSE file for details.

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-model`)
3. Commit your changes (`git commit -am 'Add new model'`)
4. Push to the branch (`git push origin feature/new-model`)
5. Open a Pull Request

---

## Acknowledgments

- Oxford Battery Intelligence Lab for the LG M50T dataset
- PyTorch team for torchdiffeq (Neural ODEs)
- Hydra team for configuration management
