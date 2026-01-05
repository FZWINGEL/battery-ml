# Code Structure

This document explains the organization of the BatteryML codebase.

## Directory Structure

```
battery-ml/
├── src/                    # Source code
│   ├── data/              # Data loading and processing
│   │   ├── expt_paths.py  # Experiment path resolution
│   │   ├── tables.py      # CSV loaders
│   │   ├── splits.py      # Data splitting strategies
│   │   ├── units.py       # Unit conversions
│   │   └── discovery.py   # File discovery utilities
│   │
│   ├── pipelines/         # Feature extraction pipelines
│   │   ├── base.py        # BasePipeline ABC
│   │   ├── sample.py      # Sample dataclass
│   │   ├── registry.py    # PipelineRegistry
│   │   ├── cache.py        # Hash-based caching
│   │   ├── summary_set.py # Summary features pipeline
│   │   ├── ica_peaks.py   # ICA feature extraction
│   │   └── latent_ode_seq.py  # Sequence pipeline
│   │
│   ├── models/             # ML models
│   │   ├── base.py        # BaseModel ABC
│   │   ├── registry.py    # ModelRegistry
│   │   ├── lgbm.py        # LightGBM wrapper
│   │   ├── mlp.py          # MLP model
│   │   ├── lstm_attn.py    # LSTM + Attention
│   │   └── neural_ode.py   # Neural ODE
│   │
│   ├── training/           # Training infrastructure
│   │   ├── trainer.py     # Training loop
│   │   ├── metrics.py     # Evaluation metrics
│   │   ├── losses.py      # Loss functions
│   │   └── callbacks.py   # Training callbacks
│   │
│   ├── tracking/           # Experiment tracking
│   │   ├── base.py        # BaseTracker ABC
│   │   ├── local.py       # Local file tracking
│   │   ├── mlflow_tracker.py  # MLflow integration
│   │   └── dual_tracker.py    # Combined tracking
│   │
│   ├── explainability/     # Interpretability
│   │   ├── shap_analysis.py   # SHAP analysis
│   │   └── attention_viz.py   # Attention visualization
│   │
│   └── config_schema.py    # Pydantic config schemas
│
├── tests/                  # Test suite
│   ├── conftest.py        # Shared fixtures
│   ├── test_pipelines.py
│   ├── test_models.py
│   └── test_cache.py
│
├── examples/              # Example scripts
│   ├── milestone_a.py
│   ├── milestone_b.py
│   └── milestone_c.py
│
├── configs/               # Hydra configurations
│   ├── config.yaml
│   ├── data/
│   ├── pipeline/
│   ├── model/
│   ├── split/
│   └── tracking/
│
└── docs/                  # Documentation
    ├── mkdocs.yml
    └── ... (documentation files)
```

## Module Organization

### Data Module (`src/data/`)

**Purpose**: Load and preprocess raw data

- **`expt_paths.py`**: Resolve file paths for different experiments
- **`tables.py`**: Load CSV files with unit normalization
- **`splits.py`**: Data splitting strategies
- **`units.py`**: Unit conversion utilities
- **`discovery.py`**: File discovery and validation

### Pipelines Module (`src/pipelines/`)

**Purpose**: Transform raw data to Sample objects

- **`base.py`**: Abstract base class for all pipelines
- **`sample.py`**: Universal Sample dataclass
- **`registry.py`**: Pipeline registration system
- **`cache.py`**: Hash-based caching for expensive computations
- **`summary_set.py`**: Summary statistics features
- **`ica_peaks.py`**: ICA peak extraction
- **`latent_ode_seq.py`**: Time-series sequences for ODEs

### Models Module (`src/models/`)

**Purpose**: Machine learning models

- **`base.py`**: Abstract base class for neural models
- **`registry.py`**: Model registration system
- **`lgbm.py`**: LightGBM gradient boosting
- **`mlp.py`**: Multi-layer perceptron
- **`lstm_attn.py`**: LSTM with self-attention
- **`neural_ode.py`**: Neural ODE for continuous-time modeling

### Training Module (`src/training/`)

**Purpose**: Training infrastructure

- **`trainer.py`**: Training loop with AMP, early stopping
- **`metrics.py`**: Evaluation metrics (RMSE, MAE, MAPE, R²)
- **`losses.py`**: Loss functions
- **`callbacks.py`**: Training callbacks

### Tracking Module (`src/tracking/`)

**Purpose**: Experiment tracking

- **`base.py`**: Abstract base class for trackers
- **`local.py`**: Local file + TensorBoard tracking
- **`mlflow_tracker.py`**: MLflow integration
- **`dual_tracker.py`**: Combined local + MLflow

### Explainability Module (`src/explainability/`)

**Purpose**: Model interpretability

- **`shap_analysis.py`**: SHAP value computation and visualization
- **`attention_viz.py`**: Attention weight visualization

## Design Patterns

### Registry Pattern

Used in:
- `PipelineRegistry` (`src/pipelines/registry.py`)
- `ModelRegistry` (`src/models/registry.py`)

**Purpose**: Enable plugin-like extensibility

### Strategy Pattern

Used in:
- Split strategies (`src/data/splits.py`)
- Models (`src/models/`)
- Pipelines (`src/pipelines/`)

**Purpose**: Interchangeable algorithms

### Template Method Pattern

Used in:
- `BasePipeline.fit_transform()` (`src/pipelines/base.py`)
- `BaseModel.predict()` (`src/models/base.py`)

**Purpose**: Define algorithm structure, allow customization

## Adding New Code

### Where to Add

- **New pipeline**: `src/pipelines/your_pipeline.py`
- **New model**: `src/models/your_model.py`
- **New split**: `src/data/splits.py` (add function)
- **New metric**: `src/training/metrics.py` (add function)
- **New tracker**: `src/tracking/your_tracker.py`

### Import Conventions

```python
# Standard library
import os
from pathlib import Path
from typing import List, Dict, Optional

# Third-party
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Local imports
from .base import BasePipeline
from .sample import Sample
from .registry import PipelineRegistry
```

## Code Style

- **Type hints**: Use type hints for all function signatures
- **Docstrings**: Google-style docstrings for all public functions/classes
- **Line length**: Maximum 100 characters
- **Naming**: 
  - Classes: `PascalCase`
  - Functions/variables: `snake_case`
  - Constants: `UPPER_SNAKE_CASE`

## Next Steps

- [Contributing Overview](overview.md) - Contribution workflow
- [Adding Pipelines](adding-pipelines.md) - Add new pipeline
- [Adding Models](adding-models.md) - Add new model
