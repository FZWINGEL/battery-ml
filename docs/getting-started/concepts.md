# Core Concepts

Understanding these core concepts will help you effectively use and extend BatteryML.

## The Sample Dataclass

The `Sample` dataclass is the **universal format** that all pipelines produce and all models consume. This decouples data processing from modeling, allowing you to swap models without changing pipeline code.

### Structure

```python
@dataclass
class Sample:
    meta: Dict[str, Any]    # Metadata (not fed to model)
    x: Tensor               # Features: (feature_dim,) or (seq_len, feature_dim)
    y: Tensor               # Target: SOH or capacity
    mask: Optional[Tensor]  # For variable-length sequences
    t: Optional[Tensor]     # Time vector for ODE models
```

### Key Properties

- **`meta`**: Contains metadata like `cell_id`, `temperature_C`, `experiment_id` - used for splits and logging, but not fed to models
- **`x`**: Features tensor - can be static `(feature_dim,)` or sequential `(seq_len, feature_dim)`
- **`y`**: Target values - typically SOH (State of Health) as a scalar
- **`mask`**: Optional boolean mask for variable-length sequences
- **`t`**: Optional time vector for ODE models (separate from `x` for clarity)

### Example Usage

```python
from src.pipelines.sample import Sample
import torch

# Static features (tabular data)
sample = Sample(
    meta={'cell_id': 'A', 'temperature_C': 25.0, 'experiment_id': 5},
    x=torch.tensor([1.5, 2.3, 0.8]),  # 3 features
    y=torch.tensor([0.95])  # SOH = 95%
)

# Sequential features (time series)
sample = Sample(
    meta={'cell_id': 'A'},
    x=torch.randn(50, 5),  # 50 time steps, 5 features per step
    y=torch.tensor([0.95]),
    t=torch.linspace(0, 100, 50),  # Time vector for ODE
    mask=torch.ones(50, dtype=torch.bool)  # All steps valid
)
```

### Helper Methods

```python
# Convert numpy arrays to tensors
sample = sample.to_tensor()

# Move to GPU
sample = sample.to_device('cuda')

# Clone
sample_copy = sample.clone()

# Get dimensions
feature_dim = sample.feature_dim  # 3 or 5
seq_len = sample.seq_len  # None or 50
```

## Registry Pattern

The registry pattern enables **extensible, plugin-like architecture**. Pipelines and models self-register using decorators, making it easy to add new components.

### Pipeline Registry

```python
from src.pipelines.registry import PipelineRegistry
from src.pipelines.base import BasePipeline

@PipelineRegistry.register("my_pipeline")
class MyPipeline(BasePipeline):
    def fit(self, data):
        # Fit scalers, etc.
        return self
    
    def transform(self, data):
        # Transform to Samples
        return samples

# Usage
pipeline = PipelineRegistry.get("my_pipeline", param1=value1)
```

### Model Registry

```python
from src.models.registry import ModelRegistry
from src.models.base import BaseModel

@ModelRegistry.register("my_model")
class MyModel(BaseModel):
    def forward(self, x, **kwargs):
        # Forward pass
        return predictions

# Usage
model = ModelRegistry.get("my_model", input_dim=10, hidden_dim=64)
```

### Loss Registry

```python
from src.training.losses import LossRegistry, BaseLoss

@LossRegistry.register("my_loss")
class MyLoss(BaseLoss):
    def forward(self, pred, target, t=None):
        # Compute loss
        return loss_value

# Usage
loss = LossRegistry.get("my_loss", reduction='mean')

# Available losses
print(LossRegistry.list_available())
# ['mse', 'physics_informed', 'huber', 'mape', 'mae']
```

### Benefits

- **Discoverability**: List all available pipelines, models, and losses
- **Consistency**: Enforced interface through base classes
- **Configuration**: Can instantiate from config strings or YAML
- **Extensibility**: Add new components without modifying core code

## Hash-Based Caching

Expensive computations (especially ICA feature extraction) are cached to disk with automatic invalidation based on input parameters.

### How It Works

```python
from src.pipelines.cache import PipelineCache

cache = PipelineCache(base_dir="artifacts/cache")

result = cache.get_or_compute(
    experiment_id=5,
    cell_id='A',
    rpt_id=3,
    pipeline_name='ica_peaks',
    pipeline_params={'sg_window': 51, 'num_peaks': 3},
    compute_fn=lambda: expensive_ica_extraction(...)
)
```

### Cache Key Generation

The cache key is generated from:

- Experiment ID
- Cell ID
- RPT ID (or other identifiers)
- Pipeline name
- Pipeline parameters (serialized and hashed)

If any of these change, the cache is invalidated and recomputed.

### Caching Benefits

- **Speed**: Skip expensive ICA computations on repeated runs
- **Reproducibility**: Same inputs always produce same cached outputs
- **Safety**: Automatic invalidation prevents stale cache issues

## Pipeline System

Pipelines transform raw data (DataFrames, arrays) into `Sample` objects. They follow a `fit/transform` pattern similar to scikit-learn.

### Pipeline Interface

```python
class BasePipeline(ABC):
    @abstractmethod
    def fit(self, data: Dict[str, Any]) -> 'BasePipeline':
        """Fit scalers/normalizers on training data."""
        pass
    
    @abstractmethod
    def transform(self, data: Dict[str, Any]) -> List[Sample]:
        """Transform raw data into Sample objects."""
        pass
    
    def fit_transform(self, data: Dict[str, Any]) -> List[Sample]:
        """Fit and transform in one call."""
        return self.fit(data).transform(data)
```

### Example Pipeline

```python
pipeline = SummarySetPipeline(include_arrhenius=True, normalize=True)

# Fit on training data
pipeline.fit({'df': train_df})

# Transform training data
train_samples = pipeline.transform({'df': train_df})

# Transform test data (uses fitted scalers)
test_samples = pipeline.transform({'df': test_df})
```

## Model System

Models consume `Sample.x` and produce predictions compatible with `Sample.y`. They can optionally use `Sample.t` for time-aware models.

### Model Interface

```python
class BaseModel(ABC, nn.Module):
    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None, **kwargs):
        """Forward pass."""
        pass
    
    def predict(self, x: torch.Tensor, **kwargs) -> np.ndarray:
        """Inference with no gradient computation."""
        pass
    
    def explain(self, x: torch.Tensor, **kwargs) -> Dict[str, Any]:
        """Return interpretability information."""
        pass
```

### Model Types

- **Tabular Models** (LightGBM, MLP): Consume static features `(batch, features)`
- **Sequence Models** (LSTM): Consume sequences `(batch, seq_len, features)`
- **ODE Models** (Neural ODE): Consume sequences with time vectors `(batch, seq_len, features)` + `t`

## Data Splitting Strategies

BatteryML supports multiple data splitting strategies:

### Temperature Holdout

Train on extreme temperatures (10°C, 40°C), validate on intermediate (25°C):

```python
from src.data.splits import temperature_split

train_samples, val_samples = temperature_split(
    all_samples,
    train_temps=[10, 40],
    val_temps=[25]
)
```

### Leave-One-Cell-Out (LOCO)

Hold out one cell for testing, train on all others:

```python
from src.data.splits import loco_split

train_samples, test_samples = loco_split(
    all_samples,
    test_cell='A'
)
```

## Configuration System

BatteryML uses Hydra for configuration management, enabling composable YAML configs:

```yaml
# configs/config.yaml
defaults:
  - data: expt5
  - pipeline: summary_set
  - model: lgbm
  - split: temp_holdout
  - loss: mse
```

Override from command line:

```bash
python run.py model=mlp loss=huber training.epochs=500
```

## Next Steps

- [User Guide](../user-guide/data-loading.md) - Detailed usage documentation
- [Architecture](../architecture/overview.md) - System design deep dive
- [API Reference](../api/pipelines.md) - Complete API documentation
