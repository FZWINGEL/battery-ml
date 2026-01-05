# Pipelines

Pipelines transform raw data (DataFrames, arrays) into `Sample` objects that models can consume. This guide covers all available pipelines and how to use them.

## Overview

Pipelines follow a `fit/transform` pattern similar to scikit-learn:

```python
pipeline = SomePipeline(param1=value1)
train_samples = pipeline.fit_transform({'df': train_df})
test_samples = pipeline.transform({'df': test_df})  # Uses fitted scalers
```

## SummarySetPipeline

Extracts features from Performance Summary CSV files.

### Features

| Feature | Description |
|---------|-------------|
| Cumulative Charge Throughput | Total charge capacity in Ah |
| Cumulative Discharge Throughput | Total discharge capacity in Ah |
| 0.1s Resistance | Fast resistance measurement (Ohms) |
| 10s Resistance | Slow resistance measurement (Ohms) |
| Temperature (K) | Temperature in Kelvin |
| Arrhenius Factor | `exp(-Ea/RT)` for temperature effects |
| Inverse Temperature | `1000/T` for linearization |

### Usage

```python
from src.pipelines.summary_set import SummarySetPipeline

pipeline = SummarySetPipeline(
    include_arrhenius=True,
    arrhenius_Ea=50000.0,  # J/mol
    normalize=True
)

samples = pipeline.fit_transform({'df': df})
```

### Parameters

- **`include_arrhenius`** (bool): Include Arrhenius temperature features
- **`arrhenius_Ea`** (float): Activation energy in J/mol (default: 50000.0)
- **`normalize`** (bool): Apply StandardScaler normalization

### When to Use

- Fast baseline experiments
- When summary statistics are sufficient
- For initial model development

## ICAPeaksPipeline

Extracts dQ/dV (Incremental Capacity Analysis) peak features from voltage curves.

### Features

For each detected peak:
- **Peak Voltage**: Position of peak (V)
- **Peak Height**: Magnitude of dQ/dV at peak
- **Peak Width**: Full-width at half-maximum (FWHM)
- **Peak Area**: Integrated area under peak

Additional features:
- **Total Area**: Total integrated dQ/dV curve
- **Number of Peaks**: Count of detected peaks
- **Voltage at Max dQ/dV**: Voltage at maximum dQ/dV value

### Usage

```python
from src.pipelines.ica_peaks import ICAPeaksPipeline

pipeline = ICAPeaksPipeline(
    sg_window=51,      # Savitzky-Golay window (must be odd)
    sg_order=3,        # Polynomial order
    num_peaks=3,       # Number of peaks to extract
    voltage_range=(3.0, 4.2),
    normalize=True,
    use_cache=True     # Cache expensive computations
)

samples = pipeline.fit_transform({
    'curves': voltage_curves,
    'targets': capacity_targets
})
```

### Parameters

- **`sg_window`** (int): Savitzky-Golay smoothing window (must be odd, default: 51)
- **`sg_order`** (int): Polynomial order for smoothing (default: 3)
- **`num_peaks`** (int): Number of peaks to extract features for (default: 3)
- **`voltage_range`** (tuple): Voltage range for analysis (default: (3.0, 4.2))
- **`resample_points`** (int): Points to resample curves to (default: 500)
- **`normalize`** (bool): Apply StandardScaler (default: True)
- **`use_cache`** (bool): Cache computed features (default: True)

### ICA Theory

ICA features are highly diagnostic for degradation mechanisms:

- **Peak Shifts**: Indicate Loss of Lithium Inventory (LLI)
- **Peak Height Changes**: Indicate Loss of Active Material (LAM)
- **Peak Width Changes**: Indicate kinetic degradation / impedance rise

See [ICA Analysis Theory](../theory/ica-analysis.md) for more details.

### Caching

ICA computation is expensive. The pipeline automatically caches results:

```python
# First run: computes and caches
samples1 = pipeline.fit_transform({'curves': curves, 'targets': targets})

# Second run: loads from cache (much faster)
samples2 = pipeline.fit_transform({'curves': curves, 'targets': targets})
```

Cache is invalidated if pipeline parameters change.

### When to Use

- Degradation mechanism analysis
- When voltage curve data is available
- For interpretable features (SHAP analysis)

## LatentODESequencePipeline

Creates time-series sequences with explicit time vectors for Neural ODE models.

### Features

- **Sequential Features**: Time-series of summary statistics
- **Time Vector**: Explicit time values for ODE integration
- **Variable Length**: Supports variable-length sequences with masking

### Usage

```python
from src.pipelines.latent_ode_seq import LatentODESequencePipeline

pipeline = LatentODESequencePipeline(
    time_unit="days",      # or "throughput_Ah"
    max_seq_len=50,        # Maximum sequence length
    normalize=True
)

# One sample per cell (entire degradation trajectory)
samples = pipeline.fit_transform({'df': df})
```

### Parameters

- **`time_unit`** (str): Time unit - "days" or "throughput_Ah" (default: "days")
- **`max_seq_len`** (int): Maximum sequence length (default: 50)
- **`normalize`** (bool): Apply StandardScaler (default: True)

### Output Format

Each sample represents one cell's degradation trajectory:

```python
sample.x.shape  # (seq_len, feature_dim)
sample.t.shape  # (seq_len,) - time vector
sample.mask.shape  # (seq_len,) - boolean mask for valid steps
```

### When to Use

- Neural ODE models
- Continuous-time degradation modeling
- When temporal dynamics are important

## Creating Custom Pipelines

See [Custom Pipeline Guide](../examples/custom-pipeline.md) for step-by-step instructions.

### Pipeline Interface

All pipelines must inherit from `BasePipeline`:

```python
from src.pipelines.base import BasePipeline
from src.pipelines.sample import Sample
from src.pipelines.registry import PipelineRegistry

@PipelineRegistry.register("my_pipeline")
class MyPipeline(BasePipeline):
    def fit(self, data: Dict[str, Any]) -> 'BasePipeline':
        # Fit scalers, compute statistics, etc.
        return self
    
    def transform(self, data: Dict[str, Any]) -> List[Sample]:
        # Transform to Sample objects
        samples = []
        # ... create samples ...
        return samples
    
    def get_feature_names(self) -> List[str]:
        # Return feature names for interpretability
        return ['feature1', 'feature2', ...]
```

## Pipeline Registry

List available pipelines:

```python
from src.pipelines.registry import PipelineRegistry

available = PipelineRegistry.list_available()
print(available)  # ['summary_set', 'ica_peaks', 'latent_ode_seq']
```

Get pipeline by name:

```python
pipeline = PipelineRegistry.get("summary_set", include_arrhenius=True)
```

## Best Practices

1. **Always fit on training data first**: Use `fit_transform` on training, `transform` on test
2. **Use caching for expensive pipelines**: Enable `use_cache=True` for ICA pipelines
3. **Normalize features**: Most models benefit from normalized features
4. **Check feature names**: Use `get_feature_names()` for interpretability
5. **Validate samples**: Check `sample.feature_dim` and `sample.seq_len` match expectations

## Next Steps

- [Models](models.md) - Using models with pipeline outputs
- [Training](training.md) - Training workflows
- [Custom Pipeline](../examples/custom-pipeline.md) - Creating custom pipelines
- [API Reference](../api/pipelines.md) - Complete API documentation
