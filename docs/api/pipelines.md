# Pipelines API Reference

Pipelines in BatteryML are responsible for transforming raw time-series or summary data into machine-learning-ready `Sample` objects. They utilize a hash-based caching mechanism to avoid recomputing expensive features like ICA peaks.

## Usage Example

```python
from src.pipelines.ica_peaks import ICAPeaksPipeline

# Data objects should contain 'curves' and 'targets' keys
pipeline = ICAPeaksPipeline(num_peaks=3, sg_window=51)
samples = pipeline.fit_transform(raw_data)
```

::: src.pipelines.sample
    options:
      show_root_heading: true
      show_source: true

::: src.pipelines.base
    options:
      show_root_heading: true
      show_source: true

::: src.pipelines.registry
    options:
      show_root_heading: true
      show_source: true

::: src.pipelines.cache
    options:
      show_root_heading: true
      show_source: true

::: src.pipelines.summary_set
    options:
      show_root_heading: true
      show_source: true

::: src.pipelines.ica_peaks
    options:
      show_root_heading: true
      show_source: true

::: src.pipelines.latent_ode_seq
    options:
      show_root_heading: true
      show_source: true
