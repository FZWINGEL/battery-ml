# Creating a Custom Pipeline

This guide walks through creating a custom pipeline step-by-step.

## Overview

Custom pipelines allow you to extract domain-specific features or integrate new data sources.

## Step 1: Create Pipeline Class

Create a new file `src/pipelines/my_pipeline.py`:

```python
"""Custom pipeline example."""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler

from .base import BasePipeline
from .sample import Sample
from .registry import PipelineRegistry


@PipelineRegistry.register("my_pipeline")
class MyPipeline(BasePipeline):
    """Custom pipeline for extracting specific features.
    
    This pipeline demonstrates:
    - Feature extraction from DataFrame
    - Normalization
    - Sample creation
    """
    
    def __init__(self, normalize: bool = True, custom_param: float = 1.0):
        """Initialize the pipeline.
        
        Args:
            normalize: Whether to normalize features
            custom_param: Custom parameter for feature extraction
        """
        self.normalize = normalize
        self.custom_param = custom_param
        self.scaler: Optional[StandardScaler] = None
        self.feature_names_: List[str] = []
    
    def get_params(self) -> dict:
        """Return pipeline parameters for caching."""
        return {
            'normalize': self.normalize,
            'custom_param': self.custom_param,
        }
    
    def fit(self, data: Dict[str, Any]) -> 'MyPipeline':
        """Fit scalers on training data.
        
        Args:
            data: Dictionary with 'df' key containing DataFrame
        
        Returns:
            self: Fitted pipeline
        """
        df = data['df']
        
        # Extract features to determine feature dimension
        sample_features = self._extract_features(df.iloc[0], df.iloc[0]['temperature_C'])
        
        # Fit scaler if normalization is enabled
        if self.normalize:
            # Collect all features for fitting
            all_features = []
            for _, row in df.iterrows():
                features = self._extract_features(row, row['temperature_C'])
                all_features.append(features)
            
            self.scaler = StandardScaler()
            self.scaler.fit(np.array(all_features))
        
        return self
    
    def transform(self, data: Dict[str, Any]) -> List[Sample]:
        """Transform DataFrame to Sample objects.
        
        Args:
            data: Dictionary with 'df' key containing DataFrame
        
        Returns:
            List of Sample objects
        """
        df = data['df']
        samples = []
        
        for _, row in df.iterrows():
            # Extract features
            features = self._extract_features(row, row['temperature_C'])
            
            # Normalize if enabled
            if self.normalize and self.scaler is not None:
                features = self.scaler.transform(features.reshape(1, -1))[0]
            
            # Create Sample
            sample = Sample(
                meta={
                    'cell_id': row['cell_id'],
                    'temperature_C': row['temperature_C'],
                    'experiment_id': row['experiment_id'],
                },
                x=features,
                y=np.array([row['Cell Capacity [mA h]'] / 1000.0])  # Convert to Ah
            )
            samples.append(sample)
        
        return samples
    
    def _extract_features(self, row: pd.Series, temp_C: float) -> np.ndarray:
        """Extract feature vector from a row.
        
        Args:
            row: DataFrame row
            temp_C: Temperature in Celsius
        
        Returns:
            Feature vector as numpy array
        """
        features = []
        
        # Example: Extract specific columns
        if 'Cumulative Charge Throughput [A h]' in row.index:
            features.append(row['Cumulative Charge Throughput [A h]'])
        else:
            features.append(0.0)
        
        # Add custom feature
        features.append(temp_C * self.custom_param)
        
        return np.array(features, dtype=np.float32)
    
    def get_feature_names(self) -> List[str]:
        """Return feature names for interpretability.
        
        Returns:
            List of feature name strings
        """
        if not self.feature_names_:
            self.feature_names_ = [
                'cumulative_charge_throughput',
                'temperature_scaled',
            ]
        return self.feature_names_
```

## Step 2: Register Pipeline

The `@PipelineRegistry.register("my_pipeline")` decorator automatically registers your pipeline.

## Step 3: Use Your Pipeline

```python
from src.pipelines.registry import PipelineRegistry

# Get pipeline from registry
pipeline = PipelineRegistry.get("my_pipeline", normalize=True, custom_param=2.0)

# Use it
samples = pipeline.fit_transform({'df': df})
```

## Step 4: Add Configuration

Create `configs/pipeline/my_pipeline.yaml`:

```yaml
normalize: true
custom_param: 2.0
```

Use with Hydra:

```bash
python run.py pipeline=my_pipeline
```

## Best Practices

### 1. Handle Missing Values

```python
def _extract_features(self, row: pd.Series, temp_C: float) -> np.ndarray:
    features = []
    
    # Handle missing values
    val = row.get('column_name', 0.0)
    if pd.isna(val):
        val = 0.0  # or use mean, median, etc.
    
    features.append(val)
    return np.array(features)
```

### 2. Support Caching

If your pipeline has expensive computations:

```python
from .cache import get_cache

def transform(self, data: Dict[str, Any]) -> List[Sample]:
    cache = get_cache()
    
    # Check cache
    result = cache.get_or_compute(
        experiment_id=row['experiment_id'],
        cell_id=row['cell_id'],
        pipeline_name='my_pipeline',
        pipeline_params=self.get_params(),
        compute_fn=lambda: expensive_computation(row)
    )
    
    # Use cached result
    # ...
```

### 3. Validate Inputs

```python
def transform(self, data: Dict[str, Any]) -> List[Sample]:
    if 'df' not in data:
        raise ValueError("Data must contain 'df' key")
    
    df = data['df']
    required_cols = ['cell_id', 'temperature_C']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Continue with transformation
    # ...
```

### 4. Document Features

```python
def get_feature_names(self) -> List[str]:
    """Return feature names.
    
    Features:
        1. cumulative_charge_throughput: Total charge capacity (Ah)
        2. temperature_scaled: Temperature scaled by custom_param
    """
    return self.feature_names_
```

## Testing Your Pipeline

Create a test file `tests/test_my_pipeline.py`:

```python
import pytest
from src.pipelines.registry import PipelineRegistry
import pandas as pd
import numpy as np

def test_my_pipeline():
    # Create test data
    df = pd.DataFrame({
        'cell_id': ['A', 'B'],
        'temperature_C': [25, 40],
        'experiment_id': [5, 5],
        'Cumulative Charge Throughput [A h]': [10.0, 20.0],
        'Cell Capacity [mA h]': [2000, 1900],
    })
    
    # Create pipeline
    pipeline = PipelineRegistry.get("my_pipeline", normalize=False)
    
    # Transform
    samples = pipeline.fit_transform({'df': df})
    
    # Assertions
    assert len(samples) == 2
    assert samples[0].feature_dim == 2
    assert samples[0].meta['cell_id'] == 'A'
```

## Next Steps

- [Pipeline API](../api/pipelines.md) - Complete API reference
- [Pipeline Guide](../user-guide/pipelines.md) - Pipeline usage guide
- [Contributing](../contributing/adding-pipelines.md) - More pipeline examples
