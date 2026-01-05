# Adding a New Pipeline

This guide shows how to add a new pipeline to BatteryML.

## Pipeline Template

```python
"""Your pipeline description."""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler

from .base import BasePipeline
from .sample import Sample
from .registry import PipelineRegistry


@PipelineRegistry.register("your_pipeline_name")
class YourPipeline(BasePipeline):
    """Brief description of your pipeline.
    
    Longer description explaining what features it extracts
    and when to use it.
    
    Example usage:
        >>> pipeline = YourPipeline(param1=value1)
        >>> samples = pipeline.fit_transform({'df': df})
    """
    
    def __init__(self, param1: float = 1.0, normalize: bool = True):
        """Initialize the pipeline.
        
        Args:
            param1: Description of param1
            normalize: Whether to normalize features
        """
        self.param1 = param1
        self.normalize = normalize
        self.scaler: Optional[StandardScaler] = None
        self.feature_names_: List[str] = []
    
    def get_params(self) -> dict:
        """Return pipeline parameters for caching."""
        return {
            'param1': self.param1,
            'normalize': self.normalize,
        }
    
    def fit(self, data: Dict[str, Any]) -> 'YourPipeline':
        """Fit scalers on training data.
        
        Args:
            data: Dictionary with data (e.g., {'df': DataFrame})
        
        Returns:
            self: Fitted pipeline
        """
        df = data['df']
        
        # Fit scaler if normalization is enabled
        if self.normalize:
            # Collect all features for fitting
            all_features = []
            for _, row in df.iterrows():
                features = self._extract_features(row)
                all_features.append(features)
            
            self.scaler = StandardScaler()
            self.scaler.fit(np.array(all_features))
        
        return self
    
    def transform(self, data: Dict[str, Any]) -> List[Sample]:
        """Transform data to Sample objects.
        
        Args:
            data: Dictionary with data
        
        Returns:
            List of Sample objects
        """
        df = data['df']
        samples = []
        
        for _, row in df.iterrows():
            # Extract features
            features = self._extract_features(row)
            
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
                y=np.array([row['target_column']])
            )
            samples.append(sample)
        
        return samples
    
    def _extract_features(self, row: pd.Series) -> np.ndarray:
        """Extract feature vector from a row.
        
        Args:
            row: DataFrame row
        
        Returns:
            Feature vector as numpy array
        """
        features = []
        
        # Extract your features here
        # Example:
        # features.append(row['column1'])
        # features.append(row['column2'] * self.param1)
        
        return np.array(features, dtype=np.float32)
    
    def get_feature_names(self) -> List[str]:
        """Return feature names for interpretability.
        
        Returns:
            List of feature name strings
        """
        if not self.feature_names_:
            self.feature_names_ = [
                'feature1',
                'feature2',
                # ... more feature names
            ]
        return self.feature_names_
```

## Steps to Add Pipeline

### 1. Create Pipeline File

Create `src/pipelines/your_pipeline.py` with the template above.

### 2. Implement Methods

- **`__init__`**: Initialize parameters
- **`fit`**: Fit scalers/normalizers
- **`transform`**: Transform to Samples
- **`_extract_features`**: Extract features from row
- **`get_feature_names`**: Return feature names
- **`get_params`**: Return parameters for caching

### 3. Add Tests

Create `tests/test_your_pipeline.py`:

```python
import pytest
from src.pipelines.registry import PipelineRegistry

def test_your_pipeline():
    """Test your pipeline."""
    pipeline = PipelineRegistry.get("your_pipeline_name", param1=1.0)
    
    # Create test data
    df = create_test_dataframe()
    
    # Test fit_transform
    samples = pipeline.fit_transform({'df': df})
    assert len(samples) > 0
    assert samples[0].feature_dim > 0
    
    # Test feature names
    feature_names = pipeline.get_feature_names()
    assert len(feature_names) > 0
```

### 4. Add Configuration

Create `configs/pipeline/your_pipeline.yaml`:

```yaml
param1: 1.0
normalize: true
```

### 5. Update Documentation

- Add to [Pipelines Guide](../user-guide/pipelines.md)
- Add example to [Custom Pipeline](../examples/custom-pipeline.md)
- Ensure docstrings are complete

## Best Practices

1. **Handle Missing Values**: Check for NaN and handle appropriately
2. **Validate Inputs**: Check required columns exist
3. **Support Caching**: Use cache for expensive computations
4. **Document Features**: Clearly document what features are extracted
5. **Add Tests**: Write comprehensive tests

## Checklist

- [ ] Pipeline file created
- [ ] Registered with `@PipelineRegistry.register`
- [ ] Tests written and passing
- [ ] Configuration file added
- [ ] Documentation updated
- [ ] Code follows style guidelines

## Next Steps

- [Pipeline API](../api/pipelines.md) - Complete API reference
- [Testing](testing.md) - Testing guidelines
- [Code Structure](code-structure.md) - Codebase organization
