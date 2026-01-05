# Testing Guidelines

This guide covers testing practices for BatteryML.

## Running Tests

### Run All Tests

```bash
pytest tests/ -v
```

### Run with Coverage

```bash
pytest tests/ --cov=src --cov-report=html
```

### Run Specific Test

```bash
pytest tests/test_pipelines.py -v
pytest tests/test_pipelines.py::test_summary_set_pipeline -v
```

## Writing Tests

### Test Structure

```python
import pytest
from src.pipelines.registry import PipelineRegistry

def test_pipeline_name():
    """Test description."""
    # Arrange
    pipeline = PipelineRegistry.get("pipeline_name")
    data = create_test_data()
    
    # Act
    result = pipeline.fit_transform(data)
    
    # Assert
    assert len(result) > 0
    assert result[0].feature_dim > 0
```

### Test Fixtures

Use fixtures for common setup:

```python
import pytest
from src.pipelines.sample import Sample

@pytest.fixture
def sample_dataframe():
    """Create test DataFrame."""
    import pandas as pd
    return pd.DataFrame({
        'cell_id': ['A', 'B'],
        'temperature_C': [25, 40],
        'experiment_id': [5, 5],
        # ... more columns
    })

def test_pipeline_with_fixture(sample_dataframe):
    """Test using fixture."""
    pipeline = SummarySetPipeline()
    samples = pipeline.fit_transform({'df': sample_dataframe})
    assert len(samples) == 2
```

### Test Models

```python
import pytest
import torch
from src.models.registry import ModelRegistry

def test_model_forward():
    """Test model forward pass."""
    model = ModelRegistry.get("mlp", input_dim=10, hidden_dim=64)
    
    x = torch.randn(32, 10)
    output = model(x)
    
    assert output.shape == (32, 1)
    assert not torch.isnan(output).any()
```

### Test Edge Cases

```python
def test_pipeline_empty_data():
    """Test pipeline with empty data."""
    pipeline = SummarySetPipeline()
    df = pd.DataFrame()  # Empty
    
    with pytest.raises(ValueError):
        pipeline.fit_transform({'df': df})

def test_pipeline_missing_columns():
    """Test pipeline with missing columns."""
    pipeline = SummarySetPipeline()
    df = pd.DataFrame({'cell_id': ['A']})  # Missing required columns
    
    with pytest.raises(KeyError):
        pipeline.fit_transform({'df': df})
```

## Test Organization

### File Structure

```
tests/
├── conftest.py          # Shared fixtures
├── test_pipelines.py    # Pipeline tests
├── test_models.py       # Model tests
├── test_cache.py         # Cache tests
└── test_data.py         # Data loading tests
```

### conftest.py

```python
import pytest
import pandas as pd
from pathlib import Path

@pytest.fixture
def sample_dataframe():
    """Create sample DataFrame for testing."""
    return pd.DataFrame({
        'cell_id': ['A', 'B', 'C'],
        'temperature_C': [25, 25, 40],
        'experiment_id': [5, 5, 5],
        'Cumulative Charge Throughput [A h]': [10.0, 20.0, 30.0],
        'Cell Capacity [mA h]': [2000, 1900, 1800],
    })

@pytest.fixture
def sample_samples():
    """Create sample Sample objects for testing."""
    from src.pipelines.sample import Sample
    import torch
    
    return [
        Sample(
            meta={'cell_id': 'A', 'temperature_C': 25},
            x=torch.randn(10),
            y=torch.tensor([0.95])
        ),
        Sample(
            meta={'cell_id': 'B', 'temperature_C': 40},
            x=torch.randn(10),
            y=torch.tensor([0.90])
        ),
    ]
```

## Test Categories

### Unit Tests

Test individual functions/classes:

```python
def test_feature_extraction():
    """Test feature extraction function."""
    pipeline = SummarySetPipeline()
    row = create_test_row()
    features = pipeline._extract_features(row, 25.0)
    assert len(features) > 0
```

### Integration Tests

Test component interactions:

```python
def test_pipeline_to_model():
    """Test pipeline output works with model."""
    pipeline = SummarySetPipeline()
    samples = pipeline.fit_transform({'df': df})
    
    model = MLPModel(input_dim=samples[0].feature_dim)
    x = torch.stack([s.x for s in samples])
    output = model(x)
    assert output.shape[0] == len(samples)
```

### Regression Tests

Test for bugs that were fixed:

```python
def test_regression_cache_invalidation():
    """Test that cache is invalidated when params change."""
    # This bug was fixed in PR #123
    pipeline1 = ICAPeaksPipeline(sg_window=51)
    pipeline2 = ICAPeaksPipeline(sg_window=53)
    
    # Should use different cache keys
    assert pipeline1.get_params() != pipeline2.get_params()
```

## Best Practices

1. **Test Names**: Use descriptive names (`test_what_when_then`)
2. **One Assertion**: One concept per test (when possible)
3. **Independent Tests**: Tests should not depend on each other
4. **Fast Tests**: Keep tests fast (use fixtures, mock expensive operations)
5. **Coverage**: Aim for high coverage of core functionality

## Mocking

Mock external dependencies:

```python
from unittest.mock import patch, MagicMock

@patch('src.pipelines.cache.get_cache')
def test_pipeline_with_cache(mock_cache):
    """Test pipeline with mocked cache."""
    mock_cache.return_value.get_or_compute.return_value = cached_result
    
    pipeline = ICAPeaksPipeline(use_cache=True)
    # Test pipeline behavior with cache
```

## Continuous Integration

Tests should run in CI:

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
      - run: pip install -r requirements.txt
      - run: pytest tests/ --cov=src
```

## Next Steps

- [Contributing Overview](overview.md) - Contribution workflow
- [Code Structure](code-structure.md) - Codebase organization
