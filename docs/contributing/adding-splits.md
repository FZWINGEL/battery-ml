# Adding a New Split Strategy

This guide shows how to add a new data splitting strategy.

## Split Function Template

```python
"""Data split strategies."""

from typing import List, Tuple
from ..pipelines.sample import Sample


def your_split_strategy(samples: List[Sample], 
                        param1: str = "default",
                        param2: int = 10) -> Tuple[List[Sample], List[Sample]]:
    """Your split strategy description.
    
    Brief description of what the split does and when to use it.
    
    Args:
        samples: List of Sample objects
        param1: Description of param1
        param2: Description of param2
    
    Returns:
        Tuple of (train_samples, test_samples)
    
    Example:
        >>> train, test = your_split_strategy(samples, param1="value")
    """
    train = []
    test = []
    
    # Implement your split logic
    for sample in samples:
        # Your splitting criteria
        if some_condition(sample, param1, param2):
            train.append(sample)
        else:
            test.append(sample)
    
    return train, test
```

## Steps to Add Split

### 1. Add Function to splits.py

Add your function to `src/data/splits.py`:

```python
def your_split_strategy(samples: List[Sample], **kwargs) -> Tuple[List[Sample], List[Sample]]:
    """Your split strategy."""
    # Implementation
    pass
```

### 2. Handle Metadata

Ensure samples have required metadata:

```python
def your_split_strategy(samples: List[Sample], 
                        split_key: str) -> Tuple[List[Sample], List[Sample]]:
    """Split based on metadata key."""
    train = []
    test = []
    
    for sample in samples:
        # Access metadata
        value = sample.meta.get(split_key)
        
        if value is None:
            raise ValueError(f"Sample missing required metadata: {split_key}")
        
        # Split logic
        if some_condition(value):
            train.append(sample)
        else:
            test.append(sample)
    
    return train, test
```

### 3. Add Tests

Create `tests/test_splits.py` or add to existing:

```python
import pytest
from src.data.splits import your_split_strategy
from src.pipelines.sample import Sample

def test_your_split_strategy():
    """Test your split strategy."""
    # Create test samples
    samples = [
        Sample(meta={'key': 'value1'}, x=None, y=None),
        Sample(meta={'key': 'value2'}, x=None, y=None),
    ]
    
    # Test split
    train, test = your_split_strategy(samples, param1="value")
    
    assert len(train) > 0
    assert len(test) > 0
    assert len(train) + len(test) == len(samples)
```

### 4. Add Configuration (Optional)

If using Hydra, create `configs/split/your_split.yaml`:

```yaml
strategy: "your_split"
param1: "default"
param2: 10
```

### 5. Update Documentation

- Add to [Splits Guide](../user-guide/splits.md)
- Add example usage
- Document parameters

## Common Split Patterns

### By Metadata Value

```python
def split_by_metadata(samples: List[Sample], 
                      key: str, 
                      train_values: List[Any]) -> Tuple[List[Sample], List[Sample]]:
    """Split by metadata value."""
    train = [s for s in samples if s.meta.get(key) in train_values]
    test = [s for s in samples if s.meta.get(key) not in train_values]
    return train, test
```

### By Percentage

```python
def split_by_percentage(samples: List[Sample], 
                        train_fraction: float = 0.8) -> Tuple[List[Sample], List[Sample]]:
    """Split by percentage."""
    n_train = int(len(samples) * train_fraction)
    train = samples[:n_train]
    test = samples[n_train:]
    return train, test
```

### By Index Range

```python
def split_by_index(samples: List[Sample], 
                   train_indices: List[int]) -> Tuple[List[Sample], List[Sample]]:
    """Split by sample indices."""
    train = [samples[i] for i in train_indices]
    test = [samples[i] for i in range(len(samples)) if i not in train_indices]
    return train, test
```

## Best Practices

1. **Validate Inputs**: Check samples have required metadata
2. **Handle Edge Cases**: Empty splits, single sample, etc.
3. **Document Clearly**: Explain split logic and use cases
4. **Return Consistent**: Always return (train, test) tuple
5. **Add Tests**: Test various scenarios

## Checklist

- [ ] Split function added to `splits.py`
- [ ] Tests written and passing
- [ ] Documentation updated
- [ ] Handles edge cases
- [ ] Code follows style guidelines

## Next Steps

- [Splits Guide](../user-guide/splits.md) - Split usage guide
- [Testing](testing.md) - Testing guidelines
- [Code Structure](code-structure.md) - Codebase organization
