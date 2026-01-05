# Adding a New Model

This guide shows how to add a new model to BatteryML.

## Model Template

```python
"""Your model description."""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from .base import BaseModel
from .registry import ModelRegistry


@ModelRegistry.register("your_model_name")
class YourModel(BaseModel):
    """Brief description of your model.
    
    Longer description explaining the architecture
    and when to use it.
    
    Example usage:
        >>> model = YourModel(input_dim=10, hidden_dim=64)
        >>> output = model(x)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, 
                 output_dim: int = 1, **kwargs):
        """Initialize the model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (default: 1 for SOH)
            **kwargs: Additional arguments
        """
        super().__init__(input_dim, output_dim)
        
        self.hidden_dim = hidden_dim
        
        # Define your architecture
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        ])
    
    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None, 
                **kwargs) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input features (batch, features) or (batch, seq_len, features)
            t: Optional time tensor (for time-aware models)
            **kwargs: Additional arguments
        
        Returns:
            Predictions tensor (batch, output_dim)
        """
        # Handle both 2D and 3D inputs
        if len(x.shape) == 3:
            # Sequence input: use mean pooling or RNN
            x = x.mean(dim=1)
        
        # Forward through layers
        for layer in self.layers:
            x = layer(x)
        
        return x
    
    def explain(self, x: torch.Tensor, **kwargs) -> Dict[str, Any]:
        """Return interpretability information.
        
        Args:
            x: Input features
            **kwargs: Additional arguments
        
        Returns:
            Dictionary with explanation data
        """
        # Implement model-specific explainability
        # Example: gradients, attention weights, etc.
        return {}
```

## Steps to Add Model

### 1. Create Model File

Create `src/models/your_model.py` with the template above.

### 2. Implement Methods

- **`__init__`**: Initialize architecture
- **`forward`**: Forward pass implementation
- **`explain`** (optional): Model-specific explainability

### 3. Handle Input Types

#### Tabular Input (2D)

```python
def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
    # x shape: (batch, features)
    assert len(x.shape) == 2
    # Process directly
    return self.network(x)
```

#### Sequence Input (3D)

```python
def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
    # x shape: (batch, seq_len, features)
    if len(x.shape) == 3:
        # Option 1: Mean pooling
        x = x.mean(dim=1)
        # Option 2: Use RNN/Transformer
        # x = self.rnn(x)[0][:, -1, :]
    
    return self.network(x)
```

#### Time-Aware Models

```python
def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None, 
            **kwargs) -> torch.Tensor:
    if t is not None:
        # Use time information
        # Concatenate time to features or use in ODE
        pass
    
    return self.network(x)
```

### 4. Add Tests

Create `tests/test_your_model.py`:

```python
import pytest
import torch
from src.models.registry import ModelRegistry

def test_your_model():
    """Test your model."""
    model = ModelRegistry.get("your_model_name", input_dim=10, hidden_dim=64)
    
    # Test forward pass
    x = torch.randn(32, 10)  # batch=32, features=10
    output = model(x)
    
    assert output.shape == (32, 1)
    
    # Test explain
    explanation = model.explain(x)
    assert isinstance(explanation, dict)
```

### 5. Add Configuration

Create `configs/model/your_model.yaml`:

```yaml
input_dim: 15
hidden_dim: 64
output_dim: 1
```

### 6. Update Documentation

- Add to [Models Guide](../user-guide/models.md)
- Add example to [Custom Model](../examples/custom-model.md)
- Ensure docstrings are complete

## Special Cases

### Tree-Based Models (LightGBM-style)

Tree-based models don't inherit from `BaseModel`:

```python
class YourTreeModel:
    """Tree-based model."""
    
    def fit(self, X, y, X_val=None, y_val=None):
        """Train the model."""
        # Training logic
        pass
    
    def predict(self, X):
        """Make predictions."""
        # Prediction logic
        return predictions
    
    @property
    def feature_importances_(self):
        """Feature importances."""
        return importances
```

### Sequence-to-Sequence Models

For models that output sequences:

```python
def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
    # Output shape: (batch, seq_len, output_dim)
    output = self.seq2seq(x)
    return output
```

## Best Practices

1. **Initialize Weights**: Use proper weight initialization
2. **Handle Variable Length**: Support masking for variable-length sequences
3. **Support Explainability**: Implement `explain` method
4. **Document Architecture**: Clearly document model architecture
5. **Add Tests**: Write comprehensive tests

## Checklist

- [ ] Model file created
- [ ] Registered with `@ModelRegistry.register`
- [ ] Tests written and passing
- [ ] Configuration file added
- [ ] Documentation updated
- [ ] Code follows style guidelines

## Next Steps

- [Model API](../api/models.md) - Complete API reference
- [Testing](testing.md) - Testing guidelines
- [Code Structure](code-structure.md) - Codebase organization
