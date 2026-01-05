# Creating a Custom Model

This guide walks through creating a custom model step-by-step.

## Overview

Custom models allow you to implement new architectures or integrate external models.

## Step 1: Create Model Class

Create a new file `src/models/my_model.py`:

```python
"""Custom model example."""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from .base import BaseModel
from .registry import ModelRegistry


@ModelRegistry.register("my_model")
class MyModel(BaseModel):
    """Custom neural network model.
    
    This model demonstrates:
    - Custom architecture
    - Forward pass implementation
    - Explainability hook
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, 
                 dropout: float = 0.1, output_dim: int = 1):
        """Initialize the model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
            output_dim: Output dimension (default: 1 for SOH)
        """
        super().__init__(input_dim, output_dim)
        
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # Define layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.decoder = nn.Linear(hidden_dim // 2, output_dim)
    
    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None, 
                **kwargs) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input features (batch, features) or (batch, seq_len, features)
            t: Optional time tensor (ignored for this model)
            **kwargs: Additional arguments
        
        Returns:
            Predictions tensor (batch, output_dim)
        """
        # Handle both 2D and 3D inputs
        if len(x.shape) == 3:
            # Sequence input: use mean pooling
            x = x.mean(dim=1)
        
        # Encode
        hidden = self.encoder(x)
        
        # Decode
        output = self.decoder(hidden)
        
        return output
    
    def explain(self, x: torch.Tensor, **kwargs) -> Dict[str, Any]:
        """Return interpretability information.
        
        Args:
            x: Input features
            **kwargs: Additional arguments
        
        Returns:
            Dictionary with explanation data
        """
        # Example: return gradient-based importance
        x.requires_grad_(True)
        output = self.forward(x)
        
        # Compute gradients
        output.backward(torch.ones_like(output))
        gradients = x.grad.abs()
        
        return {
            'gradient_importance': gradients.cpu().numpy(),
            'prediction': output.detach().cpu().numpy(),
        }
```

## Step 2: Register Model

The `@ModelRegistry.register("my_model")` decorator automatically registers your model.

## Step 3: Use Your Model

```python
from src.models.registry import ModelRegistry
from src.training.trainer import Trainer

# Get model from registry
model = ModelRegistry.get("my_model", input_dim=15, hidden_dim=128)

# Use with trainer
trainer = Trainer(model, config, tracker)
history = trainer.fit(train_samples, val_samples)
```

## Step 4: Add Configuration

Create `configs/model/my_model.yaml`:

```yaml
input_dim: 15
hidden_dim: 128
dropout: 0.1
output_dim: 1
```

Use with Hydra:

```bash
python run.py model=my_model
```

## Handling Different Input Types

### Tabular Input (2D)

```python
def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
    # x shape: (batch, features)
    assert len(x.shape) == 2
    # Process directly
    return self.network(x)
```

### Sequence Input (3D)

```python
def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None, 
            **kwargs) -> torch.Tensor:
    # x shape: (batch, seq_len, features)
    if len(x.shape) == 3:
        # Option 1: Mean pooling
        x = x.mean(dim=1)
        # Option 2: Use RNN/Transformer
        # x = self.rnn(x)[0][:, -1, :]
    
    return self.network(x)
```

### Time-Aware Models

```python
def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None, 
            **kwargs) -> torch.Tensor:
    if t is not None:
        # Use time information
        # Concatenate time to features
        t_expanded = t.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        x = torch.cat([x, t_expanded], dim=-1)
    
    return self.network(x)
```

## Best Practices

### 1. Initialize Weights

```python
def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 1):
    super().__init__(input_dim, output_dim)
    
    # Define layers
    self.layers = nn.ModuleList([
        nn.Linear(input_dim, hidden_dim),
        nn.Linear(hidden_dim, output_dim),
    ])
    
    # Initialize weights
    self._initialize_weights()
    
def _initialize_weights(self):
    for layer in self.layers:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
```

### 2. Handle Variable-Length Sequences

```python
def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
            **kwargs) -> torch.Tensor:
    if mask is not None:
        # Mask out padding
        x = x * mask.unsqueeze(-1)
        # Compute lengths
        lengths = mask.sum(dim=1)
        # Use lengths for RNN or attention
        # ...
    
    return self.network(x)
```

### 3. Support Explainability

```python
def explain(self, x: torch.Tensor, **kwargs) -> Dict[str, Any]:
    """Return model-specific explanations."""
    # Example: Attention weights for attention-based models
    # Example: Feature importance for tree-based models
    # Example: Gradients for gradient-based methods
    
    return {
        'attention_weights': attention_weights,
        'feature_importance': importance_scores,
    }
```

### 4. Count Parameters

The base class provides `count_parameters()`, but you can override:

```python
def count_parameters(self) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in self.parameters() if p.requires_grad)
```

## Testing Your Model

Create a test file `tests/test_my_model.py`:

```python
import pytest
import torch
from src.models.registry import ModelRegistry

def test_my_model():
    # Create model
    model = ModelRegistry.get("my_model", input_dim=10, hidden_dim=64)
    
    # Test forward pass
    x = torch.randn(32, 10)  # batch=32, features=10
    output = model(x)
    
    assert output.shape == (32, 1)
    
    # Test explain
    explanation = model.explain(x)
    assert 'gradient_importance' in explanation
```

## Special Cases

### LightGBM-Style Models

LightGBM doesn't inherit from `BaseModel` but follows similar interface:

```python
class MyTreeModel:
    """Tree-based model (doesn't inherit BaseModel)."""
    
    def fit(self, X, y, X_val=None, y_val=None):
        # Training logic
        pass
    
    def predict(self, X):
        # Prediction logic
        return predictions
```

### Sequence-to-Sequence Models

For models that output sequences:

```python
def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
    # Output shape: (batch, seq_len, output_dim)
    output = self.seq2seq(x)
    return output
```

## Next Steps

- [Model API](../api/models.md) - Complete API reference
- [Model Guide](../user-guide/models.md) - Model usage guide
- [Contributing](../contributing/adding-models.md) - More model examples
