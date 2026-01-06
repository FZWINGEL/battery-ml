# Models API Reference

The `models` module contains the model zoo, including gradient boosting (`LGBMModel`), deep learning (`MLPModel`, `LSTMAttentionModel`), and continuous-time models (`NeuralODEModel`). All models follow a consistent interface for fitting and predicting `Sample` objects.

## Usage Example

```python
from src.models.neural_ode import NeuralODEModel

model = NeuralODEModel(input_dim=5, latent_dim=32, solver='dopri5')
# Trainer will handle the forward/backward passes
```

## Special Models

### ACLA (Attention-CNN-LSTM-ANODE)

The `ACLAModel` is a hybrid architecture that integrates attention for feature weighting, CNN-LSTM for hierarchical temporal feature extraction, and Augmented Neural ODEs for continuous-time degradation modeling.

For a deep dive into the theory, see the [ACLA Model Theory](../theory/acla-model.md) page.

::: src.models.base
    options:
      show_root_heading: true
      show_source: true

::: src.models.registry
    options:
      show_root_heading: true
      show_source: true

::: src.models.lgbm
    options:
      show_root_heading: true
      show_source: true

::: src.models.mlp
    options:
      show_root_heading: true
      show_source: true

::: src.models.lstm_attn
    options:
      show_root_heading: true
      show_source: true

::: src.models.neural_ode
    options:
      show_root_heading: true
      show_source: true

::: src.models.acla
    options:
      show_root_heading: true
      show_source: true
