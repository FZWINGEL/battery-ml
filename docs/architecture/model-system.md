# Model System

The model system provides a unified interface for different model architectures.

## Model Architecture

```mermaid
graph TB
    Sample[Sample Object] --> BaseModel[BaseModel]
    BaseModel -->|forward| Predictions[Predictions]
    BaseModel -->|explain| Explanations[Explanations]
    Registry[ModelRegistry] -->|manages| BaseModel
    Trainer[Trainer] -->|uses| BaseModel
```

## Model Hierarchy

```mermaid
classDiagram
    class BaseModel {
        +forward(x, t) Tensor
        +predict(x) ndarray
        +explain(x) dict
        +count_parameters() int
    }
    class LGBMModel {
        +fit(X, y) None
        +predict(X) ndarray
        +feature_importances_ array
    }
    class MLPModel {
        +hidden_dims: List[int]
        +dropout: float
    }
    class LSTMAttentionModel {
        +hidden_dim: int
        +num_heads: int
    }
    class NeuralODEModel {
        +latent_dim: int
        +solver: str
        +use_adjoint: bool
    }
    
    BaseModel <|-- MLPModel
    BaseModel <|-- LSTMAttentionModel
    BaseModel <|-- NeuralODEModel
    note for LGBMModel "Special case: doesn't inherit BaseModel"
```

## Forward Pass Flow

### Tabular Models (MLP)

```mermaid
flowchart TD
    Input[Sample.x: batch, features] --> Linear1[Linear Layer 1]
    Linear1 --> Activation1[ReLU]
    Activation1 --> Dropout1[Dropout]
    Dropout1 --> Linear2[Linear Layer 2]
    Linear2 --> Activation2[ReLU]
    Activation2 --> Dropout2[Dropout]
    Dropout2 --> Output[Output Layer]
    Output --> Predictions[Predictions: batch, 1]
```

### Sequence Models (LSTM)

```mermaid
flowchart TD
    Input[Sample.x: batch, seq_len, features] --> LSTM[BiLSTM Layers]
    LSTM --> Hidden[Hidden States]
    Hidden --> Attention[Self-Attention]
    Attention --> Weights[Attention Weights]
    Weights --> Context[Context Vector]
    Context --> Output[Output Layer]
    Output --> Predictions[Predictions: batch, 1]
```

### ODE Models (Neural ODE)

```mermaid
flowchart TD
    Input[Sample.x: batch, seq_len, features] --> Encoder[Encoder]
    Encoder --> Latent[Latent State]
    Latent --> ODEFunc[ODE Function]
    ODEFunc --> Solver[ODE Solver]
    Solver --> Trajectory[Trajectory]
    Trajectory --> Decoder[Decoder]
    Decoder --> Predictions[Predictions: batch, 1]
    Sample.t --> Solver
```

## Training Flow

```mermaid
sequenceDiagram
    participant Trainer
    participant Model
    participant Loss
    participant Optimizer
    participant Scheduler
    
    Trainer->>Model: forward(x)
    Model-->>Trainer: predictions
    Trainer->>Loss: compute(predictions, targets)
    Loss-->>Trainer: loss
    Trainer->>Optimizer: backward(loss)
    Optimizer->>Model: update weights
    Trainer->>Scheduler: step()
    Scheduler->>Optimizer: update lr
```

## Model Registration

```mermaid
sequenceDiagram
    participant Developer
    participant ModelClass
    participant Registry
    participant User
    
    Developer->>ModelClass: @ModelRegistry.register("name")
    ModelClass->>Registry: Register class
    Registry->>Registry: Store in _models dict
    User->>Registry: get("name", **kwargs)
    Registry->>ModelClass: Instantiate
    ModelClass-->>User: Model instance
```

## Model Inference

```mermaid
flowchart TD
    Sample[Sample Object] --> Extract[Extract x, t]
    Extract --> Batch[Create Batch]
    Batch --> Model[Model Forward]
    Model --> Predictions[Predictions]
    Predictions --> PostProcess[Post-process]
    PostProcess --> Return[Return Results]
```

## Explainability Flow

```mermaid
flowchart TD
    Model[Model] --> Explain[explain method]
    Explain --> Type{Model Type?}
    Type -->|LightGBM| SHAP[SHAP Values]
    Type -->|LSTM| Attention[Attention Weights]
    Type -->|MLP| Gradient[Gradient-based]
    Type -->|ODE| Trajectory[Trajectory Analysis]
    SHAP --> Visualize[Visualize]
    Attention --> Visualize
    Gradient --> Visualize
    Trajectory --> Visualize
```

## Model Selection

```mermaid
graph TB
    DataType[Data Type] --> Decision{Tabular or Sequence?}
    Decision -->|Tabular| TabularModels[Tabular Models]
    Decision -->|Sequence| SequenceModels[Sequence Models]
    TabularModels --> Fast{Need Speed?}
    Fast -->|Yes| LightGBM[LightGBM]
    Fast -->|No| MLP[MLP]
    SequenceModels --> Physics{Physics-aware?}
    Physics -->|Yes| NeuralODE[Neural ODE]
    Physics -->|No| LSTM[LSTM + Attention]
```

## Next Steps

- [Models Guide](../user-guide/models.md) - Model usage guide
- [Custom Model](../examples/custom-model.md) - Create custom model
- [API Reference](../api/models.md) - Complete API docs
