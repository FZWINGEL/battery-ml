# Architecture Overview

BatteryML is designed with modularity, extensibility, and reproducibility in mind. This document provides a high-level overview of the system architecture.

## System Architecture

```mermaid
graph TB
    subgraph DataLayer[Data Layer]
        RawData[Raw CSV Files]
        DataLoader[Data Loaders]
        Splits[Split Strategies]
    end
    
    subgraph PipelineLayer[Pipeline Layer]
        Pipelines[Feature Pipelines]
        Cache[Hash-Based Cache]
        Sample[Sample Objects]
    end
    
    subgraph ModelLayer[Model Layer]
        Models[ML Models]
        Registry[Model Registry]
    end
    
    subgraph TrainingLayer[Training Layer]
        Trainer[Trainer]
        Metrics[Metrics]
        Callbacks[Callbacks]
    end
    
    subgraph TrackingLayer[Tracking Layer]
        LocalTracker[Local Tracker]
        MLflowTracker[MLflow Tracker]
        DualTracker[Dual Tracker]
    end
    
    RawData --> DataLoader
    DataLoader --> Splits
    Splits --> Pipelines
    Pipelines --> Cache
    Cache --> Sample
    Sample --> Models
    Models --> Registry
    Models --> Trainer
    Trainer --> Metrics
    Trainer --> Callbacks
    Trainer --> LocalTracker
    Trainer --> MLflowTracker
    LocalTracker --> DualTracker
    MLflowTracker --> DualTracker
```

## Component Overview

### Data Layer

- **Data Loaders**: Load CSV files from experiments
- **Unit Conversion**: Normalize units (mAh → Ah, °C → K)
- **Split Strategies**: Temperature holdout, LOCO, temporal splits

### Pipeline Layer

- **Feature Extraction**: Transform raw data to features
- **Caching**: Cache expensive computations (ICA)
- **Sample Schema**: Universal data format

### Model Layer

- **Model Zoo**: LightGBM, MLP, LSTM, Neural ODE
- **Registry Pattern**: Extensible model registration
- **Base Interface**: Consistent model API

### Training Layer

- **Trainer**: Training loop with AMP, early stopping
- **Metrics**: RMSE, MAE, MAPE, R²
- **Callbacks**: Checkpointing, scheduling

### Tracking Layer

- **Local**: JSON + TensorBoard
- **MLflow**: Experiment management
- **Dual**: Combined tracking

## Design Principles

1. **Modularity**: Components are independent and composable
2. **Extensibility**: Easy to add new pipelines/models
3. **Reproducibility**: Hash-based caching, config management
4. **Type Safety**: Pydantic validation, type hints
5. **Documentation**: Comprehensive docstrings and guides

## Data Flow

```mermaid
sequenceDiagram
    participant User
    participant DataLoader
    participant Pipeline
    participant Cache
    participant Model
    participant Trainer
    participant Tracker
    
    User->>DataLoader: Load CSV files
    DataLoader->>Pipeline: Raw DataFrame
    Pipeline->>Cache: Check cache
    Cache-->>Pipeline: Cached or compute
    Pipeline->>User: Sample objects
    User->>Model: Initialize model
    User->>Trainer: Create trainer
    Trainer->>Model: Forward pass
    Model-->>Trainer: Predictions
    Trainer->>Tracker: Log metrics
    Tracker-->>User: Results
```

## Key Design Patterns

### Registry Pattern

The Registry Pattern decoupled the configuration from the implementation, allows for runtime discovery and instantiation of components.

```mermaid
sequenceDiagram
    participant Config as Experiment Config
    participant Reg as Component Registry
    participant Base as Base Class
    participant Conc as Concrete Implementation
    
    Conc->>Reg: @Registry.register("name")
    Config->>Reg: Registry.get("name", params)
    Reg->>Conc: Instantiate(**params)
    Conc-->>Reg: Component Instance
    Reg-->>Config: Instance
```

### Sample Schema

```mermaid
graph TB
    RawData[Raw Data] --> Pipeline[Pipeline]
    Pipeline --> Sample[Sample Object]
    Sample --> Model[Model]
    Sample --> Split[Split Strategy]
    Sample --> Tracker[Tracker]
```

### Caching Strategy

```mermaid
graph TB
    Request[Pipeline Request] --> Check{Check Cache}
    Check -->|Hit| Load[Load from Cache]
    Check -->|Miss| Compute[Compute Features]
    Compute --> Save[Save to Cache]
    Load --> Return[Return Samples]
    Save --> Return
```

## Extension Points

The architecture provides several extension points:

1. **Pipelines**: Add new feature extraction methods
2. **Models**: Add new model architectures
3. **Splits**: Add new data splitting strategies
4. **Trackers**: Add new tracking backends
5. **Metrics**: Add new evaluation metrics

## Next Steps

- [Data Flow](data-flow.md) - Detailed data flow documentation
- [Design Patterns](design-patterns.md) - Design pattern deep dive
- [Pipeline System](pipeline-system.md) - Pipeline architecture
- [Model System](model-system.md) - Model architecture
