# Data Flow

This document describes the complete data flow from raw CSV files to model predictions.

## High-Level Flow

```mermaid
flowchart TD
    Start[Raw CSV Files] --> Load[Data Loader]
    Load --> Normalize[Unit Normalization]
    Normalize --> Split[Data Splitting]
    Split --> Pipeline[Feature Pipeline]
    Pipeline --> Cache{Cache Check}
    Cache -->|Hit| LoadCache[Load from Cache]
    Cache -->|Miss| Compute[Compute Features]
    Compute --> SaveCache[Save to Cache]
    LoadCache --> Sample[Sample Objects]
    SaveCache --> Sample
    Sample --> Model[Model Training]
    Model --> Predict[Predictions]
    Predict --> Metrics[Evaluation Metrics]
    Metrics --> Track[Experiment Tracking]
```

## Detailed Data Flow

### 1. Data Loading

```mermaid
sequenceDiagram
    participant User
    participant ExperimentPaths
    participant SummaryDataLoader
    participant UnitConverter
    participant DataFrame
    
    User->>ExperimentPaths: Get paths for experiment
    ExperimentPaths-->>User: File paths
    User->>SummaryDataLoader: Load CSV
    SummaryDataLoader->>DataFrame: Read CSV
    SummaryDataLoader->>UnitConverter: Normalize units
    UnitConverter-->>SummaryDataLoader: Normalized DataFrame
    SummaryDataLoader-->>User: DataFrame with metadata
```

### 2. Feature Extraction

```mermaid
sequenceDiagram
    participant User
    participant Pipeline
    participant Cache
    participant FeatureExtractor
    participant Sample
    
    User->>Pipeline: fit_transform(df)
    Pipeline->>Cache: Check cache key
    Cache-->>Pipeline: Cache miss
    Pipeline->>FeatureExtractor: Extract features
    FeatureExtractor-->>Pipeline: Feature arrays
    Pipeline->>Sample: Create Sample objects
    Sample-->>User: List of Samples
    Pipeline->>Cache: Save to cache
```

### 3. Model Training

```mermaid
sequenceDiagram
    participant User
    participant Trainer
    participant DataLoader
    participant Model
    participant Optimizer
    participant Tracker
    
    User->>Trainer: fit(train_samples, val_samples)
    Trainer->>DataLoader: Create DataLoader
    loop Each Epoch
        DataLoader->>Model: Batch of Samples
        Model->>Model: Forward pass
        Model-->>Trainer: Predictions
        Trainer->>Optimizer: Backward pass
        Optimizer->>Model: Update weights
        Trainer->>Tracker: Log metrics
    end
    Trainer-->>User: Training history
```

## Sample Object Flow

```mermaid
graph TB
    RawData[Raw Data] --> Pipeline[Pipeline]
    Pipeline --> Sample1[Sample: meta, x, y]
    Sample1 --> Split[Split Strategy]
    Split --> TrainSamples[Train Samples]
    Split --> ValSamples[Val Samples]
    TrainSamples --> Model[Model]
    ValSamples --> Model
    Model --> Predictions[Predictions]
```

## Cache Flow

```mermaid
flowchart TD
    Request[Pipeline Request] --> Hash[Hash Parameters]
    Hash --> Key[Cache Key]
    Key --> Check{File Exists?}
    Check -->|Yes| Load[Load Pickle]
    Check -->|No| Compute[Compute Features]
    Load --> Validate{Valid?}
    Validate -->|Yes| Return[Return Cached]
    Validate -->|No| Compute
    Compute --> Save[Save Pickle]
    Save --> Return
```

## Model Inference Flow

```mermaid
sequenceDiagram
    participant User
    participant Sample
    participant Model
    participant Forward[Forward Pass]
    participant Output
    
    User->>Sample: Get sample
    Sample->>Model: x, t (optional)
    Model->>Forward: forward(x, t)
    Forward->>Output: Predictions
    Output-->>User: y_pred
```

## Batch Processing

```mermaid
graph TB
    Samples[List of Samples] --> Batch[Create Batches]
    Batch --> Collate[Collate Function]
    Collate --> Tensor[Batch Tensor]
    Tensor --> Model[Model Forward]
    Model --> Loss[Compute Loss]
    Loss --> Backward[Backward Pass]
    Backward --> Update[Update Weights]
```

## Tracking Flow

```mermaid
sequenceDiagram
    participant Trainer
    participant DualTracker
    participant LocalTracker
    participant MLflowTracker
    participant TensorBoard
    participant MLflowServer
    
    Trainer->>DualTracker: Log metrics
    DualTracker->>LocalTracker: Log to local
    DualTracker->>MLflowTracker: Log to MLflow
    LocalTracker->>TensorBoard: Write logs
    MLflowTracker->>MLflowServer: HTTP request
```

## Error Handling Flow

```mermaid
flowchart TD
    Operation[Operation] --> Try{Try}
    Try -->|Success| Return[Return Result]
    Try -->|Error| Catch[Catch Exception]
    Catch --> Log[Log Error]
    Log --> Handle{Handle?}
    Handle -->|Yes| Retry[Retry or Fallback]
    Handle -->|No| Raise[Raise Exception]
    Retry --> Operation
```

## Next Steps

- [Design Patterns](design-patterns.md) - Design pattern details
- [Pipeline System](pipeline-system.md) - Pipeline internals
- [Model System](model-system.md) - Model internals
