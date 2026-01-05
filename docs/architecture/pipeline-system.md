# Pipeline System

The pipeline system transforms raw data into `Sample` objects that models can consume.

## Pipeline Architecture

```mermaid
graph TB
    RawData[Raw Data] --> BasePipeline[BasePipeline]
    BasePipeline -->|fit| FitState[Fitted State]
    BasePipeline -->|transform| Sample[Sample Objects]
    FitState -->|used by| Transform
    Registry[PipelineRegistry] -->|manages| BasePipeline
    Cache[PipelineCache] -->|caches| BasePipeline
```

## Pipeline Interface

All pipelines inherit from `BasePipeline`:

```mermaid
classDiagram
    class BasePipeline {
        +fit(data) BasePipeline
        +transform(data) List[Sample]
        +fit_transform(data) List[Sample]
        +get_feature_names() List[str]
        +get_params() dict
    }
    class SummarySetPipeline {
        +include_arrhenius: bool
        +arrhenius_Ea: float
        +normalize: bool
    }
    class ICAPeaksPipeline {
        +sg_window: int
        +num_peaks: int
        +use_cache: bool
    }
    class LatentODESequencePipeline {
        +time_unit: str
        +max_seq_len: int
    }
    
    BasePipeline <|-- SummarySetPipeline
    BasePipeline <|-- ICAPeaksPipeline
    BasePipeline <|-- LatentODESequencePipeline
```

## Pipeline Lifecycle

```mermaid
sequenceDiagram
    participant User
    participant Pipeline
    participant Scaler
    participant Cache
    participant Sample
    
    User->>Pipeline: fit_transform(train_data)
    Pipeline->>Scaler: fit(train_data)
    Scaler-->>Pipeline: Fitted scaler
    Pipeline->>Cache: Check cache
    Cache-->>Pipeline: Cache miss
    Pipeline->>Sample: Create samples
    Sample-->>Pipeline: Sample objects
    Pipeline->>Cache: Save to cache
    Pipeline-->>User: Samples
    
    User->>Pipeline: transform(test_data)
    Pipeline->>Scaler: transform(test_data)
    Scaler-->>Pipeline: Scaled data
    Pipeline->>Sample: Create samples
    Sample-->>User: Samples
```

## Sample Creation

```mermaid
flowchart TD
    Row[DataFrame Row] --> Extract[Extract Features]
    Extract --> Features[Feature Array]
    Features --> Normalize{Normalize?}
    Normalize -->|Yes| Scaler[Apply Scaler]
    Normalize -->|No| Sample
    Scaler --> Sample[Create Sample]
    Sample --> Meta[Add Metadata]
    Meta --> Return[Return Sample]
```

## Caching Strategy

```mermaid
flowchart TD
    Request[Pipeline Request] --> Params[Get Parameters]
    Params --> Hash[Hash Parameters]
    Hash --> Key[Cache Key]
    Key --> Check{File Exists?}
    Check -->|Yes| Load[Load Pickle]
    Check -->|No| Compute[Compute]
    Load --> Validate{Valid?}
    Validate -->|Yes| Return[Return]
    Validate -->|No| Compute
    Compute --> Save[Save Pickle]
    Save --> Return
```

## Feature Extraction Flow

### SummarySetPipeline

```mermaid
flowchart TD
    DataFrame[DataFrame] --> Iterate[Iterate Rows]
    Iterate --> Extract[Extract Features]
    Extract --> Throughput[Cumulative Throughput]
    Extract --> Resistance[Resistance Values]
    Extract --> Temperature[Temperature]
    Throughput --> Combine[Combine Features]
    Resistance --> Combine
    Temperature --> Combine
    Combine --> Arrhenius{Include Arrhenius?}
    Arrhenius -->|Yes| ComputeArrhenius[Compute Arrhenius Factor]
    Arrhenius -->|No| Normalize
    ComputeArrhenius --> Normalize[Normalize Features]
    Normalize --> Sample[Create Sample]
```

### ICAPeaksPipeline

```mermaid
flowchart TD
    Curves[Voltage Curves] --> Cache{Check Cache}
    Cache -->|Hit| Load[Load Cached]
    Cache -->|Miss| ComputeICA[Compute ICA]
    ComputeICA --> Smooth[Savitzky-Golay Smoothing]
    Smooth --> Peaks[Find Peaks]
    Peaks --> Extract[Extract Peak Features]
    Extract --> Save[Save to Cache]
    Load --> Features[Feature Vector]
    Save --> Features
    Features --> Normalize[Normalize]
    Normalize --> Sample[Create Sample]
```

## Pipeline Registration

```mermaid
sequenceDiagram
    participant Developer
    participant PipelineClass
    participant Registry
    participant User
    
    Developer->>PipelineClass: @PipelineRegistry.register("name")
    PipelineClass->>Registry: Register class
    Registry->>Registry: Store in _pipelines dict
    User->>Registry: get("name", **kwargs)
    Registry->>PipelineClass: Instantiate
    PipelineClass-->>User: Pipeline instance
```

## Error Handling

```mermaid
flowchart TD
    Transform[Transform] --> Validate[Validate Input]
    Validate -->|Invalid| Error[Raise Error]
    Validate -->|Valid| Process[Process Data]
    Process --> Check{Check Errors}
    Check -->|Error| Handle[Handle Error]
    Check -->|OK| Return[Return Samples]
    Handle -->|Recoverable| Retry[Retry]
    Handle -->|Fatal| Error
    Retry --> Process
```

## Next Steps

- [Model System](model-system.md) - Model architecture
- [Custom Pipeline](../examples/custom-pipeline.md) - Create custom pipeline
- [API Reference](../api/pipelines.md) - Complete API docs
