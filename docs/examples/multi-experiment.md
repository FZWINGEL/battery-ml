# Working with Multiple Experiments

This guide shows how to work with multiple experiments (1-5) in the LG M50T dataset.

## Overview

The LG M50T dataset contains 5 experiments with different degradation mechanisms:
- **Experiment 1**: Si-based Degradation
- **Experiment 2**: C-based Degradation
- **Experiment 3**: Cathode Degradation and Li-Plating
- **Experiment 4**: Drive Cycle Aging (Control)
- **Experiment 5**: Standard Cycle Aging (Control)

## Loading Multiple Experiments

### Individual Experiment Loading

```python
from pathlib import Path
from src.data.tables import SummaryDataLoader

BASE_PATH = Path("Raw Data")

# Load Experiment 5
loader5 = SummaryDataLoader(experiment_id=5, base_path=BASE_PATH)
df5 = loader5.load_all_cells(
    cells=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
    temp_map={10: ['A', 'B', 'C'], 25: ['D', 'E'], 40: ['F', 'G', 'H']}
)

# Load Experiment 1
loader1 = SummaryDataLoader(experiment_id=1, base_path=BASE_PATH)
df1 = loader1.load_all_cells(
    cells=['A', 'B', 'C'],  # Experiment-specific cells
    temp_map={25: ['A', 'B', 'C']}
)
```

### Combining Experiments

```python
import pandas as pd

# Combine multiple experiments
all_data = pd.concat([df1, df5], ignore_index=True)

# Verify
print(f"Total samples: {len(all_data)}")
print(f"Experiments: {all_data['experiment_id'].unique()}")
```

## Experiment-Specific Configurations

### Create Config Files

```yaml
# configs/data/expt1.yaml
experiment_id: 1
base_path: "Raw Data"
cells: ['A', 'B', 'C']
temp_map:
  25: ['A', 'B', 'C']
```

```yaml
# configs/data/expt5.yaml
experiment_id: 5
base_path: "Raw Data"
cells: ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
temp_map:
  10: ['A', 'B', 'C']
  25: ['D', 'E']
  40: ['F', 'G', 'H']
```

### Switch Experiments

```bash
# Run with Experiment 1
python run.py data=expt1

# Run with Experiment 5
python run.py data=expt5
```

## Cross-Experiment Analysis

### Train on One, Test on Another

```python
# Train on Experiment 5
train_loader = SummaryDataLoader(5, BASE_PATH)
train_df = train_loader.load_all_cells(...)

# Test on Experiment 1
test_loader = SummaryDataLoader(1, BASE_PATH)
test_df = test_loader.load_all_cells(...)

# Create pipeline
pipeline = SummarySetPipeline()
train_samples = pipeline.fit_transform({'df': train_df})
test_samples = pipeline.transform({'df': test_df})

# Train and evaluate
model = train_model(train_samples)
metrics = evaluate_model(model, test_samples)
```

### Combined Training

```python
# Load multiple experiments
loaders = [
    SummaryDataLoader(1, BASE_PATH),
    SummaryDataLoader(5, BASE_PATH),
]

all_dfs = []
for loader in loaders:
    df = loader.load_all_cells(...)
    all_dfs.append(df)

# Combine
combined_df = pd.concat(all_dfs, ignore_index=True)

# Train on combined data
pipeline = SummarySetPipeline()
samples = pipeline.fit_transform({'df': combined_df})
```

## Experiment Metadata

### Access Experiment Information

```python
# Each sample has experiment_id in metadata
for sample in samples:
    exp_id = sample.meta['experiment_id']
    cell_id = sample.meta['cell_id']
    temp = sample.meta['temperature_C']
    
    print(f"Exp {exp_id}, Cell {cell_id}, {temp}°C")
```

### Filter by Experiment

```python
# Filter samples by experiment
exp5_samples = [s for s in samples if s.meta['experiment_id'] == 5]
exp1_samples = [s for s in samples if s.meta['experiment_id'] == 1]
```

## Handling Different Structures

Different experiments may have:
- Different cell counts
- Different temperature conditions
- Different column names

### Robust Loading

```python
def load_experiment_robust(experiment_id: int, base_path: Path):
    """Load experiment with error handling."""
    try:
        loader = SummaryDataLoader(experiment_id, base_path)
        
        # Get available cells (may vary by experiment)
        # Check metadata or file system
        available_cells = discover_available_cells(experiment_id, base_path)
        
        df = loader.load_all_cells(cells=available_cells, ...)
        return df
    except FileNotFoundError as e:
        print(f"Experiment {experiment_id} not found: {e}")
        return None
```

## Best Practices

1. **Check Data Availability**: Verify experiments exist before loading
2. **Handle Missing Columns**: Different experiments may have different columns
3. **Normalize Units**: Ensure consistent units across experiments
4. **Document Differences**: Note experiment-specific characteristics
5. **Validate Metadata**: Ensure experiment_id is correctly set

## Example: Multi-Experiment Workflow

```python
from pathlib import Path
from src.data.tables import SummaryDataLoader
from src.pipelines.summary_set import SummarySetPipeline
from src.data.splits import temperature_split

BASE_PATH = Path("Raw Data")
EXPERIMENTS = [1, 5]

# Load all experiments
all_samples = []
for exp_id in EXPERIMENTS:
    loader = SummaryDataLoader(exp_id, BASE_PATH)
    df = loader.load_all_cells(...)
    
    pipeline = SummarySetPipeline()
    samples = pipeline.fit_transform({'df': df})
    all_samples.extend(samples)

# Split across experiments
train_samples = [s for s in all_samples if s.meta['experiment_id'] == 5]
test_samples = [s for s in all_samples if s.meta['experiment_id'] == 1]

# Train and evaluate
model = train_model(train_samples)
metrics = evaluate_model(model, test_samples)
```

## Next Steps

- [Data Loading](../user-guide/data-loading.md) - Data loading guide
- [Pipelines](../user-guide/pipelines.md) - Pipeline usage
- [API Reference](../api/data.md) - Complete API docs
