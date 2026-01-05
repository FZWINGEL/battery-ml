# Data Loading

This guide covers how to load data from the LG M50T dataset for use with BatteryML.

## Data Structure

The LG M50T dataset is organized by experiment (1-5), with each experiment containing:

```
Raw Data/
└── Expt N - [Experiment Name]/
    ├── Summary Data/
    │   ├── Performance Summary/
    │   │   └── Cell_[ID]_[Temp]C_PerformanceSummary.csv
    │   └── Ageing Sets Summary/
    │       └── Cell_[ID]_AgeingSetsSummary.csv
    └── Processed Timeseries Data/
        └── 0.1C Voltage Curves/
            └── Cell_[ID]_RPT_[N]_0.1C_Discharge.csv
```

## Loading Summary Data

### Basic Usage

```python
from pathlib import Path
from src.data.tables import SummaryDataLoader

# Initialize loader for Experiment 5
loader = SummaryDataLoader(experiment_id=5, base_path=Path("Raw Data"))

# Load single cell
df = loader.load_performance_summary(cell_id='A', temp_C=25)
```

### Loading Multiple Cells

```python
# Define temperature mapping
temp_map = {
    10: ['A', 'B', 'C'],
    25: ['D', 'E'],
    40: ['F', 'G', 'H']
}

# Load all cells
df = loader.load_all_cells(
    cells=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
    temp_map=temp_map
)
```

### Available Methods

#### `load_performance_summary(cell_id, temp_C)`

Loads Performance Summary CSV with:
- Cumulative throughput (charge/discharge)
- Resistance measurements (0.1s, 10s)
- Capacity measurements
- Cycle counts

**Returns**: DataFrame with normalized units (mAh → Ah) and metadata columns

#### `load_summary_per_cycle(cell_id)`

Loads cycle-level summary with per-cycle metrics.

#### `load_summary_per_set(cell_id)`

Loads ageing set-level summary (one row per RPT measurement).

#### `load_all_cells(cells, temp_map)`

Convenience method to load multiple cells and combine into single DataFrame.

## Unit Normalization

BatteryML automatically normalizes units:

- **Capacity**: mAh → Ah
- **Temperature**: °C → K (Kelvin)
- **Time**: Various formats → consistent units

This ensures consistency across experiments and prevents unit-related bugs.

## Experiment Path Resolution

The `ExperimentPaths` class handles path resolution for different experiment naming conventions:

```python
from src.data.expt_paths import ExperimentPaths

paths = ExperimentPaths(experiment_id=5, base_path=Path("Raw Data"))

# Get paths
perf_summary_path = paths.performance_summary(cell_id='A', temp_C=25)
voltage_curve_path = paths.voltage_curve(cell_id='A', rpt_id=1)
```

### Supported Experiments

- **Experiment 1**: Si-based Degradation
- **Experiment 2**: C-based Degradation
- **Experiment 3**: Cathode Degradation and Li-Plating
- **Experiment 4**: Drive Cycle Aging (Control)
- **Experiment 5**: Standard Cycle Aging (Control)

## Loading Voltage Curves

For ICA analysis, load 0.1C discharge curves:

```python
from src.data.discovery import find_voltage_curves

# Find all voltage curves for a cell
curves = find_voltage_curves(
    experiment_id=5,
    cell_id='A',
    base_path=Path("Raw Data")
)

# Load specific curve
import pandas as pd
curve_df = pd.read_csv(curves[0])  # First RPT
```

## Data Validation

The loader performs basic validation:

- Checks file existence before loading
- Validates required columns
- Handles missing values gracefully
- Logs warnings for data quality issues

## Common Issues

### File Not Found

**Error**: `FileNotFoundError: Performance summary not found`

**Solutions**:
1. Verify data path matches expected structure
2. Check experiment ID is correct (1-5)
3. Verify cell ID and temperature match file naming convention
4. Use `ExperimentPaths` to debug path resolution

### Missing Columns

**Error**: KeyError for expected columns

**Solutions**:
1. Check CSV file structure matches expected format
2. Verify column names match exactly (case-sensitive)
3. Some experiments may have different column names

### Unit Mismatches

**Issue**: Values seem incorrect (e.g., capacity in thousands)

**Solution**: Unit normalization should handle this automatically. Check that `UnitConverter` is being used.

## Example: Complete Data Loading Workflow

```python
from pathlib import Path
from src.data.tables import SummaryDataLoader

# Setup
BASE_PATH = Path("Raw Data")
EXPERIMENT_ID = 5
CELLS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
TEMP_MAP = {
    10: ['A', 'B', 'C'],
    25: ['D', 'E'],
    40: ['F', 'G', 'H']
}

# Load data
loader = SummaryDataLoader(EXPERIMENT_ID, BASE_PATH)
df = loader.load_all_cells(cells=CELLS, temp_map=TEMP_MAP)

# Verify
print(f"Loaded {len(df)} samples")
print(f"Columns: {df.columns.tolist()}")
print(f"Cells: {df['cell_id'].unique()}")
print(f"Temperatures: {df['temperature_C'].unique()}")
```

## Next Steps

- [Pipelines](pipelines.md) - Transform data into features
- [Splits](splits.md) - Split data for training/validation
- [API Reference](../api/data.md) - Complete API documentation
