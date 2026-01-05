# Quick Start

This guide will walk you through running your first experiment with BatteryML in under 5 minutes.

## Prerequisites

- BatteryML installed (see [Installation](installation.md))
- LG M50T dataset available in `Raw Data/` directory

## Step 1: Verify Data Location

Ensure your data is structured as follows:

```
Raw Data/
└── Expt 5 - Standard Cycle Aging (Control)/
    ├── Summary Data/
    │   ├── Performance Summary/
    │   └── Ageing Sets Summary/
    └── Processed Timeseries Data/
        └── 0.1C Voltage Curves/
```

## Step 2: Run Milestone A

Milestone A demonstrates a complete end-to-end workflow with a LightGBM baseline:

```bash
python examples/milestone_a.py
```

### Expected Output

```
============================================================
Milestone A: LGBM Baseline on Summary Data
============================================================

[1/6] Loading data...
  ✓ Loaded 168 samples from 8 cells

[2/6] Creating feature pipeline...
  ✓ Pipeline: SummarySetPipeline(include_arrhenius=True, ...)

[3/6] Transforming data to samples...
  ✓ Created 168 samples
  ✓ Feature dimension: 15

[4/6] Splitting by temperature...
  ✓ Train: 126 samples (10°C + 40°C)
  ✓ Val:   42 samples (25°C)

[5/6] Training LightGBM model...
  ✓ Model trained: LGBMModel(...)

[6/6] Evaluating...

========================================
Results (25°C Holdout)
========================================
  RMSE: 0.03425
  MAE:  0.02891
  MAPE: 0.61%
  R²:   0.9847

========================================
Top Features
========================================
  cumulative_throughput_Ah: 245.3
  temperature_K: 189.2
  ...
```

## Step 3: View Results

### TensorBoard

View training curves and metrics:

```bash
tensorboard --logdir artifacts/runs
```

Open your browser to `http://localhost:6006`

### MLflow

Compare experiments:

```bash
mlflow ui --backend-store-uri file:./artifacts/mlruns
```

Open your browser to `http://localhost:5000`

## Understanding the Output

### Metrics Explained

- **RMSE** (Root Mean Squared Error): Lower is better, measures prediction error
- **MAE** (Mean Absolute Error): Average absolute difference between predictions and targets
- **MAPE** (Mean Absolute Percentage Error): Percentage error, useful for relative comparison
- **R²** (Coefficient of Determination): Closer to 1.0 is better, measures explained variance

### What Happened?

1. **Data Loading**: Loaded Performance Summary CSV files for all cells
2. **Feature Extraction**: Created features from summary statistics (throughput, resistance, temperature)
3. **Data Splitting**: Split by temperature (train on 10°C+40°C, validate on 25°C)
4. **Model Training**: Trained LightGBM gradient boosting model
5. **Evaluation**: Computed metrics on validation set
6. **Tracking**: Logged results to local files and MLflow

## Next Steps

### Explore More Examples

- **Milestone B**: ICA features + SHAP analysis
  ```bash
  python examples/milestone_b.py
  ```

- **Milestone C**: Neural ODE vs LSTM comparison
  ```bash
  python examples/milestone_c.py
  ```

### Learn More

- [Core Concepts](concepts.md) - Understand Sample, Registry, Caching
- [User Guide](../user-guide/data-loading.md) - Detailed usage documentation
- [API Reference](../api/data.md) - Complete API documentation

## Common Issues

**Issue**: `FileNotFoundError` for data files
- **Solution**: Verify data path matches expected structure (see Step 1)

**Issue**: Import errors
- **Solution**: Ensure virtual environment is activated and dependencies installed

**Issue**: CUDA out of memory
- **Solution**: Reduce batch size or use CPU (models will auto-detect device)

For more troubleshooting help, see [Troubleshooting](../troubleshooting/common-issues.md).
