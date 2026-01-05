# Common Issues

This guide covers common issues and their solutions.

## Installation Issues

### Import Errors

**Error**: `ModuleNotFoundError: No module named 'src'`

**Solution**:
```bash
# Ensure you're in the project root
cd battery-ml

# Install in development mode
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### CUDA Not Available

**Error**: `CUDA not available` or `torch.cuda.is_available() == False`

**Solutions**:
1. Verify CUDA installation:
   ```bash
   nvidia-smi
   ```

2. Reinstall PyTorch with CUDA:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. Check PyTorch CUDA version matches your CUDA version

## Data Loading Issues

### File Not Found

**Error**: `FileNotFoundError: Performance summary not found`

**Solutions**:
1. Verify data path:
   ```python
   from pathlib import Path
   base_path = Path("Raw Data")
   print(base_path.exists())  # Should be True
   ```

2. Check experiment ID (should be 1-5)

3. Verify file naming convention matches expected format

### Missing Columns

**Error**: `KeyError: 'column_name'`

**Solutions**:
1. Check CSV file structure:
   ```python
   import pandas as pd
   df = pd.read_csv("path/to/file.csv")
   print(df.columns.tolist())
   ```

2. Some experiments may have different column names

3. Check unit normalization is applied correctly

## Pipeline Issues

### Cache Errors

**Error**: `PickleError` or cache corruption

**Solutions**:
1. Clear cache:
   ```python
   from pathlib import Path
   cache_dir = Path("artifacts/cache")
   cache_dir.rmdir()  # Remove cache directory
   ```

2. Disable caching temporarily:
   ```python
   pipeline = ICAPeaksPipeline(use_cache=False)
   ```

### Feature Dimension Mismatch

**Error**: `RuntimeError: Expected input size ... but got ...`

**Solutions**:
1. Check feature dimensions:
   ```python
   print(sample.feature_dim)
   print(model.input_dim)
   ```

2. Ensure pipeline is fitted before transforming test data

3. Verify pipeline parameters match between fit and transform

## Model Issues

### Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Reduce batch size:
   ```python
   config = {'batch_size': 16}  # Instead of 32
   ```

2. Use gradient accumulation:
   ```python
   # Accumulate gradients over multiple batches
   ```

3. Enable mixed precision:
   ```python
   config = {'use_amp': True}
   ```

4. Use CPU if GPU memory insufficient

### NaN Losses

**Error**: Loss becomes NaN during training

**Solutions**:
1. Check data for NaN/inf:
   ```python
   import numpy as np
   print(np.isnan(X).any())
   print(np.isinf(X).any())
   ```

2. Reduce learning rate:
   ```python
   config = {'learning_rate': 1e-4}  # Instead of 1e-3
   ```

3. Increase gradient clipping:
   ```python
   config = {'gradient_clip': 2.0}
   ```

4. Check model initialization

### Model Not Learning

**Issue**: Loss doesn't decrease

**Solutions**:
1. Check learning rate (may be too high or too low)

2. Verify data is normalized:
   ```python
   pipeline = SummarySetPipeline(normalize=True)
   ```

3. Check model capacity (may be too small)

4. Verify data quality and labels

## Training Issues

### Early Stopping Too Early

**Issue**: Training stops too early

**Solutions**:
1. Increase patience:
   ```python
   config = {'early_stopping_patience': 50}  # Instead of 20
   ```

2. Check if validation loss is actually improving

3. Verify validation set is representative

### Slow Training

**Issue**: Training is very slow

**Solutions**:
1. Use GPU if available:
   ```python
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   ```

2. Enable mixed precision:
   ```python
   config = {'use_amp': True}
   ```

3. Increase batch size (if memory allows)

4. Use faster model (e.g., LightGBM for baselines)

## Tracking Issues

### TensorBoard Not Showing Data

**Issue**: TensorBoard shows no data

**Solutions**:
1. Check log directory:
   ```bash
   tensorboard --logdir artifacts/runs
   ```

2. Verify files exist:
   ```bash
   ls artifacts/runs/*/tensorboard/
   ```

3. Refresh TensorBoard or restart

### MLflow Connection Issues

**Error**: `MLflowException: Unable to connect`

**Solutions**:
1. Check MLflow URI:
   ```python
   tracker = MLflowTracker(tracking_uri="file:./artifacts/mlruns")
   ```

2. For remote MLflow, check network connectivity

3. Verify MLflow server is running (if using remote)

## Getting Help

If you encounter issues not covered here:

1. Check [Data Issues](data-issues.md) for data-specific problems
2. Check [Training Issues](training-issues.md) for training problems
3. Check [FAQ](faq.md) for frequently asked questions
4. Open an issue on GitHub with:
   - Error message
   - Steps to reproduce
   - Environment details (Python version, OS, etc.)
