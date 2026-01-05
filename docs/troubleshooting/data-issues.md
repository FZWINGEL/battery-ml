# Data Issues

This guide covers data loading and processing issues.

## Path Resolution Issues

### Experiment Path Not Found

**Error**: `FileNotFoundError: Experiment path not found`

**Solutions**:
1. Verify experiment ID is correct (1-5):
   ```python
   from src.data.expt_paths import ExperimentPaths
   paths = ExperimentPaths(experiment_id=5, base_path=Path("Raw Data"))
   print(paths.base_dir)  # Check resolved path
   ```

2. Check base path:
   ```python
   base_path = Path("Raw Data")
   print(base_path.exists())  # Should be True
   print(base_path.is_absolute() or base_path.resolve())
   ```

3. Verify experiment directory exists:
   ```bash
   ls "Raw Data/Expt 5 - Standard Cycle Aging (Control)"
   ```

### Cell ID Not Found

**Error**: `FileNotFoundError: Cell A not found`

**Solutions**:
1. List available cells:
   ```python
   from src.data.discovery import discover_available_cells
   cells = discover_available_cells(experiment_id=5, base_path=Path("Raw Data"))
   print(cells)
   ```

2. Check cell naming convention (may vary by experiment)

3. Verify cell exists in experiment directory

## CSV Loading Issues

### Encoding Errors

**Error**: `UnicodeDecodeError`

**Solutions**:
1. Specify encoding:
   ```python
   df = pd.read_csv(path, encoding='utf-8')
   # or
   df = pd.read_csv(path, encoding='latin-1')
   ```

2. Handle encoding in loader:
   ```python
   # Modify loader to try multiple encodings
   ```

### Missing Columns

**Error**: `KeyError: 'column_name'`

**Solutions**:
1. Check CSV structure:
   ```python
   df = pd.read_csv(path)
   print(df.columns.tolist())
   ```

2. Different experiments may have different columns

3. Handle missing columns gracefully:
   ```python
   if 'column_name' in df.columns:
       # Use column
   else:
       # Use default value
   ```

### Data Type Issues

**Error**: `ValueError: could not convert string to float`

**Solutions**:
1. Check data types:
   ```python
   print(df.dtypes)
   ```

2. Clean data:
   ```python
   df = df.replace(['N/A', 'nan', ''], np.nan)
   df = df.astype(float, errors='ignore')
   ```

## Unit Conversion Issues

### Incorrect Units

**Issue**: Values seem wrong (e.g., capacity in thousands)

**Solutions**:
1. Verify unit conversion:
   ```python
   from src.data.units import UnitConverter
   df_normalized = UnitConverter.normalize_all_capacity_columns(df)
   ```

2. Check original units in CSV headers

3. Verify conversion factors are correct

### Temperature Conversion

**Issue**: Temperature values incorrect

**Solutions**:
1. Check temperature is in Celsius:
   ```python
   print(df['temperature_C'].unique())
   ```

2. Verify conversion to Kelvin:
   ```python
   temp_K = df['temperature_C'] + 273.15
   ```

## Sample Creation Issues

### Missing Metadata

**Error**: `KeyError: 'cell_id'` in sample.meta

**Solutions**:
1. Ensure metadata is added:
   ```python
   sample = Sample(
       meta={
           'cell_id': row['cell_id'],
           'temperature_C': row['temperature_C'],
           'experiment_id': row['experiment_id'],
       },
       x=features,
       y=target
   )
   ```

2. Check DataFrame has required columns before creating samples

### Feature Dimension Mismatch

**Error**: Samples have different feature dimensions

**Solutions**:
1. Verify feature extraction is consistent:
   ```python
   feature_dims = [s.feature_dim for s in samples]
   print(set(feature_dims))  # Should be single value
   ```

2. Handle missing values consistently:
   ```python
   # Replace NaN with 0 or mean
   features = np.nan_to_num(features, nan=0.0)
   ```

## Split Issues

### Empty Splits

**Error**: Split returns empty list

**Solutions**:
1. Check metadata values:
   ```python
   temps = [s.meta.get('temperature_C') for s in samples]
   print(set(temps))
   ```

2. Verify split criteria match available data

3. Check for None values in metadata

### Imbalanced Splits

**Issue**: One split much larger than other

**Solutions**:
1. Check data distribution:
   ```python
   from collections import Counter
   temps = [s.meta['temperature_C'] for s in samples]
   print(Counter(temps))
   ```

2. Consider alternative split strategies

3. Use stratification if possible

## Best Practices

1. **Validate Data Early**: Check data quality before processing
2. **Handle Missing Values**: Replace NaN/inf appropriately
3. **Check Units**: Verify unit conversions are correct
4. **Log Warnings**: Log data quality issues
5. **Test with Small Data**: Test pipelines on small subset first

## Next Steps

- [Common Issues](common-issues.md) - Other common problems
- [Training Issues](training-issues.md) - Training-specific issues
- [Data Loading Guide](../user-guide/data-loading.md) - Data loading documentation
