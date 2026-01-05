# Performance Optimization

This guide covers performance optimization tips for BatteryML.

## Data Loading Optimization

### Use Caching

Enable caching for expensive computations:

```python
pipeline = ICAPeaksPipeline(use_cache=True)  # Cache ICA features
```

### Batch Data Loading

Load data in batches for large datasets:

```python
def load_in_batches(loader, batch_size=100):
    """Load data in batches."""
    all_data = []
    for i in range(0, len(cells), batch_size):
        batch_cells = cells[i:i+batch_size]
        batch_data = loader.load_cells(batch_cells)
        all_data.append(batch_data)
    return pd.concat(all_data)
```

## Training Optimization

### Use GPU

Always use GPU when available:

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
```

### Mixed Precision Training

Enable AMP for faster GPU training:

```python
config = {'use_amp': True}  # 2x speedup on modern GPUs
```

### Optimize Batch Size

Find optimal batch size:

```python
# Start with 32, increase if memory allows
batch_sizes = [16, 32, 64, 128]
for bs in batch_sizes:
    try:
        config = {'batch_size': bs}
        # Train and measure time
    except RuntimeError:  # Out of memory
        break
```

### DataLoader Workers

Use multiple workers for data loading:

```python
# In Trainer, set num_workers
dataloader = DataLoader(dataset, batch_size=32, num_workers=4)
```

## Model Optimization

### Choose Right Model

- **LightGBM**: Fastest for tabular data
- **MLP**: Fast neural baseline
- **LSTM**: Slower, for sequences
- **Neural ODE**: Slowest, for continuous-time

### Model Pruning

Reduce model size:

```python
# Smaller hidden dimensions
model = MLPModel(input_dim=15, hidden_dims=[32, 16])  # Instead of [64, 32]
```

### Quantization (Future)

Post-training quantization can speed up inference:

```python
# Quantize model (if supported)
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

## Pipeline Optimization

### Vectorize Operations

Use vectorized NumPy operations:

```python
# Good: Vectorized
features = df[['col1', 'col2']].values

# Bad: Loop
features = []
for _, row in df.iterrows():
    features.append([row['col1'], row['col2']])
```

### Avoid Redundant Computations

Cache intermediate results:

```python
# Compute once, reuse
ica_features = compute_ica(curve)
# Reuse ica_features instead of recomputing
```

## Memory Optimization

### Gradient Checkpointing

For very large models:

```python
# Trade compute for memory
from torch.utils.checkpoint import checkpoint
output = checkpoint(model, x)
```

### Clear Cache

Clear GPU cache if needed:

```python
import torch
torch.cuda.empty_cache()
```

## Profiling

### Profile Training

Identify bottlenecks:

```python
import torch.profiler as profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    trainer.fit(train_samples, val_samples)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### Time Operations

```python
import time

start = time.time()
# Your operation
duration = time.time() - start
print(f"Operation took {duration:.2f}s")
```

## Best Practices

1. **Profile First**: Identify bottlenecks before optimizing
2. **Use GPU**: Always use GPU when available
3. **Enable AMP**: Use mixed precision for GPU training
4. **Cache Expensive Operations**: Cache ICA and other expensive computations
5. **Batch Operations**: Process data in batches
6. **Choose Right Model**: Use fastest model that meets accuracy requirements

## Benchmarking

### Compare Configurations

```python
configs = [
    {'batch_size': 16, 'use_amp': False},
    {'batch_size': 32, 'use_amp': False},
    {'batch_size': 32, 'use_amp': True},
]

for config in configs:
    start = time.time()
    trainer = Trainer(model, config, tracker)
    trainer.fit(train_samples, val_samples)
    duration = time.time() - start
    print(f"Config {config}: {duration:.2f}s")
```

## Next Steps

- [Training Issues](training-issues.md) - Training problems
- [Common Issues](common-issues.md) - Other issues
- [Training Guide](../user-guide/training.md) - Training documentation
