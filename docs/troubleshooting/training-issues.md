# Training Issues

This guide covers training-specific issues and solutions.

## Loss Issues

### Loss Not Decreasing

**Issue**: Loss plateaus or increases

**Solutions**:
1. **Check Learning Rate**:
   ```python
   # Too high: loss explodes
   # Too low: loss decreases very slowly
   config = {'learning_rate': 1e-4}  # Try different values
   ```

2. **Verify Data Normalization**:
   ```python
   pipeline = SummarySetPipeline(normalize=True)
   ```

3. **Check Model Capacity**:
   ```python
   # Model may be too small
   model = MLPModel(input_dim=15, hidden_dims=[128, 64, 32])
   ```

4. **Check Data Quality**:
   ```python
   # Verify targets are reasonable
   print(y_train.min(), y_train.max(), y_train.mean())
   ```

### Loss Becomes NaN

**Issue**: Loss becomes NaN during training

**Solutions**:
1. **Check for NaN in Data**:
   ```python
   import numpy as np
   print(np.isnan(X_train).any())
   print(np.isnan(y_train).any())
   ```

2. **Reduce Learning Rate**:
   ```python
   config = {'learning_rate': 1e-4}  # Lower learning rate
   ```

3. **Increase Gradient Clipping**:
   ```python
   config = {'gradient_clip': 2.0}  # Stronger clipping
   ```

4. **Check Model Initialization**:
   ```python
   # Verify weights are initialized correctly
   for param in model.parameters():
       print(param.data.mean(), param.data.std())
   ```

### Loss Oscillates

**Issue**: Loss oscillates wildly

**Solutions**:
1. **Reduce Learning Rate**:
   ```python
   config = {'learning_rate': 5e-4}  # Lower learning rate
   ```

2. **Increase Batch Size**:
   ```python
   config = {'batch_size': 64}  # Larger batches = more stable
   ```

3. **Use Learning Rate Schedule**:
   ```python
   # Cosine annealing helps stabilize training
   config = {'scheduler_T0': 50}
   ```

## Convergence Issues

### Overfitting

**Issue**: Training loss decreases but validation loss increases

**Solutions**:
1. **Add Dropout**:
   ```python
   model = MLPModel(input_dim=15, hidden_dims=[64, 32], dropout=0.3)
   ```

2. **Increase Weight Decay**:
   ```python
   config = {'weight_decay': 0.1}  # Stronger regularization
   ```

3. **Early Stopping**:
   ```python
   config = {'early_stopping_patience': 20}
   ```

4. **Reduce Model Capacity**:
   ```python
   model = MLPModel(input_dim=15, hidden_dims=[32, 16])  # Smaller model
   ```

### Underfitting

**Issue**: Both training and validation loss are high

**Solutions**:
1. **Increase Model Capacity**:
   ```python
   model = MLPModel(input_dim=15, hidden_dims=[128, 64, 32])  # Larger model
   ```

2. **Reduce Regularization**:
   ```python
   config = {'weight_decay': 0.001}  # Less regularization
   model = MLPModel(dropout=0.1)  # Less dropout
   ```

3. **Train Longer**:
   ```python
   config = {'epochs': 500}  # More epochs
   ```

4. **Check Feature Engineering**:
   ```python
   # May need better features
   pipeline = SummarySetPipeline(include_arrhenius=True)
   ```

## Performance Issues

### Slow Training

**Issue**: Training is very slow

**Solutions**:
1. **Use GPU**:
   ```python
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   ```

2. **Enable Mixed Precision**:
   ```python
   config = {'use_amp': True}  # Faster GPU training
   ```

3. **Increase Batch Size**:
   ```python
   config = {'batch_size': 64}  # Larger batches = faster
   ```

4. **Use Faster Model**:
   ```python
   # LightGBM is faster than neural networks for tabular data
   model = LGBMModel()
   ```

### Out of Memory

**Error**: `CUDA out of memory`

**Solutions**:
1. **Reduce Batch Size**:
   ```python
   config = {'batch_size': 16}  # Smaller batches
   ```

2. **Use Gradient Accumulation**:
   ```python
   # Accumulate gradients over multiple batches
   ```

3. **Enable Mixed Precision**:
   ```python
   config = {'use_amp': True}  # Saves memory
   ```

4. **Use Adjoint Method (for ODEs)**:
   ```python
   model = NeuralODEModel(use_adjoint=True)  # Memory efficient
   ```

## Early Stopping Issues

### Stops Too Early

**Issue**: Training stops before convergence

**Solutions**:
1. **Increase Patience**:
   ```python
   config = {'early_stopping_patience': 50}  # More patience
   ```

2. **Check Validation Set**:
   ```python
   # Ensure validation set is representative
   ```

3. **Monitor Training Loss**:
   ```python
   # May need to monitor training loss instead
   ```

### Never Stops

**Issue**: Training never stops (no improvement)

**Solutions**:
1. **Check Validation Metrics**:
   ```python
   # Verify validation metrics are being computed correctly
   ```

2. **Reduce Patience**:
   ```python
   config = {'early_stopping_patience': 10}  # Less patience
   ```

3. **Check for Bugs**:
   ```python
   # Verify early stopping logic is correct
   ```

## Debugging Tips

### Monitor Training

```python
# Use TensorBoard to monitor training
tensorboard --logdir artifacts/runs

# Check:
# - Loss curves
# - Learning rate schedule
# - Gradient norms
```

### Check Gradients

```python
# Monitor gradient norms
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm().item()}")
```

### Validate Data

```python
# Check data before training
print(f"X shape: {X_train.shape}")
print(f"y shape: {y_train.shape}")
print(f"X range: [{X_train.min():.2f}, {X_train.max():.2f}]")
print(f"y range: [{y_train.min():.2f}, {y_train.max():.2f}]")
```

## Best Practices

1. **Start Small**: Begin with small model and simple data
2. **Monitor Closely**: Watch training curves in TensorBoard
3. **Validate Early**: Check validation metrics frequently
4. **Save Checkpoints**: Save model checkpoints regularly
5. **Experiment Systematically**: Change one thing at a time

## Next Steps

- [Common Issues](common-issues.md) - Other common problems
- [Performance](performance.md) - Performance optimization
- [Training Guide](../user-guide/training.md) - Training documentation
