# Neural ODEs for Battery Degradation

This document explains Neural Ordinary Differential Equations (ODEs) and their application to battery degradation modeling.

## Overview

Neural ODEs enable continuous-time modeling of battery degradation trajectories, allowing predictions at arbitrary time points and physics-informed modeling.

## Ordinary Differential Equations (ODEs)

### Basic Form

An ODE describes how a state vector $\mathbf{z}(t) \in \mathbb{R}^d$ evolves over time:

\[
\frac{d\mathbf{z}(t)}{dt} = f(\mathbf{z}(t), t, \theta)
\]

### Solution

Given an initial condition $\mathbf{z}(t_0)$, the state at any time $t_i$ is found by solving the Initial Value Problem (IVP):

\[
\mathbf{z}(t_i) = \mathbf{z}(t_0) + \int_{t_0}^{t_i} f(\mathbf{z}(s), s, \theta) ds = \text{ODESolve}(\mathbf{z}(t_0), f, t_0, t_i, \theta)
\]

## Neural ODEs

### Concept

In a Neural ODE, the dynamics function $f$ is approximated by a neural network with parameters $\theta$:

\[
\frac{d\mathbf{z}(t)}{dt} = \text{NN}(\mathbf{z}(t), t, \theta)
\]

### Optimization via Adjoint Sensitivity

To train the model, we need gradients of a loss $L$ with respect to $\theta$. The Adjoint Method avoids backpropagating through the internal stages of the ODE solver by solving a second, backwards-in-time ODE for the "adjoint" state $\mathbf{a}(t) = \partial L / \partial \mathbf{z}(t)$:

\[
\frac{d\mathbf{a}(t)}{dt} = -\mathbf{a}(t)^\top \frac{\partial f(\mathbf{z}(t), t, \theta)}{\partial \mathbf{z}}
\]

This allows for constant memory cost $O(1)$ relative to the number of solver steps.

## Application to Battery Degradation

### Degradation Trajectory

Battery degradation can be modeled as:

```text
dSOH/dt = f(SOH, conditions, t)
```

Where:

- `SOH`: State of Health
- `conditions`: Temperature, C-rate, etc.
- `f`: Degradation dynamics (learned by neural network)

### Latent State

Use latent state to capture degradation mechanisms:

```text
dz/dt = NN(z, conditions, t)
SOH = Decoder(z)
```

Where:

- `z`: Latent degradation state
- `Decoder`: Maps latent state to SOH

## Implementation in BatteryML

### NeuralODEModel

The `NeuralODEModel` implements continuous-time degradation:

```python
from src.models.neural_ode import NeuralODEModel

model = NeuralODEModel(
    input_dim=5,           # Features per time step
    latent_dim=32,         # Latent state dimension
    hidden_dim=64,         # ODE function hidden dimension
    solver='dopri5',       # ODE solver
    use_adjoint=True       # Memory-efficient gradients
)
```

### Architecture

1. **Encoder**: Maps input sequence to initial latent state
2. **ODE Function**: Neural network defining dynamics
3. **ODE Solver**: Numerical integration (e.g., Runge-Kutta)
4. **Decoder**: Maps latent state to predictions

### Training

```python
from src.training.trainer import Trainer

trainer = Trainer(model, config, tracker)
history = trainer.fit(train_samples, val_samples)
```

## ODE Solvers

### Adaptive Solvers

- **dopri5**: Adaptive Runge-Kutta (most accurate, slower)
- **adaptive_heun**: Adaptive Heun method

### Fixed-Step Solvers

- **euler**: Euler method (fastest, less accurate)
- **rk4**: 4th-order Runge-Kutta (balanced)

### Solver Selection

- **Accuracy**: Use `dopri5` for high accuracy
- **Speed**: Use `euler` for fast training
- **Balance**: Use `rk4` for balanced performance

## Adjoint Method

### Memory Efficiency

The adjoint method enables memory-efficient backpropagation:

```python
model = NeuralODEModel(use_adjoint=True)  # Saves memory
```

### How It Works

Instead of storing intermediate states, the adjoint method:

1. Solves forward ODE
2. Solves backward adjoint ODE
3. Computes gradients efficiently

## Advantages for Battery Modeling

1. **Interpolation**: Predict at arbitrary time points
2. **Extrapolation**: Extend trajectories beyond training data
3. **Physics**: Can incorporate physical constraints
4. **Uncertainty**: Can quantify prediction uncertainty

## Challenges

1. **Training Time**: Slower than discrete models
2. **Memory**: Can be memory-intensive
3. **Stability**: Requires careful initialization
4. **Hyperparameters**: Many hyperparameters to tune

## Best Practices

1. **Start Simple**: Begin with simple architecture
2. **Tune Solver**: Choose appropriate solver
3. **Use Adjoint**: Enable adjoint method for memory
4. **Monitor Training**: Watch for NaN losses
5. **Validate**: Verify predictions are reasonable

## Comparison with Other Models

### vs. LSTM

- **Neural ODE**: Continuous-time, physics-informed
- **LSTM**: Discrete-time, sequence modeling

### vs. LightGBM

- **Neural ODE**: Continuous-time, flexible
- **LightGBM**: Fast, interpretable

## References

- Neural ODE paper (Chen et al., 2018)
- ODE solver literature
- Battery degradation modeling papers

## Next Steps

- [Neural ODE Tuning](../examples/neural-ode-tuning.md) - Hyperparameter tuning
- [Model Guide](../user-guide/models.md) - Model usage
- [Battery Degradation](battery-degradation.md) - Degradation theory
