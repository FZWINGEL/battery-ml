# Glossary

This glossary defines key terms used in BatteryML documentation.

## A

**Arrhenius Factor**: Temperature-dependent factor `exp(-Ea/RT)` used to model temperature effects on degradation.

## B

**BaseModel**: Abstract base class for all neural network models in BatteryML.

**BasePipeline**: Abstract base class for all feature extraction pipelines.

## C

**Cache**: Hash-based caching system for expensive computations (especially ICA).

**Cell ID**: Identifier for individual battery cells (e.g., 'A', 'B', 'C').

**C-rate**: Charge/discharge rate normalized by capacity (1C = full capacity in 1 hour).

## D

**dQ/dV**: Incremental capacity analysis metric, derivative of capacity with respect to voltage.

**Degradation**: Gradual loss of battery capacity and performance over time.

**DualTracker**: Experiment tracker that logs to both local files and MLflow simultaneously.

## E

**Experiment ID**: Identifier for experiments (1-5) in the LG M50T dataset.

**Early Stopping**: Training technique that stops training when validation loss stops improving.

## F

**Feature Dimension**: Number of features in a sample (e.g., 15 for summary features).

**Fit/Transform**: Pattern used by pipelines: fit on training data, transform on test data.

## G

**GPU**: Graphics Processing Unit, used for accelerated neural network training.

## H

**Hash-Based Caching**: Caching system that uses hash of parameters as cache key.

**Hydra**: Configuration management framework used in BatteryML.

## I

**ICA**: Incremental Capacity Analysis, technique for analyzing battery degradation.

**Impedance**: Internal resistance of battery.

## L

**LAM**: Loss of Active Material, degradation mechanism.

**Latent Dimension**: Dimension of latent state in Neural ODE models.

**LightGBM**: Gradient boosting framework, fastest model in BatteryML.

**LLI**: Loss of Lithium Inventory, degradation mechanism.

**LOCO**: Leave-One-Cell-Out, cross-validation strategy.

**LossRegistry**: Registry system for managing loss functions (MSE, Huber, Physics-Informed, etc.).

## M

**MAPE**: Mean Absolute Percentage Error, evaluation metric.

**MAE**: Mean Absolute Error, evaluation metric.

**MLflow**: Experiment tracking and model management platform.

**MLP**: Multi-Layer Perceptron, simple neural network model.

## N

**Neural ODE**: Neural Ordinary Differential Equation, continuous-time model.

**Normalization**: Scaling features to zero mean and unit variance.

## O

**ODE**: Ordinary Differential Equation.

**Output Dimension**: Number of model outputs (typically 1 for SOH prediction).

## P

**Pipeline**: Feature extraction component that transforms raw data to Samples.

**PipelineRegistry**: Registry system for managing pipelines.

## R

**R²**: Coefficient of Determination, evaluation metric.

**Registry Pattern**: Design pattern for extensible component registration.

**RMSE**: Root Mean Squared Error, evaluation metric.

**RPT**: Reference Performance Test, periodic capacity measurement.

## S

**Sample**: Universal dataclass format for data in BatteryML.

**Savitzky-Golay**: Smoothing filter used in ICA analysis.

**SHAP**: SHapley Additive exPlanations, model interpretability method.

**SOH**: State of Health, remaining capacity as fraction of initial capacity.

**Split Strategy**: Method for splitting data into train/validation/test sets.

## T

**TensorBoard**: Visualization tool for training metrics.

**Temperature Holdout**: Split strategy that trains on extreme temperatures, validates on intermediate.

## U

**Unit Normalization**: Automatic conversion of units (e.g., mAh → Ah, °C → K).

## V

**Validation Set**: Data used to evaluate model during training.

**Voltage Curve**: Voltage vs. capacity curve from discharge test.

## Next Steps

- [Getting Started](../getting-started/installation.md) - Installation guide
- [User Guide](../user-guide/data-loading.md) - Usage documentation
