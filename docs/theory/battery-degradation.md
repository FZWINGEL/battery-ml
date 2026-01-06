# Battery Degradation Theory

This document provides background on battery degradation modeling.

## Overview

Battery degradation refers to the gradual loss of capacity and performance over time and use. Understanding degradation mechanisms is crucial for:

- Predicting remaining useful life
- Optimizing battery usage
- Designing better batteries

## Degradation Mechanisms

### Loss of Lithium Inventory (LLI)

**Description**: Loss of active lithium ions that can participate in charge/discharge cycles.

**Causes**:

- SEI (Solid Electrolyte Interphase) growth
- Lithium plating
- Side reactions

**Indicators**:

- Capacity fade
- Voltage curve shifts (in ICA analysis)

### Loss of Active Material (LAM)

**Description**: Loss of active electrode material (anode or cathode).

**Causes**:

- Particle cracking
- Electrode delamination
- Material dissolution

**Indicators**:

- Capacity fade
- Peak height changes in ICA

### Impedance Rise

**Description**: Increase in internal resistance.

**Causes**:

- SEI growth
- Contact loss
- Electrolyte degradation

**Indicators**:

- Voltage drop under load
- Peak width changes in ICA

## State of Health (SOH)

SOH is a key metric for battery health, defined as the ratio of current capacity to the nominal initial capacity:

$$
\text{SOH} = \frac{Q_{\text{current}}}{Q_{\text{nominal}}}
$$

- **SOH = 1.0**: New battery (100% capacity)
- **SOH = 0.8**: 80% capacity remaining
- **SOH < 0.8**: Often considered end-of-life

## Factors Affecting Degradation

### Temperature

- **High temperature**: Accelerates degradation
- **Low temperature**: Can cause lithium plating
- **Optimal**: Moderate temperatures (20-30°C)

### Charge/Discharge Rate (C-rate)

- **High C-rate**: Increases degradation
- **Low C-rate**: Slower degradation
- **Optimal**: Low to moderate C-rates

### Depth of Discharge (DOD)

- **Deep discharge**: Increases degradation
- **Shallow discharge**: Reduces degradation
- **Optimal**: Moderate DOD (20-80%)

### Cycle Count

- More cycles = more degradation
- Degradation rate may change over time

## Modeling Approaches

### Empirical Models

- **Arrhenius equation**: Temperature dependence
- **Power law**: Cycle count dependence
- **Linear models**: Simple capacity fade

### Physics-Based Models

- **P2D model**: Pseudo-2D electrochemical model
- **Equivalent circuit models**: Electrical circuit analogs
- **Degradation mechanism models**: Explicit mechanism modeling

### Data-Driven Models

- **Machine learning**: Learn from data
- **Neural networks**: Flexible function approximation
- **Gradient boosting**: Tree-based models

## Feature Engineering

### Summary Statistics

- Cumulative throughput
- Resistance measurements
- Temperature statistics

### Incremental Capacity Analysis (ICA)

- Peak positions (LLI indicator)
- Peak heights (LAM indicator)
- Peak widths (impedance indicator)

### Time-Series Features

- Degradation trajectories
- Temporal patterns
- Sequence modeling

## Evaluation Metrics

### Regression Metrics

- **RMSE**: Root mean squared error
- **MAE**: Mean absolute error
- **MAPE**: Mean absolute percentage error
- **R²**: Coefficient of determination

### Domain-Specific Metrics

- **Capacity retention**: Percentage of initial capacity
- **Cycle life**: Number of cycles to end-of-life
- **Energy efficiency**: Energy in / energy out

## Research Challenges

1. **Generalization**: Models trained on one condition may not generalize
2. **Interpretability**: Understanding model predictions
3. **Uncertainty**: Quantifying prediction uncertainty
4. **Data scarcity**: Limited degradation data
5. **Multi-mechanism**: Multiple degradation mechanisms interact

## References

- Oxford Battery Intelligence Lab: LG M50T dataset
- Battery degradation literature
- Electrochemistry textbooks

## Next Steps

- [ICA Analysis](ica-analysis.md) - ICA theory
- [Neural ODEs](neural-odes.md) - Continuous-time modeling
- [User Guide](../user-guide/pipelines.md) - Using BatteryML
