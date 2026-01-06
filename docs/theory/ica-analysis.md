# Incremental Capacity Analysis (ICA)

This document explains Incremental Capacity Analysis and its use in battery degradation modeling.

## Overview

Incremental Capacity Analysis (ICA) is a powerful technique for analyzing battery degradation by examining the derivative of capacity with respect to voltage (dQ/dV).

## Theory

### dQ/dV Curve

The Incremental Capacity ($dQ/dV$) is the numerical derivative of battery capacity with respect to the terminal voltage:

$$
\text{IC}(V) = \frac{dQ}{dV} \approx \frac{Q_{i+1} - Q_i}{V_{i+1} - V_i}
$$

Since raw voltage data is often noisy, numerical differentiation can amplify noise. BatteryML uses high-order Savitzky-Golay filters to smooth the signal:

$$
Y_j = \frac{\sum_{i=-k}^k C_i y_{j+i}}{N}
$$

Where $C_i$ are the filter coefficients and $2k+1$ is the window size.

### Physical Interpretation

- **Peaks**: Correspond to phase transition plateaus in the $V-Q$ curve (Gibbs Phase Rule).
- **Peak positions**: Indicate the state of equilibrium potentials where phase transitions occur.
- **Peak heights**: Proportional to the amount of active material undergoing phase transformation.

## Degradation Indicators

### Loss of Lithium Inventory (LLI)

**Indicator**: Peak shifts in the voltage domain.

If the amount of cyclable lithium decreases due to SEI growth, the corresponding phase transitions occur at different stoichiometry points, leading to a shift:

$$
\Delta V_{peak} \propto \Delta \text{Li}_{\text{inv}}
$$

### Loss of Active Material (LAM)

**Indicator**: Peak height decreases

**Explanation**: Less active material reduces peak magnitude

**Example**: Peak height decreases from 100 to 80

### Impedance Rise

**Indicator**: Peak width increases

**Explanation**: Higher resistance broadens peaks

**Example**: Peak FWHM increases from 0.1V to 0.15V

## ICA Feature Extraction

### Peak Detection

1. **Smoothing**: Apply Savitzky-Golay filter to reduce noise
2. **Peak Finding**: Use peak detection algorithms
3. **Peak Characterization**: Extract position, height, width, area

### Features

For each peak:

- **Voltage**: Peak position (V)
- **Height**: Peak magnitude (dQ/dV)
- **Width**: Full-width at half-maximum (FWHM)
- **Area**: Integrated area under peak

### Additional Features

- **Total area**: Total integrated dQ/dV curve
- **Number of peaks**: Count of detected peaks
- **Voltage at max dQ/dV**: Voltage at maximum dQ/dV value

## Implementation in BatteryML

### ICAPeaksPipeline

The `ICAPeaksPipeline` extracts ICA features:

```python
from src.pipelines.ica_peaks import ICAPeaksPipeline

pipeline = ICAPeaksPipeline(
    sg_window=51,      # Smoothing window
    sg_order=3,        # Polynomial order
    num_peaks=3,       # Number of peaks to extract
    voltage_range=(3.0, 4.2)
)

samples = pipeline.fit_transform({'curves': voltage_curves, 'targets': targets})
```

### Processing Steps

1. **Load voltage curve**: 0.1C discharge curve
2. **Compute dQ/dV**: Numerical differentiation
3. **Smooth**: Savitzky-Golay filtering
4. **Find peaks**: Peak detection algorithm
5. **Extract features**: Position, height, width, area
6. **Normalize**: StandardScaler normalization

## Interpretation

### Peak Shifts

- **To higher voltage**: LLI (less lithium)
- **To lower voltage**: Unusual, may indicate other mechanisms

### Peak Height Changes

- **Decrease**: LAM (less active material)
- **Increase**: Unusual, may indicate measurement issues

### Peak Width Changes

- **Increase**: Impedance rise (higher resistance)
- **Decrease**: Unusual, may indicate improved kinetics

## Best Practices

1. **Smoothing**: Use appropriate smoothing to reduce noise
2. **Voltage Range**: Focus on relevant voltage range (e.g., 3.0-4.2V)
3. **Peak Selection**: Extract consistent number of peaks
4. **Validation**: Verify peaks correspond to known transitions
5. **Caching**: Cache expensive ICA computations

## Limitations

1. **Noise**: Sensitive to measurement noise
2. **Smoothing**: May obscure fine details
3. **Peak Detection**: May miss peaks or detect false peaks
4. **Interpretation**: Requires domain knowledge

## Applications

1. **Degradation Diagnosis**: Identify degradation mechanisms
2. **Feature Engineering**: Extract features for ML models
3. **Quality Control**: Detect manufacturing defects
4. **Research**: Understand battery behavior

## References

- Battery electrochemistry literature
- ICA analysis papers
- Oxford Battery Intelligence Lab documentation

## Next Steps

- [Battery Degradation](battery-degradation.md) - Degradation theory
- [ICA Pipeline](../user-guide/pipelines.md) - Using ICA pipeline
- [SHAP Analysis](../examples/shap-analysis.md) - Interpreting ICA features
