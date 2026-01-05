# BatteryML Documentation

Welcome to the BatteryML documentation! BatteryML is a modular machine learning platform for battery degradation modeling, designed for research on the LG M50T dataset from Oxford University's Battery Intelligence Lab.

## What is BatteryML?

BatteryML addresses the key challenge in battery degradation research: **building reproducible, extensible ML pipelines** that can leverage multiple data modalities (summary statistics, ICA curves, time-series sequences) while supporting various model architectures.

### Key Features

- **Canonical Sample Schema**: Universal `Sample` dataclass decoupling pipelines from models
- **Registry Pattern**: Decorator-based registration for extensible pipelines, models, and losses
- **Hash-Based Caching**: Expensive ICA computations cached to disk with automatic invalidation
- **Hydra Configuration**: Composable YAML configs for reproducible experiments
- **Multi-Experiment Support**: Path resolution for Experiments 1-5 with naming convention handling
- **Model Zoo**: LightGBM, MLP, LSTM+Attention, Neural ODE
- **Modular Loss Functions**: Registry-based loss selection (MSE, Huber, Physics-Informed, MAPE)
- **Dual Tracking**: Simultaneous local JSON/TensorBoard + MLflow logging
- **Interpretability**: SHAP analysis and attention visualization

## Quick Links

- [Installation Guide](getting-started/installation.md) - Get started with BatteryML
- [Quick Start Tutorial](getting-started/quickstart.md) - Run your first experiment
- [User Guide](user-guide/data-loading.md) - Comprehensive usage documentation
- [API Reference](api/data.md) - Complete API documentation
- [Architecture Overview](architecture/overview.md) - System design and patterns
- [Contributing Guide](contributing/overview.md) - How to extend BatteryML

## Research Goals

- **SOH Prediction**: Predict State of Health (remaining capacity) from operational data
- **Temperature Generalization**: Train on extreme temperatures (10°C, 40°C), validate on intermediate (25°C)
- **Degradation Mechanism Analysis**: Use SHAP and ICA features to understand degradation patterns
- **Continuous-Time Modeling**: Neural ODEs for physics-informed degradation trajectories

## Documentation Structure

This documentation is organized into several sections:

1. **Getting Started** - Installation, quick start, and core concepts
2. **User Guide** - Detailed tutorials and workflows
3. **API Reference** - Complete API documentation for all modules
4. **Architecture** - System design, data flow, and design patterns
5. **Examples** - Extended examples and use cases
6. **Contributing** - How to add models, pipelines, and extend the platform
7. **Troubleshooting** - Common issues and solutions
8. **Theory** - Background on battery degradation, ICA, and Neural ODEs

## Getting Help

- Check the [Troubleshooting](troubleshooting/common-issues.md) section for common issues
- Review the [FAQ](troubleshooting/faq.md) for frequently asked questions
- Explore the [Examples](examples/custom-pipeline.md) for code samples

## Citation

If you use BatteryML in your research, please cite:

```bibtex
@misc{batteryml2024,
  title={BatteryML: A Modular Platform for Battery Degradation Modeling},
  author={Research Module},
  year={2024},
  publisher={GitHub}
}
```

---

**Ready to get started?** Head over to the [Installation Guide](getting-started/installation.md)!
