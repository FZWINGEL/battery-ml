# Installation

This guide will help you install BatteryML and set up your development environment.

## Prerequisites

- **Python**: 3.8 or higher
- **CUDA**: (Optional) For GPU acceleration with PyTorch models
- **Git**: For cloning the repository

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/battery-ml.git
cd battery-ml
```

### 2. Create a Virtual Environment

We recommend using a virtual environment to isolate dependencies:

```bash
# Using venv (Python 3.8+)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

### 3. Install Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

### 4. (Optional) Install Development Dependencies

For running tests and contributing:

```bash
pip install pytest pytest-cov
```

### 5. Verify Installation

Test that everything is installed correctly:

```python
import torch
import lightgbm
import hydra
from src.pipelines.sample import Sample

print("✓ All imports successful!")
```

## Dependencies Overview

### Core Dependencies

| Package | Purpose | Version |
|---------|---------|---------|
| `torch` | Deep learning framework | ≥2.0 |
| `numpy` | Numerical computing | ≥1.24 |
| `pandas` | Data manipulation | ≥2.0 |
| `scipy` | Scientific computing | ≥1.10 |
| `scikit-learn` | Machine learning utilities | ≥1.3 |

### Configuration

| Package | Purpose | Version |
|---------|---------|---------|
| `hydra-core` | Configuration management | ≥1.3 |
| `omegaconf` | Configuration objects | ≥2.3 |
| `pydantic` | Data validation | ≥2.0 |

### Models

| Package | Purpose | Version |
|---------|---------|---------|
| `lightgbm` | Gradient boosting | ≥4.0 |
| `torchdiffeq` | Neural ODE solvers | ≥0.2.3 |

### Tracking & Visualization

| Package | Purpose | Version |
|---------|---------|---------|
| `tensorboard` | Training visualization | ≥2.14 |
| `mlflow` | Experiment tracking | ≥2.5 |
| `shap` | Model interpretability | ≥0.42 |
| `matplotlib` | Plotting | ≥3.7 |
| `seaborn` | Statistical visualization | ≥0.12 |
| `plotly` | Interactive plots | Latest |

## GPU Setup (Optional)

For GPU acceleration with PyTorch models:

### CUDA Installation

1. Install CUDA Toolkit from [NVIDIA](https://developer.nvidia.com/cuda-downloads)
2. Verify CUDA installation:

   ```bash
   nvidia-smi
   ```

### PyTorch with CUDA

Install PyTorch with CUDA support:

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify GPU availability:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

## Troubleshooting

### Common Issues

**Issue**: `ImportError` for torch or other packages

- **Solution**: Ensure virtual environment is activated and packages are installed

**Issue**: CUDA not detected

- **Solution**: Verify CUDA installation and PyTorch CUDA version matches your CUDA version

**Issue**: Permission errors on Windows

- **Solution**: Run terminal as administrator or use user installation: `pip install --user -r requirements.txt`

### Getting Help

If you encounter issues not covered here, check the [Troubleshooting](../troubleshooting/common-issues.md) section or open an issue on GitHub.

## Next Steps

Once installation is complete, proceed to:

- [Quick Start Guide](quickstart.md) - Run your first experiment
- [Core Concepts](concepts.md) - Understand key concepts
- [User Guide](../user-guide/data-loading.md) - Comprehensive usage guide
