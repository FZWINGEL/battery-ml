"""Test fixtures for BatteryML tests."""

import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# Add src to path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_summary_df():
    """Create a sample summary DataFrame for testing."""
    np.random.seed(42)
    n_samples = 20

    df = pd.DataFrame(
        {
            "Cumulative Charge Throughput [A h]": np.linspace(0, 500, n_samples),
            "Cumulative Discharge Throughput [A h]": np.linspace(0, 500, n_samples),
            "0.1s Resistance [Ohms]": 0.01 + np.random.randn(n_samples) * 0.001,
            "10s Resistance [Ohms]": 0.015 + np.random.randn(n_samples) * 0.002,
            "Cell Capacity [mA h]": 5000
            - np.linspace(0, 500, n_samples)
            + np.random.randn(n_samples) * 20,
            "cell_id": ["A"] * 10 + ["B"] * 10,
            "temperature_C": [10] * 10 + [25] * 10,
            "experiment_id": [5] * n_samples,
        }
    )

    return df


@pytest.fixture
def sample_voltage_curve():
    """Create a sample voltage curve for testing."""
    np.random.seed(42)
    n_points = 500

    # Simulate a discharge curve
    capacity = np.linspace(0, 4.5, n_points)
    voltage = (
        4.2
        - 0.3 * capacity
        + 0.05 * np.sin(capacity * 2)
        + np.random.randn(n_points) * 0.01
    )

    return voltage, capacity


@pytest.fixture
def sample_curves_with_meta():
    """Create sample curves with metadata for ICA testing."""
    np.random.seed(42)
    curves = []

    for cell_id in ["A", "B"]:
        for rpt_id in range(3):
            n_points = 500
            capacity = np.linspace(0, 4.5, n_points)
            voltage = 4.2 - 0.3 * capacity + np.random.randn(n_points) * 0.01

            meta = {
                "experiment_id": 5,
                "cell_id": cell_id,
                "rpt_id": rpt_id,
                "temperature_C": 10 if cell_id == "A" else 25,
            }

            curves.append((voltage, capacity, meta))

    targets = {
        ("A", 0): 4.8,
        ("A", 1): 4.7,
        ("A", 2): 4.6,
        ("B", 0): 4.9,
        ("B", 1): 4.8,
        ("B", 2): 4.7,
    }

    return curves, targets


@pytest.fixture
def sample_samples():
    """Create sample Sample objects for testing."""
    from src.pipelines.sample import Sample

    samples = []
    for i in range(10):
        temp = 10 if i < 4 else (25 if i < 6 else 40)
        sample = Sample(
            meta={
                "experiment_id": 5,
                "cell_id": chr(ord("A") + i % 8),
                "temperature_C": temp,
                "set_idx": i,
            },
            x=np.random.randn(7).astype(np.float32),
            y=np.array([4.5 - i * 0.1], dtype=np.float32),
        )
        samples.append(sample)

    return samples


@pytest.fixture
def device():
    """Get available device."""
    return "cuda" if torch.cuda.is_available() else "cpu"
