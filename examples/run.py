"""Convenient entry point for running BatteryML experiments.

This script provides a simple way to run experiments using the unified experiment runner.

Usage:
    # Run with default config (LGBM on summary data)
    python examples/run.py
    
    # Override model
    python examples/run.py model=mlp
    
    # Override pipeline and model
    python examples/run.py pipeline=ica_peaks model=lstm_attn
    
    # Override split strategy
    python examples/run.py split=loco split.test_cell=A
    
    # Multiple overrides
    python examples/run.py model=mlp pipeline=ica_peaks training.epochs=100
"""

import sys
import subprocess
from pathlib import Path

# Change to project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Run the experiment module
if __name__ == "__main__":
    cmd = [sys.executable, "-m", "src.experiments.run"] + sys.argv[1:]
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        sys.stderr.write(
            f"Experiment runner failed with exit code {result.returncode} for command: {cmd}\n"
        )
    sys.exit(result.returncode)
