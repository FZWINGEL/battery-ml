"""Benchmark script for NODE model optimizations.

This script benchmarks different optimization configurations for ACLA and Neural ODE models,
comparing training time, inference time, memory usage, accuracy metrics, and ODE solver performance.

The script supports three data split modes:
1. Temperature split (default): Trains on 10°C and 40°C cells, validates on 25°C cells
2. LOCO CV: Leave-one-cell-out cross-validation across all cells
3. Cross-experiment: Trains on one experiment, tests on a different experiment

Examples:
    # Basic usage - temperature split (default)
    python examples/benchmark_node_optimizations.py --model neural_ode
    python examples/benchmark_node_optimizations.py --model acla

    # Benchmark specific optimization (always includes baseline for comparison)
    python examples/benchmark_node_optimizations.py --model neural_ode --optimization solver_rk4
    python examples/benchmark_node_optimizations.py --model acla --optimization solver_euler

    # Override training parameters
    python examples/benchmark_node_optimizations.py --model neural_ode --epochs 50 --batch_size 8

    # Include slower configurations (with_adjoint, mixed_precision_bf16)
    python examples/benchmark_node_optimizations.py --model neural_ode --include-slow

    # Cross-validation mode (LOCO CV)
    python examples/benchmark_node_optimizations.py --model neural_ode --split-mode loco_cv
    python examples/benchmark_node_optimizations.py --model acla --split-mode loco_cv --cv-folds A B C

    # Cross-experiment testing (train on Expt 5, test on Expt 1)
    python examples/benchmark_node_optimizations.py --model neural_ode --split-mode cross_experiment --test-experiment 1
    python examples/benchmark_node_optimizations.py --model acla --split-mode cross_experiment --test-experiment 2

    # Cross-experiment testing with holdout cells (train on ALL experiments, test on specific cells)
    python examples/benchmark_node_optimizations.py --model neural_ode --split-mode cross_experiment --holdout-cells A B
    python examples/benchmark_node_optimizations.py --model acla --split-mode cross_experiment --holdout-cells D E
    python examples/benchmark_node_optimizations.py --model acla --split-mode cross_experiment --holdout-cells D E --optimization solver_euler
    python examples/benchmark_node_optimizations.py --model neural_ode --split-mode cross_experiment --holdout-cells D E --optimization solver_euler

    # Use different training experiment (default is Expt 5)
    python examples/benchmark_node_optimizations.py --model neural_ode --experiment_id 4 --split-mode cross_experiment --test-experiment 1

    # Log results to MLflow
    python examples/benchmark_node_optimizations.py --model neural_ode --mlflow

    # Custom output path
    python examples/benchmark_node_optimizations.py --model neural_ode --output results/my_benchmark.csv

    # Combined options
    python examples/benchmark_node_optimizations.py --model neural_ode --split-mode loco_cv --epochs 100 --mlflow --cv-folds A B C D

Available optimizations (for --optimization flag):
    - baseline: Optimized baseline (Dopri5 + relaxed tolerance + no_adjoint)
    - solver_dopri5_strict: Dopri5 with strict tolerance vs relaxed baseline
    - solver_rk4: RK4 fixed-step solver vs Dopri5+relaxed baseline
    - solver_euler: Euler solver vs baseline (fastest)
    - with_adjoint: Dopri5+relaxed with adjoint vs direct backprop baseline (requires --include-slow)
    - mixed_precision_bf16: Baseline with BF16 mixed precision (requires --include-slow)
    - combined: RK4 + no_adjoint vs Dopri5+relaxed baseline

Split modes:
    - temperature (default): Train on 10°C and 40°C, validate on 25°C cells
    - loco_cv: Leave-one-cell-out cross-validation (reports mean ± std across folds)
    - cross_experiment: Two options:
        * Train on one experiment, test on another (requires --test-experiment)
        * Train on ALL experiments, test on holdout cells (requires --holdout-cells)

Visualization:
    - Temperature mode: Visualizes predictions for Cell E (or first validation cell)
    - Cross-experiment mode: Visualizes predictions for test cell (Cell E preferred, fallback to first test cell)
    - LOCO CV mode: No visualization (focuses on aggregated statistics)

Output:
    - CSV file with benchmark results saved to artifacts/benchmarks/
    - Visualization figure saved to artifacts/figures/ (for temperature and cross-experiment modes)
    - Log file saved to artifacts/logs/benchmark_{model}.log
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import torch
import pandas as pd
import logging
import time
import json
from contextlib import contextmanager
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from copy import deepcopy
import csv

# Try to import psutil for memory tracking (optional)
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Set matplotlib backend to non-interactive 'Agg' before importing pyplot
import matplotlib
matplotlib.use('Agg')

from src.data.tables import SummaryDataLoader
from src.pipelines.latent_ode_seq import LatentODESequencePipeline
from src.models.registry import ModelRegistry
# Import models to trigger registration decorators
from src.models.neural_ode import NeuralODEModel
from src.models.acla import ACLAModel
from src.data.splits import temperature_split, loco_cv_splits
from src.data.experiment_config import get_experiment_config, get_experiment_name
from src.training.trainer import Trainer
from src.training.metrics import compute_metrics
from src.tracking.dual_tracker import DualTracker

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Logging Configuration
# ─────────────────────────────────────────────────────────────────

def setup_logging(log_file: Optional[Path] = None):
    """Configure logging with console and optional file output."""
    handlers = [logging.StreamHandler()]
    
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S',
        handlers=handlers,
        force=True
    )


# ─────────────────────────────────────────────────────────────────
# Performance Tracking Utilities
# ─────────────────────────────────────────────────────────────────

@contextmanager
def timer(name: str):
    """Context manager for timing code blocks."""
    start = time.time()
    yield
    elapsed = time.time() - start
    logger.debug(f"[TIME] {name}: {elapsed:.2f}s")


def get_gpu_memory_mb() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0.0


def get_gpu_memory_peak_mb() -> float:
    """Get peak GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**2
    return 0.0


def reset_gpu_memory_stats():
    """Reset GPU memory statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def get_cpu_memory_mb() -> float:
    """Get current CPU memory usage in MB."""
    if HAS_PSUTIL:
        process = psutil.Process()
        return process.memory_info().rss / 1024**2
    return 0.0


# ─────────────────────────────────────────────────────────────────
# Configuration System
# ─────────────────────────────────────────────────────────────────

@dataclass
class OptimizationConfig:
    """Configuration for a single optimization benchmark."""
    name: str
    model_params: Dict[str, Any]
    training_params: Dict[str, Any]
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


def get_baseline_config(model_type: str, input_dim: int, output_dim: int) -> OptimizationConfig:
    """Get baseline configuration (optimized for accuracy based on benchmarks)."""
    if model_type == 'acla':
        model_params = {
            'input_dim': input_dim,
            'output_dim': output_dim,
            'hidden_dim': 64,
            'augment_dim': 20,
            'cnn_filters': [64, 32],
            'solver': 'dopri5',
            'rtol': 1e-4,
            'atol': 1e-5,
            'use_adjoint': False,  # Direct backprop: better accuracy, 5x faster
        }
    elif model_type == 'neural_ode':
        model_params = {
            'input_dim': input_dim,
            'output_dim': output_dim,
            'latent_dim': 32,
            'hidden_dim': 64,
            'solver': 'dopri5',  # Dopri5 with relaxed tolerance: best accuracy (R²=0.9692, RMSE=0.00881)
            'rtol': 1e-3,  # Relaxed tolerance for adaptive solver
            'atol': 1e-4,
            'use_adjoint': False,  # Direct backprop: best accuracy, faster for short sequences
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    training_params = {
        'epochs': 500,  # Default to 100 epochs for full training
        'batch_size': 4,
        'learning_rate': 1e-3,
        'weight_decay': 0.01,
        'early_stopping_patience': 100,  # Set to epochs to disable early stopping (always train full epochs)
        'gradient_clip': 1.0,
        'use_amp': False,  # Disabled for ODE compatibility by default
    }
    
    return OptimizationConfig(
        name='baseline',
        model_params=model_params,
        training_params=training_params,
        description='Optimized baseline (Dopri5 + relaxed tolerance + no_adjoint: R²=0.9692, RMSE=0.00881)'
    )


def get_optimization_configs(model_type: str, input_dim: int, output_dim: int) -> List[OptimizationConfig]:
    """Get list of optimization configurations to benchmark."""
    configs = []
    
    # Baseline (Dopri5 + relaxed tolerance + no_adjoint - best accuracy)
    configs.append(get_baseline_config(model_type, input_dim, output_dim))
    baseline = get_baseline_config(model_type, input_dim, output_dim)
    
    # Dopri5 with strict tolerance (tests strict vs relaxed tolerance)
    # Baseline uses relaxed tolerance, this tests if strict tolerance helps
    dopri5_strict = OptimizationConfig(
        name='solver_dopri5_strict',
        model_params={**baseline.model_params, 'rtol': 1e-4, 'atol': 1e-5},
        training_params=baseline.training_params.copy(),
        description='Dopri5 with strict tolerance (rtol=1e-4, atol=1e-5) vs relaxed tolerance baseline'
    )
    configs.append(dopri5_strict)
    
    # RK4 solver (fixed-step vs adaptive baseline)
    # Tests if fixed-step RK4 can match adaptive Dopri5+relaxed accuracy
    solver_rk4 = OptimizationConfig(
        name='solver_rk4',
        model_params={**baseline.model_params, 'solver': 'rk4'},
        training_params=baseline.training_params.copy(),
        description='RK4 fixed-step solver vs Dopri5+relaxed baseline (tests fixed vs adaptive)'
    )
    configs.append(solver_rk4)
    
    # Euler solver (fastest vs baseline)
    # Tests speed vs accuracy tradeoff
    solver_euler = OptimizationConfig(
        name='solver_euler',
        model_params={**baseline.model_params, 'solver': 'euler'},
        training_params=baseline.training_params.copy(),
        description='Euler solver vs baseline (fastest, tests speed vs accuracy tradeoff)'
    )
    configs.append(solver_euler)
    
    # With adjoint (adjoint vs direct backprop, both with Dopri5+relaxed)
    # Tests gradient computation method impact
    with_adjoint = OptimizationConfig(
        name='with_adjoint',
        model_params={**baseline.model_params, 'use_adjoint': True},
        training_params=baseline.training_params.copy(),
        description='Dopri5+relaxed + adjoint vs direct backprop baseline (tests gradient method)'
    )
    configs.append(with_adjoint)
    
    # Mixed precision BF16 (baseline + mixed precision)
    # Tests if mixed precision helps with Dopri5+relaxed
    mixed_precision_bf16 = OptimizationConfig(
        name='mixed_precision_bf16',
        model_params=baseline.model_params.copy(),
        training_params={**baseline.training_params, 'use_amp': True},
        description='Baseline + BF16 mixed precision (tests precision impact)'
    )
    configs.append(mixed_precision_bf16)
    
    # Combined: RK4 + no_adjoint (tests if RK4 can match baseline with more training)
    # Since baseline is now Dopri5+relaxed, this tests fixed-step alternative
    combined = OptimizationConfig(
        name='combined',
        model_params={
            **baseline.model_params,
            'solver': 'rk4',  # Fixed-step RK4
            'use_adjoint': False,  # Direct backprop (same as baseline)
        },
        training_params=baseline.training_params.copy(),
        description='RK4 + no_adjoint vs Dopri5+relaxed baseline (tests fixed-step alternative)'
    )
    configs.append(combined)
    
    return configs


# ─────────────────────────────────────────────────────────────────
# Benchmark Runner
# ─────────────────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    config_name: str
    model_type: str
    training_time: float
    inference_time: float
    memory_peak_mb: float
    memory_final_mb: float
    rmse: float
    mae: float
    mape: float
    r2: float
    max_ae: float
    nfe: Optional[int] = None  # Number of function evaluations (ODE-specific)
    epochs_trained: int = 0
    best_val_loss: float = float('inf')
    cell_e_predictions: Optional[np.ndarray] = None  # Predictions for visualization cell (seq_len, output_dim)
    cell_e_actual: Optional[np.ndarray] = None  # Actual values for visualization cell (seq_len, output_dim)
    cell_e_time: Optional[np.ndarray] = None  # Time values for visualization cell (seq_len,)
    cell_e_id: Optional[str] = None  # Cell ID used for visualization (e.g., 'E' or first val cell)
    cv_metrics: Optional[Dict[str, float]] = None  # CV statistics (mean/std) if from cross-validation
    n_folds: Optional[int] = None  # Number of CV folds if from cross-validation
    # Per-cell metrics for holdout cell mode
    per_cell_metrics: Optional[Dict[str, Dict[str, float]]] = None  # {cell_id: {rmse, mae, r2, ...}}
    per_cell_predictions: Optional[Dict[str, Dict[str, np.ndarray]]] = None  # {cell_id: {predictions, actual, time}}
    

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        # Replace None with NaN for CSV compatibility
        # Exclude numpy arrays and complex objects from CSV export
        for key, value in list(result.items()):
            if value is None:
                result[key] = float('nan')
            elif isinstance(value, np.ndarray):
                del result[key]  # Don't save arrays to CSV
            elif isinstance(value, dict) and key == 'cv_metrics':
                # Flatten CV metrics into separate columns
                if value:
                    for metric_name, metric_value in value.items():
                        result[f'cv_{metric_name}'] = metric_value
                del result[key]  # Remove original dict
            elif isinstance(value, dict) and key == 'per_cell_metrics':
                # Flatten per-cell metrics into separate columns
                if value:
                    for cell_id, cell_metrics in value.items():
                        for metric_name, metric_value in cell_metrics.items():
                            result[f'cell_{cell_id}_{metric_name}'] = metric_value
                del result[key]  # Remove original dict
            elif isinstance(value, dict) and key == 'per_cell_predictions':
                del result[key]  # Don't save predictions dict to CSV
        return result



class BenchmarkRunner:
    """Runs benchmarks for optimization configurations."""
    
    def __init__(self, device: str = 'auto'):
        """Initialize benchmark runner.
        
        Args:
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        logger.info(f"Using device: {device}")
    
    def _pad_sequences_if_needed(self, samples: List) -> List:
        """Pad sequences to the same length if they have different lengths.
        
        Args:
            samples: List of Sample objects with sequences
            
        Returns:
            List of Sample objects with padded sequences (if needed)
        """
        if not samples:
            return samples
        
        # Ensure samples are tensors for consistent handling
        samples = [s.to_tensor() if hasattr(s, 'to_tensor') else s for s in samples]
        
        # Check if sequences have different lengths
        seq_lens = []
        for s in samples:
            if hasattr(s, 'x') and s.x is not None:
                if isinstance(s.x, torch.Tensor) and s.x.dim() >= 2:
                    seq_lens.append(s.x.shape[0])
                elif isinstance(s.x, np.ndarray) and s.x.ndim >= 2:
                    seq_lens.append(s.x.shape[0])
                else:
                    seq_lens.append(0)
            else:
                seq_lens.append(0)
        
        if len(set(seq_lens)) <= 1:
            # All sequences have the same length, no padding needed
            return samples
        
        # Find max length
        max_len = max(seq_lens)
        if max_len == 0:
            return samples
        
        # Pad sequences to max length
        padded_samples = []
        for s in samples:
            # Clone sample to avoid modifying original
            s_copy = s.clone() if hasattr(s, 'clone') else deepcopy(s)
            
            if hasattr(s_copy, 'x') and s_copy.x is not None:
                # Ensure x is a tensor
                if isinstance(s_copy.x, np.ndarray):
                    s_copy.x = torch.from_numpy(s_copy.x).float()
                
                if isinstance(s_copy.x, torch.Tensor) and s_copy.x.dim() >= 2:
                    current_len = s_copy.x.shape[0]
                    if current_len < max_len:
                        # Pad x by repeating last timestep
                        pad_size = max_len - current_len
                        last_timestep = s_copy.x[-1:]  # (1, feature_dim)
                        padding = last_timestep.repeat(pad_size, 1)  # (pad_size, feature_dim)
                        s_copy.x = torch.cat([s_copy.x, padding], dim=0)
                        
                        # Pad y if it exists and is a sequence
                        if hasattr(s_copy, 'y') and s_copy.y is not None:
                            if isinstance(s_copy.y, np.ndarray):
                                s_copy.y = torch.from_numpy(s_copy.y).float()
                            
                            if isinstance(s_copy.y, torch.Tensor):
                                if s_copy.y.dim() == 1:
                                    # Scalar target - pad with last value
                                    last_y = s_copy.y[-1:]
                                    padding_y = last_y.repeat(pad_size)
                                    s_copy.y = torch.cat([s_copy.y, padding_y], dim=0)
                                elif s_copy.y.dim() >= 2 and s_copy.y.shape[0] < max_len:
                                    # Sequence target
                                    pad_size_y = max_len - s_copy.y.shape[0]
                                    last_y = s_copy.y[-1:]  # (1, output_dim)
                                    padding_y = last_y.repeat(pad_size_y, 1)
                                    s_copy.y = torch.cat([s_copy.y, padding_y], dim=0)
                        
                        # Pad t if it exists - MUST be strictly increasing for ODE
                        if hasattr(s_copy, 't') and s_copy.t is not None:
                            if isinstance(s_copy.t, np.ndarray):
                                s_copy.t = torch.from_numpy(s_copy.t).float()
                            
                            if isinstance(s_copy.t, torch.Tensor) and len(s_copy.t) >= 2:
                                pad_size_t = max_len - current_len
                                # Extrapolate time values (maintain average delta)
                                dt = s_copy.t[-1] - s_copy.t[-2]
                                if dt <= 0:
                                    dt = torch.tensor(1.0, dtype=s_copy.t.dtype, device=s_copy.t.device)
                                last_t = s_copy.t[-1]
                                # Create strictly increasing time values
                                padding_t = torch.arange(1, pad_size_t + 1, dtype=s_copy.t.dtype, device=s_copy.t.device) * dt + last_t
                                s_copy.t = torch.cat([s_copy.t, padding_t], dim=0)
            
            padded_samples.append(s_copy)
        
        logger.info(f"Padded sequences to max length: {max_len} (was {min(seq_lens)}-{max_len})")
        return padded_samples
    
    def run_benchmark(
        self,
        config: OptimizationConfig,
        model_type: str,
        train_samples: List,
        val_samples: List,
        use_mlflow: bool = False,
        mlflow_experiment: str = "benchmark_optimizations"
    ) -> BenchmarkResult:
        """Run a single benchmark configuration.
        
        Args:
            config: Optimization configuration
            model_type: Model type ('acla' or 'neural_ode')
            train_samples: Training samples
            val_samples: Validation samples
            use_mlflow: Whether to log to MLflow
            mlflow_experiment: MLflow experiment name
        
        Returns:
            BenchmarkResult with all metrics
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Benchmarking: {config.name}")
        logger.info(f"Description: {config.description}")
        logger.info(f"{'='*60}")
        
        # Reset memory stats
        reset_gpu_memory_stats()
        
        # Initialize model
        model_class = ModelRegistry.get_class(model_type)
        if model_class is None:
            available = ModelRegistry.list_available()
            raise ValueError(
                f"Model type '{model_type}' not found in registry. "
                f"Available models: {available}. "
                f"Make sure the model module is imported."
            )
        
        try:
            model = model_class(**config.model_params)
        except Exception as e:
            raise RuntimeError(f"Failed to create model '{model_type}': {e}") from e
        
        model = model.to(self.device)
        
        logger.info(f"Model parameters: {model.count_parameters():,}")
        
        # Setup tracker
        tracker = None
        if use_mlflow:
            project_root = Path(__file__).parent.parent
            artifacts_dir = project_root / "artifacts"
            tracker = DualTracker(
                local_base_dir=str(artifacts_dir / "runs"),
                use_tensorboard=False,
                mlflow_tracking_uri=f"sqlite:///{artifacts_dir / 'mlflow.db'}",
                mlflow_experiment_name=mlflow_experiment
            )
            tracker.start_run(f"{model_type}_{config.name}", {
                'model': model_type,
                'config': config.name,
                **config.model_params,
                **config.training_params
            })
        
        # Training benchmark
        training_start = time.time()
        epochs_trained = 0
        best_val_loss = float('inf')
        
        try:
            # Handle variable sequence lengths (common in cross-experiment mode)
            # Pad sequences to the same length if needed
            train_samples_padded = self._pad_sequences_if_needed(train_samples)
            val_samples_padded = self._pad_sequences_if_needed(val_samples)
            
            trainer = Trainer(
                model,
                config.training_params,
                tracker=tracker,
                device=self.device,
                verbose=False
            )
            
            # Run training
            history = trainer.fit(train_samples_padded, val_samples_padded)
            epochs_trained = len(history['train_loss'])
            best_val_loss = trainer.best_val_loss
            
            training_time = time.time() - training_start
            
            logger.info(f"Training completed in {training_time:.2f}s")
            logger.info(f"Epochs trained: {epochs_trained}")
            logger.info(f"Best validation loss: {best_val_loss:.5f}")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
            training_time = time.time() - training_start
            epochs_trained = 0
        
        # Memory after training
        memory_peak_mb = get_gpu_memory_peak_mb()
        memory_final_mb = get_gpu_memory_mb()
        
        # Inference benchmark
        inference_start = time.time()
        try:
            # Warmup
            with torch.no_grad():
                sample = train_samples[0]
                x = sample.x.unsqueeze(0).to(self.device)
                t = sample.t.to(self.device) if hasattr(sample, 't') and sample.t is not None else None
                _ = model(x, t=t)
            
            # Benchmark inference
            n_inference_runs = 10
            inference_times = []
            
            for _ in range(n_inference_runs):
                start = time.time()
                with torch.no_grad():
                    sample = val_samples[0]
                    x = sample.x.unsqueeze(0).to(self.device)
                    t = sample.t.to(self.device) if hasattr(sample, 't') and sample.t is not None else None
                    _ = model(x, t=t)
                inference_times.append(time.time() - start)
            
            inference_time = np.mean(inference_times)
            logger.info(f"Inference time (avg over {n_inference_runs} runs): {inference_time*1000:.2f}ms")
            
        except Exception as e:
            logger.error(f"Inference benchmark failed: {e}")
            inference_time = float('nan')
        
        # Accuracy evaluation
        try:
            # Use padded samples for prediction
            val_samples_padded = self._pad_sequences_if_needed(val_samples)
            
            trainer = Trainer(
                model,
                config.training_params,
                device=self.device,
                verbose=False
            )
            predictions_padded = trainer.predict(val_samples_padded)
            
            # Trim predictions back to original lengths and collect targets
            # This handles variable-length sequences properly
            y_true_list = []
            y_pred_list = []
            cell_ids = []
            
            for i, sample in enumerate(val_samples):
                # Get original sequence length
                orig_len = sample.x.shape[0] if hasattr(sample.x, 'shape') else len(sample.x)
                
                # Get target (original, not padded)
                y_sample = sample.y.numpy() if isinstance(sample.y, torch.Tensor) else sample.y
                y_true_list.append(y_sample)
                
                # Get prediction and trim to original length
                if predictions_padded.ndim == 3:  # (batch, seq_len, output_dim)
                    pred_sample = predictions_padded[i, :orig_len, :]
                else:
                    pred_sample = predictions_padded[i]
                y_pred_list.append(pred_sample)
                
                # Track cell ID
                cell_id = sample.meta.get('cell_id', f'cell_{i}')
                cell_ids.append(cell_id)
            
            # Compute per-cell metrics
            per_cell_metrics = {}
            per_cell_predictions = {}
            
            for i, cell_id in enumerate(cell_ids):
                y_true_cell = y_true_list[i]
                y_pred_cell = y_pred_list[i]
                
                # Get time values for this cell
                sample = val_samples[i]
                if hasattr(sample, 't') and sample.t is not None:
                    t_cell = sample.t.cpu().numpy() if isinstance(sample.t, torch.Tensor) else sample.t
                else:
                    t_cell = np.arange(len(y_pred_cell))
                
                # Compute metrics for this cell
                if y_true_cell.shape[-1] == 1:
                    cell_metrics = compute_metrics(y_true_cell.flatten(), y_pred_cell.flatten())
                else:
                    # Multi-target: average across targets
                    all_metrics = []
                    for j in range(y_true_cell.shape[-1]):
                        m = compute_metrics(y_true_cell[:, j], y_pred_cell[:, j])
                        all_metrics.append(m)
                    cell_metrics = {
                        'rmse': np.mean([m['rmse'] for m in all_metrics]),
                        'mae': np.mean([m['mae'] for m in all_metrics]),
                        'mape': np.mean([m['mape'] for m in all_metrics]),
                        'r2': np.mean([m['r2'] for m in all_metrics]),
                        'max_ae': np.mean([m['max_ae'] for m in all_metrics]),
                    }
                
                per_cell_metrics[cell_id] = cell_metrics
                per_cell_predictions[cell_id] = {
                    'predictions': y_pred_cell,
                    'actual': y_true_cell,
                    'time': t_cell
                }
                
                # Log per-cell metrics
                logger.info(f"Cell {cell_id}: RMSE={cell_metrics['rmse']:.5f}, MAE={cell_metrics['mae']:.5f}, R²={cell_metrics['r2']:.4f}")
            
            # Use first cell for summary metrics (or average if multiple)
            if len(per_cell_metrics) == 1:
                metrics = list(per_cell_metrics.values())[0]
            else:
                # Average across cells
                metrics = {
                    'rmse': np.mean([m['rmse'] for m in per_cell_metrics.values()]),
                    'mae': np.mean([m['mae'] for m in per_cell_metrics.values()]),
                    'mape': np.mean([m['mape'] for m in per_cell_metrics.values()]),
                    'r2': np.mean([m['r2'] for m in per_cell_metrics.values()]),
                    'max_ae': np.mean([m['max_ae'] for m in per_cell_metrics.values()]),
                }
                logger.info(f"Average across {len(per_cell_metrics)} cells: RMSE={metrics['rmse']:.5f}, R²={metrics['r2']:.4f}")
            
            # Extract predictions for visualization (prefer Cell E, fallback to first validation cell)
            cell_e_predictions = None
            cell_e_actual = None
            cell_e_time = None
            cell_e_id = None  # Initialize before try block
            
            try:
                # Try to find Cell E first, then fallback to first validation cell
                if 'E' in per_cell_predictions:
                    cell_e_id = 'E'
                    cell_e_predictions = per_cell_predictions['E']['predictions']
                    cell_e_actual = per_cell_predictions['E']['actual']
                    cell_e_time = per_cell_predictions['E']['time']
                elif len(per_cell_predictions) > 0:
                    # Fallback to first validation cell
                    cell_e_id = list(per_cell_predictions.keys())[0]
                    cell_e_predictions = per_cell_predictions[cell_e_id]['predictions']
                    cell_e_actual = per_cell_predictions[cell_e_id]['actual']
                    cell_e_time = per_cell_predictions[cell_e_id]['time']
            except Exception as e:
                logger.debug(f"Could not extract cell predictions: {e}")
                cell_e_predictions = None
                cell_e_actual = None
                cell_e_time = None
                cell_e_id = None
            

        except Exception as e:
            logger.error(f"Accuracy evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            metrics = {
                'rmse': float('nan'),
                'mae': float('nan'),
                'mape': float('nan'),
                'r2': float('nan'),
                'max_ae': float('nan'),
            }
            cell_e_predictions = None
            cell_e_actual = None
            cell_e_time = None
            cell_e_id = None
            per_cell_metrics = None
            per_cell_predictions = None
        

        # Try to extract ODE NFE (Number of Function Evaluations)
        nfe = None
        try:
            # This is tricky - torchdiffeq doesn't expose NFE directly
            # We can try to access it from the solver state if available
            # For now, we'll leave it as None and note it in the documentation
            pass
        except Exception:
            pass
        
        # Log to MLflow
        if tracker:
            tracker.log_metrics({
                'training_time': training_time,
                'inference_time': inference_time,
                'memory_peak_mb': memory_peak_mb,
                'memory_final_mb': memory_final_mb,
                'rmse': metrics['rmse'],
                'mae': metrics['mae'],
                'mape': metrics['mape'],
                'r2': metrics['r2'],
                'max_ae': metrics['max_ae'],
                'epochs_trained': epochs_trained,
                'best_val_loss': best_val_loss,
            })
            tracker.end_run()
        
        result = BenchmarkResult(
            config_name=config.name,
            model_type=model_type,
            training_time=training_time,
            inference_time=inference_time,
            memory_peak_mb=memory_peak_mb,
            memory_final_mb=memory_final_mb,
            rmse=metrics['rmse'],
            mae=metrics['mae'],
            mape=metrics['mape'],
            r2=metrics['r2'],
            max_ae=metrics['max_ae'],
            nfe=nfe,
            epochs_trained=epochs_trained,
            best_val_loss=best_val_loss,
            cell_e_predictions=cell_e_predictions,
            cell_e_actual=cell_e_actual,
            cell_e_time=cell_e_time,
            cell_e_id=cell_e_id,
            per_cell_metrics=per_cell_metrics,
            per_cell_predictions=per_cell_predictions,
        )


        
        return result
    
    def run_cross_validation(
        self,
        configs: List[OptimizationConfig],
        model_type: str,
        cv_splits: List[Tuple[str, List, List]],
        input_dim: int,
        output_dim: int,
        use_mlflow: bool = False,
        cv_folds: Optional[List[str]] = None
    ) -> List[BenchmarkResult]:
        """Run cross-validation benchmarks across all folds.
        
        Args:
            configs: List of optimization configurations to benchmark
            model_type: Model type ('acla' or 'neural_ode')
            cv_splits: List of (cell_id, train_samples, val_samples) tuples
            input_dim: Input dimension
            output_dim: Output dimension
            use_mlflow: Whether to log to MLflow
            cv_folds: Optional list of cell IDs to limit CV folds (e.g., ['A', 'B', 'C'])
        
        Returns:
            List of aggregated BenchmarkResult objects (one per config)
        """
        if cv_folds:
            # Filter to requested folds
            cv_splits = [(cell_id, train, val) for cell_id, train, val in cv_splits 
                        if cell_id in cv_folds]
            logger.info(f"Limited to {len(cv_folds)} CV folds: {cv_folds}")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Running {len(cv_splits)}-fold LOCO Cross-Validation")
        logger.info(f"{'='*60}")
        
        # Store results per config per fold
        config_results: Dict[str, List[BenchmarkResult]] = {}
        
        for fold_idx, (cell_id, train_samples, val_samples) in enumerate(cv_splits, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"CV Fold {fold_idx}/{len(cv_splits)}: Testing on Cell {cell_id}")
            logger.info(f"{'='*60}")
            
            # Run benchmarks for this fold
            for config in configs:
                try:
                    result = self.run_benchmark(
                        config,
                        model_type,
                        train_samples,
                        val_samples,
                        use_mlflow=use_mlflow
                    )
                    
                    # Store result with fold info
                    if config.name not in config_results:
                        config_results[config.name] = []
                    config_results[config.name].append(result)
                    
                except Exception as e:
                    logger.error(f"Benchmark failed for {config.name} on fold {cell_id}: {e}")
                    continue
        
        # Aggregate results across folds
        aggregated_results = []
        for config_name, fold_results in config_results.items():
            if not fold_results:
                continue
            
            # Compute mean and std across folds
            metrics = ['training_time', 'inference_time', 'memory_peak_mb', 'memory_final_mb',
                      'rmse', 'mae', 'mape', 'r2', 'max_ae', 'epochs_trained', 'best_val_loss']
            
            aggregated = {}
            for metric in metrics:
                values = [getattr(r, metric) for r in fold_results if not np.isnan(getattr(r, metric))]
                if values:
                    aggregated[f'{metric}_mean'] = np.mean(values)
                    aggregated[f'{metric}_std'] = np.std(values)
                else:
                    aggregated[f'{metric}_mean'] = float('nan')
                    aggregated[f'{metric}_std'] = float('nan')
            
            # Create aggregated result (use mean values for main fields)
            aggregated_result = BenchmarkResult(
                config_name=config_name,
                model_type=model_type,
                training_time=aggregated['training_time_mean'],
                inference_time=aggregated['inference_time_mean'],
                memory_peak_mb=aggregated['memory_peak_mb_mean'],
                memory_final_mb=aggregated['memory_final_mb_mean'],
                rmse=aggregated['rmse_mean'],
                mae=aggregated['mae_mean'],
                mape=aggregated['mape_mean'],
                r2=aggregated['r2_mean'],
                max_ae=aggregated['max_ae_mean'],
                nfe=None,
                epochs_trained=int(aggregated['epochs_trained_mean']) if not np.isnan(aggregated['epochs_trained_mean']) else 0,
                best_val_loss=aggregated['best_val_loss_mean'],
                cell_e_predictions=None,  # Don't store predictions for CV
                cell_e_actual=None,
                cell_e_time=None
            )
            
            # Store aggregated metrics as attributes for later use
            aggregated_result.cv_metrics = aggregated
            aggregated_result.n_folds = len(fold_results)
            
            aggregated_results.append(aggregated_result)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Cross-Validation Complete: {len(cv_splits)} folds")
        logger.info(f"{'='*60}")
        
        return aggregated_results


# ─────────────────────────────────────────────────────────────────
# Results Storage and Reporting
# ─────────────────────────────────────────────────────────────────

def visualize_cell_e_predictions(
    results: List[BenchmarkResult], 
    output_path: Path, 
    model_type: str,
    split_mode: str = 'temperature',
    test_experiment_id: Optional[int] = None
):
    """Create visualization of predictions for validation/test cells from all configurations.
    
    When multiple holdout cells are present, creates a subplot for each cell.
    
    Args:
        results: List of benchmark results with cell predictions
        output_path: Path to save the figure
        model_type: Type of model being benchmarked ('neural_ode' or 'acla')
        split_mode: Split mode used ('temperature', 'cross_experiment')
        test_experiment_id: Test experiment ID (for cross_experiment mode)
    """
    import matplotlib.pyplot as plt
    
    # Filter results that have predictions
    results_with_preds = [r for r in results if r.cell_e_predictions is not None]
    
    if not results_with_preds:
        logger.warning("No cell predictions available for visualization")
        return
    
    # Check if we have per-cell predictions for multiple cells
    first_result = results_with_preds[0]
    
    if first_result.per_cell_predictions and len(first_result.per_cell_predictions) > 1:
        # Multiple holdout cells - create subplot for each cell
        cell_ids = list(first_result.per_cell_predictions.keys())
        n_cells = len(cell_ids)
        n_configs = len(results_with_preds)
        
        # Create figure with one row per cell
        fig, axes = plt.subplots(n_cells, 1, figsize=(12, 5 * n_cells))
        if n_cells == 1:
            axes = [axes]
        
        # Color palette for different configurations
        colors = plt.cm.tab10(np.linspace(0, 1, n_configs))
        
        for cell_idx, cell_id in enumerate(cell_ids):
            ax = axes[cell_idx]
            
            # Get data for this cell from first result (actual values are same for all configs)
            cell_data = first_result.per_cell_predictions.get(cell_id)
            if cell_data is None:
                continue
                
            time_values = cell_data['time']
            actual_values = cell_data['actual']
            
            # Plot actual values
            if actual_values.shape[-1] == 1:
                y_actual = actual_values.flatten()
            else:
                y_actual = actual_values[:, 0]  # First target for multi-output
            ax.plot(time_values, y_actual, 'ko-', label='Actual', 
                   linewidth=2, markersize=6, alpha=0.7)
            
            # Plot predictions from each configuration
            for result, color in zip(results_with_preds, colors):
                if result.per_cell_predictions is None or cell_id not in result.per_cell_predictions:
                    continue
                    
                pred_data = result.per_cell_predictions[cell_id]
                y_pred = pred_data['predictions']
                
                if y_pred.shape[-1] == 1:
                    y_pred = y_pred.flatten()
                else:
                    y_pred = y_pred[:, 0]  # First target
                
                # Get per-cell R² for this config
                cell_r2 = float('nan')
                if result.per_cell_metrics and cell_id in result.per_cell_metrics:
                    cell_r2 = result.per_cell_metrics[cell_id].get('r2', float('nan'))
                
                label = result.config_name
                if not np.isnan(cell_r2):
                    label += f" (R²={cell_r2:.3f})"
                
                ax.plot(time_values, y_pred, '-', label=label, 
                       linewidth=2, alpha=0.8, color=color)
            
            ax.set_xlabel('Time (days)' if time_values.max() > 10 else 'Timestep', fontsize=11)
            ax.set_ylabel('SOH', fontsize=11)
            ax.set_title(f'Cell {cell_id} Predictions', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9, loc='best', ncol=1)
            ax.grid(True, alpha=0.3)
        
        # Create overall title
        if split_mode == 'cross_experiment' and test_experiment_id:
            from src.data.experiment_config import get_experiment_name
            test_exp_name = get_experiment_name(test_experiment_id)
            title = f'{model_type.upper()} Model - Holdout Cells {", ".join(cell_ids)} (Test: Expt {test_experiment_id})'
        else:
            title = f'{model_type.upper()} Model - Holdout Cells {", ".join(cell_ids)}'
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        # Save figure
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
        logger.info(f"Cells {', '.join(cell_ids)} visualization saved to {output_path}")
        plt.close()
        
    else:
        # Single cell - use original visualization logic
        cell_id = first_result.cell_e_id or 'E'
        time_values = first_result.cell_e_time
        if time_values is None:
            seq_len = first_result.cell_e_predictions.shape[0]
            time_values = np.arange(seq_len)
        
        actual_values = first_result.cell_e_actual
        output_dim = first_result.cell_e_predictions.shape[-1]
        
        # Create figure
        if output_dim == 1:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            axes = [ax]
            target_names = ['Prediction']
        else:
            fig, axes = plt.subplots(output_dim, 1, figsize=(10, 4 * output_dim))
            if not isinstance(axes, np.ndarray):
                axes = [axes]
            target_names = ['LAM_NE', 'LAM_PE', 'LLI'][:output_dim]
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(results_with_preds)))
        
        for ax_idx, (ax, target_name) in enumerate(zip(axes, target_names)):
            if actual_values is not None:
                if output_dim == 1:
                    y_actual = actual_values.flatten()
                else:
                    y_actual = actual_values[:, ax_idx]
                ax.plot(time_values, y_actual, 'ko-', label='Actual', 
                       linewidth=2, markersize=6, alpha=0.7)
            
            for result, color in zip(results_with_preds, colors):
                if output_dim == 1:
                    y_pred = result.cell_e_predictions.flatten()
                else:
                    y_pred = result.cell_e_predictions[:, ax_idx]
                
                # Get per-cell R² if available
                cell_r2 = result.r2
                if result.per_cell_metrics and cell_id in result.per_cell_metrics:
                    cell_r2 = result.per_cell_metrics[cell_id].get('r2', result.r2)
                
                label = result.config_name
                if not np.isnan(cell_r2):
                    label += f" (R²={cell_r2:.3f})"
                
                ax.plot(time_values, y_pred, '-', label=label, 
                       linewidth=2, alpha=0.8, color=color)
            
            ax.set_xlabel('Time (days)' if time_values.max() > 10 else 'Timestep', fontsize=11)
            ax.set_ylabel(target_name, fontsize=11)
            ax.set_title(f'{target_name} Predictions - Cell {cell_id}', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9, loc='best', ncol=1)
            ax.grid(True, alpha=0.3)
            
            if actual_values is not None:
                if output_dim == 1:
                    y_max = max(actual_values.flatten().max(), 
                               max([r.cell_e_predictions.flatten().max() for r in results_with_preds]))
                    y_min = min(actual_values.flatten().min(),
                               min([r.cell_e_predictions.flatten().min() for r in results_with_preds]))
                else:
                    y_max = max(actual_values[:, ax_idx].max(),
                               max([r.cell_e_predictions[:, ax_idx].max() for r in results_with_preds]))
                    y_min = min(actual_values[:, ax_idx].min(),
                               min([r.cell_e_predictions[:, ax_idx].min() for r in results_with_preds]))
                ax.set_ylim([y_min - 0.05 * abs(y_max - y_min), y_max * 1.1])
        
        if split_mode == 'cross_experiment' and test_experiment_id:
            from src.data.experiment_config import get_experiment_name
            test_exp_name = get_experiment_name(test_experiment_id)
            title = f'{model_type.upper()} Model Predictions - Cell {cell_id} (Test: Expt {test_experiment_id} - {test_exp_name})'
        elif split_mode == 'temperature':
            title = f'{model_type.upper()} Model Predictions - Cell {cell_id} (25°C Holdout)'
        else:
            title = f'{model_type.upper()} Model Predictions - Cell {cell_id}'
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
        logger.info(f"Cell {cell_id} visualization saved to {output_path}")
        plt.close()




def save_results(results: List[BenchmarkResult], output_path: Path):
    """Save benchmark results to CSV.
    
    Args:
        results: List of benchmark results
        output_path: Path to save CSV file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    rows = [r.to_dict() for r in results]
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")


def print_comparison_table(results: List[BenchmarkResult]):
    """Print formatted comparison table of results.
    
    Args:
        results: List of benchmark results
    """
    if not results:
        return
    
    # Check if results are from CV (have cv_metrics)
    is_cv = any(r.cv_metrics is not None for r in results)
    
    logger.info("\n" + "="*80)
    logger.info("BENCHMARK COMPARISON" + (" (Cross-Validation)" if is_cv else ""))
    logger.info("="*80)
    
    # Find baseline for relative comparisons
    baseline = next((r for r in results if r.config_name == 'baseline'), None)
    
    # Format table header
    if is_cv:
        header = f"{'Config':<20} {'Train Time':<18} {'Inf Time':<18} {'Memory':<12} {'RMSE':<18} {'R²':<18}"
        logger.info(header)
        logger.info("-"*80)
    else:
        header = f"{'Config':<20} {'Train Time':<12} {'Inf Time':<12} {'Memory':<12} {'RMSE':<10} {'R²':<10}"
        logger.info(header)
        logger.info("-"*80)
    
    for result in results:
        # Format training time
        if is_cv and result.cv_metrics:
            train_time_mean = result.cv_metrics.get('training_time_mean', result.training_time)
            train_time_std = result.cv_metrics.get('training_time_std', 0.0)
            train_time_str = f"{train_time_mean:.2f}±{train_time_std:.2f}s"
            if baseline and baseline.config_name != result.config_name and baseline.cv_metrics:
                baseline_train_mean = baseline.cv_metrics.get('training_time_mean', baseline.training_time)
                speedup = baseline_train_mean / train_time_mean if train_time_mean > 0 else float('inf')
                train_time_str += f" ({speedup:.2f}x)"
        else:
            train_time_str = f"{result.training_time:.2f}s"
            if baseline and baseline.config_name != result.config_name:
                speedup = baseline.training_time / result.training_time if result.training_time > 0 else float('inf')
                train_time_str += f" ({speedup:.2f}x)"
        
        # Format inference time
        if is_cv and result.cv_metrics:
            inf_time_mean = result.cv_metrics.get('inference_time_mean', result.inference_time)
            inf_time_std = result.cv_metrics.get('inference_time_std', 0.0)
            if not np.isnan(inf_time_mean):
                inf_time_str = f"{inf_time_mean*1000:.2f}±{inf_time_std*1000:.2f}ms"
                if baseline and baseline.config_name != result.config_name and baseline.cv_metrics:
                    baseline_inf_mean = baseline.cv_metrics.get('inference_time_mean', baseline.inference_time)
                    speedup = baseline_inf_mean / inf_time_mean if inf_time_mean > 0 else float('inf')
                    inf_time_str += f" ({speedup:.2f}x)"
            else:
                inf_time_str = "N/A"
        else:
            inf_time_str = f"{result.inference_time*1000:.2f}ms" if not np.isnan(result.inference_time) else "N/A"
            if baseline and baseline.config_name != result.config_name and not np.isnan(result.inference_time) and not np.isnan(baseline.inference_time):
                speedup = baseline.inference_time / result.inference_time if result.inference_time > 0 else float('inf')
                inf_time_str += f" ({speedup:.2f}x)"
        
        memory_str = f"{result.memory_peak_mb:.0f}MB"
        
        # Format RMSE
        if is_cv and result.cv_metrics:
            rmse_mean = result.cv_metrics.get('rmse_mean', result.rmse)
            rmse_std = result.cv_metrics.get('rmse_std', 0.0)
            rmse_str = f"{rmse_mean:.5f}±{rmse_std:.5f}" if not np.isnan(rmse_mean) else "N/A"
        else:
            rmse_str = f"{result.rmse:.5f}" if not np.isnan(result.rmse) else "N/A"
        
        # Format R²
        if is_cv and result.cv_metrics:
            r2_mean = result.cv_metrics.get('r2_mean', result.r2)
            r2_std = result.cv_metrics.get('r2_std', 0.0)
            r2_str = f"{r2_mean:.4f}±{r2_std:.4f}" if not np.isnan(r2_mean) else "N/A"
        else:
            r2_str = f"{result.r2:.4f}" if not np.isnan(result.r2) else "N/A"
        
        logger.info(f"{result.config_name:<20} {train_time_str:<18} {inf_time_str:<18} {memory_str:<12} {rmse_str:<18} {r2_str:<18}")
    
    logger.info("="*80)
    
    # Summary statistics
    if baseline and not is_cv:
        logger.info("\nSpeedup Summary (relative to baseline):")
        for result in results:
            if result.config_name != baseline.config_name:
                if result.training_time > 0 and baseline.training_time > 0:
                    speedup = baseline.training_time / result.training_time
                    logger.info(f"  {result.config_name}: {speedup:.2f}x training speedup")
    
    if is_cv:
        n_folds = results[0].n_folds if results else 0
        logger.info(f"\nCross-Validation: {n_folds} folds (mean ± std)")


# ─────────────────────────────────────────────────────────────────
# Main Function
# ─────────────────────────────────────────────────────────────────

def load_data(
    experiment_id: int = 5,
    base_path: Optional[Path] = None,
    split_mode: str = 'temperature'
):
    """Load and prepare data for benchmarking.
    
    Args:
        experiment_id: Experiment ID
        base_path: Base path to raw data
        split_mode: Split mode - 'temperature', 'loco_cv', or 'cross_experiment'
    
    Returns:
        For 'temperature': (train_samples, val_samples, input_dim, output_dim)
        For 'loco_cv': (cv_splits, input_dim, output_dim) where cv_splits is list of (cell_id, train, val) tuples
        For 'cross_experiment': (train_samples, test_samples, input_dim, output_dim)
    """
    if base_path is None:
        project_root = Path(__file__).parent.parent
        base_path = project_root / "Raw Data"
    
    logger.info(f"Loading data from Experiment {experiment_id} ({get_experiment_name(experiment_id)})...")
    
    # Get experiment-specific configuration
    exp_config = get_experiment_config(experiment_id)
    cells = exp_config['cells']
    temp_map = exp_config['temp_map']
    
    # Load summary data
    loader = SummaryDataLoader(experiment_id, base_path)
    df = loader.load_all_cells(cells=cells, temp_map=temp_map)
    
    # Create sequence pipeline
    pipeline = LatentODESequencePipeline(
        time_unit="days",
        max_seq_len=None,
        normalize=True
    )
    
    samples = pipeline.fit_transform({'df': df})
    logger.info(f"Created {len(samples)} sequence samples")
    
    if len(samples) > 0:
        input_dim = samples[0].feature_dim
        output_dim = samples[0].y.shape[-1] if hasattr(samples[0], 'y') else 1
        logger.info(f"Input dimension: {input_dim}, Output dimension: {output_dim}")
    else:
        raise ValueError("No samples created")
    
    # Split data based on mode
    if split_mode == 'temperature':
        train_samples, val_samples = temperature_split(
            samples,
            train_temps=[10, 40],
            val_temps=[25]
        )
        logger.info(f"Train: {len(train_samples)} cells, Val: {len(val_samples)} cells")
        return train_samples, val_samples, input_dim, output_dim
    
    elif split_mode == 'loco_cv':
        cv_splits = loco_cv_splits(samples)
        logger.info(f"Created {len(cv_splits)} LOCO CV folds")
        for cell_id, train, val in cv_splits:
            logger.info(f"  Fold {cell_id}: Train {len(train)} cells, Val {len(val)} cells")
        return cv_splits, input_dim, output_dim
    
    else:
        raise ValueError(f"Unknown split_mode: {split_mode}. Use 'temperature' or 'loco_cv'")


def load_cross_experiment_data(
    train_experiment_id: int = 5,
    test_experiment_id: int = 1,
    base_path: Optional[Path] = None,
    holdout_cells: Optional[List[str]] = None
):
    """Load data from different experiments for cross-experiment testing.
    
    If holdout_cells is None: Train on train_experiment, test on test_experiment (original behavior)
    If holdout_cells is provided: Train on ALL experiments except holdout_cells, test on holdout_cells
    
    Args:
        train_experiment_id: Experiment ID for training data (or starting experiment if holdout_cells provided)
        test_experiment_id: Experiment ID for test data (ignored if holdout_cells provided)
        base_path: Base path to raw data
        holdout_cells: Optional list of cell IDs to hold out (e.g., ['A', 'B']). 
                      If provided, loads all experiments and creates LOCO splits.
    
    Returns:
        Tuple of (train_samples, test_samples, input_dim, output_dim)
    """
    if base_path is None:
        project_root = Path(__file__).parent.parent
        base_path = project_root / "Raw Data"
    
    if holdout_cells is None:
        # Original behavior: train on one experiment, test on another
        logger.info(f"Loading cross-experiment data:")
        logger.info(f"  Train: Experiment {train_experiment_id} ({get_experiment_name(train_experiment_id)})")
        logger.info(f"  Test:  Experiment {test_experiment_id} ({get_experiment_name(test_experiment_id)})")
        
        # Load training experiment
        train_config = get_experiment_config(train_experiment_id)
        train_loader = SummaryDataLoader(train_experiment_id, base_path)
        train_df = train_loader.load_all_cells(
            cells=train_config['cells'],
            temp_map=train_config['temp_map']
        )
        
        # Load test experiment
        test_config = get_experiment_config(test_experiment_id)
        test_loader = SummaryDataLoader(test_experiment_id, base_path)
        test_df = test_loader.load_all_cells(
            cells=test_config['cells'],
            temp_map=test_config['temp_map']
        )
        
        # Create pipeline and fit on training data only
        pipeline = LatentODESequencePipeline(
            time_unit="days",
            max_seq_len=None,
            normalize=True
        )
        
        # Fit pipeline on training data
        train_samples = pipeline.fit_transform({'df': train_df})
        logger.info(f"Created {len(train_samples)} training samples")
        
        # Transform test data (using scaler fitted on training data)
    else:
        # New behavior: train on all experiments, hold out specific cells
        logger.info(f"Loading cross-experiment data with holdout cells:")
        logger.info(f"  Holdout cells: {holdout_cells}")
        logger.info(f"  Loading ALL experiments (1-5) and holding out specified cells")
        
        # Load all experiments (skip experiments that fail to load)
        all_dfs = []
        loaded_experiments = []
        skipped_experiments = []
        
        for exp_id in range(1, 6):
            try:
                exp_config = get_experiment_config(exp_id)
                loader = SummaryDataLoader(exp_id, base_path)
                df = loader.load_all_cells(
                    cells=exp_config['cells'],
                    temp_map=exp_config['temp_map']
                )
                if len(df) > 0:
                    all_dfs.append(df)
                    loaded_experiments.append(exp_id)
                    logger.info(f"  Loaded Experiment {exp_id} ({get_experiment_name(exp_id)}): {len(df)} samples")
                else:
                    skipped_experiments.append(exp_id)
                    logger.warning(f"  Skipping Experiment {exp_id} ({get_experiment_name(exp_id)}): No data loaded")
            except (ValueError, FileNotFoundError, KeyError) as e:
                skipped_experiments.append(exp_id)
                logger.warning(f"  Skipping Experiment {exp_id} ({get_experiment_name(exp_id)}): {e}")
                continue
        
        # Combine all dataframes
        if not all_dfs:
            raise ValueError("No experiments could be loaded. Check data paths and cell IDs.")
        
        combined_df = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"  Combined: {len(combined_df)} total samples from {len(all_dfs)} experiments")
        
        # Summary of loaded vs skipped experiments
        if loaded_experiments:
            logger.info(f"  ✓ Successfully loaded experiments: {loaded_experiments}")
        if skipped_experiments:
            logger.warning(f"  ✗ Skipped experiments (missing data): {skipped_experiments}")
        
        # Split dataframes FIRST (before fitting pipeline) to ensure holdout cells are completely unseen
        train_df = combined_df[~combined_df['cell_id'].isin(holdout_cells)].copy()
        test_df = combined_df[combined_df['cell_id'].isin(holdout_cells)].copy()
        
        logger.info(f"  Split: {len(train_df)} training samples, {len(test_df)} test samples")
        
        # Create pipeline and fit ONLY on training data (holdout cells completely excluded)
        pipeline = LatentODESequencePipeline(
            time_unit="days",
            max_seq_len=None,
            normalize=True
        )
        
        # Fit pipeline on training data only (holdout cells not seen at all)
        train_samples = pipeline.fit_transform({'df': train_df})
        logger.info(f"Created {len(train_samples)} training sequence samples")
        
        # Transform test data using scaler fitted on training data only
        test_samples = pipeline.transform({'df': test_df})
        logger.info(f"Created {len(test_samples)} test sequence samples")
        
        logger.info(f"  Train: {len(train_samples)} samples (all cells except {holdout_cells})")
        logger.info(f"  Test:  {len(test_samples)} samples (cells {holdout_cells})")
        
        # Get dimensions from samples
        if len(train_samples) > 0:
            input_dim = train_samples[0].feature_dim
            output_dim = train_samples[0].y.shape[-1] if hasattr(train_samples[0], 'y') else 1
            logger.info(f"Input dimension: {input_dim}, Output dimension: {output_dim}")
            return train_samples, test_samples, input_dim, output_dim
        else:
            raise ValueError("No training samples created")


def main():
    """Main benchmark function."""
    parser = argparse.ArgumentParser(description='Benchmark NODE model optimizations')
    parser.add_argument('--model', type=str, required=True, choices=['acla', 'neural_ode'],
                       help='Model type to benchmark')
    parser.add_argument('--optimization', type=str, default=None,
                       help='Specific optimization to benchmark (default: all)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Override batch size')
    parser.add_argument('--output', type=str, default=None,
                       help='Output CSV path (default: artifacts/benchmarks/benchmark_<model>_<timestamp>.csv)')
    parser.add_argument('--mlflow', action='store_true',
                       help='Log results to MLflow')
    parser.add_argument('--experiment_id', type=int, default=5,
                       help='Experiment ID for data loading')
    parser.add_argument('--include-slow', action='store_true',
                       help='Include slower configurations (with_adjoint, mixed_precision_bf16)')
    parser.add_argument('--split-mode', type=str, default='temperature',
                       choices=['temperature', 'loco_cv', 'cross_experiment'],
                       help='Data split mode: temperature (default), loco_cv (cross-validation), or cross_experiment')
    parser.add_argument('--test-experiment', type=int, default=None,
                       help='Test experiment ID for cross-experiment mode (required if --split-mode cross_experiment and --holdout-cells not provided)')
    parser.add_argument('--holdout-cells', type=str, nargs='+', default=None,
                       help='Cell IDs to hold out for testing (e.g., --holdout-cells A B). If provided, trains on ALL experiments and tests on these cells.')
    parser.add_argument('--cv-folds', type=str, nargs='+', default=None,
                       help='Limit CV folds to specific cell IDs (e.g., --cv-folds A B C)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.split_mode == 'cross_experiment':
        if args.holdout_cells is None and args.test_experiment is None:
            parser.error("--test-experiment or --holdout-cells is required when --split-mode is cross_experiment")
        if args.holdout_cells is not None and args.test_experiment is not None:
            logger.warning("--test-experiment ignored when --holdout-cells is provided")
    
    if args.test_experiment is not None and args.split_mode != 'cross_experiment':
        logger.warning("--test-experiment ignored (only used with --split-mode cross_experiment)")
    
    if args.holdout_cells is not None and args.split_mode != 'cross_experiment':
        logger.warning("--holdout-cells ignored (only used with --split-mode cross_experiment)")
    
    # Setup logging
    project_root = Path(__file__).parent.parent
    log_file = project_root / "artifacts" / "logs" / f"benchmark_{args.model}.log"
    setup_logging(log_file)
    
    logger.info("="*60)
    logger.info(f"Benchmarking {args.model.upper()} Model Optimizations")
    logger.info(f"Split Mode: {args.split_mode}")
    logger.info("="*60)
    
    # Load data based on split mode
    if args.split_mode == 'cross_experiment':
        train_samples, val_samples, input_dim, output_dim = load_cross_experiment_data(
            train_experiment_id=args.experiment_id,
            test_experiment_id=args.test_experiment,
            holdout_cells=args.holdout_cells
        )
    else:
        data_result = load_data(args.experiment_id, split_mode=args.split_mode)
        if args.split_mode == 'loco_cv':
            cv_splits, input_dim, output_dim = data_result
            train_samples = None
            val_samples = None
        else:
            train_samples, val_samples, input_dim, output_dim = data_result
            cv_splits = None
    
    # Get configurations
    all_configs = get_optimization_configs(args.model, input_dim, output_dim)
    
    # Filter out slow configurations unless --include-slow flag is set
    slow_configs = {'with_adjoint', 'mixed_precision_bf16'}
    if not args.include_slow:
        all_configs = [c for c in all_configs if c.name not in slow_configs]
        logger.info(f"Excluding slow configurations: {slow_configs}")
        logger.info("Use --include-slow to include them")
    
    # Filter to specific optimization if requested, but always include baseline for comparison
    if args.optimization:
        baseline_config = next((c for c in all_configs if c.name == 'baseline'), None)
        optimization_config = next((c for c in all_configs if c.name == args.optimization), None)
        
        if not optimization_config:
            if args.optimization in slow_configs:
                logger.error(f"Optimization '{args.optimization}' requires --include-slow flag")
                logger.info(f"Use: --optimization {args.optimization} --include-slow")
            else:
                logger.error(f"Optimization '{args.optimization}' not found")
                logger.info(f"Available optimizations: {[c.name for c in all_configs]}")
            return
        
        # Always include baseline first, then the requested optimization
        configs = []
        if baseline_config:
            configs.append(baseline_config)
        configs.append(optimization_config)
        logger.info(f"Running baseline + {args.optimization} for comparison")
    else:
        configs = all_configs
    
    # Override training parameters if specified
    if args.epochs:
        for config in configs:
            config.training_params['epochs'] = args.epochs
    if args.batch_size:
        for config in configs:
            config.training_params['batch_size'] = args.batch_size
    
    # Run benchmarks based on split mode
    runner = BenchmarkRunner()
    results = []
    
    if args.split_mode == 'loco_cv':
        # Run cross-validation
        results = runner.run_cross_validation(
            configs,
            args.model,
            cv_splits,
            input_dim,
            output_dim,
            use_mlflow=args.mlflow,
            cv_folds=args.cv_folds
        )
    else:
        # Run standard benchmark
        for config in configs:
            try:
                result = runner.run_benchmark(
                    config,
                    args.model,
                    train_samples,
                    val_samples,
                    use_mlflow=args.mlflow
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Benchmark failed for {config.name}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        split_suffix = f"_{args.split_mode}" if args.split_mode != 'temperature' else ""
        if args.split_mode == 'cross_experiment':
            split_suffix += f"_expt{args.test_experiment}"
        output_path = project_root / "artifacts" / "benchmarks" / f"benchmark_{args.model}{split_suffix}_{timestamp}.csv"
    
    save_results(results, output_path)
    
    # Print comparison
    print_comparison_table(results)
    
    # Create visualization for Cell E predictions (skip for CV mode)
    if results and args.split_mode != 'loco_cv':
        figures_dir = project_root / "artifacts" / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        split_suffix = f"_{args.split_mode}" if args.split_mode != 'temperature' else ""
        if args.split_mode == 'cross_experiment':
            split_suffix += f"_expt{args.test_experiment}"
        viz_path = figures_dir / f"benchmark_{args.model}_cell_e{split_suffix}_{timestamp}.png"
        
        try:
            visualize_cell_e_predictions(
                results, 
                viz_path, 
                args.model,
                split_mode=args.split_mode,
                test_experiment_id=args.test_experiment if args.split_mode == 'cross_experiment' else None
            )
        except Exception as e:
            logger.warning(f"Could not create visualization: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info("\nBenchmark complete!")
    logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
