"""Milestone D: Neural ODE for multi-target degradation mode prediction.

This script demonstrates:
1. Create sequences with explicit time using LatentODESequencePipeline
2. Train Neural ODE model to predict LAM_NE, LAM_PE, and LLI
3. Compare vs LSTM baseline
4. Visualize degradation mode trajectories

Usage:
    python examples/milestone_d.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Set matplotlib backend to non-interactive 'Agg' before importing pyplot
# This prevents tkinter threading conflicts
import matplotlib
matplotlib.use('Agg')

import numpy as np
import torch
import pandas as pd
import logging
import time
from contextlib import contextmanager
from typing import Dict, Optional

# Try to import psutil for memory tracking (optional)
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

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
        force=True  # Override any existing configuration
    )

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# Performance Tracking Utilities
# ─────────────────────────────────────────────────────────────────

# Global timing dictionary to track all stages
TIMINGS: Dict[str, float] = {}


@contextmanager
def timer(name: str):
    """Context manager for timing code blocks.
    
    Args:
        name: Name of the operation being timed
    
    Usage:
        with timer("Data Loading"):
            # code to time
    """
    start = time.time()
    yield
    elapsed = time.time() - start
    TIMINGS[name] = elapsed
    logger.info(f"[TIME] {name}: {elapsed:.2f}s")


def log_gpu_memory(label: str = ""):
    """Log current GPU memory usage if CUDA is available."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        prefix = f"[{label}] " if label else ""
        logger.debug(f"{prefix}GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {max_allocated:.2f}GB peak")


def log_cpu_memory(label: str = ""):
    """Log current CPU memory usage if psutil is available."""
    if HAS_PSUTIL:
        process = psutil.Process()
        memory_gb = process.memory_info().rss / 1024**3
        prefix = f"[{label}] " if label else ""
        logger.debug(f"{prefix}CPU Memory: {memory_gb:.2f}GB")


def log_memory(label: str = ""):
    """Log both GPU and CPU memory usage."""
    log_gpu_memory(label)
    log_cpu_memory(label)


def print_performance_summary():
    """Print a formatted summary table of all timing measurements."""
    if not TIMINGS:
        return
    
    logger.info("")
    logger.info("=" * 50)
    logger.info("PERFORMANCE SUMMARY")
    logger.info("=" * 50)
    logger.info(f"{'Stage':<30} {'Time (s)':<12} {'%':>6}")
    logger.info("-" * 50)
    
    total = sum(TIMINGS.values())
    for name, elapsed in TIMINGS.items():
        pct = (elapsed / total * 100) if total > 0 else 0
        logger.info(f"{name:<30} {elapsed:<12.2f} {pct:>5.1f}%")
    
    logger.info("-" * 50)
    logger.info(f"{'TOTAL':<30} {total:<12.2f} {100.0:>5.1f}%")
    logger.info("=" * 50)
    
    # Log memory summary if available
    if torch.cuda.is_available():
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        logger.info(f"Peak GPU Memory: {max_allocated:.2f}GB")
    
    if HAS_PSUTIL:
        process = psutil.Process()
        memory_gb = process.memory_info().rss / 1024**3
        logger.info(f"Final CPU Memory: {memory_gb:.2f}GB")

from src.data.tables import SummaryDataLoader, TimeseriesDataLoader
from src.data.expt_paths import ExperimentPaths
from src.pipelines.latent_ode_seq import LatentODESequencePipeline
from src.pipelines.ica_peaks import ICAPeaksPipeline
from src.models.neural_ode import NeuralODEModel
from src.models.lstm_attn import LSTMAttentionModel
from src.models.cnn_lstm import CNNLSTMModel
from src.models.acla import ACLAModel
from src.data.splits import temperature_split
from src.training.trainer import Trainer
from src.training.metrics import compute_metrics, print_metrics
from src.tracking.dual_tracker import DualTracker


def load_ica_features_for_cells(base_path: Path, experiment_id: int, 
                                 temp_map: dict, ica_pipeline: ICAPeaksPipeline):
    """Load and extract ICA features for all cells, organized by (cell_id, rpt_id).
    
    Args:
        base_path: Base path to raw data
        experiment_id: Experiment ID
        temp_map: Mapping from temperature to cell IDs
        ica_pipeline: ICAPeaksPipeline instance for feature extraction
    
    Returns:
        Dictionary mapping (cell_id, rpt_id) to ICA feature vector
    """
    ts_loader = TimeseriesDataLoader(experiment_id, base_path)
    paths = ExperimentPaths(experiment_id, base_path)
    
    ica_features = {}
    
    for temp_C, cells in temp_map.items():
        for cell_id in cells:
            # Get available RPTs
            rpts = paths.list_available_rpts(cell_id)
            
            if not rpts:
                continue
            
            for rpt_id in rpts:
                try:
                    # Load voltage curve
                    df = ts_loader.load_voltage_curve(cell_id, rpt_id)
                    
                    if df.empty:
                        continue
                    
                    # Get voltage and capacity columns
                    voltage_cols = [c for c in df.columns if 'Voltage' in c or 'voltage' in c.lower()]
                    capacity_cols = [c for c in df.columns if 'Capacity' in c or 'capacity' in c.lower() 
                                     or 'Charge' in c or 'charge' in c.lower()]
                    
                    if not voltage_cols or not capacity_cols:
                        continue
                    
                    voltage = df[voltage_cols[0]].values
                    capacity = df[capacity_cols[0]].values
                    
                    if len(voltage) == 0 or len(capacity) == 0:
                        continue
                    
                    # Extract ICA features (without normalization yet - will normalize after fitting scaler)
                    features = ica_pipeline._process_single_curve(
                        experiment_id, cell_id, rpt_id, voltage, capacity
                    )
                    
                    ica_features[(cell_id, rpt_id)] = features
                    
                except Exception as e:
                    # Skip if curve can't be loaded or processed
                    continue
    
    return ica_features


def add_ica_features_to_samples(samples, ica_features: dict, ica_pipeline: ICAPeaksPipeline):
    """Add ICA features to sequence samples by matching RPT indices.
    
    Args:
        samples: List of Sample objects (will be modified in-place)
        ica_features: Dictionary mapping (cell_id, rpt_id) to ICA feature vector
        ica_pipeline: ICAPeaksPipeline instance (for feature dimension)
    """
    ica_feature_dim = len(ica_pipeline.get_feature_names())
    
    for sample in samples:
        cell_id = sample.meta['cell_id']
        seq_len = sample.seq_len
        
        # Initialize ICA feature matrix
        ica_matrix = np.zeros((seq_len, ica_feature_dim), dtype=np.float32)
        
        # Match ICA features to sequence timesteps
        # Each timestep corresponds to an ageing set/RPT (indexed by sequence position)
        for t_idx in range(seq_len):
            # Try to find matching RPT (RPT indices typically start at 0 or 1)
            # Try both 0-indexed and 1-indexed RPT IDs
            for rpt_offset in [0, 1]:
                rpt_id = t_idx + rpt_offset
                key = (cell_id, rpt_id)
                
                if key in ica_features:
                    ica_matrix[t_idx] = ica_features[key]
                    break
            
            # If no exact match, try to find closest available RPT
            if np.allclose(ica_matrix[t_idx], 0):
                # Find closest RPT
                available_rpts = [rpt for (cid, rpt), _ in ica_features.items() if cid == cell_id]
                if available_rpts:
                    closest_rpt = min(available_rpts, key=lambda r: abs(r - t_idx))
                    key = (cell_id, closest_rpt)
                    if key in ica_features:
                        ica_matrix[t_idx] = ica_features[key]
        
        # Concatenate ICA features with existing features
        if isinstance(sample.x, torch.Tensor):
            x_np = sample.x.numpy()
        else:
            x_np = sample.x
        
        # Concatenate along feature dimension: (seq_len, summary_features + ica_features)
        x_combined = np.concatenate([x_np, ica_matrix], axis=1)
        
        # Update sample
        if isinstance(sample.x, torch.Tensor):
            sample.x = torch.from_numpy(x_combined).float()
        else:
            sample.x = x_combined.astype(np.float32)


def extract_degradation_modes(df: pd.DataFrame, samples, loader: SummaryDataLoader):
    """Extract LAM_NE, LAM_PE, and LLI from dataframe and update samples.
    
    Args:
        df: Original dataframe with all cells
        samples: List of Sample objects (will be modified in-place)
        loader: SummaryDataLoader instance
    """
    target_cols = {
        'LAM_NE': ['LAM NE_tot', 'LAM NE', 'LAM_NE'],
        'LAM_PE': ['LAM PE', 'LAM_PE', 'LAM PE [%]'],
        'LLI': ['LLI', 'LLI [%]']
    }
    
    for sample in samples:
        cell_id = sample.meta['cell_id']
        temp_C = sample.meta['temperature_C']
        
        try:
            # Load cell-specific dataframe
            cell_df = loader.load_performance_summary(cell_id, int(temp_C))
            cell_df = cell_df.sort_index()
            
            seq_len = sample.seq_len
            
            # Extract each degradation mode
            degradation_values = {}
            for mode, possible_cols in target_cols.items():
                values = None
                for col in possible_cols:
                    if col in cell_df.columns:
                        values = cell_df[col].values.astype(float)
                        values = np.nan_to_num(values, nan=0.0)
                        # Convert percentage to fraction if needed
                        if values.max() > 1.0:
                            values = values / 100.0
                        break
                
                if values is None:
                    # Fallback: use zeros if column not found
                    values = np.zeros(len(cell_df))
                
                # Match length with sequence
                if len(values) > seq_len:
                    values = values[:seq_len]
                elif len(values) < seq_len:
                    # Pad with last value if needed
                    values = np.pad(values, (0, seq_len - len(values)), mode='edge')
                
                degradation_values[mode] = values
            
            # Stack into (seq_len, 3) array: [LAM_NE, LAM_PE, LLI]
            y_new = np.stack([
                degradation_values['LAM_NE'],
                degradation_values['LAM_PE'],
                degradation_values['LLI']
            ], axis=1).astype(np.float32)
            
            # Update sample y values
            if isinstance(sample.y, torch.Tensor):
                sample.y = torch.from_numpy(y_new)
            else:
                sample.y = y_new
                
        except Exception as e:
            logger.warning(f"Could not extract degradation modes for cell {cell_id}: {e}")
            # Fallback: use zeros
            y_new = np.zeros((seq_len, 3), dtype=np.float32)
            if isinstance(sample.y, torch.Tensor):
                sample.y = torch.from_numpy(y_new)
            else:
                sample.y = y_new


def compute_multi_target_metrics(y_true, y_pred):
    """Compute metrics for each target separately and averaged.
    
    Args:
        y_true: True values of shape (n_samples, n_targets) or (n_samples * seq_len, n_targets)
        y_pred: Predicted values of same shape
    
    Returns:
        Dictionary with per-target and averaged metrics
    """
    n_targets = y_true.shape[-1]
    target_names = ['LAM_NE', 'LAM_PE', 'LLI']
    
    all_metrics = {}
    
    # Compute metrics for each target
    for i, name in enumerate(target_names):
        y_true_i = y_true[:, i].flatten()
        y_pred_i = y_pred[:, i].flatten()
        
        metrics_i = compute_metrics(y_true_i, y_pred_i)
        for key, value in metrics_i.items():
            all_metrics[f'{name}_{key}'] = value
    
    # Compute averaged metrics across all targets
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    avg_metrics = compute_metrics(y_true_flat, y_pred_flat)
    for key, value in avg_metrics.items():
        all_metrics[f'avg_{key}'] = value
    
    return all_metrics


def print_multi_target_metrics(metrics, target_names=['LAM_NE', 'LAM_PE', 'LLI']):
    """Print metrics in a formatted table."""
    logger.info(f"{'Target':<12} {'RMSE':<12} {'MAE':<12} {'R²':<12}")
    logger.info("-" * 50)
    
    for name in target_names:
        rmse = metrics.get(f'{name}_rmse', 0.0)
        mae = metrics.get(f'{name}_mae', 0.0)
        r2 = metrics.get(f'{name}_r2', 0.0)
        logger.info(f"{name:<12} {rmse:<12.5f} {mae:<12.5f} {r2:<12.4f}")
    
    logger.info("-" * 50)
    avg_rmse = metrics.get('avg_rmse', 0.0)
    avg_mae = metrics.get('avg_mae', 0.0)
    avg_r2 = metrics.get('avg_r2', 0.0)
    logger.info(f"{'Average':<12} {avg_rmse:<12.5f} {avg_mae:<12.5f} {avg_r2:<12.4f}")


def visualize_acla_attention(model, val_samples, device, save_path=None):
    """Visualize ACLA's native attention weights.
    
    ACLA uses self-attention across timesteps, making it more interpretable
    than SHAP for understanding which timesteps matter for predictions.
    
    Args:
        model: Trained ACLA model
        val_samples: Validation samples
        device: Device to run on
        save_path: Optional path to save figure
    
    Returns:
        Dictionary with attention analysis results
    """
    import matplotlib.pyplot as plt
    
    model.eval()
    all_attn_weights = []
    cell_ids = []
    
    with torch.no_grad():
        for sample in val_samples:
            x = sample.x.unsqueeze(0).to(device)  # (1, seq_len, features)
            t = sample.t.to(device) if hasattr(sample, 't') else None
            
            # Get attention weights via explain()
            explain_result = model.explain(x, t)
            attn_weights = explain_result.get('attention_weights')
            
            if attn_weights is not None:
                all_attn_weights.append(attn_weights)
                cell_ids.append(sample.meta.get('cell_id', f'Sample_{len(cell_ids)}'))
    
    if not all_attn_weights:
        logger.warning("  No attention weights available from ACLA model")
        return {'error': 'No attention weights'}
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Average attention pattern across all samples
    avg_attn = np.mean([w.mean(axis=0) for w in all_attn_weights], axis=0)  # (seq_len, seq_len)
    
    im1 = axes[0].imshow(avg_attn, cmap='viridis', aspect='auto')
    axes[0].set_xlabel('Key Timestep (Source)')
    axes[0].set_ylabel('Query Timestep (Target)')
    axes[0].set_title('ACLA: Average Attention Pattern\n(Which timesteps inform each prediction)')
    plt.colorbar(im1, ax=axes[0], label='Attention Weight')
    
    # Plot 2: Attention importance per timestep (how much each timestep is attended to)
    # Sum attention received by each timestep (column sum)
    timestep_importance = avg_attn.sum(axis=0)  # How much each source timestep is attended
    timestep_importance = timestep_importance / timestep_importance.sum()  # Normalize
    
    seq_len = len(timestep_importance)
    timesteps = np.arange(seq_len)
    
    bars = axes[1].bar(timesteps, timestep_importance, color='steelblue', alpha=0.7, edgecolor='navy')
    axes[1].set_xlabel('Timestep (RPT Index)')
    axes[1].set_ylabel('Normalized Attention Importance')
    axes[1].set_title('ACLA: Timestep Importance\n(Which RPT measurements contribute most)')
    axes[1].grid(True, alpha=0.3)
    
    # Highlight most important timesteps
    top_k = 3
    sorted_indices = np.argsort(timestep_importance)[-top_k:]
    for idx in sorted_indices:
        bars[idx].set_color('coral')
        bars[idx].set_edgecolor('darkred')
    
    axes[1].legend([bars[sorted_indices[-1]], bars[0]], 
                   [f'Top {top_k} Important', 'Other Timesteps'], loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved ACLA attention to {save_path}")
    
    plt.close()
    
    # Return analysis results
    return {
        'avg_attention': avg_attn,
        'timestep_importance': timestep_importance,
        'top_timesteps': sorted_indices.tolist(),
        'cell_ids': cell_ids
    }


def compute_gradient_feature_importance(model, val_samples, device, feature_names=None, save_path=None):
    """Compute gradient-based feature importance for neural network models.
    
    This method computes how sensitive the model output is to each input feature
    by computing gradients. Works well for any differentiable model architecture.
    
    Args:
        model: Trained PyTorch model
        val_samples: Validation samples
        device: Device to run on
        feature_names: List of feature names
        save_path: Optional path to save figure
    
    Returns:
        Dictionary with feature importance results
    """
    import matplotlib.pyplot as plt
    
    # cuDNN LSTM requires train mode for backward pass
    # We use train mode but disable dropout effects by using torch.inference_mode for RNG
    was_training = model.training
    model.train()  # Required for cuDNN LSTM backward
    
    all_gradients = []
    
    for sample in val_samples:
        model.zero_grad()  # Clear any existing gradients
        
        x = sample.x.clone().unsqueeze(0).to(device)  # (1, seq_len, features)
        x.requires_grad_(True)
        
        t = sample.t.to(device) if hasattr(sample, 't') else None
        
        # Forward pass
        output = model(x, t=t)  # (1, seq_len, output_dim)
        
        # Compute gradients with respect to input
        # Sum all outputs to get scalar for gradient computation
        output_sum = output.sum()
        output_sum.backward()
        
        # Get gradients: (1, seq_len, features)
        grad = x.grad.detach().cpu().numpy()
        all_gradients.append(grad)
        
        # Clear gradients for next sample
        x.grad = None
    
    # Stack gradients: (n_samples, 1, seq_len, features)
    all_gradients = np.array(all_gradients)
    
    # Restore original model mode
    if not was_training:
        model.eval()
    
    # Compute mean absolute gradient per feature (averaged across samples and timesteps)
    # Shape: (n_samples, 1, seq_len, features) -> (features,)
    feature_importance = np.mean(np.abs(all_gradients), axis=(0, 1, 2))
    
    # Normalize to sum to 1
    feature_importance = feature_importance / feature_importance.sum()
    
    # Generate feature names if not provided
    n_features = len(feature_importance)
    if feature_names is None or len(feature_names) != n_features:
        feature_names = [f'feature_{i}' for i in range(n_features)]
    
    # Sort by importance
    sorted_indices = np.argsort(feature_importance)[::-1]
    sorted_names = [feature_names[i] for i in sorted_indices]
    sorted_importance = feature_importance[sorted_indices]
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Show top 20 features (or all if fewer)
    n_display = min(20, len(sorted_names))
    y_pos = np.arange(n_display)
    
    colors = ['coral' if i < 5 else 'steelblue' for i in range(n_display)]
    
    bars = ax.barh(y_pos, sorted_importance[:n_display], color=colors, alpha=0.7, edgecolor='navy')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names[:n_display])
    ax.invert_yaxis()  # Highest at top
    ax.set_xlabel('Normalized Gradient Importance')
    ax.set_title('ACLA: Gradient-Based Feature Importance\n(Which input features most affect predictions)')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, sorted_importance[:n_display])):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2, 
                f'{val:.4f}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved gradient feature importance to {save_path}")
    
    plt.close()
    
    # Return results
    importance_dict = {name: float(imp) for name, imp in zip(feature_names, feature_importance)}
    
    return {
        'feature_importance': importance_dict,
        'sorted_features': sorted_names,
        'sorted_importance': sorted_importance.tolist(),
        'top_5': list(zip(sorted_names[:5], sorted_importance[:5].tolist()))
    }


def main():
    """Run Milestone D: Multi-target degradation mode prediction."""
    
    # Get project root (parent of examples directory)
    PROJECT_ROOT = Path(__file__).parent.parent
    BASE_PATH = PROJECT_ROOT / "Raw Data"
    EXPERIMENT_ID = 5
    CELLS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    TEMP_MAP = {
        10: ['A', 'B', 'C'],
        25: ['D', 'E'],
        40: ['F', 'G', 'H']
    }
    
    # Training config
    CONFIG = {
        'epochs': 500,
        'batch_size': 4,  # Small batch for sequence data
        'learning_rate': 1e-3,
        'weight_decay': 0.01,
        'early_stopping_patience': 20,
        'gradient_clip': 1.0,
        'use_amp': False,  # Disable for ODE compatibility
        'scheduler_T0': 30,
    }
    
    logger.info("=" * 60)
    logger.info("Milestone D: Neural ODE for Degradation Mode Prediction")
    logger.info("=" * 60)
    
    # ─────────────────────────────────────────────────────────────────
    # 1. Load Data
    # ─────────────────────────────────────────────────────────────────
    
    logger.info("[1/8] Loading data...")
    
    with timer("Data Loading"):
        try:
            loader = SummaryDataLoader(EXPERIMENT_ID, BASE_PATH)
            df = loader.load_all_cells(cells=CELLS, temp_map=TEMP_MAP)
            logger.info(f"  [OK] Loaded {len(df)} samples from {df['cell_id'].nunique()} cells")
        except FileNotFoundError as e:
            logger.error(f"  [ERR] Data not found: {e}")
            return
    
    # ─────────────────────────────────────────────────────────────────
    # 2. Create ICA Pipeline and Extract Features
    # ─────────────────────────────────────────────────────────────────
    
    logger.info("[2/8] Creating ICA pipeline and extracting features...")
    
    with timer("ICA Feature Extraction"):
        ica_pipeline = ICAPeaksPipeline(
            sg_window=1541,
            sg_order=3,
            num_peaks=3,
            voltage_range=(3.0, 4.2),
            resample_points=2000,
            normalize=True,
            use_cache=True
        )
        
        # Load voltage curves and extract ICA features
        logger.info("  Loading voltage curves and extracting ICA features...")
        ica_features = load_ica_features_for_cells(BASE_PATH, EXPERIMENT_ID, TEMP_MAP, ica_pipeline)
        logger.info(f"  [OK] Extracted ICA features for {len(ica_features)} RPT measurements")
        
        # Fit ICA scaler on all extracted features and normalize
        if ica_features and ica_pipeline.normalize:
            from sklearn.preprocessing import StandardScaler
            all_ica_features = np.vstack(list(ica_features.values()))
            if ica_pipeline.scaler is None:
                ica_pipeline.scaler = StandardScaler()
            ica_pipeline.scaler.fit(all_ica_features)
            
            # Normalize all features
            normalized_features = ica_pipeline.scaler.transform(all_ica_features)
            for i, key in enumerate(ica_features.keys()):
                ica_features[key] = normalized_features[i]
            
            logger.info(f"  [OK] Fitted and normalized ICA feature scaler")
    
    # ─────────────────────────────────────────────────────────────────
    # 3. Create Sequence Pipeline
    # ─────────────────────────────────────────────────────────────────
    
    logger.info("[3/8] Creating sequence pipeline...")
    
    with timer("Pipeline Creation"):
        pipeline = LatentODESequencePipeline(
            time_unit="days",
            max_seq_len=None,
            normalize=True
        )
        
        samples = pipeline.fit_transform({'df': df})
        logger.info(f"  [OK] Created {len(samples)} sequence samples (one per cell)")
        
        if len(samples) > 0:
            logger.info(f"  [OK] Sequence length: {samples[0].seq_len}")
            logger.info(f"  [OK] Base feature dimension: {samples[0].feature_dim}")
            logger.info(f"  [OK] Time range: 0 to {samples[0].t[-1].item():.1f} days")
        
        # ─────────────────────────────────────────────────────────────────
        # 3.5. Add ICA Features to Sequences
        # ─────────────────────────────────────────────────────────────────
        
        logger.info("[3.5/8] Adding ICA features to sequences...")
        
        add_ica_features_to_samples(samples, ica_features, ica_pipeline)
        logger.info(f"  [OK] Added ICA features to sequences")
        
        if len(samples) > 0:
            logger.info(f"  [OK] Updated feature dimension: {samples[0].feature_dim}")
            logger.info(f"    (Base: {pipeline.get_feature_names()}, ICA: {len(ica_pipeline.get_feature_names())} features)")
        
        # ─────────────────────────────────────────────────────────────────
        # 4. Extract Degradation Modes
        # ─────────────────────────────────────────────────────────────────
        
        logger.info("[4/8] Extracting degradation modes (LAM_NE, LAM_PE, LLI)...")
        
        extract_degradation_modes(df, samples, loader)
        logger.info(f"  [OK] Updated samples with multi-target degradation modes")
        
        if len(samples) > 0:
            logger.info(f"  [OK] Target dimension: {samples[0].y.shape[-1]} (LAM_NE, LAM_PE, LLI)")
    
    # ─────────────────────────────────────────────────────────────────
    # 5. Split Data
    # ─────────────────────────────────────────────────────────────────
    
    logger.info("[5/8] Splitting by temperature...")
    
    train_samples, val_samples = temperature_split(
        samples,
        train_temps=[10, 40],
        val_temps=[25]
    )
    
    logger.info(f"  [OK] Train: {len(train_samples)} cells")
    logger.info(f"  [OK] Val:   {len(val_samples)} cells")
    
    if len(train_samples) == 0 or len(val_samples) == 0:
        logger.error("  [ERR] Not enough samples for train/val split")
        return
    
    # ─────────────────────────────────────────────────────────────────
    # 6. Train Neural ODE
    # ─────────────────────────────────────────────────────────────────
    
    logger.info("[6/8] Training Neural ODE...")
    
    device = 'cuda' 
    input_dim = samples[0].feature_dim
    output_dim = 3  # LAM_NE, LAM_PE, LLI
    log_memory("Before Neural ODE")
    
    try:
        with timer("Neural ODE Training"):
            ode_model = NeuralODEModel(
                input_dim=input_dim,
                output_dim=output_dim,
                latent_dim=32,
                hidden_dim=64,
                solver='dopri5',
                rtol=1e-4,
                atol=1e-5,
                use_adjoint=True
            )
            
            logger.info(f"  [OK] Neural ODE: {ode_model.count_parameters():,} parameters")
            logger.info(f"  [OK] Training on {len(train_samples)} cells, validating on {len(val_samples)} cells")
            logger.info(f"  [OK] Device: {device}")
            logger.info(f"  [OK] Epochs: {CONFIG['epochs']}, Batch size: {CONFIG['batch_size']}, LR: {CONFIG['learning_rate']}")
            
            # Use absolute paths for tracking to work correctly from any directory
            artifacts_dir = PROJECT_ROOT / "artifacts"
            tracker = DualTracker(
                local_base_dir=str(artifacts_dir / "runs"),
                use_tensorboard=True,
                mlflow_tracking_uri=f"sqlite:///{artifacts_dir / 'mlflow.db'}",
                mlflow_experiment_name="battery_degradation"
            )
            tracker.start_run("milestone_d_neural_ode", {
                'model': 'neural_ode',
                'pipeline': 'latent_ode_seq',
                'loss': 'physics_informed',
                'output_dim': output_dim,
                'include_ica': True,
                **CONFIG
            })
            
            # Use modular loss function configuration
            ode_trainer = Trainer(
                ode_model, CONFIG, tracker, device=device,
                loss_config={
                    'name': 'physics_informed',
                    'monotonicity_weight': 0.1,
                    'smoothness_weight': 0.01
                },
                verbose=True  # Print progress every epoch
            )
            
            logger.info("  Starting Neural ODE training...")
            logger.info("  " + "-" * 56)
            ode_history = ode_trainer.fit(train_samples, val_samples)
            
            # Print training summary
            logger.info("  " + "-" * 56)
            logger.info(f"  Training completed!")
            logger.info(f"  Best validation loss: {ode_trainer.best_val_loss:.5f}")
            logger.info(f"  Final train loss: {ode_history['train_loss'][-1]:.5f}")
            logger.info(f"  Final validation loss: {ode_history['val_loss'][-1]:.5f}")
            logger.info(f"  Total epochs trained: {len(ode_history['train_loss'])}")
            
            # Evaluate
            ode_predictions = ode_trainer.predict(val_samples)
            y_val = np.vstack([s.y.numpy() for s in val_samples])
            
            # Reshape for multi-target metrics: (n_samples * seq_len, n_targets)
            seq_len = val_samples[0].seq_len
            n_samples = len(val_samples)
            y_val_reshaped = y_val.reshape(-1, output_dim)
            ode_pred_reshaped = ode_predictions.reshape(-1, output_dim)
            
            ode_metrics = compute_multi_target_metrics(y_val_reshaped, ode_pred_reshaped)
            
            tracker.log_metrics({'final_' + k: v for k, v in ode_metrics.items()})
            tracker.end_run()
            
            logger.info("=" * 40)
            logger.info("Neural ODE Results (25°C Holdout)")
            logger.info("=" * 40)
            print_multi_target_metrics(ode_metrics)
            log_memory("After Neural ODE")
            
            # Gradient-based Feature Importance
            logger.info("\n[6.5/9] Computing gradient-based feature importance for Neural ODE...")
            try:
                # Get feature names
                base_feature_names = pipeline.get_feature_names()
                ica_feature_names = ica_pipeline.get_feature_names()
                all_feature_names = base_feature_names + ica_feature_names
                
                figures_dir = PROJECT_ROOT / "artifacts" / "figures"
                figures_dir.mkdir(parents=True, exist_ok=True)
                grad_path = figures_dir / "neural_ode_feature_importance.png"
                
                grad_result = compute_gradient_feature_importance(
                    ode_model, val_samples, device,
                    feature_names=all_feature_names,
                    save_path=str(grad_path)
                )
                
                if 'top_5' in grad_result:
                    logger.info("  [OK] Gradient feature importance computed")
                    logger.info("\n  Top 5 Features by Gradient Importance:")
                    for name, imp in grad_result['top_5']:
                        logger.info(f"    {name}: {imp:.4f}")
                    
                    logger.info(f"\n  [OK] Feature importance plot saved to {grad_path}")
            except Exception as e:
                logger.warning(f"  [WARN] Gradient feature importance failed: {e}")
                import traceback
                traceback.print_exc()
        
    except ImportError as e:
        logger.error(f"  [ERR] torchdiffeq not installed: {e}")
        logger.error("  Install with: pip install torchdiffeq")
        ode_metrics = None
    except Exception as e:
        logger.error(f"  [ERR] Neural ODE training failed: {e}")
        import traceback
        traceback.print_exc()
        ode_metrics = None
    
    # ─────────────────────────────────────────────────────────────────
    # 7. Train LSTM Baseline for Comparison
    # ─────────────────────────────────────────────────────────────────
    
    logger.info("[7/8] Training LSTM baseline for comparison...")
    log_memory("Before LSTM")
    
    with timer("LSTM Training"):
        lstm_model = LSTMAttentionModel(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=64,
            num_layers=2,
            num_heads=4
        )
        
        logger.info(f"  [OK] LSTM: {lstm_model.count_parameters():,} parameters")
        logger.info(f"  [OK] Training on {len(train_samples)} cells, validating on {len(val_samples)} cells")
        logger.info(f"  [OK] Device: {device}")
        logger.info(f"  [OK] Epochs: {CONFIG['epochs']}, Batch size: {CONFIG['batch_size']}, LR: {CONFIG['learning_rate']}")
        
        # Use absolute paths for tracking to work correctly from any directory
        artifacts_dir = PROJECT_ROOT / "artifacts"
        tracker2 = DualTracker(
            local_base_dir=str(artifacts_dir / "runs"),
            use_tensorboard=True,
            mlflow_tracking_uri=f"sqlite:///{artifacts_dir / 'mlflow.db'}",
            mlflow_experiment_name="battery_degradation"
        )
        tracker2.start_run("milestone_d_lstm_baseline", {
            'model': 'lstm_attn',
            'pipeline': 'latent_ode_seq',
            'loss': 'physics_informed',
            'output_dim': output_dim,
            'include_ica': True,
            **CONFIG
        })
        
        lstm_trainer = Trainer(
            lstm_model, CONFIG, tracker2, device=device,
            loss_config={
                'name': 'physics_informed',
                'monotonicity_weight': 0.1,
                'smoothness_weight': 0.01
            },
            verbose=True  # Print progress every epoch
        )
        
        logger.info("  Starting LSTM training...")
        logger.info("  " + "-" * 56)
        lstm_history = lstm_trainer.fit(train_samples, val_samples)
        
        # Print training summary
        logger.info("  " + "-" * 56)
        logger.info(f"  Training completed!")
        logger.info(f"  Best validation loss: {lstm_trainer.best_val_loss:.5f}")
        logger.info(f"  Final train loss: {lstm_history['train_loss'][-1]:.5f}")
        logger.info(f"  Final validation loss: {lstm_history['val_loss'][-1]:.5f}")
        logger.info(f"  Total epochs trained: {len(lstm_history['train_loss'])}")
        
        lstm_predictions = lstm_trainer.predict(val_samples)
        lstm_pred_reshaped = lstm_predictions.reshape(-1, output_dim)
        lstm_metrics = compute_multi_target_metrics(y_val_reshaped, lstm_pred_reshaped)
        
        tracker2.log_metrics({'final_' + k: v for k, v in lstm_metrics.items()})
        tracker2.end_run()
        
        logger.info("=" * 40)
        logger.info("LSTM Baseline Results (25°C Holdout)")
        logger.info("=" * 40)
        print_multi_target_metrics(lstm_metrics)
        log_memory("After LSTM")
        
        # Gradient-based Feature Importance
        logger.info("\n[7.5/9] Computing gradient-based feature importance for LSTM...")
        try:
            # Get feature names
            base_feature_names = pipeline.get_feature_names()
            ica_feature_names = ica_pipeline.get_feature_names()
            all_feature_names = base_feature_names + ica_feature_names
            
            figures_dir = PROJECT_ROOT / "artifacts" / "figures"
            figures_dir.mkdir(parents=True, exist_ok=True)
            grad_path = figures_dir / "lstm_feature_importance.png"
            
            grad_result = compute_gradient_feature_importance(
                lstm_model, val_samples, device,
                feature_names=all_feature_names,
                save_path=str(grad_path)
            )
            
            if 'top_5' in grad_result:
                logger.info("  [OK] Gradient feature importance computed")
                logger.info("\n  Top 5 Features by Gradient Importance:")
                for name, imp in grad_result['top_5']:
                    logger.info(f"    {name}: {imp:.4f}")
                
                logger.info(f"\n  [OK] Feature importance plot saved to {grad_path}")
        except Exception as e:
            logger.warning(f"  [WARN] Gradient feature importance failed: {e}")
            import traceback
            traceback.print_exc()
    
    # ─────────────────────────────────────────────────────────────────
    # 8. Train CNN-LSTM Model
    # ─────────────────────────────────────────────────────────────────
    
    logger.info("[8/10] Training CNN-LSTM model...")
    log_memory("Before CNN-LSTM")
    
    with timer("CNN-LSTM Training"):
        cnn_lstm_model = CNNLSTMModel(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=64,
            cnn_filters=[64, 32],
            num_layers=2,
            num_heads=4
        )
        
        logger.info(f"  [OK] CNN-LSTM: {cnn_lstm_model.count_parameters():,} parameters")
        logger.info(f"  [OK] Training on {len(train_samples)} cells, validating on {len(val_samples)} cells")
        logger.info(f"  [OK] Device: {device}")
        logger.info(f"  [OK] Epochs: {CONFIG['epochs']}, Batch size: {CONFIG['batch_size']}, LR: {CONFIG['learning_rate']}")
        
        # Use absolute paths for tracking to work correctly from any directory
        artifacts_dir = PROJECT_ROOT / "artifacts"
        tracker_cnn = DualTracker(
            local_base_dir=str(artifacts_dir / "runs"),
            use_tensorboard=True,
            mlflow_tracking_uri=f"sqlite:///{artifacts_dir / 'mlflow.db'}",
            mlflow_experiment_name="battery_degradation"
        )
        tracker_cnn.start_run("milestone_d_cnn_lstm", {
            'model': 'cnn_lstm',
            'pipeline': 'latent_ode_seq',
            'loss': 'physics_informed',
            'output_dim': output_dim,
            'include_ica': True,
            **CONFIG
        })
        
        cnn_lstm_trainer = Trainer(
            cnn_lstm_model, CONFIG, tracker_cnn, device=device,
            loss_config={
                'name': 'physics_informed',
                'monotonicity_weight': 0.1,
                'smoothness_weight': 0.01
            },
            verbose=True  # Print progress every epoch
        )
        
        logger.info("  Starting CNN-LSTM training...")
        logger.info("  " + "-" * 56)
        cnn_lstm_history = cnn_lstm_trainer.fit(train_samples, val_samples)
        
        # Print training summary
        logger.info("  " + "-" * 56)
        logger.info(f"  Training completed!")
        logger.info(f"  Best validation loss: {cnn_lstm_trainer.best_val_loss:.5f}")
        logger.info(f"  Final train loss: {cnn_lstm_history['train_loss'][-1]:.5f}")
        logger.info(f"  Final validation loss: {cnn_lstm_history['val_loss'][-1]:.5f}")
        logger.info(f"  Total epochs trained: {len(cnn_lstm_history['train_loss'])}")
        
        cnn_lstm_predictions = cnn_lstm_trainer.predict(val_samples)
        cnn_lstm_pred_reshaped = cnn_lstm_predictions.reshape(-1, output_dim)
        cnn_lstm_metrics = compute_multi_target_metrics(y_val_reshaped, cnn_lstm_pred_reshaped)
        
        tracker_cnn.log_metrics({'final_' + k: v for k, v in cnn_lstm_metrics.items()})
        tracker_cnn.end_run()
        
        logger.info("=" * 40)
        logger.info("CNN-LSTM Results (25°C Holdout)")
        logger.info("=" * 40)
        print_multi_target_metrics(cnn_lstm_metrics)
        log_memory("After CNN-LSTM")
        
        # Gradient-based Feature Importance
        logger.info("\n[8.5/10] Computing gradient-based feature importance for CNN-LSTM...")
        try:
            # Get feature names
            base_feature_names = pipeline.get_feature_names()
            ica_feature_names = ica_pipeline.get_feature_names()
            all_feature_names = base_feature_names + ica_feature_names
            
            figures_dir = PROJECT_ROOT / "artifacts" / "figures"
            figures_dir.mkdir(parents=True, exist_ok=True)
            grad_path = figures_dir / "cnn_lstm_feature_importance.png"
            
            grad_result = compute_gradient_feature_importance(
                cnn_lstm_model, val_samples, device,
                feature_names=all_feature_names,
                save_path=str(grad_path)
            )
            
            if 'top_5' in grad_result:
                logger.info("  [OK] Gradient feature importance computed")
                logger.info("\n  Top 5 Features by Gradient Importance:")
                for name, imp in grad_result['top_5']:
                    logger.info(f"    {name}: {imp:.4f}")
                
                logger.info(f"\n  [OK] Feature importance plot saved to {grad_path}")
        except Exception as e:
            logger.warning(f"  [WARN] Gradient feature importance failed: {e}")
            import traceback
            traceback.print_exc()
    
    # ─────────────────────────────────────────────────────────────────
    # 9. Comparison (before ACLA)
    # ─────────────────────────────────────────────────────────────────
    
    logger.info("=" * 40)
    logger.info("Model Comparison (Average across all targets)")
    logger.info("=" * 40)
    
    logger.info(f"{'Model':<20} {'RMSE':<12} {'MAE':<12} {'R²':<12}")
    logger.info("-" * 60)
    
    if ode_metrics:
        logger.info(f"{'Neural ODE':<20} {ode_metrics['avg_rmse']:<12.5f} {ode_metrics['avg_mae']:<12.5f} {ode_metrics['avg_r2']:<12.4f}")
    
    logger.info(f"{'LSTM+Attention':<20} {lstm_metrics['avg_rmse']:<12.5f} {lstm_metrics['avg_mae']:<12.5f} {lstm_metrics['avg_r2']:<12.4f}")
    
    logger.info(f"{'CNN-LSTM':<20} {cnn_lstm_metrics['avg_rmse']:<12.5f} {cnn_lstm_metrics['avg_mae']:<12.5f} {cnn_lstm_metrics['avg_r2']:<12.4f}")
    
    # ─────────────────────────────────────────────────────────────────
    # 10. Train ACLA Model
    # ─────────────────────────────────────────────────────────────────
    
    logger.info("[10/11] Training ACLA model...")
    log_memory("Before ACLA")
    
    try:
        with timer("ACLA Training"):
            acla_model = ACLAModel(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dim=64,
                augment_dim=20,
                cnn_filters=[64, 32],
                solver='dopri5',
                rtol=1e-4,
                atol=1e-5,
                use_adjoint=True
            )
            
            # ACLA-specific config: lower LR and tighter gradient clipping for stable ODE training
            ACLA_CONFIG = {
                **CONFIG,
                'learning_rate': 1e-4,  # Lower LR for ODE stability
                'gradient_clip': 0.5,   # Tighter clipping
            }
            
            logger.info(f"  [OK] ACLA: {acla_model.count_parameters():,} parameters")
            logger.info(f"  [OK] Training on {len(train_samples)} cells, validating on {len(val_samples)} cells")
            logger.info(f"  [OK] Device: {device}")
            logger.info(f"  [OK] Epochs: {ACLA_CONFIG['epochs']}, Batch size: {ACLA_CONFIG['batch_size']}, LR: {ACLA_CONFIG['learning_rate']}")
            
            # Use absolute paths for tracking to work correctly from any directory
            artifacts_dir = PROJECT_ROOT / "artifacts"
            tracker3 = DualTracker(
                local_base_dir=str(artifacts_dir / "runs"),
                use_tensorboard=True,
                mlflow_tracking_uri=f"sqlite:///{artifacts_dir / 'mlflow.db'}",
                mlflow_experiment_name="battery_degradation"
            )
            tracker3.start_run("milestone_d_acla", {
                'model': 'acla',
                'pipeline': 'latent_ode_seq',
                'loss': 'physics_informed',
                'output_dim': output_dim,
                'include_ica': True,
                **ACLA_CONFIG
            })
            
            acla_trainer = Trainer(
                acla_model, ACLA_CONFIG, tracker3, device=device,
                loss_config={
                    'name': 'physics_informed',
                    'monotonicity_weight': 0.1,
                    'smoothness_weight': 0.01
                },
                verbose=True  # Print progress every epoch
            )
            
            logger.info("  Starting ACLA training...")
            logger.info("  " + "-" * 56)
            acla_history = acla_trainer.fit(train_samples, val_samples)
            
            # Print training summary
            logger.info("  " + "-" * 56)
            logger.info(f"  Training completed!")
            logger.info(f"  Best validation loss: {acla_trainer.best_val_loss:.5f}")
            logger.info(f"  Final train loss: {acla_history['train_loss'][-1]:.5f}")
            logger.info(f"  Final validation loss: {acla_history['val_loss'][-1]:.5f}")
            logger.info(f"  Total epochs trained: {len(acla_history['train_loss'])}")
            
            acla_predictions = acla_trainer.predict(val_samples)
            acla_pred_reshaped = acla_predictions.reshape(-1, output_dim)
            acla_metrics = compute_multi_target_metrics(y_val_reshaped, acla_pred_reshaped)
            
            tracker3.log_metrics({'final_' + k: v for k, v in acla_metrics.items()})
            tracker3.end_run()
            
            logger.info("=" * 40)
            logger.info("ACLA Results (25°C Holdout)")
            logger.info("=" * 40)
            print_multi_target_metrics(acla_metrics)
            log_memory("After ACLA")
            
            # Native Attention Analysis (better than SHAP for ACLA architecture)
            logger.info("\n[8.5/9] Analyzing ACLA attention weights...")
            logger.info("  Note: ACLA uses self-attention, which is more interpretable than SHAP for this architecture")
            try:
                figures_dir = PROJECT_ROOT / "artifacts" / "figures"
                figures_dir.mkdir(parents=True, exist_ok=True)
                attn_path = figures_dir / "acla_attention.png"
                
                attn_result = visualize_acla_attention(
                    acla_model, val_samples, device, save_path=str(attn_path)
                )
                
                if 'error' not in attn_result:
                    logger.info("  [OK] Attention visualization created")
                    
                    # Report top important timesteps
                    top_timesteps = attn_result.get('top_timesteps', [])
                    timestep_importance = attn_result.get('timestep_importance', [])
                    
                    logger.info("\n  Top 3 Most Important Timesteps (RPT indices):")
                    for idx in reversed(top_timesteps):
                        importance = timestep_importance[idx] if idx < len(timestep_importance) else 0
                        logger.info(f"    RPT {idx}: {importance:.4f} attention weight")
                    
                    logger.info(f"\n  [OK] Attention plot saved to {attn_path}")
                    logger.info("  Interpretation: ACLA learns which RPT measurements are most")
                    logger.info("  informative for predicting degradation mode trajectories.")
                else:
                    logger.warning(f"  [WARN] Could not extract attention: {attn_result.get('error')}")
            except Exception as e:
                logger.warning(f"  [WARN] Attention analysis failed: {e}")
                import traceback
                traceback.print_exc()
            
            # Gradient-based feature importance (complementary to attention)
            logger.info("\n[8.6/9] Computing ACLA gradient-based feature importance...")
            try:
                # Get feature names
                base_feature_names = pipeline.get_feature_names()
                ica_feature_names = ica_pipeline.get_feature_names()
                all_feature_names = base_feature_names + ica_feature_names
                
                grad_path = figures_dir / "acla_feature_importance.png"
                
                grad_result = compute_gradient_feature_importance(
                    acla_model, val_samples, device,
                    feature_names=all_feature_names,
                    save_path=str(grad_path)
                )
                
                if 'top_5' in grad_result:
                    logger.info("  [OK] Gradient feature importance computed")
                    logger.info("\n  Top 5 Features by Gradient Importance:")
                    for name, imp in grad_result['top_5']:
                        logger.info(f"    {name}: {imp:.4f}")
                    
                    logger.info(f"\n  [OK] Feature importance plot saved to {grad_path}")
            except Exception as e:
                logger.warning(f"  [WARN] Gradient feature importance failed: {e}")
                import traceback
                traceback.print_exc()
        
    except ImportError as e:
        logger.error(f"  [ERR] torchdiffeq not installed: {e}")
        logger.error("  Install with: pip install torchdiffeq")
        acla_metrics = None
    except Exception as e:
        logger.error(f"  [ERR] ACLA training failed: {e}")
        import traceback
        traceback.print_exc()
        acla_metrics = None
    
    # ─────────────────────────────────────────────────────────────────
    # 11. Final Comparison
    # ─────────────────────────────────────────────────────────────────
    
    logger.info("=" * 40)
    logger.info("Final Model Comparison (Average across all targets)")
    logger.info("=" * 40)
    
    logger.info(f"{'Model':<20} {'RMSE':<12} {'MAE':<12} {'R²':<12}")
    logger.info("-" * 60)
    
    if ode_metrics:
        logger.info(f"{'Neural ODE':<20} {ode_metrics['avg_rmse']:<12.5f} {ode_metrics['avg_mae']:<12.5f} {ode_metrics['avg_r2']:<12.4f}")
    
    logger.info(f"{'LSTM+Attention':<20} {lstm_metrics['avg_rmse']:<12.5f} {lstm_metrics['avg_mae']:<12.5f} {lstm_metrics['avg_r2']:<12.4f}")
    
    logger.info(f"{'CNN-LSTM':<20} {cnn_lstm_metrics['avg_rmse']:<12.5f} {cnn_lstm_metrics['avg_mae']:<12.5f} {cnn_lstm_metrics['avg_r2']:<12.4f}")
    
    if acla_metrics:
        logger.info(f"{'ACLA':<20} {acla_metrics['avg_rmse']:<12.5f} {acla_metrics['avg_mae']:<12.5f} {acla_metrics['avg_r2']:<12.4f}")
    
    # ─────────────────────────────────────────────────────────────────
    # 12. Visualize Full Prediction Trajectory for One Cell
    # ─────────────────────────────────────────────────────────────────
    
    if (ode_metrics or acla_metrics) and len(val_samples) > 0:
        with timer("Visualization"):
            try:
                import matplotlib.pyplot as plt
                
                # Select Cell E for visualization
                sample_idx = None
                for idx, s in enumerate(val_samples):
                    if s.meta.get('cell_id') == 'E':
                        sample_idx = idx
                        break
                
                # Fallback to first sample if Cell E not found
                if sample_idx is None:
                    sample_idx = 0
                    logger.warning("  Cell E not found in validation samples, using first sample instead")
                
                sample = val_samples[sample_idx].to_device(device)
                cell_id = sample.meta['cell_id']
                
                logger.info(f"[10/10] Visualizing predictions for cell {cell_id}...")
                
                # Get time values
                t = sample.t.cpu().numpy()
                x = sample.x.unsqueeze(0)  # Add batch dim: (1, seq_len, features)
                
                # Get Neural ODE predictions at all timesteps
                ode_model.eval()
                with torch.no_grad():
                    ode_preds = ode_model(x, t=sample.t)[0].cpu().numpy()  # (seq_len, 3)
                
                # Get LSTM predictions at all timesteps
                lstm_model.eval()
                with torch.no_grad():
                    lstm_preds = lstm_model(x)[0].cpu().numpy()  # (seq_len, 3)
                
                # Get CNN-LSTM predictions at all timesteps
                cnn_lstm_model.eval()
                with torch.no_grad():
                    cnn_lstm_preds = cnn_lstm_model(x)[0].cpu().numpy()  # (seq_len, 3)
                
                # Get ACLA predictions at all timesteps
                acla_preds = None
                if acla_metrics:
                    acla_model.eval()
                    with torch.no_grad():
                        acla_preds = acla_model(x, t=sample.t)[0].cpu().numpy()  # (seq_len, 3)
                
                # Get actual degradation mode values
                try:
                    temp_C = sample.meta['temperature_C']
                    cell_df = loader.load_performance_summary(cell_id, int(temp_C))
                    cell_df = cell_df.sort_index()
                    
                    target_cols = {
                        'LAM_NE': ['LAM NE_tot', 'LAM NE', 'LAM_NE'],
                        'LAM_PE': ['LAM PE', 'LAM_PE', 'LAM PE [%]'],
                        'LLI': ['LLI', 'LLI [%]']
                    }
                    
                    actual_values = {}
                    for mode, possible_cols in target_cols.items():
                        values = None
                        for col in possible_cols:
                            if col in cell_df.columns:
                                values = cell_df[col].values.astype(float)
                                values = np.nan_to_num(values, nan=0.0)
                                if values.max() > 1.0:
                                    values = values / 100.0
                                break
                        
                        if values is None:
                            values = np.zeros(len(cell_df))
                        
                        # Match length
                        if len(values) > len(t):
                            values = values[:len(t)]
                        elif len(values) < len(t):
                            values = np.pad(values, (0, len(t) - len(values)), mode='edge')
                        
                        actual_values[mode] = values
                        
                except Exception as e:
                    logger.warning(f"  Could not load actual degradation values: {e}")
                    actual_values = None
                
                # Create visualization with 3 subplots
                figures_dir = PROJECT_ROOT / "artifacts" / "figures"
                figures_dir.mkdir(parents=True, exist_ok=True)
                
                fig, axes = plt.subplots(3, 1, figsize=(10, 12))
                target_names = ['LAM_NE', 'LAM_PE', 'LLI']
                target_labels = ['LAM NE (Loss of Active Material - Negative Electrode)',
                               'LAM PE (Loss of Active Material - Positive Electrode)',
                               'LLI (Loss of Lithium Inventory)']
                
                for i, (ax, name, label) in enumerate(zip(axes, target_names, target_labels)):
                    # Plot actual values if available
                    if actual_values and name in actual_values:
                        ax.plot(t, actual_values[name], 'ko-', label='Actual', 
                               linewidth=2, markersize=6, alpha=0.7)
                    
                    # Plot Neural ODE predictions
                    ax.plot(t, ode_preds[:, i], 'b-', label='Neural ODE', 
                           linewidth=2, alpha=0.8)
                    
                    # Plot LSTM predictions
                    ax.plot(t, lstm_preds[:, i], 'r--', label='LSTM+Attention', 
                           linewidth=2, alpha=0.8)
                    
                    # Plot CNN-LSTM predictions
                    ax.plot(t, cnn_lstm_preds[:, i], 'm:', label='CNN-LSTM', 
                           linewidth=2, alpha=0.8)
                    
                    # Plot ACLA predictions
                    if acla_preds is not None:
                        ax.plot(t, acla_preds[:, i], 'g-.', label='ACLA', 
                               linewidth=2, alpha=0.8)
                    
                    ax.set_xlabel('Time (days)', fontsize=11)
                    ax.set_ylabel(name, fontsize=11)
                    ax.set_title(label, fontsize=12, fontweight='bold')
                    ax.legend(fontsize=10, loc='best')
                    ax.grid(True, alpha=0.3)
                    
                    # Set y-axis limits
                    y_max = max(ode_preds[:, i].max(), lstm_preds[:, i].max(), cnn_lstm_preds[:, i].max())
                    if acla_preds is not None:
                        y_max = max(y_max, acla_preds[:, i].max())
                    if actual_values and name in actual_values:
                        y_max = max(y_max, actual_values[name].max())
                    ax.set_ylim([-0.05, y_max * 1.1])
                
                plt.suptitle(f'Degradation Mode Prediction Trajectories - Cell {cell_id} ({temp_C}°C)', 
                            fontsize=14, fontweight='bold', y=0.995)
                plt.tight_layout()
                trajectory_path = figures_dir / "neural_ode_degradation_modes.png"
                plt.savefig(str(trajectory_path), dpi=150, bbox_inches='tight')
                logger.info(f"  [OK] Trajectory plot saved to {trajectory_path}")
                plt.close()
                
            except Exception as e:
                import traceback
                logger.error(f"  Could not plot trajectory: {e}")
                traceback.print_exc()
    
    logger.info("=" * 60)
    logger.info("Milestone D Complete!")
    logger.info("=" * 60)
    
    # Print performance summary
    print_performance_summary()
    
    return ode_metrics, lstm_metrics, cnn_lstm_metrics, acla_metrics


if __name__ == "__main__":
    # Setup logging with file output
    PROJECT_ROOT = Path(__file__).parent.parent
    log_file = PROJECT_ROOT / "artifacts" / "logs" / "milestone_d.log"
    setup_logging(log_file)
    
    main()
