"""Milestone C: Neural ODE for continuous-time degradation modeling.

This script demonstrates:
1. Create sequences with explicit time using LatentODESequencePipeline
2. Train Neural ODE model that respects continuous time
3. Compare vs LSTM baseline
4. Visualize latent trajectory

Usage:
    python examples/milestone_c.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import pandas as pd

from src.data.tables import SummaryDataLoader
from src.pipelines.latent_ode_seq import LatentODESequencePipeline
from src.models.neural_ode import NeuralODEModel
from src.models.lstm_attn import LSTMAttentionModel
from src.data.splits import temperature_split
from src.training.trainer import Trainer
from src.training.metrics import compute_metrics, print_metrics
from src.tracking.dual_tracker import DualTracker


def main():
    """Run Milestone C: Neural ODE."""
    
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
        'epochs': 100,
        'batch_size': 4,  # Small batch for sequence data
        'learning_rate': 1e-3,
        'weight_decay': 0.01,
        'early_stopping_patience': 20,
        'gradient_clip': 1.0,
        'use_amp': False,  # Disable for ODE compatibility
        'scheduler_T0': 30,
    }
    
    print("=" * 60)
    print("Milestone C: Neural ODE for Degradation Modeling")
    print("=" * 60)
    
    # ─────────────────────────────────────────────────────────────────
    # 1. Load Data
    # ─────────────────────────────────────────────────────────────────
    
    print("\n[1/6] Loading data...")
    
    try:
        loader = SummaryDataLoader(EXPERIMENT_ID, BASE_PATH)
        df = loader.load_all_cells(cells=CELLS, temp_map=TEMP_MAP)
        print(f"  ✓ Loaded {len(df)} samples from {df['cell_id'].nunique()} cells")
    except FileNotFoundError as e:
        print(f"  ✗ Data not found: {e}")
        return
    
    # ─────────────────────────────────────────────────────────────────
    # 2. Create Sequence Pipeline
    # ─────────────────────────────────────────────────────────────────
    
    print("\n[2/6] Creating sequence pipeline...")
    
    pipeline = LatentODESequencePipeline(
        time_unit="days",
        max_seq_len=None,
        normalize=True
    )
    
    samples = pipeline.fit_transform({'df': df})
    print(f"  ✓ Created {len(samples)} sequence samples (one per cell)")
    
    if len(samples) > 0:
        print(f"  ✓ Sequence length: {samples[0].seq_len}")
        print(f"  ✓ Feature dimension: {samples[0].feature_dim}")
        print(f"  ✓ Time range: 0 to {samples[0].t[-1].item():.1f} days")
    
    # ─────────────────────────────────────────────────────────────────
    # 3. Split Data
    # ─────────────────────────────────────────────────────────────────
    
    print("\n[3/6] Splitting by temperature...")
    
    train_samples, val_samples = temperature_split(
        samples,
        train_temps=[10, 40],
        val_temps=[25]
    )
    
    print(f"  ✓ Train: {len(train_samples)} cells")
    print(f"  ✓ Val:   {len(val_samples)} cells")
    
    if len(train_samples) == 0 or len(val_samples) == 0:
        print("  ✗ Not enough samples for train/val split")
        return
    
    # ─────────────────────────────────────────────────────────────────
    # 4. Train Neural ODE
    # ─────────────────────────────────────────────────────────────────
    
    print("\n[4/6] Training Neural ODE...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_dim = samples[0].feature_dim
    
    try:
        ode_model = NeuralODEModel(
            input_dim=input_dim,
            output_dim=1,
            latent_dim=32,
            hidden_dim=64,
            solver='dopri5',
            rtol=1e-4,
            atol=1e-5,
            use_adjoint=True
        )
        
        print(f"  ✓ Neural ODE: {ode_model.count_parameters():,} parameters")
        
        # Use absolute paths for tracking to work correctly from any directory
        artifacts_dir = PROJECT_ROOT / "artifacts"
        tracker = DualTracker(
            local_base_dir=str(artifacts_dir / "runs"),
            use_tensorboard=True,
            mlflow_tracking_uri=f"sqlite:///{artifacts_dir / 'mlflow.db'}",
            mlflow_experiment_name="battery_degradation"
        )
        tracker.start_run("milestone_c_neural_ode", {
            'model': 'neural_ode',
            'pipeline': 'latent_ode_seq',
            'loss': 'mse',
            **CONFIG
        })
        
        # Use modular loss function configuration
        ode_trainer = Trainer(
            ode_model, CONFIG, tracker, device=device,
            loss_config={'name': 'mse'}  # Can swap to 'huber', 'physics_informed', etc.
        )
        ode_history = ode_trainer.fit(train_samples, val_samples)
        
        # Evaluate
        ode_predictions = ode_trainer.predict(val_samples)
        y_val = np.vstack([s.y.numpy() for s in val_samples])
        ode_metrics = compute_metrics(y_val.flatten(), ode_predictions.flatten())
        
        tracker.log_metrics({'final_' + k: v for k, v in ode_metrics.items()})
        tracker.end_run()
        
        print("\n" + "=" * 40)
        print("Neural ODE Results (25°C Holdout)")
        print("=" * 40)
        print_metrics(ode_metrics)
        
    except ImportError as e:
        print(f"  ✗ torchdiffeq not installed: {e}")
        print("  Install with: pip install torchdiffeq")
        ode_metrics = None
    except Exception as e:
        print(f"  ✗ Neural ODE training failed: {e}")
        ode_metrics = None
    
    # ─────────────────────────────────────────────────────────────────
    # 5. Train LSTM Baseline for Comparison
    # ─────────────────────────────────────────────────────────────────
    
    print("\n[5/6] Training LSTM baseline for comparison...")
    
    lstm_model = LSTMAttentionModel(
        input_dim=input_dim,
        output_dim=1,
        hidden_dim=64,
        num_layers=2,
        num_heads=4
    )
    
    print(f"  ✓ LSTM: {lstm_model.count_parameters():,} parameters")
    
    # Use absolute paths for tracking to work correctly from any directory
    artifacts_dir = PROJECT_ROOT / "artifacts"
    tracker2 = DualTracker(
        local_base_dir=str(artifacts_dir / "runs"),
        use_tensorboard=True,
        mlflow_tracking_uri=f"sqlite:///{artifacts_dir / 'mlflow.db'}",
        mlflow_experiment_name="battery_degradation"
    )
    tracker2.start_run("milestone_c_lstm_baseline", {
        'model': 'lstm_attn',
        'pipeline': 'latent_ode_seq',
        'loss': 'mse',
        **CONFIG
    })
    
    lstm_trainer = Trainer(
        lstm_model, CONFIG, tracker2, device=device,
        loss_config={'name': 'mse'}
    )
    lstm_history = lstm_trainer.fit(train_samples, val_samples)
    
    lstm_predictions = lstm_trainer.predict(val_samples)
    lstm_metrics = compute_metrics(y_val.flatten(), lstm_predictions.flatten())
    
    tracker2.log_metrics({'final_' + k: v for k, v in lstm_metrics.items()})
    tracker2.end_run()
    
    print("\n" + "=" * 40)
    print("LSTM Baseline Results (25°C Holdout)")
    print("=" * 40)
    print_metrics(lstm_metrics)
    
    # ─────────────────────────────────────────────────────────────────
    # 6. Comparison
    # ─────────────────────────────────────────────────────────────────
    
    print("\n" + "=" * 40)
    print("Model Comparison")
    print("=" * 40)
    
    print(f"\n{'Model':<20} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
    print("-" * 50)
    
    if ode_metrics:
        print(f"{'Neural ODE':<20} {ode_metrics['rmse']:<10.5f} {ode_metrics['mae']:<10.5f} {ode_metrics['r2']:<10.4f}")
    
    print(f"{'LSTM+Attention':<20} {lstm_metrics['rmse']:<10.5f} {lstm_metrics['mae']:<10.5f} {lstm_metrics['r2']:<10.4f}")
    
    # ─────────────────────────────────────────────────────────────────
    # 7. Visualize Full Prediction Trajectory for One Cell
    # ─────────────────────────────────────────────────────────────────
    
    if ode_metrics and len(val_samples) > 0:
        try:
            import matplotlib.pyplot as plt
            
            # Select one cell for visualization
            sample_idx = 0
            sample = val_samples[sample_idx].to_device(device)
            cell_id = sample.meta['cell_id']
            
            print(f"\n[7/7] Visualizing predictions for cell {cell_id}...")
            
            # Get time values
            t = sample.t.cpu().numpy()
            x = sample.x.unsqueeze(0)  # Add batch dim: (1, seq_len, features)
            
            # Get Neural ODE predictions at all timesteps
            ode_model.eval()
            with torch.no_grad():
                ode_preds = ode_model(x, t=sample.t)[0, :, 0].cpu().numpy()  # (seq_len,)
            
            # Get LSTM predictions at all timesteps
            lstm_model.eval()
            with torch.no_grad():
                lstm_preds = lstm_model(x)[0, :, 0].cpu().numpy()  # (seq_len,)
            
            # Get actual SOH values at each timestep from original data
            try:
                loader = SummaryDataLoader(EXPERIMENT_ID, BASE_PATH)
                temp_C = sample.meta['temperature_C']
                cell_df = loader.load_performance_summary(cell_id, temp_C)
                cell_df = cell_df.sort_index()
                
                # Extract capacity values
                target_col = None
                for col in ['Cell Capacity [mA h]', 'Cell Capacity [A h]', 'SoH']:
                    if col in cell_df.columns:
                        target_col = col
                        break
                
                if target_col:
                    capacity_values = cell_df[target_col].values.astype(float)
                    capacity_values = np.nan_to_num(capacity_values, nan=0.0)
                    
                    # Convert mAh to Ah if needed
                    if capacity_values[0] > 100:
                        capacity_values = capacity_values / 1000.0
                    
                    # Calculate SOH: current capacity / initial capacity
                    initial_capacity = capacity_values[0]
                    if initial_capacity > 0:
                        actual_soh = capacity_values / initial_capacity
                    else:
                        actual_soh = np.ones_like(capacity_values)
                    
                    # Match length with predictions (in case of truncation)
                    if len(actual_soh) > len(t):
                        actual_soh = actual_soh[:len(t)]
                    elif len(actual_soh) < len(t):
                        # Pad with last value if needed
                        actual_soh = np.pad(actual_soh, (0, len(t) - len(actual_soh)), 
                                           mode='edge')
                else:
                    actual_soh = None
            except Exception as e:
                print(f"  Warning: Could not load actual SOH values: {e}")
                actual_soh = None
            
            # Create visualization
            figures_dir = PROJECT_ROOT / "artifacts" / "figures"
            figures_dir.mkdir(parents=True, exist_ok=True)
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            # Plot actual SOH if available
            if actual_soh is not None:
                ax.plot(t, actual_soh, 'ko-', label='Actual SOH', 
                       linewidth=2, markersize=6, alpha=0.7)
            
            # Plot Neural ODE predictions
            ax.plot(t, ode_preds, 'b-', label='Neural ODE', 
                   linewidth=2, alpha=0.8)
            
            # Plot LSTM predictions
            ax.plot(t, lstm_preds, 'r--', label='LSTM+Attention', 
                   linewidth=2, alpha=0.8)
            
            ax.set_xlabel('Time (days)', fontsize=12)
            ax.set_ylabel('SOH', fontsize=12)
            ax.set_title(f'Prediction Trajectory Comparison - Cell {cell_id} ({temp_C}°C)', 
                        fontsize=14, fontweight='bold')
            ax.legend(fontsize=11, loc='best')
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, max(1.1, max(ode_preds.max(), lstm_preds.max()) * 1.1)])
            
            plt.tight_layout()
            trajectory_path = figures_dir / "neural_ode_trajectory.png"
            plt.savefig(str(trajectory_path), dpi=150, bbox_inches='tight')
            print(f"  ✓ Trajectory plot saved to {trajectory_path}")
            plt.close()
            
        except Exception as e:
            import traceback
            print(f"  Could not plot trajectory: {e}")
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Milestone C Complete!")
    print("=" * 60)
    
    return ode_metrics, lstm_metrics


if __name__ == "__main__":
    main()
