"""Milestone B: ICA peak features + SHAP analysis.

This script demonstrates:
1. Load voltage curve data for ICA analysis
2. Extract dQ/dV peak features using ICAPeaksPipeline
3. Train LGBM on ICA features
4. Analyze feature importance with SHAP

Usage:
    python examples/milestone_b.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths

from src.data.tables import TimeseriesDataLoader, SummaryDataLoader
from src.data.expt_paths import ExperimentPaths
from src.pipelines.ica_peaks import ICAPeaksPipeline
from src.models.lgbm import LGBMModel
from src.data.splits import temperature_split
from src.training.metrics import compute_metrics, print_metrics
from src.explainability.shap_analysis import compute_shap_values, plot_shap_summary, get_feature_importance
from src.tracking.dual_tracker import DualTracker


def load_voltage_curves(base_path: Path, experiment_id: int, temp_map: dict):
    """Load all voltage curves with targets."""
    
    ts_loader = TimeseriesDataLoader(experiment_id, base_path)
    summary_loader = SummaryDataLoader(experiment_id, base_path)
    paths = ExperimentPaths(experiment_id, base_path)
    
    curves = []
    targets = {}
    
    for temp_C, cells in temp_map.items():
        for cell_id in cells:
            # Get available RPTs
            rpts = paths.list_available_rpts(cell_id)
            
            if not rpts:
                print(f"  No RPTs found for cell {cell_id}")
                continue
            
            # Load summary to get SOH targets
            try:
                summary_df = summary_loader.load_performance_summary(cell_id, temp_C)
            except FileNotFoundError:
                continue
            
            # Get initial capacity (first capacity value) for this cell
            target_col = 'Cell Capacity [mA h]' if 'Cell Capacity [mA h]' in summary_df.columns else 'SoH'
            initial_capacity = None
            if target_col in summary_df.columns and len(summary_df) > 0:
                first_capacity = summary_df[target_col].iloc[0]
                if not pd.isna(first_capacity):
                    initial_capacity = first_capacity / 1000.0 if first_capacity > 100 else first_capacity
            
            for rpt_id in rpts:
                try:
                    df = ts_loader.load_voltage_curve(cell_id, rpt_id)
                    
                    # Check if DataFrame is empty
                    if df.empty:
                        print(f"  Warning: Empty DataFrame for {cell_id} RPT{rpt_id}")
                        continue
                    
                    # Get voltage and capacity columns
                    # Column names: "Voltage (V)" and "Charge (mA.h)" or "Capacity"
                    voltage_cols = [c for c in df.columns if 'Voltage' in c or 'voltage' in c.lower()]
                    capacity_cols = [c for c in df.columns if 'Capacity' in c or 'capacity' in c.lower() or 'Charge' in c or 'charge' in c.lower()]
                    
                    if not voltage_cols:
                        print(f"  Warning: No voltage column found in {cell_id} RPT{rpt_id}. Available columns: {list(df.columns)}")
                        continue
                    if not capacity_cols:
                        print(f"  Warning: No capacity/charge column found in {cell_id} RPT{rpt_id}. Available columns: {list(df.columns)}")
                        continue
                    
                    voltage_col = voltage_cols[0]
                    capacity_col = capacity_cols[0]
                    
                    voltage = df[voltage_col].values
                    capacity = df[capacity_col].values
                    
                    # Check if arrays are empty
                    if len(voltage) == 0 or len(capacity) == 0:
                        print(f"  Warning: Empty voltage/capacity arrays for {cell_id} RPT{rpt_id}")
                        continue
                    
                    meta = {
                        'experiment_id': experiment_id,
                        'cell_id': cell_id,
                        'rpt_id': rpt_id,
                        'temperature_C': temp_C,
                    }
                    
                    curves.append((voltage, capacity, meta))
                    
                    # Get target SOH from summary (match by RPT index if possible)
                    if rpt_id < len(summary_df):
                        target_col = 'Cell Capacity [mA h]' if 'Cell Capacity [mA h]' in summary_df.columns else 'SoH'
                        if target_col in summary_df.columns:
                            capacity = summary_df[target_col].iloc[min(rpt_id, len(summary_df)-1)]
                            if not pd.isna(capacity):
                                # Convert mAh to Ah if needed
                                if capacity > 100:
                                    capacity = capacity / 1000.0
                                
                                # Calculate SOH: current capacity / initial capacity
                                if initial_capacity and initial_capacity > 0:
                                    soh = capacity / initial_capacity
                                else:
                                    soh = 1.0
                                targets[(cell_id, rpt_id)] = soh
                
                except FileNotFoundError:
                    continue
                except Exception as e:
                    print(f"  Error loading {cell_id} RPT{rpt_id}: {e}")
                    continue
    
    return curves, targets


def visualize_ica_analysis(pipeline: ICAPeaksPipeline, curves: list, 
                            num_examples: int = 4, save_path: Path = None):
    """Visualize ICA analysis with identified peaks for example curves.
    
    Args:
        pipeline: ICAPeaksPipeline instance
        curves: List of (voltage, capacity, meta) tuples
        num_examples: Number of example curves to plot
        save_path: Path to save the figure
    """
    # Select diverse examples (different cells, different RPTs)
    example_indices = np.linspace(0, len(curves) - 1, num_examples, dtype=int)
    
    fig, axes = plt.subplots(num_examples, 2, figsize=(14, 3 * num_examples))
    if num_examples == 1:
        axes = axes.reshape(1, -1)
    
    for idx, curve_idx in enumerate(example_indices):
        voltage, capacity, meta = curves[curve_idx]
        
        # Compute ICA
        V_mid, dQdV = pipeline.compute_ica(voltage, capacity)
        
        if len(V_mid) == 0 or len(dQdV) == 0:
            axes[idx, 0].text(0.5, 0.5, f"Invalid curve\n{meta['cell_id']} RPT{meta['rpt_id']}", 
                             ha='center', va='center', transform=axes[idx, 0].transAxes)
            axes[idx, 1].text(0.5, 0.5, "No ICA data", 
                             ha='center', va='center', transform=axes[idx, 1].transAxes)
            continue
        
        # Find peaks - Use same parameters as pipeline
        peaks, properties = find_peaks(
            dQdV,
            height=0.01,
            distance=4000,
            prominence=1
        )
        
        # Get peak widths
        if len(peaks) > 0:
            try:
                widths, width_heights, left_ips, right_ips = peak_widths(dQdV, peaks, rel_height=0.5)
            except Exception:
                widths = np.zeros(len(peaks))
        else:
            widths = np.array([])
        
        # Sort peaks by height
        if len(peaks) > 0:
            sorted_idx = np.argsort(dQdV[peaks])[::-1]
            peaks = peaks[sorted_idx]
            if len(widths) == len(peaks):
                widths = widths[sorted_idx]
        
        # Plot voltage-capacity curve
        axes[idx, 0].plot(voltage, capacity, 'b-', linewidth=1.5, label='Voltage-Capacity')
        axes[idx, 0].set_xlabel('Voltage (V)')
        axes[idx, 0].set_ylabel('Capacity (Ah)')
        axes[idx, 0].set_title(f"Cell {meta['cell_id']} RPT{meta['rpt_id']} - Voltage Curve")
        axes[idx, 0].set_xlim(pipeline.voltage_range)
        axes[idx, 0].grid(True, alpha=0.3)
        axes[idx, 0].legend()
        
        # Plot ICA curve with peaks
        axes[idx, 1].plot(V_mid, dQdV, 'b-', linewidth=1.5, label='dQ/dV')
        
        # Mark identified peaks - Only draw top 3 most prominent
        if len(peaks) > 0:
            top_peaks = peaks[:3]
            top_widths = widths[:3] if len(widths) >= 3 else widths
            
            for i, peak_idx in enumerate(top_peaks):
                color = plt.cm.tab10(i)
                axes[idx, 1].plot(V_mid[peak_idx], dQdV[peak_idx], 'ro', 
                                 markersize=10, label=f'Peak {i+1}')
                # Draw peak width (if available)
                if i < len(top_widths) and top_widths[i] > 0:
                    # Convert width from index units to voltage units
                    dV = V_mid[1] - V_mid[0] if len(V_mid) > 1 else 0.001
                    width_v = top_widths[i] * dV
                    left_v = V_mid[peak_idx] - width_v / 2
                    right_v = V_mid[peak_idx] + width_v / 2
                    axes[idx, 1].axvline(left_v, color=color, linestyle='--', alpha=0.5, linewidth=1)
                    axes[idx, 1].axvline(right_v, color=color, linestyle='--', alpha=0.5, linewidth=1)
        
        axes[idx, 1].set_xlabel('Voltage (V)')
        axes[idx, 1].set_ylabel('dQ/dV (Ah/V)')
        axes[idx, 1].set_title(f"ICA Analysis - {len(peaks)} peaks detected")
        axes[idx, 1].set_xlim(pipeline.voltage_range)
        axes[idx, 1].grid(True, alpha=0.3)
        axes[idx, 1].legend()
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
        print(f"  ✓ ICA visualization saved to {save_path}")
    
    plt.close()


def main():
    """Run Milestone B: ICA features + SHAP."""
    
    # Get project root (parent of examples directory)
    PROJECT_ROOT = Path(__file__).parent.parent
    BASE_PATH = PROJECT_ROOT / "Raw Data"
    EXPERIMENT_ID = 5
    TEMP_MAP = {
        10: ['A', 'B', 'C'],
        25: ['D', 'E'],
        40: ['F', 'G', 'H']
    }
    
    print("=" * 60)
    print("Milestone B: ICA Peak Features + SHAP Analysis")
    print("=" * 60)
    
    # ─────────────────────────────────────────────────────────────────
    # 1. Load Voltage Curves
    # ─────────────────────────────────────────────────────────────────
    
    print("\n[1/5] Loading voltage curves...")
    
    try:
        curves, targets = load_voltage_curves(BASE_PATH, EXPERIMENT_ID, TEMP_MAP)
        print(f"  ✓ Loaded {len(curves)} voltage curves")
        print(f"  ✓ Targets available for {len(targets)} curves")
    except Exception as e:
        print(f"  ✗ Error loading data: {e}")
        print(f"\n  Please ensure data is at: {BASE_PATH.resolve()}")
        return
    
    if len(curves) == 0:
        print("  ✗ No curves found. Check data paths.")
        return
    
    # ─────────────────────────────────────────────────────────────────
    # 2. Create ICA Pipeline
    # ─────────────────────────────────────────────────────────────────
    
    print("\n[2/5] Creating ICA pipeline...")
    
    pipeline = ICAPeaksPipeline(
        sg_window=1541,
        sg_order=3,
        num_peaks=3,
        voltage_range=(3.0, 4.2),
        resample_points=2000,
        normalize=True,
        use_cache=True
    )
    
    print(f"  ✓ Pipeline: {pipeline.get_params()}")
    
    # ─────────────────────────────────────────────────────────────────
    # 3. Transform and Split
    # ─────────────────────────────────────────────────────────────────
    
    print("\n[3/5] Extracting ICA features...")
    
    samples = pipeline.fit_transform({'curves': curves, 'targets': targets})
    print(f"  ✓ Created {len(samples)} samples")
    print(f"  ✓ Feature names: {pipeline.get_feature_names()}")
    
    # Visualize ICA analysis
    print("\n[3.5/5] Creating ICA visualization...")
    figures_dir = PROJECT_ROOT / "artifacts" / "figures"
    ica_viz_path = figures_dir / "ica_peak_detection.png"
    visualize_ica_analysis(pipeline, curves, num_examples=4, save_path=ica_viz_path)
    
    train_samples, val_samples = temperature_split(
        samples,
        train_temps=[10, 40],
        val_temps=[25]
    )
    
    print(f"  ✓ Train: {len(train_samples)}, Val: {len(val_samples)}")
    
    if len(train_samples) == 0 or len(val_samples) == 0:
        print("  ✗ Not enough samples for train/val split")
        return
    
    # ─────────────────────────────────────────────────────────────────
    # 4. Train Model
    # ─────────────────────────────────────────────────────────────────
    
    print("\n[4/5] Training LGBM on ICA features...")
    
    X_train = np.vstack([s.x for s in train_samples])
    y_train = np.vstack([s.y for s in train_samples])
    X_val = np.vstack([s.x for s in val_samples])
    y_val = np.vstack([s.y for s in val_samples])
    
    model = LGBMModel(n_estimators=500, learning_rate=0.05)
    model.fit(X_train, y_train, X_val, y_val, feature_names=pipeline.get_feature_names())
    
    y_pred = model.predict(X_val)
    metrics = compute_metrics(y_val.flatten(), y_pred.flatten())
    
    print("\n" + "=" * 40)
    print("ICA Feature Results (25°C Holdout)")
    print("=" * 40)
    print_metrics(metrics)
    
    # ─────────────────────────────────────────────────────────────────
    # 5. SHAP Analysis
    # ─────────────────────────────────────────────────────────────────
    
    print("\n[5/5] Computing SHAP values...")
    
    shap_result = compute_shap_values(
        model, X_val,
        feature_names=pipeline.get_feature_names()
    )
    
    if 'error' in shap_result:
        print(f"  ✗ SHAP error: {shap_result['error']}")
    else:
        print("  ✓ SHAP values computed")
        
        # Feature importance from SHAP
        importance = get_feature_importance(shap_result)
        print("\n" + "=" * 40)
        print("SHAP Feature Importance")
        print("=" * 40)
        
        sorted_imp = sorted(importance.items(), key=lambda x: -x[1])
        for name, imp in sorted_imp[:5]:
            print(f"  {name}: {imp:.4f}")
        
        # Save SHAP plot
        figures_dir = PROJECT_ROOT / "artifacts" / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        shap_path = figures_dir / "shap_ica.png"
        plot_shap_summary(shap_result, X_val, save_path=str(shap_path))
        print(f"\n  ✓ SHAP plot saved to {shap_path}")
    
    # ─────────────────────────────────────────────────────────────────
    # Track Experiment
    # ─────────────────────────────────────────────────────────────────
    
    # Use absolute paths for tracking to work correctly from any directory
    artifacts_dir = PROJECT_ROOT / "artifacts"
    tracker = DualTracker(
        local_base_dir=str(artifacts_dir / "runs"),
        use_tensorboard=True,
        mlflow_tracking_uri=f"sqlite:///{artifacts_dir / 'mlflow.db'}",
        mlflow_experiment_name="battery_degradation"
    )
    run_id = tracker.start_run("milestone_b_ica_shap", {
        'experiment_id': EXPERIMENT_ID,
        'pipeline': 'ica_peaks',
        'model': 'lgbm',
        'n_curves': len(curves),
    })
    tracker.log_metrics(metrics)
    tracker.end_run()
    
    print("\n" + "=" * 60)
    print("Milestone B Complete!")
    print("=" * 60)
    
    return metrics


if __name__ == "__main__":
    main()
