"""Milestone A: End-to-end LGBM baseline on Expt 5 summary data.

This script demonstrates the complete pipeline:
1. Load Performance Summary data for all cells
2. Create features using SummarySetPipeline
3. Split by temperature (train 10°C+40°C, validate 25°C)
4. Train LightGBM model
5. Evaluate and log with dual tracking (local + MLflow)

Usage:
    python examples/milestone_a.py

Expected output:
    - Loaded X samples from 8 cells
    - RMSE, MAE, MAPE, R² metrics
    - Feature importance ranking
    - Run saved to artifacts/runs/<run_id>
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.data.tables import SummaryDataLoader
from src.pipelines.summary_set import SummarySetPipeline
from src.models.lgbm import LGBMModel
from src.data.splits import temperature_split
from src.training.metrics import compute_metrics, print_metrics
from src.tracking.dual_tracker import DualTracker


def main():
    """Run Milestone A: LGBM baseline."""
    
    # ─────────────────────────────────────────────────────────────────
    # Configuration
    # ─────────────────────────────────────────────────────────────────
    
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
    
    print("=" * 60)
    print("Milestone A: LGBM Baseline on Summary Data")
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
        print(f"\n  Please ensure data is at: {BASE_PATH.resolve()}")
        print("  Expected structure: Raw Data/Expt 5 - Standard Cycle Aging (Control)/...")
        return
    
    # ─────────────────────────────────────────────────────────────────
    # 2. Create Pipeline
    # ─────────────────────────────────────────────────────────────────
    
    print("\n[2/6] Creating feature pipeline...")
    
    pipeline = SummarySetPipeline(
        include_arrhenius=True,
        arrhenius_Ea=50000.0,
        normalize=True
    )
    
    print(f"  ✓ Pipeline: {pipeline}")
    print(f"  ✓ Features: {pipeline.get_params()}")
    
    # ─────────────────────────────────────────────────────────────────
    # 3. Create Samples
    # ─────────────────────────────────────────────────────────────────
    
    print("\n[3/6] Transforming data to samples...")
    
    all_samples = pipeline.fit_transform({'df': df})
    print(f"  ✓ Created {len(all_samples)} samples")
    print(f"  ✓ Feature dimension: {all_samples[0].feature_dim}")
    
    # ─────────────────────────────────────────────────────────────────
    # 4. Split Data
    # ─────────────────────────────────────────────────────────────────
    
    print("\n[4/6] Splitting by temperature...")
    
    train_samples, val_samples = temperature_split(
        all_samples,
        train_temps=[10, 40],
        val_temps=[25]
    )
    
    print(f"  ✓ Train: {len(train_samples)} samples (10°C + 40°C)")
    print(f"  ✓ Val:   {len(val_samples)} samples (25°C)")
    
    # ─────────────────────────────────────────────────────────────────
    # 5. Prepare Arrays for LGBM
    # ─────────────────────────────────────────────────────────────────
    
    X_train = np.vstack([s.x for s in train_samples])
    y_train = np.vstack([s.y for s in train_samples])
    X_val = np.vstack([s.x for s in val_samples])
    y_val = np.vstack([s.y for s in val_samples])
    
    print(f"  ✓ X_train shape: {X_train.shape}")
    print(f"  ✓ X_val shape:   {X_val.shape}")
    
    # ─────────────────────────────────────────────────────────────────
    # 6. Train Model
    # ─────────────────────────────────────────────────────────────────
    
    print("\n[5/6] Training LightGBM model...")
    
    model = LGBMModel(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        early_stopping_rounds=50
    )
    
    model.fit(
        X_train, y_train,
        X_val, y_val,
        feature_names=pipeline.get_feature_names()
    )
    
    print(f"  ✓ Model trained: {model}")
    
    # ─────────────────────────────────────────────────────────────────
    # 7. Evaluate
    # ─────────────────────────────────────────────────────────────────
    
    print("\n[6/6] Evaluating...")
    
    y_pred = model.predict(X_val)
    metrics = compute_metrics(y_val.flatten(), y_pred.flatten())
    
    print("\n" + "=" * 40)
    print("Results (25°C Holdout)")
    print("=" * 40)
    print_metrics(metrics)
    
    # Feature importance
    print("\n" + "=" * 40)
    print("Top Features")
    print("=" * 40)
    importances = list(zip(pipeline.get_feature_names(), model.feature_importances_))
    importances.sort(key=lambda x: -x[1])
    
    for name, imp in importances[:5]:
        print(f"  {name}: {imp:.1f}")
    
    # ─────────────────────────────────────────────────────────────────
    # 8. Track Experiment
    # ─────────────────────────────────────────────────────────────────
    
    print("\n" + "=" * 40)
    print("Tracking Experiment")
    print("=" * 40)
    
    # Use absolute paths for tracking to work correctly from any directory
    artifacts_dir = PROJECT_ROOT / "artifacts"
    tracker = DualTracker(
        local_base_dir=str(artifacts_dir / "runs"),
        use_tensorboard=True,
        mlflow_tracking_uri=f"sqlite:///{artifacts_dir / 'mlflow.db'}",
        mlflow_experiment_name="battery_degradation"
    )
    
    run_id = tracker.start_run("milestone_a_lgbm", {
        'experiment_id': EXPERIMENT_ID,
        'train_temps': [10, 40],
        'val_temps': [25],
        'model': 'lgbm',
        'pipeline': 'summary_set',
        'n_train': len(train_samples),
        'n_val': len(val_samples),
        'model_params': model.params,
    })
    
    tracker.log_metrics(metrics)
    tracker.end_run()
    
    print(f"  ✓ Run ID: {run_id}")
    print(f"  ✓ Results saved to: artifacts/runs/{run_id}")
    
    print("\n" + "=" * 60)
    print("Milestone A Complete!")
    print("=" * 60)
    
    return metrics


if __name__ == "__main__":
    main()
