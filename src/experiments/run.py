"""Unified experiment runner for BatteryML.

This module provides a centralized experiment orchestration system that replaces
manual script execution with declarative configuration-driven experiments.

Usage:
    python -m src.experiments.run --config-name=config model=mlp pipeline=ica_peaks
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
from omegaconf import DictConfig, OmegaConf

from src.data.registry import DataLoaderRegistry
from src.pipelines.registry import PipelineRegistry
from src.models.registry import ModelRegistry
from src.data.splits import temperature_split, leave_one_cell_out, random_split
from src.training.metrics import compute_metrics, print_metrics
from src.tracking.dual_tracker import DualTracker
from src.config_schema import ExperimentConfig


class ExperimentRunner:
    """Unified experiment runner that orchestrates all components."""
    
    def __init__(self, cfg: DictConfig):
        """Initialize runner with configuration.
        
        Args:
            cfg: Hydra configuration object
        """
        self.cfg = cfg
        self.config = ExperimentConfig(**OmegaConf.to_container(cfg, resolve=True))
        
        # Set random seed for reproducibility
        np.random.seed(self.config.seed)
        
        # Initialize components
        self.loader = None
        self.pipeline = None
        self.model = None
        self.tracker = None
        
    def setup_data_loader(self) -> None:
        """Setup data loader based on configuration and pipeline requirements."""
        print("[1/7] Setting up data loader...")
        
        # Determine data format based on pipeline
        self.data_format = DataLoaderRegistry.get_data_format(self.config.pipeline.name)
        
        print(f"  ✓ Data format: {self.data_format}")
        print(f"  ✓ Experiment: {self.config.data.experiment_id}")
        print(f"  ✓ Base path: {self.config.data.base_path}")
        
    def load_data(self) -> Any:
        """Load data using the appropriate loader for the pipeline."""
        print("[2/7] Loading data...")
        
        try:
            # Use registry to load data in the format expected by pipeline
            data = DataLoaderRegistry.load_data(
                pipeline_name=self.config.pipeline.name,
                experiment_id=self.config.data.experiment_id,
                base_path=self.config.data.base_path,
                cells=list(self.config.data.cells),
                temp_map={int(k): list(v) for k, v in self.config.data.temp_map.items()}
            )
            
            # Print summary based on data format
            if self.data_format == 'summary':
                df = data['df']
                print(f"  ✓ Loaded {len(df)} samples from {df['cell_id'].nunique()} cells")
            elif self.data_format == 'curves':
                curves = data['curves']
                n_cells = len(set(meta['cell_id'] for _, _, meta in curves))
                print(f"  ✓ Loaded {len(curves)} curves from {n_cells} cells")
            
            return data
        except FileNotFoundError as e:
            print(f"  ✗ Data not found: {e}")
            print(f"  Please ensure data is at: {self.config.data.base_path}")
            raise
            
    def setup_pipeline(self) -> None:
        """Setup preprocessing pipeline using registry."""
        print("[3/7] Setting up pipeline...")
        
        # Extract pipeline parameters based on pipeline type
        if self.config.pipeline.name == "summary_set":
            pipeline_params = {
                'include_arrhenius': self.config.pipeline.include_arrhenius,
                'arrhenius_Ea': self.config.pipeline.arrhenius_Ea,
                'normalize': self.config.pipeline.normalize,
            }
        elif self.config.pipeline.name == "ica_peaks":
            pipeline_params = {
                'sg_window': self.config.pipeline.sg_window,
                'sg_order': self.config.pipeline.sg_order,
                'num_peaks': self.config.pipeline.num_peaks,
                'voltage_range': tuple(self.config.pipeline.voltage_range),
                'normalize': self.config.pipeline.normalize,
                'use_cache': self.config.pipeline.use_cache,
            }
        elif self.config.pipeline.name == "latent_ode_seq":
            pipeline_params = {
                'time_unit': self.config.pipeline.time_unit,
                'normalize': self.config.pipeline.normalize,
            }
        else:
            pipeline_params = {
                'normalize': self.config.pipeline.normalize,
            }
            
        self.pipeline = PipelineRegistry.get(self.config.pipeline.name, **pipeline_params)
        print(f"  ✓ Pipeline: {self.config.pipeline.name}")
        print(f"  ✓ Parameters: {pipeline_params}")
        
    def create_samples(self, data: Any) -> List:
        """Fit pipeline and transform raw data to canonical Sample objects."""
        print("[4/7] Creating samples...")
        
        # Pass data directly - it's already in the format expected by the pipeline
        samples = self.pipeline.fit_transform(data)
        print(f"  ✓ Created {len(samples)} samples")
        if samples:
            print(f"  ✓ Feature dimension: {samples[0].feature_dim}")
        return samples
        
    def split_data(self, samples: List) -> Tuple[List, List]:
        """Split data using configured strategy."""
        print("[5/7] Splitting data...")
        
        if self.config.split.strategy == "temperature_holdout":
            train_samples, val_samples = temperature_split(
                samples,
                train_temps=self.config.split.train_temps,
                val_temps=self.config.split.val_temps
            )
            print(f"  ✓ Temperature holdout: {self.config.split.train_temps} -> {self.config.split.val_temps}")
            
        elif self.config.split.strategy == "loco":
            train_samples, val_samples = leave_one_cell_out(
                samples,
                test_cell=self.config.split.test_cell
            )
            print(f"  ✓ LOCO: test cell = {self.config.split.test_cell}")
            
        elif self.config.split.strategy == "random":
            train_samples, val_samples, _ = random_split(
                samples,
                train_fraction=1-self.config.split.val_fraction,
                val_fraction=self.config.split.val_fraction,
                seed=self.config.split.random_seed
            )
            print(f"  ✓ Random split: {self.config.split.val_fraction} validation")
            
        else:
            raise ValueError(f"Unknown split strategy: {self.config.split.strategy}")
            
        print(f"  ✓ Train: {len(train_samples)} samples")
        print(f"  ✓ Val:   {len(val_samples)} samples")
        
        return train_samples, val_samples
        
    def setup_model(self, input_dim: int = None) -> None:
        """Setup model using registry."""
        print("[6/7] Setting up model...")
        
        # Extract model parameters based on model type
        if self.config.model.name == "lgbm":
            model_params = {
                'n_estimators': self.config.model.n_estimators,
                'learning_rate': self.config.model.learning_rate,
                'max_depth': self.config.model.max_depth,
                'num_leaves': self.config.model.num_leaves,
                'reg_alpha': self.config.model.reg_alpha,
                'reg_lambda': self.config.model.reg_lambda,
                'early_stopping_rounds': self.config.model.early_stopping_rounds,
            }
        elif self.config.model.name == "mlp":
            model_params = {
                'input_dim': input_dim,
                'hidden_dims': list(self.config.model.hidden_dims),
                'dropout': self.config.model.dropout,
            }
        elif self.config.model.name == "lstm_attn":
            model_params = {
                'input_dim': input_dim,
                'hidden_dim': self.config.model.hidden_dim,
                'num_layers': self.config.model.num_layers,
                'num_heads': self.config.model.num_heads,
                'dropout': self.config.model.dropout,
            }
        elif self.config.model.name == "neural_ode":
            model_params = {
                'input_dim': input_dim,
                'latent_dim': self.config.model.latent_dim,
                'hidden_dim': self.config.model.hidden_dim,
                'solver': self.config.model.solver,
                'rtol': self.config.model.rtol,
                'atol': self.config.model.atol,
                'use_adjoint': self.config.model.use_adjoint,
            }
        else:
            model_params = {}
            
        self.model = ModelRegistry.get(self.config.model.name, **model_params)
        print(f"  ✓ Model: {self.config.model.name}")
        print(f"  ✓ Parameters: {model_params}")
        
    def setup_tracker(self) -> None:
        """Setup experiment tracking."""
        print("[7/7] Setting up tracking...")
        
        # Get project root for paths
        project_root = Path(__file__).parent.parent.parent
        
        # Setup tracking based on configuration
        if self.config.tracking.backend == "dual":
            self.tracker = DualTracker(
                local_base_dir=str(project_root / "artifacts/runs"),
                use_tensorboard=self.config.tracking.use_tensorboard,
                mlflow_tracking_uri=self.config.tracking.mlflow_uri,
                mlflow_experiment_name=self.config.tracking.experiment_name
            )
        else:
            # Fallback to local tracking
            from src.tracking.local import LocalTracker
            self.tracker = LocalTracker(
                base_dir=str(project_root / "artifacts/runs")
            )
            
        print(f"  ✓ Tracker: {self.config.tracking.backend}")
        
    def train_and_evaluate(self, train_samples: List, val_samples: List) -> Dict[str, float]:
        """Train model and evaluate on validation set."""
        print("[7/7] Training and evaluating...")
        
        # Convert samples to arrays for models that need it
        if self.config.model.name == "lgbm":
            X_train = np.vstack([s.x for s in train_samples])
            y_train = np.vstack([s.y for s in train_samples])
            X_val = np.vstack([s.x for s in val_samples])
            y_val = np.vstack([s.y for s in val_samples])
            
            # Train LGBM model
            self.model.fit(
                X_train, y_train,
                X_val, y_val,
                feature_names=self.pipeline.get_feature_names()
            )
            
            # Predict and evaluate
            y_pred = self.model.predict(X_val)
            metrics = compute_metrics(y_val.flatten(), y_pred.flatten())
            
        else:
            # For neural models, use trainer
            from src.training.trainer import Trainer
            
            # Setup loss config dict
            loss_config = {'name': self.config.loss.name}
            if self.config.loss.name == "huber":
                loss_config['delta'] = self.config.loss.delta
            elif self.config.loss.name == "mape":
                loss_config['epsilon'] = self.config.loss.epsilon
            elif self.config.loss.name == "physics_informed":
                loss_config['monotonicity_weight'] = self.config.loss.monotonicity_weight
                loss_config['smoothness_weight'] = self.config.loss.smoothness_weight
            loss_config['reduction'] = self.config.loss.reduction
            
            # Create trainer with config dict
            training_config = self.config.training.model_dump()
            trainer = Trainer(
                model=self.model,
                config=training_config,
                loss_config=loss_config
            )
            
            # Train model
            trainer.fit(train_samples, val_samples)
            
            # Get predictions and compute metrics
            y_pred = trainer.predict(val_samples)
            y_val = np.vstack([s.y for s in val_samples])
            metrics = compute_metrics(y_val.flatten(), y_pred.flatten())
            
        return metrics
        
    def log_results(self, metrics: Dict[str, float], train_samples: List, val_samples: List) -> str:
        """Log experiment results to tracking system."""
        print("\n" + "=" * 40)
        print("Logging Results")
        print("=" * 40)
        
        # Start tracking run
        run_params = {
            'experiment_id': self.config.data.experiment_id,
            'model': self.config.model.name,
            'pipeline': self.config.pipeline.name,
            'split_strategy': self.config.split.strategy,
            'n_train': len(train_samples),
            'n_val': len(val_samples),
            'seed': self.config.seed,
        }
        
        # Add model-specific parameters
        if self.config.model.name == "lgbm":
            run_params['model_params'] = self.model.params
        else:
            run_params['model_params'] = self.config.model.model_dump()
            
        run_id = self.tracker.start_run(self.config.name, run_params)
        
        # Log metrics
        self.tracker.log_metrics(metrics)
        
        # Print results
        print("\n" + "=" * 40)
        print("Results")
        print("=" * 40)
        print_metrics(metrics)
        
        # End run
        self.tracker.end_run()
        
        print(f"  ✓ Run ID: {run_id}")
        print(f"  ✓ Results saved to: artifacts/runs/{run_id}")
        
        return run_id
        
    def run(self) -> Dict[str, float]:
        """Run complete experiment pipeline."""
        print("=" * 60)
        print(f"Experiment: {self.config.name}")
        print(f"Model: {self.config.model.name}")
        print(f"Pipeline: {self.config.pipeline.name}")
        print(f"Split: {self.config.split.strategy}")
        print("=" * 60)
        
        # Setup data loader and pipeline first
        self.setup_data_loader()
        self.setup_pipeline()
        
        # Load and process data to get input dimension
        data = self.load_data()
        samples = self.create_samples(data)
        train_samples, val_samples = self.split_data(samples)
        
        # Now setup model with known input dimension
        input_dim = samples[0].feature_dim if samples and len(samples) > 0 else None
        if input_dim is None:
            raise ValueError("No samples created - cannot determine input dimension for model")
        self.setup_model(input_dim=input_dim)
        self.setup_tracker()
        
        # Train and evaluate
        metrics = self.train_and_evaluate(train_samples, val_samples)
        
        # Log results
        self.log_results(metrics, train_samples, val_samples)
        
        print("\n" + "=" * 60)
        print("Experiment Complete!")
        print("=" * 60)
        
        return metrics


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> Dict[str, float]:
    """Main entry point for experiment runner.
    
    Args:
        cfg: Hydra configuration
        
    Returns:
        Dictionary of evaluation metrics
    """
    runner = ExperimentRunner(cfg)
    return runner.run()


if __name__ == "__main__":
    main()
