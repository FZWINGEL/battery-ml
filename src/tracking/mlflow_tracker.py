"""MLflow-based experiment tracking."""

from pathlib import Path
from typing import Dict, Any, Optional
import logging

from .base import BaseTracker, TrackerRegistry

logger = logging.getLogger(__name__)


@TrackerRegistry.register("mlflow")
class MLflowTracker(BaseTracker):
    """MLflow-based tracking.
    
    Provides:
    - Centralized experiment tracking
    - Model versioning
    - Artifact storage
    - UI for experiment comparison
    
    Example usage:
        >>> tracker = MLflowTracker(tracking_uri="file:./mlruns")
        >>> run_id = tracker.start_run("experiment_1", {"lr": 0.001})
        >>> tracker.log_metrics({"loss": 0.5}, step=1)
        >>> tracker.end_run()
    """
    
    def __init__(self,
                 tracking_uri: str = "file:./artifacts/mlruns",
                 experiment_name: str = "battery_degradation"):
        """Initialize the tracker.
        
        Args:
            tracking_uri: MLflow tracking URI
            experiment_name: Experiment name
        """
        try:
            import mlflow
            self.mlflow = mlflow
        except ImportError:
            raise ImportError("mlflow required. Install with: pip install mlflow")
        
        self.mlflow.set_tracking_uri(tracking_uri)
        self.mlflow.set_experiment(experiment_name)
        
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.run = None
        self.run_id: Optional[str] = None
        
        logger.info(f"MLflow tracking URI: {tracking_uri}")
    
    def start_run(self, run_name: str, config: Dict[str, Any]) -> str:
        """Start a new run.
        
        Args:
            run_name: Name for this run
            config: Configuration dictionary
        
        Returns:
            Run ID
        """
        # End any existing run first
        if self.run is not None:
            try:
                self.mlflow.end_run()
            except Exception:
                pass
        
        try:
            self.run = self.mlflow.start_run(run_name=run_name)
            self.run_id = self.run.info.run_id
            
            # Log config as params (flatten nested dicts)
            flat_config = self._flatten_dict(config)
            
            # MLflow has a 500 char limit on param values
            for k, v in flat_config.items():
                str_val = str(v)
                if len(str_val) > 500:
                    str_val = str_val[:497] + "..."
                try:
                    self.mlflow.log_param(k, str_val)
                except Exception as e:
                    logger.warning(f"Failed to log param {k}: {e}")
            
            logger.info(f"Started MLflow run: {self.run_id}")
            return self.run_id
        except Exception as e:
            logger.error(f"Failed to start MLflow run: {e}")
            self.run = None
            self.run_id = None
            raise
    
    def _flatten_dict(self, d: dict, parent_key: str = '', sep: str = '.') -> dict:
        """Flatten nested dictionary.
        
        Args:
            d: Dictionary to flatten
            parent_key: Parent key prefix
            sep: Separator between keys
        
        Returns:
            Flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log additional parameters.
        
        Args:
            params: Parameters to log
        """
        if self.run is None or self.mlflow.active_run() is None:
            logger.warning("No active MLflow run. Cannot log params.")
            return
        
        flat_params = self._flatten_dict(params)
        for k, v in flat_params.items():
            try:
                self.mlflow.log_param(k, str(v)[:500])
            except Exception as e:
                logger.warning(f"Failed to log param {k}: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics.
        
        Args:
            metrics: Metric name to value mapping
            step: Optional step number
        """
        if self.run is None or self.mlflow.active_run() is None:
            logger.warning("No active MLflow run. Cannot log metrics.")
            return
        
        try:
            self.mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            logger.warning(f"Failed to log metrics to MLflow: {e}")
    
    def log_artifact(self, path: Path, name: Optional[str] = None) -> None:
        """Log a file artifact.
        
        Args:
            path: Path to artifact
            name: Optional subfolder name
        """
        if self.run is None or self.mlflow.active_run() is None:
            logger.warning("No active MLflow run. Cannot log artifact.")
            return
        
        try:
            if name:
                self.mlflow.log_artifact(str(path), name)
            else:
                self.mlflow.log_artifact(str(path))
        except Exception as e:
            logger.warning(f"Failed to log artifact to MLflow: {e}")
    
    def log_model(self, model, artifact_path: str = "model") -> None:
        """Log a PyTorch model.
        
        Args:
            model: PyTorch model
            artifact_path: Path in artifact store
        """
        try:
            self.mlflow.pytorch.log_model(model, artifact_path)
        except Exception as e:
            # Fall back to saving state dict
            logger.warning(f"Failed to log model with pytorch flavor: {e}")
            import torch
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
                torch.save(model.state_dict(), f.name)
                self.mlflow.log_artifact(f.name, artifact_path)
    
    def end_run(self) -> None:
        """End the current run."""
        if self.run is not None:
            try:
                self.mlflow.end_run()
                logger.info(f"Ended MLflow run: {self.run_id}")
            except Exception as e:
                logger.warning(f"Failed to end MLflow run: {e}")
            finally:
                self.run = None
                self.run_id = None
    
    def get_run_id(self) -> Optional[str]:
        """Get current run ID."""
        return self.run_id
