"""Dual tracker that logs to both local and MLflow simultaneously."""

from pathlib import Path
from typing import Dict, Any, Optional
import logging

from .base import BaseTracker, TrackerRegistry
from .local import LocalTracker
from .mlflow_tracker import MLflowTracker

logger = logging.getLogger(__name__)


@TrackerRegistry.register("dual")
class DualTracker(BaseTracker):
    """Tracker that logs to both local files and MLflow.

    Provides the best of both worlds:
    - Local: Fast access, TensorBoard, works offline
    - MLflow: Centralized UI, model registry, comparison

    Example usage:
        >>> tracker = DualTracker()
        >>> run_id = tracker.start_run("experiment_1", {"lr": 0.001})
        >>> tracker.log_metrics({"loss": 0.5}, step=1)
        >>> tracker.end_run()
    """

    def __init__(
        self,
        local_base_dir: str = "artifacts/runs",
        use_tensorboard: bool = True,
        mlflow_tracking_uri: str = "file:./artifacts/mlruns",
        mlflow_experiment_name: str = "battery_degradation",
    ):
        """Initialize dual tracker.

        Args:
            local_base_dir: Base directory for local runs
            use_tensorboard: Whether to enable TensorBoard
            mlflow_tracking_uri: MLflow tracking URI
            mlflow_experiment_name: MLflow experiment name
        """
        self.local = LocalTracker(
            base_dir=local_base_dir, use_tensorboard=use_tensorboard
        )

        try:
            self.mlflow = MLflowTracker(
                tracking_uri=mlflow_tracking_uri, experiment_name=mlflow_experiment_name
            )
            self._has_mlflow = True
        except ImportError:
            logger.warning("MLflow not available, using local tracking only")
            self.mlflow = None
            self._has_mlflow = False

        self.run_id: Optional[str] = None

    def start_run(self, run_name: str, config: Dict[str, Any]) -> str:
        """Start a run on both backends.

        Args:
            run_name: Name for this run
            config: Configuration dictionary

        Returns:
            Local run ID
        """
        # Start local run first
        self.run_id = self.local.start_run(run_name, config)

        # Start MLflow run
        if self._has_mlflow:
            try:
                mlflow_run_id = self.mlflow.start_run(run_name, config)
                logger.info(
                    f"Dual tracking: local={self.run_id}, mlflow={mlflow_run_id}"
                )
            except Exception as e:
                logger.warning(f"Failed to start MLflow run: {e}")

        return self.run_id

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to both backends.

        Args:
            params: Parameters to log
        """
        self.local.log_params(params)

        if self._has_mlflow:
            try:
                self.mlflow.log_params(params)
            except Exception as e:
                logger.warning(f"Failed to log params to MLflow: {e}")

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """Log metrics to both backends.

        Args:
            metrics: Metric name to value mapping
            step: Optional step number
        """
        self.local.log_metrics(metrics, step)

        if self._has_mlflow:
            try:
                self.mlflow.log_metrics(metrics, step)
            except Exception as e:
                logger.warning(f"Failed to log metrics to MLflow: {e}")

    def log_artifact(self, path: Path, name: Optional[str] = None) -> None:
        """Log artifact to both backends.

        Args:
            path: Path to artifact
            name: Optional name
        """
        self.local.log_artifact(path, name)

        if self._has_mlflow:
            try:
                self.mlflow.log_artifact(path, name)
            except Exception as e:
                logger.warning(f"Failed to log artifact to MLflow: {e}")

    def log_figure(self, figure, name: str) -> None:
        """Log matplotlib figure.

        Args:
            figure: Matplotlib figure
            name: Figure name
        """
        self.local.log_figure(figure, name)

    def log_model(self, model, artifact_path: str = "model") -> None:
        """Log PyTorch model to MLflow.

        Args:
            model: PyTorch model
            artifact_path: Artifact path
        """
        if self._has_mlflow:
            try:
                self.mlflow.log_model(model, artifact_path)
            except Exception as e:
                logger.warning(f"Failed to log model to MLflow: {e}")

    def end_run(self) -> None:
        """End run on both backends."""
        self.local.end_run()

        if self._has_mlflow:
            try:
                self.mlflow.end_run()
            except Exception as e:
                logger.warning(f"Failed to end MLflow run: {e}")

    def get_run_dir(self) -> Optional[Path]:
        """Get local run directory."""
        return self.local.get_run_dir()
