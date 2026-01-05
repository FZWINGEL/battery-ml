"""Local file-based tracking (JSON + TensorBoard)."""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from .base import BaseTracker, TrackerRegistry

logger = logging.getLogger(__name__)


@TrackerRegistry.register("local")
class LocalTracker(BaseTracker):
    """Local file-based tracking (JSON + TensorBoard).
    
    Outputs to: artifacts/runs/<run_id>/
    - config.json
    - metrics.json
    - step_metrics.csv
    - tensorboard/
    - artifacts/
    
    Example usage:
        >>> tracker = LocalTracker()
        >>> run_id = tracker.start_run("experiment_1", {"lr": 0.001})
        >>> tracker.log_metrics({"loss": 0.5}, step=1)
        >>> tracker.end_run()
    """
    
    def __init__(self, 
                 base_dir: str = "artifacts/runs",
                 use_tensorboard: bool = True):
        """Initialize the tracker.
        
        Args:
            base_dir: Base directory for run outputs
            use_tensorboard: Whether to log to TensorBoard
        """
        self.base_dir = Path(base_dir)
        self.use_tensorboard = use_tensorboard
        
        self.run_dir: Optional[Path] = None
        self.run_id: Optional[str] = None
        self.config: Dict[str, Any] = {}
        self.metrics: Dict[str, Any] = {}
        self.step_metrics: list = []
        
        self.tb_writer = None
    
    def start_run(self, run_name: str, config: Dict[str, Any]) -> str:
        """Start a new run.
        
        Args:
            run_name: Name for this run
            config: Configuration dictionary
        
        Returns:
            Run ID
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{run_name}_{timestamp}"
        self.run_dir = self.base_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = config
        self.metrics = {}
        self.step_metrics = []
        
        # Save config
        with open(self.run_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        # Initialize TensorBoard
        if self.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(self.run_dir / "tensorboard")
                logger.info(f"TensorBoard logging to: {self.run_dir / 'tensorboard'}")
            except ImportError:
                logger.warning("TensorBoard not available")
                self.tb_writer = None
        
        logger.info(f"Started run: {self.run_id}")
        return self.run_id
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log additional parameters.
        
        Args:
            params: Parameters to log
        """
        self.config.update(params)
        with open(self.run_dir / "config.json", 'w') as f:
            json.dump(self.config, f, indent=2, default=str)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics.
        
        Args:
            metrics: Metric name to value mapping
            step: Optional step number
        """
        # Update final metrics
        self.metrics.update(metrics)
        
        # Save to file
        with open(self.run_dir / "metrics.json", 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        # Log to TensorBoard
        if self.tb_writer and step is not None:
            for k, v in metrics.items():
                self.tb_writer.add_scalar(k, v, step)
        
        # Append to step history
        if step is not None:
            self.step_metrics.append({'step': step, **metrics})
    
    def log_artifact(self, path: Path, name: Optional[str] = None) -> None:
        """Log a file artifact.
        
        Args:
            path: Path to artifact
            name: Optional name
        """
        artifacts_dir = self.run_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        
        dest_name = name or Path(path).name
        shutil.copy(path, artifacts_dir / dest_name)
        logger.debug(f"Logged artifact: {dest_name}")
    
    def log_figure(self, figure, name: str) -> None:
        """Log a matplotlib figure.
        
        Args:
            figure: Matplotlib figure
            name: Figure name
        """
        figures_dir = self.run_dir / "figures"
        figures_dir.mkdir(exist_ok=True)
        
        figure.savefig(figures_dir / f"{name}.png", dpi=150, bbox_inches='tight')
        
        if self.tb_writer:
            self.tb_writer.add_figure(name, figure)
    
    def end_run(self) -> None:
        """End the current run."""
        # Save step metrics as CSV
        if self.step_metrics:
            try:
                import pandas as pd
                df = pd.DataFrame(self.step_metrics)
                df.to_csv(self.run_dir / "step_metrics.csv", index=False)
            except ImportError:
                # Fallback to JSON
                with open(self.run_dir / "step_metrics.json", 'w') as f:
                    json.dump(self.step_metrics, f, indent=2)
        
        if self.tb_writer:
            self.tb_writer.close()
        
        logger.info(f"Run saved to: {self.run_dir}")
    
    def get_run_dir(self) -> Optional[Path]:
        """Get current run directory."""
        return self.run_dir
