"""Training callbacks for monitoring and checkpointing."""

from pathlib import Path
from typing import Dict, Any, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class Callback:
    """Base callback class."""

    def on_epoch_start(self, epoch: int, logs: Dict[str, Any] = None) -> None:
        """Called at start of each epoch."""
        pass

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None) -> None:
        """Called at end of each epoch."""
        pass

    def on_train_start(self, logs: Dict[str, Any] = None) -> None:
        """Called at start of training."""
        pass

    def on_train_end(self, logs: Dict[str, Any] = None) -> None:
        """Called at end of training."""
        pass


class EarlyStoppingCallback(Callback):
    """Stop training when a monitored metric has stopped improving."""

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 20,
        min_delta: float = 0.0,
        mode: str = "min",
    ):
        """Initialize early stopping.

        Args:
            monitor: Metric to monitor
            patience: Number of epochs to wait
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max'
        """
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.best = float("inf") if mode == "min" else float("-inf")
        self.counter = 0
        self.should_stop = False

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None) -> None:
        """Check if training should stop."""
        if logs is None or self.monitor not in logs:
            return

        current = logs[self.monitor]

        if self.mode == "min":
            improved = current < self.best - self.min_delta
        else:
            improved = current > self.best + self.min_delta

        if improved:
            self.best = current
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(f"Early stopping triggered at epoch {epoch}")


class ModelCheckpointCallback(Callback):
    """Save model checkpoints based on monitored metric."""

    def __init__(
        self,
        save_path: Path,
        monitor: str = "val_loss",
        mode: str = "min",
        save_best_only: bool = True,
        save_fn: Optional[Callable] = None,
    ):
        """Initialize checkpointing.

        Args:
            save_path: Path to save checkpoint
            monitor: Metric to monitor
            mode: 'min' or 'max'
            save_best_only: Only save when metric improves
            save_fn: Custom save function
        """
        self.save_path = Path(save_path)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_fn = save_fn

        self.best = float("inf") if mode == "min" else float("-inf")

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None) -> None:
        """Save checkpoint if metric improved."""
        if logs is None or self.monitor not in logs:
            return

        current = logs[self.monitor]

        if self.mode == "min":
            improved = current < self.best
        else:
            improved = current > self.best

        if improved or not self.save_best_only:
            self.best = current
            if self.save_fn:
                self.save_fn(self.save_path)
            logger.info(f"Checkpoint saved: {self.monitor}={current:.5f}")


class LRSchedulerCallback(Callback):
    """Learning rate scheduler callback."""

    def __init__(self, scheduler, step_on: str = "epoch"):
        """Initialize scheduler callback.

        Args:
            scheduler: PyTorch scheduler
            step_on: When to step ('epoch' or 'batch')
        """
        self.scheduler = scheduler
        self.step_on = step_on

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None) -> None:
        """Step the scheduler."""
        if self.step_on == "epoch":
            self.scheduler.step()


class ProgressCallback(Callback):
    """Print training progress."""

    def __init__(self, print_every: int = 10):
        """Initialize progress callback.

        Args:
            print_every: Print every N epochs
        """
        self.print_every = print_every

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None) -> None:
        """Print progress."""
        if (epoch + 1) % self.print_every == 0 and logs:
            metrics_str = ", ".join(f"{k}: {v:.5f}" for k, v in logs.items())
            logger.info(f"Epoch {epoch + 1}: {metrics_str}")
