"""Base class and registry for experiment trackers."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path


class BaseTracker(ABC):
    """Abstract base for experiment tracking.
    
    All trackers must implement:
    - start_run: Begin a new experiment run
    - log_params: Log hyperparameters
    - log_metrics: Log metrics (optionally at specific step)
    - log_artifact: Log file artifacts
    - end_run: Finalize the run
    """
    
    @abstractmethod
    def start_run(self, run_name: str, config: Dict[str, Any]) -> str:
        """Start a new run, return run_id.
        
        Args:
            run_name: Name for this run
            config: Configuration dictionary
        
        Returns:
            Unique run identifier
        """
        pass
    
    @abstractmethod
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters.
        
        Args:
            params: Dictionary of parameters
        """
        pass
    
    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics (optionally at a specific step).
        
        Args:
            metrics: Dictionary of metric name to value
            step: Optional step/epoch number
        """
        pass
    
    @abstractmethod
    def log_artifact(self, path: Path, name: Optional[str] = None) -> None:
        """Log a file artifact.
        
        Args:
            path: Path to the artifact file
            name: Optional name for the artifact
        """
        pass
    
    @abstractmethod
    def end_run(self) -> None:
        """End the current run."""
        pass


class TrackerRegistry:
    """Registry for tracker classes."""
    
    _trackers = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register a tracker class.
        
        Args:
            name: Registry name
        
        Returns:
            Decorator function
        """
        def decorator(tracker_class):
            cls._trackers[name] = tracker_class
            return tracker_class
        return decorator
    
    @classmethod
    def get(cls, name: str, **kwargs) -> BaseTracker:
        """Get a tracker instance by name.
        
        Args:
            name: Registry name
            **kwargs: Arguments for tracker constructor
        
        Returns:
            Tracker instance
        """
        if name not in cls._trackers:
            available = list(cls._trackers.keys())
            raise ValueError(f"Unknown tracker: {name}. Available: {available}")
        return cls._trackers[name](**kwargs)
    
    @classmethod
    def list_available(cls) -> list:
        """List all registered tracker names."""
        return list(cls._trackers.keys())
