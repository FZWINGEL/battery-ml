"""Base class for preprocessing pipelines."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from .sample import Sample


class BasePipeline(ABC):
    """Abstract base class for all preprocessing pipelines.
    
    All pipelines must:
    1. Accept raw data (DataFrames, arrays)
    2. Output Sample objects with consistent structure
    3. Support fit/transform pattern for scalers
    4. Be cacheable (expensive computations)
    
    Example usage:
        >>> pipeline = SummarySetPipeline(include_arrhenius=True)
        >>> train_samples = pipeline.fit_transform({'df': train_df})
        >>> test_samples = pipeline.transform({'df': test_df})
    """
    
    name: str = "base"
    
    @abstractmethod
    def fit(self, data: Dict[str, Any]) -> 'BasePipeline':
        """Fit any scalers/normalizers on training data.
        
        Args:
            data: Dictionary containing raw data (e.g., {'df': DataFrame})
        
        Returns:
            self: Fitted pipeline instance
        """
        pass
    
    @abstractmethod
    def transform(self, data: Dict[str, Any]) -> List[Sample]:
        """Transform raw data into Sample objects.
        
        Args:
            data: Dictionary containing raw data
        
        Returns:
            List of Sample objects
        """
        pass
    
    def fit_transform(self, data: Dict[str, Any]) -> List[Sample]:
        """Fit and transform in one call.
        
        Args:
            data: Dictionary containing raw data
        
        Returns:
            List of Sample objects
        """
        return self.fit(data).transform(data)
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Return list of feature names (for SHAP/interpretability).
        
        Returns:
            List of feature name strings
        """
        pass
    
    def get_params(self) -> dict:
        """Return pipeline parameters (for caching key).
        
        Returns:
            Dictionary of pipeline parameters
        """
        return {}
    
    def __repr__(self) -> str:
        params = self.get_params()
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        return f"{self.__class__.__name__}({param_str})"
