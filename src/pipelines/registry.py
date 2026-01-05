"""Registry for preprocessing pipelines."""

from typing import Dict, Type, Optional
from .base import BasePipeline


class PipelineRegistry:
    """Registry for pipeline classes.
    
    Example usage:
        >>> @PipelineRegistry.register("summary_set")
        ... class SummarySetPipeline(BasePipeline):
        ...     pass
        
        >>> pipeline = PipelineRegistry.get("summary_set", normalize=True)
    """
    
    _pipelines: Dict[str, Type[BasePipeline]] = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register a pipeline class.
        
        Args:
            name: Registry name for the pipeline
        
        Returns:
            Decorator function
        """
        def decorator(pipeline_class: Type[BasePipeline]):
            cls._pipelines[name] = pipeline_class
            pipeline_class.name = name
            return pipeline_class
        return decorator
    
    @classmethod
    def get(cls, name: str, **kwargs) -> BasePipeline:
        """Get a pipeline instance by name.
        
        Args:
            name: Registry name of the pipeline
            **kwargs: Arguments to pass to pipeline constructor
        
        Returns:
            Pipeline instance
        
        Raises:
            ValueError: If pipeline name is not found
        """
        if name not in cls._pipelines:
            available = list(cls._pipelines.keys())
            raise ValueError(f"Unknown pipeline: {name}. Available: {available}")
        return cls._pipelines[name](**kwargs)
    
    @classmethod
    def list_available(cls) -> list:
        """List all registered pipeline names.
        
        Returns:
            List of pipeline names
        """
        return list(cls._pipelines.keys())
    
    @classmethod
    def get_class(cls, name: str) -> Optional[Type[BasePipeline]]:
        """Get pipeline class by name (without instantiating).
        
        Args:
            name: Registry name of the pipeline
        
        Returns:
            Pipeline class or None if not found
        """
        return cls._pipelines.get(name)
