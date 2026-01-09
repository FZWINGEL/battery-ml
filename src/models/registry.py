"""Registry for models."""

from typing import Dict, Type, Optional, Union
from .base import BaseModel


class ModelRegistry:
    """Registry for model classes.

    Example usage:
        >>> @ModelRegistry.register("mlp")
        ... class MLPModel(BaseModel):
        ...     pass

        >>> model = ModelRegistry.get("mlp", input_dim=10, hidden_dims=[64, 32])
    """

    _models: Dict[str, Type] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a model class.

        Args:
            name: Registry name for the model

        Returns:
            Decorator function
        """

        def decorator(model_class):
            cls._models[name] = model_class
            model_class.name = name
            return model_class

        return decorator

    @classmethod
    def get(cls, name: str, **kwargs) -> Union[BaseModel, object]:
        """Get a model instance by name.

        Args:
            name: Registry name of the model
            **kwargs: Arguments to pass to model constructor

        Returns:
            Model instance

        Raises:
            ValueError: If model name is not found
        """
        if name not in cls._models:
            available = list(cls._models.keys())
            raise ValueError(f"Unknown model: {name}. Available: {available}")
        return cls._models[name](**kwargs)

    @classmethod
    def list_available(cls) -> list:
        """List all registered model names.

        Returns:
            List of model names
        """
        return list(cls._models.keys())

    @classmethod
    def get_class(cls, name: str) -> Optional[Type]:
        """Get model class by name (without instantiating).

        Args:
            name: Registry name of the model

        Returns:
            Model class or None if not found
        """
        return cls._models.get(name)
