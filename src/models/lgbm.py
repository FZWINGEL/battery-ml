"""LightGBM wrapper compatible with pipeline interface."""

import numpy as np
import pandas as pd
import re
from typing import Optional, Dict, Any, List
import logging

from .registry import ModelRegistry

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    logger.warning("LightGBM not installed. Install with: pip install lightgbm")


@ModelRegistry.register("lgbm")
class LGBMModel:
    """LightGBM wrapper compatible with pipeline.
    
    Not a nn.Module (sklearn-style interface), but follows same pattern.
    Excellent for:
    - Fast baselines
    - SHAP interpretability
    - Summary + ICA peak features
    
    Example usage:
        >>> model = LGBMModel(n_estimators=500)
        >>> model.fit(X_train, y_train, X_val, y_val)
        >>> predictions = model.predict(X_test)
    """
    
    name = "lgbm"
    
    def __init__(self,
                 input_dim: int = None,  # Unused, for interface compatibility
                 output_dim: int = 1,
                 n_estimators: int = 1000,
                 learning_rate: float = 0.05,
                 max_depth: int = 6,
                 num_leaves: int = 31,
                 reg_alpha: float = 0.1,
                 reg_lambda: float = 0.1,
                 early_stopping_rounds: int = 50,
                 random_state: int = 42):
        """Initialize the model.
        
        Args:
            input_dim: Input dimension (unused, for interface compatibility)
            output_dim: Output dimension
            n_estimators: Number of boosting rounds
            learning_rate: Learning rate
            max_depth: Maximum tree depth
            num_leaves: Maximum number of leaves
            reg_alpha: L1 regularization
            reg_lambda: L2 regularization
            early_stopping_rounds: Early stopping patience
            random_state: Random seed
        """
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM required. Install with: pip install lightgbm")
        
        self.params = {
            'objective': 'regression',
            'metric': 'rmse',
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'num_leaves': num_leaves,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'verbosity': -1,
            'random_state': random_state,
        }
        self.early_stopping_rounds = early_stopping_rounds
        self.model: Optional[lgb.LGBMRegressor] = None
        self.feature_names_: List[str] = []
        self._use_feature_names: bool = False  # Track if feature names were explicitly provided
        self.input_dim = input_dim
        self.output_dim = output_dim
    
    @staticmethod
    def _sanitize_feature_names(feature_names: List[str]) -> List[str]:
        """Sanitize feature names for LightGBM compatibility.
        
        LightGBM doesn't support special JSON characters like brackets, spaces, etc.
        Replaces them with underscores.
        
        Args:
            feature_names: Original feature names
        
        Returns:
            Sanitized feature names
        """
        sanitized = []
        for name in feature_names:
            # Replace brackets, spaces, and other special characters with underscores
            sanitized_name = re.sub(r'[\[\](){}<>,\s]+', '_', name)
            # Remove multiple consecutive underscores
            sanitized_name = re.sub(r'_+', '_', sanitized_name)
            # Remove leading/trailing underscores
            sanitized_name = sanitized_name.strip('_')
            sanitized.append(sanitized_name)
        return sanitized
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            feature_names: Optional[List[str]] = None) -> 'LGBMModel':
        """Fit the model.
        
        Args:
            X: Training features
            y: Training targets
            X_val: Validation features (for early stopping)
            y_val: Validation targets
            feature_names: Feature names for interpretability
        
        Returns:
            self
        """
        # Store original feature names for interpretability
        original_feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]
        # Sanitize feature names for LightGBM compatibility
        self.feature_names_ = self._sanitize_feature_names(original_feature_names)
        self._use_feature_names = feature_names is not None
        
        # Convert to pandas DataFrames if feature names are provided to avoid warnings
        if self._use_feature_names:
            X_df = pd.DataFrame(X, columns=self.feature_names_)
            X_val_df = pd.DataFrame(X_val, columns=self.feature_names_) if X_val is not None else None
        else:
            X_df = X
            X_val_df = X_val
        
        self.model = lgb.LGBMRegressor(**self.params)
        
        eval_set = [(X_val_df, y_val.ravel())] if X_val is not None else None
        callbacks = [lgb.early_stopping(self.early_stopping_rounds)] if eval_set else None
        
        self.model.fit(
            X_df, y.ravel(),
            eval_set=eval_set,
            callbacks=callbacks,
        )
        
        logger.info(f"Trained LGBM with {self.model.n_estimators_} estimators")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict.
        
        Args:
            X: Input features
        
        Returns:
            Predictions shaped (n_samples, 1)
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        # Use DataFrame if feature names were explicitly provided during fit
        if self._use_feature_names:
            X_df = pd.DataFrame(X, columns=self.feature_names_)
            return self.model.predict(X_df).reshape(-1, 1)
        else:
            return self.model.predict(X).reshape(-1, 1)
    
    def explain(self, X: np.ndarray) -> Dict[str, Any]:
        """Return SHAP values for interpretability.
        
        Args:
            X: Input features
        
        Returns:
            Dictionary with SHAP values, expected value, feature names
        """
        try:
            import shap
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X)
            return {
                'shap_values': shap_values,
                'expected_value': explainer.expected_value,
                'feature_names': self.feature_names_,
            }
        except ImportError:
            return {'error': 'shap not installed'}
        except Exception as e:
            return {'error': str(e)}
    
    @property
    def feature_importances_(self) -> np.ndarray:
        """Get feature importances."""
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        return self.model.feature_importances_
    
    def save(self, path: str) -> None:
        """Save model to file.
        
        Args:
            path: Path to save model
        """
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        self.model.booster_.save_model(path)
    
    def load(self, path: str) -> 'LGBMModel':
        """Load model from file.
        
        Args:
            path: Path to load model from
        
        Returns:
            self
        """
        self.model = lgb.LGBMRegressor(**self.params)
        self.model.booster_ = lgb.Booster(model_file=path)
        return self
    
    def __repr__(self) -> str:
        return f"LGBMModel(n_estimators={self.params['n_estimators']}, lr={self.params['learning_rate']})"
