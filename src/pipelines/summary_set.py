"""Summary Set Pipeline - Features from Performance Summary data."""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler

from .base import BasePipeline
from .sample import Sample
from .registry import PipelineRegistry


@PipelineRegistry.register("summary_set")
class SummarySetPipeline(BasePipeline):
    """Pipeline using Performance Summary / Summary per Set data.
    
    Features per set:
    - Cumulative throughput (Ah)
    - Average temperature (K, plus Arrhenius transform)
    - Resistance measurements
    - Cycle count
    - Previous SOH (if available)
    
    Target: SOH or Cell Capacity
    
    Example usage:
        >>> pipeline = SummarySetPipeline(include_arrhenius=True)
        >>> samples = pipeline.fit_transform({'df': df})
    """
    
    FEATURE_COLS = [
        'Cumulative Charge Throughput [A h]',
        'Cumulative Discharge Throughput [A h]',
        '0.1s Resistance [Ohms]',
        '10s Resistance [Ohms]',
    ]
    
    TARGET_COL = 'Cell Capacity [mA h]'  # Will be normalized to Ah
    
    def __init__(self, 
                 include_arrhenius: bool = True,
                 arrhenius_Ea: float = 50000.0,
                 normalize: bool = True):
        """Initialize the pipeline.
        
        Args:
            include_arrhenius: Include Arrhenius temperature features
            arrhenius_Ea: Activation energy for Arrhenius (J/mol)
            normalize: Whether to apply StandardScaler normalization
        """
        self.include_arrhenius = include_arrhenius
        self.arrhenius_Ea = arrhenius_Ea
        self.normalize = normalize
        self.scaler: Optional[StandardScaler] = None
        self.feature_names_: List[str] = []
        
    def get_params(self) -> dict:
        """Return pipeline parameters for caching."""
        return {
            'include_arrhenius': self.include_arrhenius,
            'arrhenius_Ea': self.arrhenius_Ea,
            'normalize': self.normalize,
        }
    
    def _extract_features(self, row: pd.Series, temp_C: float) -> np.ndarray:
        """Extract feature vector from a single row.
        
        Args:
            row: DataFrame row with feature columns
            temp_C: Temperature in Celsius
        
        Returns:
            Feature vector as numpy array
        """
        features = []
        
        # Base features
        for col in self.FEATURE_COLS:
            if col in row.index:
                val = row[col]
                # Handle NaN
                if pd.isna(val):
                    val = 0.0
                features.append(float(val))
            else:
                features.append(0.0)
        
        # Temperature features
        temp_K = temp_C + 273.15
        features.append(temp_K)
        
        if self.include_arrhenius:
            # Arrhenius factor: exp(-Ea / RT)
            R = 8.314  # J/(mol·K)
            arrhenius = np.exp(-self.arrhenius_Ea / (R * temp_K))
            features.append(arrhenius)
            
            # Inverse temperature (useful for linearizing)
            features.append(1000.0 / temp_K)
        
        return np.array(features, dtype=np.float32)
    
    def fit(self, data: Dict[str, Any]) -> 'SummarySetPipeline':
        """Fit scaler on training data.
        
        Args:
            data: Dictionary with 'df' key containing DataFrame
        
        Returns:
            self
        """
        df = data['df']
        
        # Build feature names
        self.feature_names_ = list(self.FEATURE_COLS) + ['temp_K']
        if self.include_arrhenius:
            self.feature_names_.extend(['arrhenius', 'inv_temp'])
        
        # Extract all features for scaling
        X = []
        for idx, row in df.iterrows():
            temp_C = row.get('temperature_C', 25)
            X.append(self._extract_features(row, temp_C))
        X = np.vstack(X)
        
        if self.normalize:
            self.scaler = StandardScaler()
            self.scaler.fit(X)
        
        return self
    
    def transform(self, data: Dict[str, Any]) -> List[Sample]:
        """Transform data into Sample objects.
        
        Args:
            data: Dictionary with 'df' key containing DataFrame
        
        Returns:
            List of Sample objects
        """
        df = data['df']
        samples = []
        
        # Pre-compute initial capacity for each cell (first capacity value)
        initial_capacities = {}
        for cell_id in df['cell_id'].unique():
            cell_df = df[df['cell_id'] == cell_id].sort_index()
            target_col = self.TARGET_COL if self.TARGET_COL in cell_df.columns else 'SoH'
            
            if target_col in cell_df.columns:
                # Get first capacity value
                first_capacity = cell_df[target_col].iloc[0]
                if pd.isna(first_capacity):
                    initial_capacities[cell_id] = None
                else:
                    # Convert mAh to Ah if needed
                    if first_capacity > 100:
                        first_capacity = first_capacity / 1000.0
                    initial_capacities[cell_id] = first_capacity
            else:
                initial_capacities[cell_id] = None
        
        for idx, row in df.iterrows():
            temp_C = row.get('temperature_C', 25)
            
            # Extract features
            x = self._extract_features(row, temp_C)
            if self.normalize and self.scaler:
                x = self.scaler.transform(x.reshape(1, -1)).flatten()
            
            # Extract target and normalize to SOH
            cell_id = str(row.get('cell_id', 'unknown'))
            y_raw = row.get(self.TARGET_COL, row.get('SoH', 1.0))
            
            if pd.isna(y_raw):
                y = 1.0
            else:
                # Convert mAh to Ah if needed
                if y_raw > 100:
                    y_raw = y_raw / 1000.0
                
                # Calculate SOH: current capacity / initial capacity
                initial_capacity = initial_capacities.get(cell_id)
                if initial_capacity and initial_capacity > 0:
                    y = y_raw / initial_capacity
                else:
                    y = 1.0
            
            # Build sample
            sample = Sample(
                meta={
                    'experiment_id': int(row.get('experiment_id', 5)),
                    'cell_id': cell_id,
                    'temperature_C': float(temp_C),
                    'set_idx': int(idx) if isinstance(idx, (int, np.integer)) else int(row.get('set_idx', 0)),
                },
                x=x,
                y=np.array([y], dtype=np.float32),
            )
            samples.append(sample)
        
        return samples
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        return self.feature_names_
