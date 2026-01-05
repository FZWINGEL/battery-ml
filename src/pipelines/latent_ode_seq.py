"""Latent ODE Sequence Pipeline - Sequences for Neural ODE models."""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler

from .base import BasePipeline
from .sample import Sample
from .registry import PipelineRegistry


@PipelineRegistry.register("latent_ode_seq")
class LatentODESequencePipeline(BasePipeline):
    """Pipeline creating sequences for Neural ODE / Latent ODE models.
    
    Key insight: Use ageing SET index as timeline (not cycles).
    - t = cumulative days or throughput (continuous time)
    - x_t = features at each set (summary + optionally ICA peaks)
    - Provides explicit time deltas for ODE integration
    
    This is more appropriate than cycle-level for ODEs because:
    1. Sets have meaningful time gaps (days/weeks)
    2. RPT measurements provide consistent feature snapshots
    3. Fewer points = faster ODE integration
    
    Example usage:
        >>> pipeline = LatentODESequencePipeline(time_unit="days")
        >>> samples = pipeline.fit_transform({'df': df})
    """
    
    def __init__(self,
                 time_unit: str = "days",  # "days" or "throughput_Ah"
                 include_ica: bool = False,
                 max_seq_len: Optional[int] = None,
                 normalize: bool = True):
        """Initialize the pipeline.
        
        Args:
            time_unit: Time unit for ODE integration ("days" or "throughput_Ah")
            include_ica: Whether to include ICA peak features
            max_seq_len: Maximum sequence length (truncate if longer)
            normalize: Whether to apply StandardScaler
        """
        self.time_unit = time_unit
        self.include_ica = include_ica
        self.max_seq_len = max_seq_len
        self.normalize = normalize
        
        self.scaler: Optional[StandardScaler] = None
        self.feature_names_: List[str] = []
    
    def get_params(self) -> dict:
        """Return pipeline parameters for caching."""
        return {
            'time_unit': self.time_unit,
            'include_ica': self.include_ica,
            'max_seq_len': self.max_seq_len,
        }
    
    def _build_sequence(self, cell_df: pd.DataFrame, cell_id: str, 
                        temp_C: float, experiment_id: int) -> Sample:
        """Build a single sequence sample for one cell.
        
        Args:
            cell_df: DataFrame with data for one cell
            cell_id: Cell identifier
            temp_C: Temperature in Celsius
            experiment_id: Experiment ID
        
        Returns:
            Sample object with sequence data
        """
        # Sort by set index
        cell_df = cell_df.sort_index()
        
        seq_len = len(cell_df)
        if self.max_seq_len and seq_len > self.max_seq_len:
            cell_df = cell_df.iloc[-self.max_seq_len:]
            seq_len = self.max_seq_len
        
        # Build time vector
        if self.time_unit == "throughput_Ah":
            # Use cumulative throughput as time proxy
            throughput_col = None
            for col in cell_df.columns:
                if 'Discharge Throughput' in col:
                    throughput_col = col
                    break
            
            if throughput_col:
                t = cell_df[throughput_col].values.astype(float)
            else:
                # Fallback to index-based time
                t = cell_df.index.values.astype(float)
        else:
            # Approximate days from set index (each set ~7 days for Expt 5)
            t = cell_df.index.values.astype(float) * 7.0
        
        # Normalize time to start at 0
        t = t - t[0]
        
        # Build feature matrix
        feature_cols = [
            'Cumulative Discharge Throughput [A h]',
            '0.1s Resistance [Ohms]',
            '10s Resistance [Ohms]',
        ]
        
        X = []
        for col in feature_cols:
            if col in cell_df.columns:
                values = cell_df[col].values.astype(float)
                values = np.nan_to_num(values, nan=0.0)
                X.append(values)
            else:
                X.append(np.zeros(seq_len))
        
        # Add temperature (constant for each cell, but include for multi-temp training)
        temp_K = temp_C + 273.15
        X.append(np.full(seq_len, temp_K))
        
        # Arrhenius factor
        R = 8.314
        Ea = 50000.0
        arrhenius = np.exp(-Ea / (R * temp_K))
        X.append(np.full(seq_len, arrhenius))
        
        X = np.stack(X, axis=1)  # (seq_len, num_features)
        
        # Target: final SOH (normalized by first capacity value)
        target_col = None
        for col in ['Cell Capacity [mA h]', 'Cell Capacity [A h]', 'SoH']:
            if col in cell_df.columns:
                target_col = col
                break
        
        if target_col:
            # Get capacity values
            capacity_values = cell_df[target_col].values.astype(float)
            capacity_values = np.nan_to_num(capacity_values, nan=0.0)
            
            # Convert mAh to Ah if needed
            if capacity_values[0] > 100:  # Assume mAh if first value > 100
                capacity_values = capacity_values / 1000.0
            
            # Get first capacity value as nominal/initial capacity
            initial_capacity = capacity_values[0]
            
            if initial_capacity > 0:
                # Calculate SOH: current capacity / initial capacity
                # Use full sequence for trajectory matching
                y_seq = capacity_values / initial_capacity
            else:
                y_seq = np.ones(seq_len)
        else:
            y_seq = np.ones(seq_len)
        
        # Ensure y has correct length (handle truncation)
        if len(y_seq) > seq_len:
            y_seq = y_seq[-seq_len:]
        elif len(y_seq) < seq_len:
            y_seq = np.pad(y_seq, (seq_len - len(y_seq), 0), mode='edge')
        
        sample = Sample(
            meta={
                'experiment_id': experiment_id,
                'cell_id': cell_id,
                'temperature_C': temp_C,
                'seq_len': seq_len,
            },
            x=X.astype(np.float32),
            y=y_seq.astype(np.float32).reshape(-1, 1),  # (seq_len, 1)
            t=t.astype(np.float32),  # CRITICAL for ODE
        )
        
        return sample
    
    def fit(self, data: Dict[str, Any]) -> 'LatentODESequencePipeline':
        """Fit scaler on training sequences.
        
        Args:
            data: Dictionary with 'df' key containing DataFrame
        
        Returns:
            self
        """
        df = data['df']
        
        # Build feature names
        self.feature_names_ = [
            'throughput_Ah', 'R_0.1s', 'R_10s', 'temp_K', 'arrhenius'
        ]
        
        # Collect all feature vectors for scaling
        all_X = []
        for cell_id in df['cell_id'].unique():
            cell_df = df[df['cell_id'] == cell_id]
            sample = self._build_sequence(
                cell_df, cell_id, 
                cell_df['temperature_C'].iloc[0],
                cell_df['experiment_id'].iloc[0]
            )
            all_X.append(sample.x)
        
        if all_X:
            all_X = np.vstack(all_X)
            
            if self.normalize:
                self.scaler = StandardScaler()
                self.scaler.fit(all_X)
        
        return self
    
    def transform(self, data: Dict[str, Any]) -> List[Sample]:
        """Transform data into sequence samples (one per cell).
        
        Args:
            data: Dictionary with 'df' key containing DataFrame
        
        Returns:
            List of Sample objects (one per cell)
        """
        df = data['df']
        samples = []
        
        for cell_id in df['cell_id'].unique():
            cell_df = df[df['cell_id'] == cell_id]
            sample = self._build_sequence(
                cell_df, cell_id,
                cell_df['temperature_C'].iloc[0],
                cell_df['experiment_id'].iloc[0]
            )
            
            if self.normalize and self.scaler:
                # Normalize each timestep
                sample.x = self.scaler.transform(sample.x).astype(np.float32)
            
            samples.append(sample.to_tensor())
        
        return samples
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        return self.feature_names_
