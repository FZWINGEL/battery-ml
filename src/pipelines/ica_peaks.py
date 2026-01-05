"""ICA Peaks Pipeline - Extract dQ/dV features from voltage curves."""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from scipy.signal import savgol_filter, find_peaks, peak_widths
from sklearn.preprocessing import StandardScaler

from .base import BasePipeline
from .sample import Sample
from .registry import PipelineRegistry
from .cache import get_cache


@PipelineRegistry.register("ica_peaks")
class ICAPeaksPipeline(BasePipeline):
    """Pipeline extracting ICA (dQ/dV) peak features from voltage curves.
    
    For each RPT:
    1. Load 0.1C discharge curve
    2. Compute dQ/dV with Savitzky-Golay smoothing
    3. Extract peak positions, heights, widths, areas
    4. Output fixed-length feature vector
    
    These features are highly diagnostic for degradation mechanisms:
    - Peak shifts → Loss of Lithium Inventory (LLI)
    - Peak height changes → Loss of Active Material (LAM)
    - Peak width changes → Kinetic degradation / impedance rise
    
    Example usage:
        >>> pipeline = ICAPeaksPipeline(sg_window=51, num_peaks=3)
        >>> samples = pipeline.fit_transform({'curves': curves, 'targets': targets})
    """
    
    def __init__(self,
                 sg_window: int = 51,
                 sg_order: int = 3,
                 num_peaks: int = 3,
                 voltage_range: Tuple[float, float] = (3.0, 4.2),
                 resample_points: int = 500,
                 normalize: bool = True,
                 use_cache: bool = True):
        """Initialize the pipeline.
        
        Args:
            sg_window: Savitzky-Golay filter window size (must be odd)
            sg_order: Savitzky-Golay polynomial order
            num_peaks: Number of peaks to extract features for
            voltage_range: Voltage range for ICA analysis
            resample_points: Number of points to resample curves to
            normalize: Whether to apply StandardScaler
            use_cache: Whether to cache computed features
        """
        self.sg_window = sg_window
        self.sg_order = sg_order
        self.num_peaks = num_peaks
        self.voltage_range = voltage_range
        self.resample_points = resample_points
        self.normalize = normalize
        self.use_cache = use_cache
        
        self.scaler: Optional[StandardScaler] = None
        self.feature_names_: List[str] = []
        self._build_feature_names()
    
    def _build_feature_names(self):
        """Build list of feature names."""
        self.feature_names_ = []
        for i in range(self.num_peaks):
            self.feature_names_.extend([
                f'peak{i+1}_voltage',
                f'peak{i+1}_height',
                f'peak{i+1}_width',
                f'peak{i+1}_area',
            ])
        self.feature_names_.extend([
            'total_area',
            'num_peaks_detected',
            'voltage_at_max_dqdv',
        ])
    
    def get_params(self) -> dict:
        """Return pipeline parameters for caching."""
        return {
            'sg_window': self.sg_window,
            'sg_order': self.sg_order,
            'num_peaks': self.num_peaks,
            'voltage_range': self.voltage_range,
            'resample_points': self.resample_points,
        }
    
    def compute_ica(self, voltage: np.ndarray, capacity: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute ICA curve (dQ/dV) with smoothing.
        
        Args:
            voltage: Voltage array
            capacity: Capacity array
        
        Returns:
            Tuple of (voltage_midpoints, dQ/dV values)
        """
        # Sort by voltage
        sorted_idx = np.argsort(voltage)
        V = voltage[sorted_idx]
        Q = capacity[sorted_idx]
        
        # Filter to voltage range
        mask = (V >= self.voltage_range[0]) & (V <= self.voltage_range[1])
        V, Q = V[mask], Q[mask]
        
        if len(V) < self.sg_window:
            return np.array([]), np.array([])
        
        # Smooth capacity
        Q_smooth = savgol_filter(Q, self.sg_window, self.sg_order)
        
        # Compute derivative
        dQ = np.diff(Q_smooth)
        dV = np.diff(V)
        dV[dV == 0] = 1e-10
        dQdV = dQ / dV
        V_mid = (V[:-1] + V[1:]) / 2
        
        # Additional smoothing on dQ/dV
        if len(dQdV) > self.sg_window:
            dQdV = savgol_filter(dQdV, self.sg_window, self.sg_order)
        
        # Take absolute value as dQ/dV is negative during discharge
        return V_mid, np.abs(dQdV)
    
    def extract_peak_features(self, V: np.ndarray, dQdV: np.ndarray) -> np.ndarray:
        """Extract fixed-length feature vector from ICA curve.
        
        Args:
            V: Voltage midpoints
            dQdV: dQ/dV values
        
        Returns:
            Feature vector
        """
        features = []
        
        if len(V) == 0 or len(dQdV) == 0:
            # Return zeros if curve is invalid
            return np.zeros(len(self.feature_names_), dtype=np.float32)
        
        # Find peaks
        peaks, properties = find_peaks(
            dQdV, 
            height=0.01,
            distance=200,
            prominence=0.01
        )
        
        # Get peak widths
        if len(peaks) > 0:
            try:
                widths, width_heights, left_ips, right_ips = peak_widths(dQdV, peaks, rel_height=0.5)
            except Exception:
                widths = np.zeros(len(peaks))
        else:
            widths = np.array([])
        
        # Sort peaks by height (descending)
        if len(peaks) > 0:
            sorted_idx = np.argsort(dQdV[peaks])[::-1]
            peaks = peaks[sorted_idx]
            if len(widths) == len(peaks):
                widths = widths[sorted_idx]
            else:
                widths = np.zeros(len(peaks))
        
        # Extract features for top N peaks
        dV = V[1] - V[0] if len(V) > 1 else 0.001
        for i in range(self.num_peaks):
            if i < len(peaks):
                peak_idx = peaks[i]
                features.extend([
                    V[peak_idx],                    # voltage
                    dQdV[peak_idx],                 # height
                    widths[i] * dV if i < len(widths) else 0,  # width in V
                    dQdV[peak_idx] * widths[i] * dV if i < len(widths) else 0,  # approx area
                ])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])
        
        # Global features
        features.append(np.trapz(np.maximum(dQdV, 0), V))  # total area
        features.append(float(len(peaks)))                  # num peaks detected
        features.append(V[np.argmax(dQdV)] if len(dQdV) > 0 else 0)  # voltage at max
        
        return np.array(features, dtype=np.float32)
    
    def _process_single_curve(self, 
                              experiment_id: int,
                              cell_id: str,
                              rpt_id: int,
                              voltage: np.ndarray,
                              capacity: np.ndarray) -> np.ndarray:
        """Process single curve with caching.
        
        Args:
            experiment_id: Experiment ID
            cell_id: Cell identifier
            rpt_id: RPT index
            voltage: Voltage array
            capacity: Capacity array
        
        Returns:
            Feature vector
        """
        if self.use_cache:
            cache = get_cache()
            return cache.get_or_compute(
                experiment_id=experiment_id,
                cell_id=cell_id,
                rpt_id=rpt_id,
                pipeline_name=self.name,
                pipeline_params=self.get_params(),
                compute_fn=lambda: self._compute_features(voltage, capacity)
            )
        else:
            return self._compute_features(voltage, capacity)
    
    def _compute_features(self, voltage: np.ndarray, capacity: np.ndarray) -> np.ndarray:
        """Compute features without caching.
        
        Args:
            voltage: Voltage array
            capacity: Capacity array
        
        Returns:
            Feature vector
        """
        V_mid, dQdV = self.compute_ica(voltage, capacity)
        return self.extract_peak_features(V_mid, dQdV)
    
    def fit(self, data: Dict[str, Any]) -> 'ICAPeaksPipeline':
        """Fit scaler on training data features.
        
        Args:
            data: Dictionary with 'curves' key containing list of (voltage, capacity, meta) tuples
        
        Returns:
            self
        """
        curves = data['curves']  # List of (voltage, capacity, meta) tuples
        
        X = []
        for voltage, capacity, meta in curves:
            features = self._process_single_curve(
                meta['experiment_id'],
                meta['cell_id'],
                meta['rpt_id'],
                voltage, capacity
            )
            X.append(features)
        X = np.vstack(X)
        
        if self.normalize:
            self.scaler = StandardScaler()
            self.scaler.fit(X)
        
        return self
    
    def transform(self, data: Dict[str, Any]) -> List[Sample]:
        """Transform curves into Sample objects.
        
        Args:
            data: Dictionary with 'curves' and optionally 'targets' keys
        
        Returns:
            List of Sample objects
        """
        curves = data['curves']
        targets = data.get('targets', {})  # {(cell_id, rpt_id): soh_value}
        
        samples = []
        for voltage, capacity, meta in curves:
            features = self._process_single_curve(
                meta['experiment_id'],
                meta['cell_id'],
                meta['rpt_id'],
                voltage, capacity
            )
            
            if self.normalize and self.scaler:
                features = self.scaler.transform(features.reshape(1, -1)).flatten()
            
            # Get target
            key = (meta['cell_id'], meta['rpt_id'])
            y_value = targets.get(key, 0.0)
            
            sample = Sample(
                meta=meta,
                x=features,
                y=np.array([y_value], dtype=np.float32),
            )
            samples.append(sample)
        
        return samples
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        return self.feature_names_
