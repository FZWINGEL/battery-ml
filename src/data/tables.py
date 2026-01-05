"""Data loaders for summary CSV files."""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import logging

from .expt_paths import ExperimentPaths
from .units import UnitConverter

logger = logging.getLogger(__name__)


class SummaryDataLoader:
    """Load and normalize summary CSV data.
    
    Example usage:
        >>> loader = SummaryDataLoader(5, Path("Raw Data"))
        >>> df = loader.load_all_cells(
        ...     cells=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
        ...     temp_map={10: ['A', 'B', 'C'], 25: ['D', 'E'], 40: ['F', 'G', 'H']}
        ... )
    """
    
    def __init__(self, experiment_id: int, base_path: Path):
        """Initialize the loader.
        
        Args:
            experiment_id: Experiment ID (1-5)
            base_path: Base path to raw data
        """
        self.paths = ExperimentPaths(experiment_id, Path(base_path))
        self.experiment_id = experiment_id
    
    def load_performance_summary(self, cell_id: str, temp_C: int) -> pd.DataFrame:
        """Load Performance Summary with unit normalization.
        
        Args:
            cell_id: Cell identifier ('A', 'B', ..., 'H')
            temp_C: Temperature in Celsius
        
        Returns:
            DataFrame with normalized units and metadata columns
        
        Raises:
            FileNotFoundError: If the summary file doesn't exist
        """
        path = self.paths.performance_summary(cell_id, temp_C)
        
        if not path.exists():
            raise FileNotFoundError(f"Performance summary not found: {path}")
        
        logger.debug(f"Loading performance summary: {path}")
        df = pd.read_csv(path, index_col=0)
        
        # Normalize capacity columns to Ah
        df = UnitConverter.normalize_all_capacity_columns(df)
        
        # Add metadata
        df['cell_id'] = cell_id
        df['temperature_C'] = temp_C
        df['experiment_id'] = self.experiment_id
        
        return df
    
    def load_summary_per_cycle(self, cell_id: str) -> pd.DataFrame:
        """Load cycle-level summary with unit normalization.
        
        Args:
            cell_id: Cell identifier
        
        Returns:
            DataFrame with cycle-level metrics
        
        Raises:
            FileNotFoundError: If the summary file doesn't exist
        """
        path = self.paths.summary_per_cycle(cell_id)
        
        if not path.exists():
            raise FileNotFoundError(f"Cycle summary not found: {path}")
        
        logger.debug(f"Loading cycle summary: {path}")
        df = pd.read_csv(path)
        
        # Normalize units
        df = UnitConverter.normalize_all_capacity_columns(df)
        
        df['cell_id'] = cell_id
        df['experiment_id'] = self.experiment_id
        
        return df
    
    def load_summary_per_set(self, cell_id: str) -> pd.DataFrame:
        """Load set-level summary with unit normalization.
        
        Args:
            cell_id: Cell identifier
        
        Returns:
            DataFrame with set-level metrics
        
        Raises:
            FileNotFoundError: If the summary file doesn't exist
        """
        path = self.paths.summary_per_set(cell_id)
        
        if not path.exists():
            raise FileNotFoundError(f"Set summary not found: {path}")
        
        logger.debug(f"Loading set summary: {path}")
        df = pd.read_csv(path)
        
        # Normalize units
        df = UnitConverter.normalize_all_capacity_columns(df)
        
        df['cell_id'] = cell_id
        df['experiment_id'] = self.experiment_id
        
        return df
    
    def load_all_cells(self, cells: List[str], 
                       temp_map: Dict[int, List[str]]) -> pd.DataFrame:
        """Load performance summary for all specified cells.
        
        Args:
            cells: List of cell IDs to load
            temp_map: Mapping from temperature (°C) to cell IDs
        
        Returns:
            Combined DataFrame with all cells
        """
        dfs = []
        
        for temp_C, temp_cells in temp_map.items():
            for cell_id in temp_cells:
                if cell_id in cells:
                    try:
                        df = self.load_performance_summary(cell_id, temp_C)
                        dfs.append(df)
                        logger.info(f"Loaded cell {cell_id} at {temp_C}°C: {len(df)} samples")
                    except FileNotFoundError as e:
                        logger.warning(f"Skipping cell {cell_id}: {e}")
        
        if not dfs:
            raise ValueError("No data loaded. Check paths and cell IDs.")
        
        combined = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded {len(combined)} total samples from {len(dfs)} cells")
        
        return combined
    
    def get_available_cells(self) -> List[str]:
        """Get list of cells with available data.
        
        Returns:
            List of cell IDs that have data files
        """
        available = []
        
        for cell_id in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
            # Check common temperatures
            for temp_C in [10, 25, 40]:
                path = self.paths.performance_summary(cell_id, temp_C)
                if path.exists():
                    available.append(cell_id)
                    break
        
        return available


class TimeseriesDataLoader:
    """Load voltage curve timeseries data."""
    
    def __init__(self, experiment_id: int, base_path: Path):
        """Initialize the loader.
        
        Args:
            experiment_id: Experiment ID (1-5)
            base_path: Base path to raw data
        """
        self.paths = ExperimentPaths(experiment_id, Path(base_path))
        self.experiment_id = experiment_id
    
    def load_voltage_curve(self, cell_id: str, rpt: int,
                           curve_type: str = "0.1C",
                           direction: str = "discharge") -> pd.DataFrame:
        """Load a single voltage curve.
        
        Args:
            cell_id: Cell identifier
            rpt: RPT measurement index
            curve_type: Type of curve (e.g., "0.1C")
            direction: "discharge" or "charge"
        
        Returns:
            DataFrame with voltage curve data
        
        Raises:
            FileNotFoundError: If the curve file doesn't exist
        """
        path = self.paths.voltage_curve(cell_id, rpt, curve_type, direction)
        
        if not path.exists():
            raise FileNotFoundError(f"Voltage curve not found: {path}")
        
        logger.debug(f"Loading voltage curve: {path}")
        df = pd.read_csv(path)
        
        # Typical columns: Voltage [V], Capacity [mA h], ...
        df = UnitConverter.normalize_all_capacity_columns(df)
        
        df['cell_id'] = cell_id
        df['rpt_id'] = rpt
        df['experiment_id'] = self.experiment_id
        
        return df
    
    def load_all_curves(self, cell_id: str, 
                        curve_type: str = "0.1C",
                        direction: str = "discharge") -> Dict[int, pd.DataFrame]:
        """Load all available voltage curves for a cell.
        
        Args:
            cell_id: Cell identifier
            curve_type: Type of curve
            direction: "discharge" or "charge"
        
        Returns:
            Dictionary mapping RPT index to DataFrame
        """
        rpts = self.paths.list_available_rpts(cell_id, curve_type)
        
        curves = {}
        for rpt in rpts:
            try:
                curves[rpt] = self.load_voltage_curve(cell_id, rpt, curve_type, direction)
            except FileNotFoundError:
                logger.warning(f"Could not load RPT {rpt} for cell {cell_id}")
        
        logger.info(f"Loaded {len(curves)} curves for cell {cell_id}")
        return curves
