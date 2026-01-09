"""Data loader registry for pipeline-specific data loading."""

from typing import Dict, Type, Any, List, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import logging

from .tables import SummaryDataLoader, TimeseriesDataLoader

logger = logging.getLogger(__name__)


class DataLoaderRegistry:
    """Registry for data loaders with pipeline-to-loader mapping.
    
    Maps pipeline types to their compatible data loaders and provides
    unified interface for loading data in the format expected by each pipeline.
    """
    
    # Pipeline to data format mapping
    PIPELINE_DATA_FORMAT = {
        'summary_set': 'summary',      # Uses summary statistics (df)
        'summary_cycle': 'summary',    # Uses summary statistics (df)
        'ica_peaks': 'curves',         # Uses voltage curves
        'latent_ode_seq': 'curves',    # Uses voltage curves as sequences
    }
    
    @classmethod
    def get_data_format(cls, pipeline_name: str) -> str:
        """Get the data format required by a pipeline.
        
        Args:
            pipeline_name: Name of the pipeline
            
        Returns:
            Data format string ('summary' or 'curves')
        """
        return cls.PIPELINE_DATA_FORMAT.get(pipeline_name, 'summary')
    
    @classmethod
    def load_summary_data(cls, experiment_id: int, base_path: str,
                          cells: List[str], temp_map: Dict[int, List[str]]) -> pd.DataFrame:
        """Load summary data for summary-based pipelines.
        
        Args:
            experiment_id: Experiment ID (1-5)
            base_path: Base path to raw data
            cells: List of cell IDs to load
            temp_map: Mapping from temperature to cell IDs
            
        Returns:
            DataFrame with summary data
        """
        loader = SummaryDataLoader(experiment_id, Path(base_path))
        return loader.load_all_cells(cells=cells, temp_map=temp_map)
    
    @classmethod
    def load_curves_data(cls, experiment_id: int, base_path: str,
                         cells: List[str], temp_map: Dict[int, List[str]],
                         curve_type: str = "0.1C",
                         direction: str = "discharge") -> Dict[str, Any]:
        """Load voltage curve data for curve-based pipelines.
        
        Args:
            experiment_id: Experiment ID (1-5)
            base_path: Base path to raw data
            cells: List of cell IDs to load
            temp_map: Mapping from temperature to cell IDs
            curve_type: Type of curve (e.g., "0.1C")
            direction: "discharge" or "charge"
            
        Returns:
            Dictionary with 'curves' key containing list of (voltage, capacity, meta) tuples,
            and 'targets' key containing SOH values
        """
        timeseries_loader = TimeseriesDataLoader(experiment_id, Path(base_path))
        summary_loader = SummaryDataLoader(experiment_id, Path(base_path))
        
        # Build reverse temp_map for lookup
        cell_to_temp = {}
        for temp, cell_list in temp_map.items():
            for cell in cell_list:
                cell_to_temp[cell] = temp
        
        curves = []
        targets = {}
        
        for cell_id in cells:
            temp_C = cell_to_temp.get(cell_id, 25)
            
            try:
                # Load all voltage curves for this cell
                cell_curves = timeseries_loader.load_all_curves(cell_id, curve_type, direction)
                
                # Load summary data to get SOH targets
                try:
                    summary_df = summary_loader.load_performance_summary(cell_id, temp_C)
                except FileNotFoundError:
                    summary_df = None
                    logger.warning(f"No summary data for cell {cell_id}, using capacity as target")
                
                for rpt_id, curve_df in cell_curves.items():
                    # Extract voltage and capacity arrays
                    # Column names: 'Voltage (V)', 'Charge (mA.h)'
                    voltage_col = None
                    capacity_col = None
                    
                    for c in curve_df.columns:
                        if 'voltage' in c.lower():
                            voltage_col = c
                        elif 'charge' in c.lower() or 'capacity' in c.lower():
                            capacity_col = c
                    
                    if voltage_col is None or capacity_col is None:
                        logger.warning(f"Could not find voltage/capacity columns for cell {cell_id} RPT {rpt_id}. Columns: {curve_df.columns.tolist()}")
                        continue
                    
                    voltage = curve_df[voltage_col].values
                    capacity = curve_df[capacity_col].values
                    
                    meta = {
                        'experiment_id': experiment_id,
                        'cell_id': cell_id,
                        'rpt_id': rpt_id,
                        'temperature_C': temp_C,
                    }
                    
                    curves.append((voltage, capacity, meta))
                    
                    # Get SOH target from summary if available
                    if summary_df is not None:
                        # Use the pre-computed SoH column if available
                        if 'SoH' in summary_df.columns:
                            # Map RPT index to summary row (approximate mapping)
                            # Use the RPT index to select a row from summary data
                            if rpt_id < len(summary_df):
                                soh = summary_df['SoH'].iloc[rpt_id]
                                targets[(cell_id, rpt_id)] = soh
                            else:
                                # If RPT index exceeds summary length, use last available SOH
                                soh = summary_df['SoH'].iloc[-1]
                                targets[(cell_id, rpt_id)] = soh
                        else:
                            # Fallback: compute SOH from capacity
                            if 'Cell Capacity [mA h]' in summary_df.columns:
                                capacity_val = summary_df['Cell Capacity [mA h]'].iloc[min(rpt_id, len(summary_df)-1)]
                                # Normalize to SOH (assuming initial capacity is max capacity)
                                initial_capacity = summary_df['Cell Capacity [mA h]'].max()
                                soh = capacity_val / initial_capacity if initial_capacity > 0 else 1.0
                                targets[(cell_id, rpt_id)] = soh
                                    
            except Exception as e:
                logger.warning(f"Error loading curves for cell {cell_id}: {e}")
                continue
        
        logger.info(f"Loaded {len(curves)} curves from {len(cells)} cells")
        
        return {
            'curves': curves,
            'targets': targets,
        }
    
    @classmethod
    def load_data(cls, pipeline_name: str, experiment_id: int, base_path: str,
                  cells: List[str], temp_map: Dict[int, List[str]]) -> Any:
        """Load data in the format required by the specified pipeline.
        
        Args:
            pipeline_name: Name of the pipeline
            experiment_id: Experiment ID (1-5)
            base_path: Base path to raw data
            cells: List of cell IDs to load
            temp_map: Mapping from temperature to cell IDs
            
        Returns:
            Data in the format expected by the pipeline
        """
        data_format = cls.get_data_format(pipeline_name)
        
        if data_format == 'summary':
            df = cls.load_summary_data(experiment_id, base_path, cells, temp_map)
            return {'df': df}
        elif data_format == 'curves':
            return cls.load_curves_data(experiment_id, base_path, cells, temp_map)
        else:
            raise ValueError(f"Unknown data format: {data_format}")
