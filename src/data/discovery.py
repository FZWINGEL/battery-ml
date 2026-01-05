"""File discovery utilities for experiment data."""

from pathlib import Path
from typing import List, Dict, Any, Optional
import re
import logging

from .expt_paths import ExperimentPaths

logger = logging.getLogger(__name__)


def discover_experiment_files(base_path: Path, 
                               experiment_id: int) -> Dict[str, List[Path]]:
    """Discover all available data files for an experiment.
    
    Args:
        base_path: Base path to data
        experiment_id: Experiment ID (1-5)
    
    Returns:
        Dictionary with keys:
        - 'performance_summary': List of performance summary files
        - 'cycle_summary': List of cycle summary files
        - 'set_summary': List of set summary files
        - 'voltage_curves': List of voltage curve files
    """
    paths = ExperimentPaths(experiment_id, base_path)
    
    result = {
        'performance_summary': [],
        'cycle_summary': [],
        'set_summary': [],
        'voltage_curves': [],
    }
    
    if not paths.exists():
        logger.warning(f"Experiment directory not found: {paths.expt_path}")
        return result
    
    # Find performance summaries
    perf_dir = paths.expt_path / "Summary Data" / "Performance Summary"
    if perf_dir.exists():
        result['performance_summary'] = list(perf_dir.glob("*.csv"))
    
    # Find cycle summaries
    cycle_dir = (paths.expt_path / "Summary Data" / "Ageing Sets Summary" / 
                 "Summary per Cycle")
    if cycle_dir.exists():
        result['cycle_summary'] = list(cycle_dir.glob("*.csv"))
    
    # Find set summaries
    set_dir = (paths.expt_path / "Summary Data" / "Ageing Sets Summary" / 
               "Summary per Set")
    if set_dir.exists():
        result['set_summary'] = list(set_dir.glob("*.csv"))
    
    # Find voltage curves
    ts_dir = paths.expt_path / "Processed Timeseries Data"
    if ts_dir.exists():
        result['voltage_curves'] = list(ts_dir.rglob("*.csv"))
    
    for key, files in result.items():
        logger.info(f"Found {len(files)} {key} files")
    
    return result


def parse_filename_metadata(filename: str) -> Dict[str, Any]:
    """Extract metadata from standardized filename.
    
    Args:
        filename: Filename to parse
    
    Returns:
        Dictionary with extracted metadata (experiment_id, cell_id, temp_C, rpt_id, etc.)
    
    Example:
        >>> meta = parse_filename_metadata("Expt 5 - cell A (10degC) - Processed Data.csv")
        >>> print(meta)  # {'experiment_id': 5, 'cell_id': 'A', 'temperature_C': 10}
    """
    meta = {}
    
    # Extract experiment ID
    expt_match = re.search(r'[Ee]xpt\s*(\d+)', filename)
    if expt_match:
        meta['experiment_id'] = int(expt_match.group(1))
    
    # Extract cell ID
    cell_match = re.search(r'cell\s*([A-Ha-h])', filename, re.IGNORECASE)
    if cell_match:
        meta['cell_id'] = cell_match.group(1).upper()
    
    # Extract temperature
    temp_match = re.search(r'(\d+)\s*deg[Cc]', filename)
    if temp_match:
        meta['temperature_C'] = int(temp_match.group(1))
    
    # Extract RPT index
    rpt_match = re.search(r'RPT\s*(\d+)', filename)
    if rpt_match:
        meta['rpt_id'] = int(rpt_match.group(1))
    
    # Extract curve type
    curve_match = re.search(r'(\d+\.?\d*)[Cc]', filename)
    if curve_match and 'deg' not in filename[max(0, filename.find(curve_match.group(0))-3):]:
        meta['curve_type'] = curve_match.group(1) + "C"
    
    # Extract direction
    if 'discharge' in filename.lower():
        meta['direction'] = 'discharge'
    elif 'charge' in filename.lower():
        meta['direction'] = 'charge'
    
    return meta


def validate_data_structure(base_path: Path, experiment_id: int) -> Dict[str, Any]:
    """Validate that expected data structure exists.
    
    Args:
        base_path: Base path to data
        experiment_id: Experiment ID
    
    Returns:
        Dictionary with validation results:
        - 'valid': bool
        - 'missing': List of missing paths
        - 'found': List of found paths
        - 'cells': List of cells with data
    """
    paths = ExperimentPaths(experiment_id, base_path)
    
    result = {
        'valid': True,
        'missing': [],
        'found': [],
        'cells': [],
    }
    
    # Check base directory
    if not paths.exists():
        result['valid'] = False
        result['missing'].append(str(paths.expt_path))
        return result
    
    result['found'].append(str(paths.expt_path))
    
    # Check for each expected cell
    temp_map = {
        5: {10: ['A', 'B', 'C'], 25: ['D', 'E'], 40: ['F', 'G', 'H']},
        # Add other experiments as needed
    }
    
    if experiment_id in temp_map:
        for temp_C, cells in temp_map[experiment_id].items():
            for cell_id in cells:
                path = paths.performance_summary(cell_id, temp_C)
                if path.exists():
                    result['found'].append(str(path))
                    if cell_id not in result['cells']:
                        result['cells'].append(cell_id)
                else:
                    result['missing'].append(str(path))
    
    result['valid'] = len(result['cells']) > 0
    
    return result
