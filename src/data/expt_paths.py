"""Path resolution for experiment data following Dataset.md conventions."""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
import re


@dataclass
class ExperimentPaths:
    """Resolves all paths for a given experiment following Dataset.md conventions.
    
    Handles differences between experiments:
    - Expt 5 uses "Cell A", Expt 1-4 use "cell A"
    - Different folder naming conventions
    
    Example usage:
        >>> paths = ExperimentPaths(5, Path("Raw Data"))
        >>> summary_path = paths.performance_summary("A", 10)
        >>> print(summary_path)
    """
    
    EXPERIMENT_FOLDERS = {
        1: "Expt 1 - Si-based Degradation",
        2: "Expt 2,2 - C-based Degradation 2",
        3: "Expt 3 - Cathode Degradation and Li-Plating",
        4: "Expt 4 - Drive Cycle Aging (Control)",
        5: "Expt 5 - Standard Cycle Aging (Control)",
    }
    
    # CRITICAL: Expt 5 uses "Cell A", Expt 1-4 use "cell A"
    CELL_DIR_PREFIX = {1: "cell", 2: "cell", 3: "cell", 4: "cell", 5: "Cell"}
    
    experiment_id: int
    base_path: Path = field(default_factory=lambda: Path("Raw Data"))
    
    expt_path: Path = field(init=False)
    cell_prefix: str = field(init=False)
    
    def __post_init__(self):
        self.base_path = Path(self.base_path)
        
        if self.experiment_id not in self.EXPERIMENT_FOLDERS:
            raise ValueError(f"Invalid experiment_id: {self.experiment_id}. Must be 1-5.")
        
        self.expt_path = self.base_path / self.EXPERIMENT_FOLDERS[self.experiment_id]
        self.cell_prefix = self.CELL_DIR_PREFIX[self.experiment_id]
    
    # ─────────────────────────────────────────────────────────────────
    # Summary Data Paths
    # ─────────────────────────────────────────────────────────────────
    
    def performance_summary(self, cell_id: str, temp_C: int) -> Path:
        """Performance Summary CSV (set-level health indicators).
        
        Args:
            cell_id: Cell identifier ('A', 'B', ..., 'H')
            temp_C: Temperature in Celsius
        
        Returns:
            Path to the Performance Summary CSV file
        """
        # Experiment 2 uses "2,2" in filenames, not "2"
        exp_label = "2,2" if self.experiment_id == 2 else str(self.experiment_id)
        filename = f"Expt {exp_label} - cell {cell_id} ({temp_C}degC) - Processed Data.csv"
        return self.expt_path / "Summary Data" / "Performance Summary" / filename
    
    def summary_per_cycle(self, cell_id: str) -> Path:
        """Summary per Cycle CSV (cycle-level metrics).
        
        Args:
            cell_id: Cell identifier
        
        Returns:
            Path to the Summary per Cycle CSV file
        """
        # Experiment 2 uses "2,2" in filenames, not "2"
        exp_label = "2,2" if self.experiment_id == 2 else str(self.experiment_id)
        filename = f"expt {exp_label} - cell {cell_id} - cycle_data.csv"
        return (self.expt_path / "Summary Data" / "Ageing Sets Summary" / 
                "Summary per Cycle" / filename)
    
    def summary_per_set(self, cell_id: str) -> Path:
        """Summary per Set CSV.
        
        Args:
            cell_id: Cell identifier
        
        Returns:
            Path to the Summary per Set CSV file
        """
        # Experiment 2 uses "2,2" in filenames, not "2"
        exp_label = "2,2" if self.experiment_id == 2 else str(self.experiment_id)
        filename = f"expt {exp_label} - cell {cell_id} - set_data.csv"
        return (self.expt_path / "Summary Data" / "Ageing Sets Summary" / 
                "Summary per Set" / filename)
    
    # ─────────────────────────────────────────────────────────────────
    # Timeseries Data Paths
    # ─────────────────────────────────────────────────────────────────
    
    def voltage_curve(self, cell_id: str, rpt: int, 
                      curve_type: str = "0.1C",
                      direction: str = "discharge") -> Path:
        """Processed voltage curve CSV.
        
        Args:
            cell_id: Cell identifier
            rpt: RPT measurement index
            curve_type: Type of curve (e.g., "0.1C")
            direction: "discharge" or "charge"
        
        Returns:
            Path to the voltage curve CSV file
        """
        cell_dir = f"{self.cell_prefix} {cell_id}"
        # Experiment 2 uses "2,2" in filenames, not "2"
        exp_label = "2,2" if self.experiment_id == 2 else str(self.experiment_id)
        filename = f"Expt {exp_label} - cell {cell_id} - RPT{rpt} - {curve_type} {direction} data.csv"
        return (self.expt_path / "Processed Timeseries Data" / 
                f"{curve_type} Voltage Curves" / cell_dir / filename)
    
    def list_available_rpts(self, cell_id: str, curve_type: str = "0.1C") -> List[int]:
        """List all available RPT indices for a cell.
        
        Args:
            cell_id: Cell identifier
            curve_type: Type of curve
        
        Returns:
            Sorted list of available RPT indices
        """
        cell_dir = f"{self.cell_prefix} {cell_id}"
        curve_dir = (self.expt_path / "Processed Timeseries Data" / 
                     f"{curve_type} Voltage Curves" / cell_dir)
        
        if not curve_dir.exists():
            return []
        
        rpts = []
        for f in curve_dir.glob(f"*RPT*discharge*.csv"):
            match = re.search(r'RPT(\d+)', f.name)
            if match:
                rpts.append(int(match.group(1)))
        return sorted(rpts)
    
    def list_all_files(self, pattern: str = "*.csv") -> List[Path]:
        """List all files matching pattern in experiment directory.
        
        Args:
            pattern: Glob pattern (default: "*.csv")
        
        Returns:
            List of matching file paths
        """
        if not self.expt_path.exists():
            return []
        return list(self.expt_path.rglob(pattern))
    
    def exists(self) -> bool:
        """Check if the experiment directory exists.
        
        Returns:
            True if directory exists
        """
        return self.expt_path.exists()
    
    def __repr__(self) -> str:
        return f"ExperimentPaths(experiment_id={self.experiment_id}, base_path='{self.base_path}')"
