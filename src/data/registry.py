"""Data loader registry for pipeline-specific data loading."""

from typing import Dict, Any, List
from pathlib import Path
import logging
import pandas as pd

from .tables import SummaryDataLoader, TimeseriesDataLoader

logger = logging.getLogger(__name__)


class DataLoaderRegistry:
    """Registry for data loaders with pipeline-to-loader mapping.

    Maps pipeline types to their compatible data loaders and provides
    unified interface for loading data in the format expected by each pipeline.
    """

    # Pipeline to data format mapping
    PIPELINE_DATA_FORMAT = {
        "summary_set": "summary",  # Uses summary statistics (df)
        "summary_cycle": "summary",  # Uses summary statistics (df)
        "ica_peaks": "curves",  # Uses voltage curves
        "latent_ode_seq": "curves",  # Uses voltage curves as sequences
    }

    @classmethod
    def get_data_format(cls, pipeline_name: str) -> str:
        """Get the data format required by a pipeline.

        Args:
            pipeline_name: Name of the pipeline

        Returns:
            Data format string ('summary' or 'curves')
        """
        return cls.PIPELINE_DATA_FORMAT.get(pipeline_name, "summary")

    @classmethod
    def load_summary_data(
        cls,
        experiment_id: int,
        base_path: str,
        cells: List[str],
        temp_map: Dict[int, List[str]],
    ) -> pd.DataFrame:
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
    def load_curves_data(
        cls,
        experiment_id: int,
        base_path: str,
        cells: List[str],
        temp_map: Dict[int, List[str]],
        curve_type: str = "0.1C",
        direction: str = "discharge",
    ) -> Dict[str, Any]:
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

        # Cache summary data per temperature to avoid repeated file I/O
        summary_cache = {}

        curves = []
        targets = {}

        for cell_id in cells:
            temp_C = cell_to_temp.get(cell_id, 25)

            try:
                # Load all voltage curves for this cell
                cell_curves = timeseries_loader.load_all_curves(
                    cell_id, curve_type, direction
                )

                # Load summary data to get SOH targets (use cache)
                if temp_C not in summary_cache:
                    try:
                        summary_cache[temp_C] = summary_loader.load_performance_summary(
                            cell_id, temp_C
                        )
                    except FileNotFoundError:
                        summary_cache[temp_C] = None
                        logger.warning(
                            f"No summary data for cell {cell_id} at {temp_C}°C"
                        )

                summary_df = summary_cache[temp_C]

                for rpt_id, curve_df in cell_curves.items():
                    # Extract voltage and capacity arrays with robust column detection
                    voltage_col = None
                    capacity_col = None

                    # Define possible column name patterns
                    voltage_patterns = ["voltage", "volt", "v"]
                    capacity_patterns = ["charge", "capacity", "cap", "ah", "amp"]

                    for c in curve_df.columns:
                        c_lower = c.lower()
                        if voltage_col is None and any(
                            pattern in c_lower for pattern in voltage_patterns
                        ):
                            voltage_col = c
                        elif capacity_col is None and any(
                            pattern in c_lower for pattern in capacity_patterns
                        ):
                            capacity_col = c

                    if voltage_col is None or capacity_col is None:
                        logger.warning(
                            f"Could not find voltage/capacity columns for cell {cell_id} RPT {rpt_id}. Columns: {curve_df.columns.tolist()}"
                        )
                        continue

                    voltage = curve_df[voltage_col].values
                    capacity = curve_df[capacity_col].values

                    meta = {
                        "experiment_id": experiment_id,
                        "cell_id": cell_id,
                        "rpt_id": rpt_id,
                        "temperature_C": temp_C,
                    }

                    curves.append((voltage, capacity, meta))

                    # Get SOH target from summary if available
                    if summary_df is not None:
                        # Use the pre-computed SoH column if available
                        if "SoH" in summary_df.columns:
                            # Try to map this RPT to a specific summary row using an explicit identifier
                            summary_row_idx = None
                            id_columns = [
                                "rpt_id",
                                "RPT",
                                "cycle",
                                "Cycle",
                                "cycle_index",
                                "Cycle_Index",
                            ]
                            for id_col in id_columns:
                                if id_col in summary_df.columns:
                                    matches = summary_df[summary_df[id_col] == rpt_id]
                                    if not matches.empty:
                                        summary_row_idx = matches.index[0]
                                        break

                            if summary_row_idx is not None:
                                # Prefer an exact identifier-based match when available
                                soh = summary_df["SoH"].loc[summary_row_idx]
                                targets[(cell_id, rpt_id)] = soh
                            else:
                                # Fall back to approximate positional mapping between RPT and summary index
                                logger.warning(
                                    "Approximate RPT-to-summary mapping for cell %s RPT %s; "
                                    "using positional index with last-value fallback.",
                                    cell_id,
                                    rpt_id,
                                )
                                if len(summary_df) > 0:
                                    if rpt_id < len(summary_df):
                                        soh = summary_df["SoH"].iloc[rpt_id]
                                    else:
                                        # If RPT index exceeds summary length, use last available SOH
                                        soh = summary_df["SoH"].iloc[-1]
                                    targets[(cell_id, rpt_id)] = soh
                        else:
                            # Fallback: compute SOH from capacity
                            if "Cell Capacity [mA h]" in summary_df.columns:
                                capacity_series = summary_df["Cell Capacity [mA h]"]
                                capacity_val = capacity_series.iloc[
                                    min(rpt_id, len(capacity_series) - 1)
                                ]

                                # Normalize to SOH using a more robust initial capacity estimate
                                # Use the first available capacity measurement as initial capacity
                                initial_capacity = capacity_series.iloc[0]
                                if pd.isna(initial_capacity):
                                    non_nan = capacity_series.dropna()
                                    if not non_nan.empty:
                                        initial_capacity = non_nan.iloc[0]

                                if initial_capacity and initial_capacity > 0:
                                    soh = capacity_val / initial_capacity
                                else:
                                    # If we cannot determine a valid initial capacity, fall back to 1.0
                                    logger.warning(
                                        f"Could not determine initial capacity for cell {cell_id}, using SOH=1.0"
                                    )
                                    soh = 1.0

                                targets[(cell_id, rpt_id)] = soh

            except Exception as e:
                logger.warning(f"Error loading curves for cell {cell_id}: {e}")
                continue

        logger.info(f"Loaded {len(curves)} curves from {len(cells)} cells")

        return {
            "curves": curves,
            "targets": targets,
        }

    @classmethod
    def load_data(
        cls,
        pipeline_name: str,
        experiment_id: int,
        base_path: str,
        cells: List[str],
        temp_map: Dict[int, List[str]],
    ) -> Any:
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

        if data_format == "summary":
            df = cls.load_summary_data(experiment_id, base_path, cells, temp_map)
            return {"df": df}
        elif data_format == "curves":
            return cls.load_curves_data(experiment_id, base_path, cells, temp_map)
        else:
            raise ValueError(f"Unknown data format: {data_format}")
