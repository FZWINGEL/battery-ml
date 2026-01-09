"""Centralized unit conversions for battery data.

Call these ONCE during data loading to ensure consistent internal units.
"""

from typing import Union
import numpy as np
import pandas as pd


class UnitConverter:
    """Ensures all data uses consistent units internally.

    Internal units (after conversion):
    - Capacity: Ah (not mAh)
    - Current: A (not mA)
    - Temperature: K (for Arrhenius calculations)
    - Time: seconds (or days for long-term analysis)
    - Resistance: Ohms

    Example usage:
        >>> capacity_mAh = 4800
        >>> capacity_Ah = UnitConverter.mAh_to_Ah(capacity_mAh)
        >>> print(capacity_Ah)  # 4.8
    """

    @staticmethod
    def mAh_to_Ah(
        value: Union[float, np.ndarray, pd.Series],
    ) -> Union[float, np.ndarray, pd.Series]:
        """Convert milliamp-hours to amp-hours.

        Args:
            value: Value(s) in mAh

        Returns:
            Value(s) in Ah
        """
        return value / 1000.0

    @staticmethod
    def Ah_to_mAh(
        value: Union[float, np.ndarray, pd.Series],
    ) -> Union[float, np.ndarray, pd.Series]:
        """Convert amp-hours to milliamp-hours.

        Args:
            value: Value(s) in Ah

        Returns:
            Value(s) in mAh
        """
        return value * 1000.0

    @staticmethod
    def mA_to_A(
        value: Union[float, np.ndarray, pd.Series],
    ) -> Union[float, np.ndarray, pd.Series]:
        """Convert milliamps to amps.

        Args:
            value: Value(s) in mA

        Returns:
            Value(s) in A
        """
        return value / 1000.0

    @staticmethod
    def A_to_mA(
        value: Union[float, np.ndarray, pd.Series],
    ) -> Union[float, np.ndarray, pd.Series]:
        """Convert amps to milliamps.

        Args:
            value: Value(s) in A

        Returns:
            Value(s) in mA
        """
        return value * 1000.0

    @staticmethod
    def celsius_to_kelvin(
        value: Union[float, np.ndarray, pd.Series],
    ) -> Union[float, np.ndarray, pd.Series]:
        """Convert Celsius to Kelvin.

        Args:
            value: Temperature(s) in °C

        Returns:
            Temperature(s) in K
        """
        return value + 273.15

    @staticmethod
    def kelvin_to_celsius(
        value: Union[float, np.ndarray, pd.Series],
    ) -> Union[float, np.ndarray, pd.Series]:
        """Convert Kelvin to Celsius.

        Args:
            value: Temperature(s) in K

        Returns:
            Temperature(s) in °C
        """
        return value - 273.15

    @staticmethod
    def normalize_capacity_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Auto-detect mAh vs Ah and normalize to Ah.

        Heuristic: if column name contains 'mA' or mean value > 100,
        assume mAh and convert.

        Args:
            df: DataFrame containing the column
            col: Column name to normalize

        Returns:
            DataFrame with normalized column (modified copy)
        """
        df = df.copy()

        # Check column name and values
        is_mAh = (
            "[mA h]" in col
            or "[mAh]" in col
            or "mAh" in col
            or (col in df.columns and df[col].mean() > 100)
        )

        if is_mAh and col in df.columns:
            df[col] = df[col] / 1000.0

        return df

    @staticmethod
    def normalize_all_capacity_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize all capacity-related columns to Ah.

        Args:
            df: DataFrame to normalize

        Returns:
            DataFrame with all capacity columns normalized
        """
        df = df.copy()

        for col in df.columns:
            if any(
                kw in col.lower()
                for kw in ["capacity", "throughput", "charge", "discharge"]
            ):
                df = UnitConverter.normalize_capacity_column(df, col)

        return df

    @staticmethod
    def compute_arrhenius_factor(
        temp_K: Union[float, np.ndarray], Ea: float = 50000.0
    ) -> Union[float, np.ndarray]:
        """Compute Arrhenius factor exp(-Ea/RT).

        Args:
            temp_K: Temperature in Kelvin
            Ea: Activation energy in J/mol (default: 50000)

        Returns:
            Arrhenius factor
        """
        R = 8.314  # J/(mol·K)
        return np.exp(-Ea / (R * temp_K))


# Metadata to store with artifacts for documentation
UNIT_METADATA = {
    "capacity": "Ah",
    "current": "A",
    "temperature": "K",
    "resistance": "Ohm",
    "time": "s",
    "voltage": "V",
    "power": "W",
    "energy": "Wh",
}
