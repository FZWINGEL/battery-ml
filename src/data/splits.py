"""Data split strategies for battery degradation experiments."""

from typing import List, Tuple, Dict
from ..pipelines.sample import Sample


def temperature_split(
    samples: List[Sample], train_temps: List[int], val_temps: List[int]
) -> Tuple[List[Sample], List[Sample]]:
    """Split samples by temperature.

    Default for Expt 5: train on [10, 40], val on [25].
    Tests temperature interpolation capability.

    Args:
        samples: List of Sample objects
        train_temps: Temperatures for training (e.g., [10, 40])
        val_temps: Temperatures for validation (e.g., [25])

    Returns:
        Tuple of (train_samples, val_samples)

    Example:
        >>> train, val = temperature_split(samples, train_temps=[10, 40], val_temps=[25])
    """
    train = [s for s in samples if s.meta.get("temperature_C") in train_temps]
    val = [s for s in samples if s.meta.get("temperature_C") in val_temps]

    return train, val


def leave_one_cell_out(
    samples: List[Sample], test_cell: str
) -> Tuple[List[Sample], List[Sample]]:
    """Leave-one-cell-out split.

    Use for testing generalization to unseen cells.

    Args:
        samples: List of Sample objects
        test_cell: Cell ID to hold out for testing

    Returns:
        Tuple of (train_samples, test_samples)

    Example:
        >>> train, test = leave_one_cell_out(samples, test_cell='A')
    """
    train = [s for s in samples if s.meta.get("cell_id") != test_cell]
    test = [s for s in samples if s.meta.get("cell_id") == test_cell]

    return train, test


def loco_cv_splits(
    samples: List[Sample],
) -> List[Tuple[str, List[Sample], List[Sample]]]:
    """Generate all leave-one-cell-out cross-validation splits.

    Args:
        samples: List of Sample objects

    Returns:
        List of (cell_id, train_samples, test_samples) tuples

    Example:
        >>> for cell_id, train, test in loco_cv_splits(samples):
        ...     model.fit(train)
        ...     score = model.evaluate(test)
    """
    cells = sorted(set(s.meta.get("cell_id") for s in samples if "cell_id" in s.meta))

    splits = []
    for cell in cells:
        train, test = leave_one_cell_out(samples, cell)
        splits.append((cell, train, test))

    return splits


def temporal_split(
    samples: List[Sample], train_fraction: float = 0.7, val_fraction: float = 0.15
) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    """Split samples temporally (early cycles for train, later for val/test).

    Useful for testing extrapolation to future degradation states.

    Args:
        samples: List of Sample objects (should have 'set_idx' or 'cycle_idx' in meta)
        train_fraction: Fraction of samples for training
        val_fraction: Fraction of samples for validation

    Returns:
        Tuple of (train_samples, val_samples, test_samples)
    """

    # Sort by time index
    def get_time_idx(s: Sample) -> int:
        return s.meta.get("set_idx", s.meta.get("cycle_idx", 0))

    sorted_samples = sorted(samples, key=get_time_idx)
    n = len(sorted_samples)

    train_end = int(n * train_fraction)
    val_end = int(n * (train_fraction + val_fraction))

    train = sorted_samples[:train_end]
    val = sorted_samples[train_end:val_end]
    test = sorted_samples[val_end:]

    return train, val, test


def random_split(
    samples: List[Sample],
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    """Random split with fixed seed.

    Args:
        samples: List of Sample objects
        train_fraction: Fraction for training
        val_fraction: Fraction for validation
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_samples, val_samples, test_samples)
    """
    import random

    rng = random.Random(seed)
    shuffled = samples.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * train_fraction)
    val_end = int(n * (train_fraction + val_fraction))

    train = shuffled[:train_end]
    val = shuffled[train_end:val_end]
    test = shuffled[val_end:]

    return train, val, test


def stratified_temperature_split(
    samples: List[Sample], val_fraction: float = 0.2, seed: int = 42
) -> Tuple[List[Sample], List[Sample]]:
    """Stratified split maintaining temperature distribution.

    Args:
        samples: List of Sample objects
        val_fraction: Fraction for validation
        seed: Random seed

    Returns:
        Tuple of (train_samples, val_samples)
    """
    import random

    rng = random.Random(seed)

    # Group by temperature
    by_temp: Dict[int, List[Sample]] = {}
    for s in samples:
        temp = s.meta.get("temperature_C", 25)
        if temp not in by_temp:
            by_temp[temp] = []
        by_temp[temp].append(s)

    train, val = [], []

    for temp, temp_samples in by_temp.items():
        shuffled = temp_samples.copy()
        rng.shuffle(shuffled)

        n_val = max(1, int(len(shuffled) * val_fraction))
        val.extend(shuffled[:n_val])
        train.extend(shuffled[n_val:])

    return train, val
