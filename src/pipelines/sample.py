"""Canonical Sample schema for battery degradation data.

This is the universal format that all pipelines must produce and all models consume.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union
import torch
import numpy as np


@dataclass
class Sample:
    """Universal format every pipeline must produce.

    This contract enables:
    - Model-agnostic pipelines (swap models without changing data code)
    - Time-aware models (ODEs use delta_t; LSTMs ignore it)
    - Clean validation splits (meta contains grouping keys)
    - Composable features (x can be dict of multiple feature types)

    Attributes:
        meta: Metadata for splits, logging, debugging — not fed to model.
            Required keys:
            - 'experiment_id': int (1-5)
            - 'cell_id': str ('A', 'B', ..., 'H')
            - 'temperature_C': float (10, 25, 40)
            Optional keys:
            - 'set_idx': int (ageing set index)
            - 'rpt_id': int (RPT measurement index)
            - 'cycle_idx': int (absolute cycle number)
            - 'timestamp': float (seconds or days since experiment start)
            - 'delta_t': float (time since previous sample)
            - 'cumulative_throughput_Ah': float

        x: Features the model sees. Can be:
            - Tensor shape (feature_dim,): static features
            - Tensor shape (seq_len, feature_dim): sequences
            - Dict: {'summary': tensor, 'ica_peaks': tensor, ...}

        y: Target values. Shape (1,) for SOH regression or (n_targets,) for multi-task.

        mask: Optional boolean mask for variable-length sequences. Shape (seq_len,).

        t: Optional time vector for ODE models. Shape (seq_len,).
    """

    # Metadata (not fed to model)
    meta: Dict[str, Any] = field(default_factory=dict)

    # Features
    x: Union[torch.Tensor, Dict[str, torch.Tensor], np.ndarray, None] = None

    # Target(s)
    y: Union[torch.Tensor, np.ndarray, None] = None

    # Optional: for variable-length sequences
    mask: Optional[torch.Tensor] = None

    # Time vector: for ODE models (separate from x for clarity)
    t: Optional[torch.Tensor] = None

    def to_tensor(self) -> "Sample":
        """Convert numpy arrays to torch tensors (in-place)."""
        if isinstance(self.x, np.ndarray):
            self.x = torch.from_numpy(self.x).float()
        elif isinstance(self.x, dict):
            self.x = {
                k: torch.from_numpy(v).float() if isinstance(v, np.ndarray) else v
                for k, v in self.x.items()
            }

        if isinstance(self.y, np.ndarray):
            self.y = torch.from_numpy(self.y).float()

        if isinstance(self.t, np.ndarray):
            self.t = torch.from_numpy(self.t).float()

        if isinstance(self.mask, np.ndarray):
            self.mask = torch.from_numpy(self.mask).bool()

        return self

    def to_device(self, device: str) -> "Sample":
        """Move tensors to specified device (in-place)."""
        if isinstance(self.x, torch.Tensor):
            self.x = self.x.to(device)
        elif isinstance(self.x, dict):
            self.x = {k: v.to(device) for k, v in self.x.items()}

        if self.y is not None and isinstance(self.y, torch.Tensor):
            self.y = self.y.to(device)

        if self.t is not None and isinstance(self.t, torch.Tensor):
            self.t = self.t.to(device)

        if self.mask is not None and isinstance(self.mask, torch.Tensor):
            self.mask = self.mask.to(device)

        return self

    def clone(self) -> "Sample":
        """Create a deep copy of the sample."""
        import copy

        return copy.deepcopy(self)

    @property
    def feature_dim(self) -> int:
        """Get feature dimension."""
        if self.x is None:
            return 0
        if isinstance(self.x, dict):
            # Sum all feature dims
            total = 0
            for v in self.x.values():
                if hasattr(v, "shape"):
                    total += v.shape[-1]
            return total
        if hasattr(self.x, "shape"):
            return self.x.shape[-1]
        return 0

    @property
    def seq_len(self) -> Optional[int]:
        """Get sequence length (None if not a sequence)."""
        if self.x is None:
            return None
        if isinstance(self.x, dict):
            for v in self.x.values():
                if hasattr(v, "shape") and len(v.shape) >= 2:
                    return v.shape[0]
            return None
        if hasattr(self.x, "shape") and len(self.x.shape) >= 2:
            return self.x.shape[0]
        return None
