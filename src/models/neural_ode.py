"""Neural ODE for continuous-time degradation modeling."""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import logging

from .base import BaseModel
from .registry import ModelRegistry

logger = logging.getLogger(__name__)

try:
    from torchdiffeq import odeint, odeint_adjoint
    HAS_TORCHDIFFEQ = True
except ImportError:
    HAS_TORCHDIFFEQ = False
    logger.warning("torchdiffeq not installed. Install with: pip install torchdiffeq")


class ODEFunc(nn.Module):
    """Neural network defining dz/dt = f(z, t).
    
    The ODE function takes the current state z and time t,
    and returns the rate of change dz/dt.
    """
    
    def __init__(self, latent_dim: int, hidden_dim: int = 64):
        """Initialize the ODE function.
        
        Args:
            latent_dim: Dimension of the latent state
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim),  # +1 for time
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )
    
    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Compute dz/dt.
        
        Args:
            t: Current time (scalar or batch)
            z: Current state of shape (batch, latent_dim)
        
        Returns:
            dz/dt of shape (batch, latent_dim)
        """
        # Concatenate time as feature
        if t.dim() == 0:
            t_expanded = t.expand(z.shape[0], 1)
        else:
            t_expanded = t.unsqueeze(-1) if t.dim() == 1 else t
        
        zt = torch.cat([z, t_expanded], dim=-1)
        return self.net(zt)


@ModelRegistry.register("neural_ode")
class NeuralODEModel(BaseModel):
    """Latent ODE for continuous-time degradation modeling.
    
    Architecture:
    1. Encoder: maps x_0 → z_0 (initial latent state)
    2. ODE: integrates dz/dt = f(z,t) from t_0 to t_N
    3. Decoder: maps z_N → y (SOH prediction)
    
    Uses actual time values from Sample.t for integration.
    
    Example usage:
        >>> model = NeuralODEModel(input_dim=5, latent_dim=32)
        >>> output = model(x, t=t)  # x: (batch, seq_len, features), t: (seq_len,)
    """
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int = 1,
                 latent_dim: int = 32,
                 hidden_dim: int = 64,
                 solver: str = "dopri5",
                 rtol: float = 1e-4,
                 atol: float = 1e-5,
                 use_adjoint: bool = True):
        """Initialize the model.
        
        Args:
            input_dim: Number of input features per timestep
            output_dim: Number of output predictions
            latent_dim: Dimension of latent ODE state
            hidden_dim: Hidden dimension in networks
            solver: ODE solver ("dopri5", "euler", "rk4", etc.)
            rtol: Relative tolerance for solver
            atol: Absolute tolerance for solver
            use_adjoint: Use adjoint method for memory-efficient gradients
        """
        super().__init__(input_dim, output_dim)
        
        if not HAS_TORCHDIFFEQ:
            raise ImportError("torchdiffeq required. Install with: pip install torchdiffeq")
        
        self.latent_dim = latent_dim
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.use_adjoint = use_adjoint
        
        # Encoder: x_0 → z_0
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )
        
        # ODE function
        self.ode_func = ODEFunc(latent_dim, hidden_dim)
        
        # Decoder: z_N → y
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, output_dim),
        )
        
        self._trajectory: Optional[torch.Tensor] = None
    
    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        """Forward pass with ODE integration.
        
        Args:
            x: Input sequence of shape (batch, seq_len, input_dim)
            t: Time points of shape (batch, seq_len) or (seq_len,)
        
        Returns:
            Predictions of shape (batch, output_dim)
        """
        batch_size = x.shape[0]
        
        # Encode initial state
        z0 = self.encoder(x[:, 0, :])  # (batch, latent_dim)
        
        # Get integration times
        if t is None:
            # Default: uniform time points
            t_span = torch.linspace(0, 1, x.shape[1], device=x.device)
        else:
            # Use provided times (normalize to start at 0)
            if t.dim() == 2:
                t_span = t[0]  # Assume same times across batch
            else:
                t_span = t
            t_span = t_span - t_span[0]
        
        # Integrate ODE
        odeint_fn = odeint_adjoint if self.use_adjoint else odeint
        
        trajectory = odeint_fn(
            self.ode_func,
            z0,
            t_span,
            method=self.solver,
            rtol=self.rtol,
            atol=self.atol,
        )  # (seq_len, batch, latent_dim)
        
        self._trajectory = trajectory.detach()
        
        # Decode all timesteps (sequence-to-sequence)
        traj = trajectory.permute(1, 0, 2)  # (batch, seq_len, latent_dim)
        batch_size, seq_len, _ = traj.shape
        
        # Flatten, decode, reshape
        traj_flat = traj.reshape(-1, self.latent_dim)
        y_flat = self.decoder(traj_flat)
        y = y_flat.reshape(batch_size, seq_len, -1)
        
        return y
    
    def forward_trajectory(self, x: torch.Tensor, t: Optional[torch.Tensor] = None,
                           **kwargs) -> torch.Tensor:
        """Get predictions at all timesteps.
        
        Args:
            x: Input sequence
            t: Time points
        
        Returns:
            Predictions at all timesteps of shape (batch, seq_len, output_dim)
        """
        _ = self.forward(x, t)
        
        if self._trajectory is None:
            raise RuntimeError("Forward pass failed")
        
        # Decode at all timesteps
        traj = self._trajectory.permute(1, 0, 2)  # (batch, seq_len, latent_dim)
        batch_size, seq_len, _ = traj.shape
        
        # Flatten, decode, reshape
        traj_flat = traj.reshape(-1, self.latent_dim)
        y_flat = self.decoder(traj_flat)
        y = y_flat.reshape(batch_size, seq_len, -1)
        
        return y
    
    def explain(self, x: torch.Tensor, t: Optional[torch.Tensor] = None,
                **kwargs) -> Dict[str, Any]:
        """Return latent trajectory for visualization.
        
        Args:
            x: Input sequence
            t: Time points
        
        Returns:
            Dictionary with latent trajectory
        """
        _ = self.forward(x, t)
        return {
            'trajectory': self._trajectory.cpu().numpy() if self._trajectory is not None else None,
            'latent_dim': self.latent_dim,
        }
